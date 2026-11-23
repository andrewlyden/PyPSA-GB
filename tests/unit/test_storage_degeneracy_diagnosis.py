"""Unit tests for the storage round-trip / degeneracy split (plan item B2).

The diagnostic exists because two very different things look identical in the
charge and discharge totals: a low configured round-trip efficiency, which is
the model doing what it was told, and simultaneous charge/discharge from an
interior-point solve, which is energy the model loses to the solver. These tests
build one network of each kind and check the script tells them apart.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market.diagnose_storage_degeneracy import (
    diagnose,
    diagnose_network,
    price_cluster,
    reconstruct_state_of_charge,
)
from scripts.storage.add_storage import (
    DEFAULT_STORAGE_ROUND_TRIP_EFFICIENCIES,
    _leg_efficiency,
)
from config.config_loader import load_defaults


def _write_network(path, net_dispatch, *, p_nom=100.0, max_hours=4.0,
                   leg_efficiency=0.92, soc_initial=200.0, carrier="Battery",
                   p_dispatch=None, p_store=None):
    """Save a solved-looking network carrying net storage dispatch, and the
    gross split too when ``p_dispatch``/``p_store`` are given (as the balancing
    solver now writes)."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=len(net_dispatch), freq="h"))
    n.add("Bus", "bus", v_nom=400)
    n.add("Carrier", carrier)
    n.add("StorageUnit", "unit", bus="bus", carrier=carrier, p_nom=p_nom,
          max_hours=max_hours, efficiency_store=leg_efficiency,
          efficiency_dispatch=leg_efficiency, state_of_charge_initial=soc_initial,
          standing_loss=0.0)
    n.storage_units_t.p = pd.DataFrame({"unit": net_dispatch}, index=n.snapshots)
    if p_dispatch is not None and p_store is not None:
        n.storage_units_t.p_dispatch = pd.DataFrame({"unit": p_dispatch}, index=n.snapshots)
        n.storage_units_t.p_store = pd.DataFrame({"unit": p_store}, index=n.snapshots)
    n.export_to_netcdf(str(path))
    return path


@pytest.mark.unit
class TestRoundTripVersusDegeneracy:

    def test_a_clean_cycle_reports_no_dissipation(self, tmp_path):
        """Charge then discharge within capacity: loss is exactly the round trip."""
        # Store 100 MWh of throughput for four hours, then discharge for four.
        dispatch = [-100.0] * 2 + [0.0] * 2 + [84.64] * 2 + [0.0] * 2
        path = _write_network(tmp_path / "clean.nc", dispatch, soc_initial=0.0)
        summary, per_carrier = diagnose_network(path)

        assert summary["charge_mwh"] == pytest.approx(200.0)
        assert summary["dissipation_lower_bound_mwh"] == pytest.approx(0.0, abs=1e-6)
        assert summary["units_exceeding_capacity"] == 0
        # The whole gap is the configured round trip, not a defect.
        assert summary["gap_explained_by_nominal_efficiency_mwh"] > 0
        assert summary["unexplained_gap_mwh"] == pytest.approx(
            summary["expected_discharge_at_nominal_mwh"] - summary["discharge_mwh"])
        assert per_carrier.loc[0, "nominal_roundtrip"] == pytest.approx(0.8464)

    def test_charging_beyond_capacity_is_flagged_as_dissipation(self, tmp_path):
        """Net p that implies more stored energy than the unit holds can only be
        simultaneous charge and discharge: the store is real, the discharge that
        cancelled it in net p is not visible, and the difference is burnt."""
        # 8 hours charging at 100 MW into a 400 MWh unit that starts full.
        dispatch = [-100.0] * 8
        path = _write_network(tmp_path / "degenerate.nc", dispatch,
                              p_nom=100.0, max_hours=4.0, soc_initial=400.0)
        summary, _ = diagnose_network(path)

        assert summary["units_exceeding_capacity"] == 1
        assert summary["dissipation_lower_bound_mwh"] > 0
        # 800 MWh charged at 0.92 into a unit already at its 400 MWh ceiling.
        assert summary["dissipation_lower_bound_mwh"] == pytest.approx(
            800.0 * 0.92, rel=1e-6)
        assert summary["worst_implied_soc_over_capacity_ratio"] > 1.0

    def test_a_low_round_trip_carrier_is_not_called_degeneracy(self, tmp_path):
        """The LAES case: 36% round trip looks catastrophic and is not a defect."""
        dispatch = [-100.0] * 4 + [36.0] * 4
        path = _write_network(tmp_path / "laes.nc", dispatch, carrier="LAES",
                              leg_efficiency=0.60, p_nom=100.0, max_hours=6.0,
                              soc_initial=0.0)
        summary, per_carrier = diagnose_network(path)

        assert per_carrier.loc[0, "implied_roundtrip"] == pytest.approx(0.36)
        assert per_carrier.loc[0, "nominal_roundtrip"] == pytest.approx(0.36)
        assert summary["dissipation_lower_bound_mwh"] == pytest.approx(0.0, abs=1e-6)

    def test_carry_over_is_reported_separately_from_loss(self, tmp_path):
        """A month that ends fuller than it started is not a loss."""
        dispatch = [-100.0] * 2 + [0.0] * 6
        path = _write_network(tmp_path / "carry.nc", dispatch, soc_initial=0.0)
        summary, _ = diagnose_network(path)
        assert summary["discharge_mwh"] == 0.0
        assert summary["carry_over_mwh"] == pytest.approx(200.0 * 0.92)
        assert summary["dissipation_lower_bound_mwh"] == pytest.approx(0.0, abs=1e-6)

    def test_totals_aggregate_across_networks(self, tmp_path):
        paths = [
            _write_network(tmp_path / "a.nc", [-100.0] * 2 + [84.64] * 2, soc_initial=0.0),
            _write_network(tmp_path / "b.nc", [-50.0] * 2 + [42.32] * 2, soc_initial=0.0),
        ]
        per_network, by_carrier, totals = diagnose(paths)
        assert len(per_network) == 2
        assert totals["charge_mwh"] == pytest.approx(300.0)
        assert by_carrier.loc[0, "carrier"] == "Battery"
        assert totals["implied_roundtrip"] == pytest.approx(0.8464, rel=1e-3)
        # net-only networks: the exact figures are honestly absent
        assert totals["networks_with_gross_split"] == 0
        assert totals["dissipation_exact_mwh"] is None

    def test_a_simultaneous_pair_is_invisible_in_net_p_but_exact_with_the_split(
            self, tmp_path):
        """50 MW stored and 50 MW dispatched in the same hour nets to zero, so
        the lower bound sees nothing; the gross split gives the loss exactly.
        (A real 10 MW discharge in hour 3 keeps net p from being all zeros,
        which PyPSA would drop from the export as equal to its default.)"""
        hours = 2
        path = _write_network(
            tmp_path / "pair.nc", [0.0] * hours + [10.0, 0.0],
            p_dispatch=[50.0] * hours + [10.0, 0.0], p_store=[50.0] * hours + [0.0, 0.0],
            soc_initial=200.0)
        summary, per_carrier = diagnose_network(path)

        assert summary["gross_split_available"] is True
        assert summary["dissipation_lower_bound_mwh"] == pytest.approx(0.0, abs=1e-6)
        assert summary["simultaneous_unit_hours"] == hours
        assert summary["dissipation_exact_mwh"] == pytest.approx(
            hours * 50.0 * (1.0 / 0.92 - 0.92))
        assert per_carrier.loc[0, "dissipation_exact_mwh"] == pytest.approx(
            summary["dissipation_exact_mwh"])

    def test_a_clean_split_reports_zero_exact_dissipation(self, tmp_path):
        path = _write_network(
            tmp_path / "clean_split.nc", [-100.0, 0.0, 84.64, 0.0],
            p_dispatch=[0.0, 0.0, 84.64, 0.0], p_store=[100.0, 0.0, 0.0, 0.0],
            soc_initial=0.0)
        summary, _ = diagnose_network(path)
        assert summary["dissipation_exact_mwh"] == pytest.approx(0.0, abs=1e-9)
        assert summary["simultaneous_unit_hours"] == 0
        _, _, totals = diagnose([path])
        assert totals["networks_with_gross_split"] == 1
        assert totals["dissipation_exact_mwh"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
class TestHelpers:

    def test_state_of_charge_follows_the_pypsa_recursion(self):
        index = pd.date_range("2030-01-01", periods=3, freq="h")
        dispatch = pd.DataFrame({"unit": [-10.0, 0.0, 5.0]}, index=index)
        hours = pd.Series(1.0, index=index)
        attributes = pd.DataFrame({
            "efficiency_store": [0.9], "efficiency_dispatch": [0.8],
            "state_of_charge_initial": [100.0], "standing_loss": [0.1],
        }, index=["unit"])
        soc = reconstruct_state_of_charge(dispatch, hours, attributes)["unit"]
        assert soc.iloc[0] == pytest.approx(100.0 * 0.9 + 10.0 * 0.9)
        assert soc.iloc[1] == pytest.approx(soc.iloc[0] * 0.9)
        assert soc.iloc[2] == pytest.approx(soc.iloc[1] * 0.9 - 5.0 / 0.8)

    def test_price_cluster_counts_hours_at_each_threshold(self, tmp_path):
        path = tmp_path / "price.csv"
        pd.DataFrame(
            {"wholesale_price": [0.0, 0.005, 0.9, 1.4, 50.0]},
            index=pd.date_range("2030-01-01", periods=5, freq="h"),
        ).to_csv(path, index_label="snapshot")
        cluster = price_cluster([path], thresholds=(0.0, 0.01, 1.0, 1.5))
        assert cluster["hours"] == 5
        assert cluster["hours_at_or_below"] == {"0": 1, "0.01": 2, "1": 3, "1.5": 4}


@pytest.mark.unit
class TestRoundTripConfiguration:

    def test_per_leg_efficiency_is_the_symmetric_split(self):
        for carrier, round_trip in DEFAULT_STORAGE_ROUND_TRIP_EFFICIENCIES.items():
            leg = _leg_efficiency(carrier, {}, None)
            assert leg * leg == pytest.approx(round_trip)

    def test_laes_matches_the_project_technology_table(self):
        """0.55, not the 0.36 the previous per-leg 0.60 produced."""
        assert DEFAULT_STORAGE_ROUND_TRIP_EFFICIENCIES["LAES"] == pytest.approx(0.55)
        assert _leg_efficiency("LAES", {}, None) == pytest.approx(np.sqrt(0.55))

    def test_config_overrides_the_default(self):
        assert _leg_efficiency("Battery", {"Battery": 0.64}, None) == pytest.approx(0.8)

    def test_an_unknown_carrier_is_refused(self):
        with pytest.raises(ValueError, match=r"storage.round_trip_efficiencies"):
            _leg_efficiency("Flywheel", {}, None)

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
    def test_an_impossible_round_trip_is_refused(self, value):
        with pytest.raises(ValueError, match="must be in"):
            _leg_efficiency("Battery", {"Battery": value}, None)

    def test_defaults_match_the_config_file(self):
        import yaml
        configured = load_defaults()["storage"]["round_trip_efficiencies"]
        assert configured == pytest.approx(DEFAULT_STORAGE_ROUND_TRIP_EFFICIENCIES)


# --------------------------------------------------------------------------
# B2's acceptance, as a standing check on the solved run rather than a one-off
# measurement. The second half of the original wording — "no unit's implied
# state of charge exceeds its capacity" — cannot be met exactly by a barrier
# solve without crossover, so it is expressed here as a bound on the energy
# involved, which is the quantity that actually matters to the bill.
# --------------------------------------------------------------------------

CP30_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
CP30_WHOLESALE = [
    f"resources/market/CP30_FFR_2030_ETYS_market_{month}2030"
    f"_neso_es5_landing_reassignment_reinforced_wholesale.nc"
    for month in CP30_MONTHS
]

# A strict lower bound on energy lost to simultaneous charge and discharge, as a
# share of charging. Measured at 0.26% on the twelve reinforced months; 2% leaves
# room for a re-solve to move it without letting the defect back in unnoticed.
# The pre-floor magnitude cannot be re-measured - those networks are superseded -
# so the bound comes from the measured residual plus headroom, not from it.
MAX_DISSIPATION_SHARE_OF_CHARGE = 0.02


@pytest.mark.unit
@pytest.mark.skipif(
    not all(Path(path).exists() for path in CP30_WHOLESALE),
    reason="the twelve solved CP30 wholesale networks are not present",
)
class TestSolvedRunStorageBalance:

    @pytest.fixture(scope="class")
    def totals(self):
        _, by_carrier, totals = diagnose(CP30_WHOLESALE)
        return by_carrier, totals

    def test_charge_and_discharge_balance_to_the_configured_round_trip(self, totals):
        """The first half of B2's acceptance, carrier by carrier."""
        by_carrier, _ = totals
        for row in by_carrier.itertuples():
            assert row.implied_roundtrip == pytest.approx(row.nominal_roundtrip, abs=0.03), (
                f"{row.carrier} delivers a {row.implied_roundtrip:.3f} round trip against a "
                f"configured {row.nominal_roundtrip:.3f}; the gap is storage energy the run "
                "cannot account for"
            )

    def test_the_gap_is_efficiency_and_carry_over_not_lost_energy(self, totals):
        _, t = totals
        # Charged minus discharged is almost entirely the configured efficiencies.
        assert t["gap_explained_by_nominal_efficiency_mwh"] / t["charge_discharge_gap_mwh"] > 0.9
        # Carry-over cannot be large: the whole fleet holds well under a TWh.
        assert abs(t["carry_over_mwh"]) < t["fleet_energy_capacity_mwh"] * len(CP30_MONTHS)

    def test_simultaneous_charge_and_discharge_stays_bounded(self, totals):
        """The honest form of "no unit's implied SoC exceeds its capacity".

        A barrier solve without crossover returns an interior point, so a small
        residual is a property of the solver rather than of the model. The bound
        is what separates that from a storage marginal cost slipping back to zero,
        which is what let the defect in originally.
        """
        _, t = totals
        assert t["dissipation_lower_bound_share_of_charge"] < MAX_DISSIPATION_SHARE_OF_CHARGE, (
            "storage is dissipating energy through simultaneous charge and discharge; check "
            "that the storage marginal cost is strictly positive before suspecting the model"
        )


CP30_BALANCING = [path.replace("_wholesale.nc", "_balancing.nc") for path in CP30_WHOLESALE]


@pytest.mark.unit
@pytest.mark.skipif(
    not all(Path(path).exists() for path in CP30_BALANCING),
    reason="the twelve solved CP30 balancing networks are not present",
)
class TestSolvedBalancingStorageBalance:
    """The Stage 2 twin of the check above.

    The BM objective replaces PyPSA's and, until gross p_dispatch was priced in
    it, left the charge/discharge split free: 3.62 TWh (21.5% of charging) was
    dissipated across the 2030 months while Stage 1 sat at 0.27%. Balancing
    networks solved since then also carry the gross split, so the exact figure
    is checked where it is available.
    """

    @pytest.fixture(scope="class")
    def totals(self):
        _, by_carrier, totals = diagnose(CP30_BALANCING)
        return by_carrier, totals

    def test_simultaneous_charge_and_discharge_stays_bounded(self, totals):
        _, t = totals
        assert t["dissipation_lower_bound_share_of_charge"] < MAX_DISSIPATION_SHARE_OF_CHARGE, (
            "the balancing stage is dissipating storage energy; the BM objective must price "
            "gross StorageUnit-p_dispatch (market.balancing.storage_marginal_cost_floor)"
        )

    def test_exact_dissipation_is_negligible_where_measurable(self, totals):
        _, t = totals
        if t["dissipation_exact_mwh"] is None:
            pytest.skip("these balancing networks predate the saved gross split")
        assert t["dissipation_exact_mwh"] / t["charge_mwh"] < MAX_DISSIPATION_SHARE_OF_CHARGE
