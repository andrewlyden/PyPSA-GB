"""Large hydro with its power ceiling apart from inflow, on an annual path (OW3, 3 Oct 2026).

``renewables.hydro.large_hydro.power_ceiling: availability`` keeps the inflow the reservoir
model read (p_max_pu x inflow_profile_factor) in its own series and lifts p_max_pu
to the availability; ``month_end: trajectory`` starts and ends each run on an
annual path built from the inflow. Both are build-time (finalize_network) and off
by default; the CP30 2030 family turns them on.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.CP30.energy_validation import _generator_energy, network_energy_tables
from scripts.renewables.large_hydro_reservoir import (
    attach_large_hydro_trajectory,
    large_hydro_release_path,
    large_hydro_trajectory,
    prepare_large_hydro_reservoir,
    separate_large_hydro_ceiling,
)
from scripts.solve.hydro_constraints import (
    LARGE_HYDRO_INFLOW_SERIES,
    LARGE_HYDRO_TRAJECTORY_END,
    LARGE_HYDRO_TRAJECTORY_START,
    build_hydro_constraints_callback,
    initialise_large_hydro_storage_state,
    large_hydro_end_floor,
    large_hydro_inflow_pu,
    large_hydro_net_inflow,
    resolve_large_hydro_ceiling,
    resolve_large_hydro_month_end,
    resolve_large_hydro_trajectory,
    update_large_hydro_storage_state,
)

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("test_large_hydro_reservoir")
P_NOM = 100.0
FACTOR = 0.55
AVAILABILITY = 0.906


def _hydro_config(ceiling="inflow_scaled", month_end="at_least_initial", trajectory=None, availability=AVAILABILITY):
    large = {"reservoir_model": True, "inflow_profile_factor": FACTOR, "initial_storage_hours": 2.0,
             "reservoir_capacity_hours": 168.0, "month_end": month_end, "power_ceiling": ceiling}
    if ceiling == "availability":
        large["availability_pu"] = availability
    if trajectory is not None:
        large["trajectory"] = trajectory
    return {"enabled": True, "large_hydro": large}


def _inflow_pu(sns):
    """Monthly-step inflow CF (0.10 in the first month, 0.30 in the second), as the CP30 proxy."""
    first_month = sns.month == sns[0].month
    return pd.Series(np.where(first_month, 0.10, 0.30), index=sns)


def _network(hours=24 * 59, start="2030-01-01"):
    """Two months: one hydro unit in the old convention (p_max_pu = inflow / 0.55); a 150 MW CCGT,
    so the 250 MW evening peak (17:00-20:00) needs the 300 GBP/MWh peaker."""
    n = pypsa.Network()
    sns = pd.date_range(start, periods=hours, freq="h")
    n.set_snapshots(sns)
    n.add("Carrier", ["large_hydro", "CCGT", "OCGT"])
    n.add("Bus", "A", carrier="AC")
    n.add("Generator", "hydro", bus="A", carrier="large_hydro", p_nom=P_NOM, marginal_cost=0.5,
          p_max_pu=_inflow_pu(sns) / FACTOR)
    n.add("Generator", "ccgt", bus="A", carrier="CCGT", p_nom=150.0, marginal_cost=50.0)
    n.add("Generator", "peaker", bus="A", carrier="OCGT", p_nom=500.0, marginal_cost=300.0)
    peak = pd.Series(np.where(sns.hour.isin([17, 18, 19, 20]), 250.0, 150.0), index=sns)
    n.add("Load", "load", bus="A", p_set=peak)
    return n


def _energy(frame, weights):
    return float(frame.mul(weights, axis=0).to_numpy().sum())


# ── Config ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestConfig:

    def test_defaults_are_todays_behaviour(self):
        assert resolve_large_hydro_ceiling({}) == ("inflow_scaled", None)
        assert resolve_large_hydro_month_end({}) == "at_least_initial"
        assert resolve_large_hydro_trajectory({}) is None

    def test_bad_values_are_refused(self):
        with pytest.raises(ValueError, match="power_ceiling"):
            resolve_large_hydro_ceiling({"large_hydro": {"power_ceiling": "flat"}})
        with pytest.raises(ValueError, match="availability_pu"):
            resolve_large_hydro_ceiling(_hydro_config("availability", availability=None))
        with pytest.raises(ValueError, match="availability_pu"):
            resolve_large_hydro_ceiling(_hydro_config("availability", availability=1.2))
        no_reservoir = _hydro_config("availability")
        no_reservoir["large_hydro"]["reservoir_model"] = False
        with pytest.raises(ValueError, match="reservoir_model"):
            resolve_large_hydro_ceiling(no_reservoir)
        with pytest.raises(ValueError, match="needs trajectory"):
            resolve_large_hydro_trajectory(_hydro_config(month_end="trajectory", trajectory={"mean_level_hours": 84}))
        with pytest.raises(ValueError, match="unknown keys"):
            resolve_large_hydro_trajectory(_hydro_config(month_end="trajectory", trajectory={
                "mean_level_hours": 84, "smoothing_days": 30, "band": 1}))
        with pytest.raises(ValueError, match="mean_level_hours"):
            resolve_large_hydro_trajectory(_hydro_config(month_end="trajectory", trajectory={
                "mean_level_hours": 200, "smoothing_days": 30}))

    def test_only_the_cp30_2030_family_turns_it_on(self):
        sys.path.insert(0, str(ROOT / "config"))
        from config_loader import load_config

        scenarios = load_config(include_archive=True)["scenarios"]   # the year-network probe is archived
        for sid in ("CP30_FFR_2030",
                    "CP30_FFR_2030_Jan_cp30_network",
                    "CP30_FFR_2030_Jan_cp30_network_central_dispatch",
                    "CP30_FFR_2030_ETYS_market_1week_market_yearnet"):
            hydro = scenarios[sid]["renewables"]["hydro"]
            assert resolve_large_hydro_ceiling(hydro) == ("availability", pytest.approx(0.906))
            assert resolve_large_hydro_trajectory(hydro) == {
                "mean_level_hours": 84.0, "smoothing_days": 30.0, "reservoir_capacity_hours": 168.0}
        for sid in ("FES25_HT_2050_Jan_market", "Historical_2020_ETYS_market", "CP30_FFR_2030_ETYS_market_1week_neso_comparator"):
            hydro = scenarios[sid]["renewables"]["hydro"]
            assert resolve_large_hydro_ceiling(hydro) == ("inflow_scaled", None)
            assert resolve_large_hydro_month_end(hydro) != "trajectory"


# ── Build: the ceiling apart from the inflow ──────────────────────────────

@pytest.mark.unit
class TestSeparateCeiling:

    def test_default_is_a_no_op(self):
        n = _network()
        before = n.generators_t.p_max_pu.copy()
        summary = prepare_large_hydro_reservoir(n, _hydro_config(), LOG)
        assert summary["ceiling"]["units"] == 0 and summary["trajectory"] == {"trajectory": False}
        pd.testing.assert_frame_equal(n.generators_t.p_max_pu, before)
        assert LARGE_HYDRO_INFLOW_SERIES not in n.generators_t

    def test_inflow_energy_is_kept_and_the_ceiling_is_the_availability(self):
        n = _network()
        cfg = _hydro_config("availability")
        w = n.snapshot_weightings.generators
        old_inflow = large_hydro_inflow_pu(n, n.snapshots, cfg, ["hydro"])
        np.testing.assert_allclose(old_inflow["hydro"].to_numpy(), _inflow_pu(n.snapshots).to_numpy())
        summary = separate_large_hydro_ceiling(n, cfg, LOG)
        assert summary["units"] == 1
        assert summary["inflow_mwh_after"] == pytest.approx(summary["inflow_mwh_before"])
        assert "hydro" not in n.generators_t.p_max_pu.columns
        assert float(n.generators.at["hydro", "p_max_pu"]) == pytest.approx(AVAILABILITY)
        new_inflow = large_hydro_inflow_pu(n, n.snapshots, cfg, ["hydro"])
        np.testing.assert_allclose(new_inflow["hydro"].to_numpy(), old_inflow["hydro"].to_numpy())
        assert _energy(new_inflow * P_NOM, w) == pytest.approx(_energy(old_inflow * P_NOM, w))
        # Idempotent: a second pass changes nothing.
        separate_large_hydro_ceiling(n, cfg, LOG)
        np.testing.assert_allclose(large_hydro_inflow_pu(n, n.snapshots, cfg, ["hydro"])["hydro"].to_numpy(),
                                   old_inflow["hydro"].to_numpy())

    def test_series_survive_netcdf_copy_and_slicing(self, tmp_path):
        n = _network(hours=48)
        cfg = _hydro_config("availability", "trajectory", {"mean_level_hours": 10.0, "smoothing_days": 0.5})
        prepare_large_hydro_reservoir(n, cfg, LOG)
        path = tmp_path / "net.nc"
        n.export_to_netcdf(path)
        m = pypsa.Network(path)
        for attr in (LARGE_HYDRO_INFLOW_SERIES, LARGE_HYDRO_TRAJECTORY_START, LARGE_HYDRO_TRAJECTORY_END):
            np.testing.assert_allclose(m.generators_t[attr]["hydro"].to_numpy(), n.generators_t[attr]["hydro"].to_numpy())
        window = m.copy()
        window.set_snapshots(m.snapshots[24:])
        np.testing.assert_allclose(window.generators_t[LARGE_HYDRO_INFLOW_SERIES]["hydro"].to_numpy(),
                                   n.generators_t[LARGE_HYDRO_INFLOW_SERIES]["hydro"].to_numpy()[24:])

    def test_energy_validation_reads_the_inflow_not_the_ceiling(self):
        n = _network()
        cfg = _hydro_config("availability")
        before = network_energy_tables(n, cfg)["generators"].at["hydro", "available_mwh"]
        separate_large_hydro_ceiling(n, cfg, LOG)
        after = network_energy_tables(n, cfg)["generators"].at["hydro", "available_mwh"]
        assert after == pytest.approx(before)
        # Without the series the availability ceiling would have been taken for energy.
        static = n.generators.assign(carrier=n.generators.carrier.astype(str))
        w = n.snapshot_weightings.generators
        ceiling = _generator_energy(static, n.generators_t.p_max_pu, None, w, None).at["hydro", "available_mwh"]
        assert ceiling == pytest.approx(AVAILABILITY * P_NOM * len(n.snapshots))


# ── Build: the annual path ────────────────────────────────────────────────

@pytest.mark.unit
class TestTrajectory:

    def test_release_is_a_cyclic_centred_mean_that_releases_the_inflow(self):
        sns = pd.date_range("2030-01-01", periods=24 * 59, freq="h")
        inflow = _inflow_pu(sns).to_frame("hydro")
        release = large_hydro_release_path(inflow, 24 * 10)
        assert float(release.sum().iloc[0]) == pytest.approx(float(inflow.sum().iloc[0]))
        # The step at the month boundary is released across it, not at it.
        boundary = int(np.argmax(sns.month != sns[0].month))
        assert 0.10 < float(release.iloc[boundary - 1, 0]) < 0.30
        assert 0.10 < float(release.iloc[boundary, 0]) < 0.30
        pd.testing.assert_frame_equal(large_hydro_release_path(inflow, 0), inflow.astype(float))

    def test_levels_chain_average_the_mean_and_close_the_year(self):
        sns = pd.date_range("2030-01-01", periods=24 * 59, freq="h")
        inflow = _inflow_pu(sns).to_frame("hydro")
        before, after = large_hydro_trajectory(inflow, pd.Series(1.0, index=sns), 84.0, 10.0, 168.0)
        np.testing.assert_allclose(before["hydro"].to_numpy()[1:], after["hydro"].to_numpy()[:-1])
        assert float(after["hydro"].mean()) == pytest.approx(84.0)
        # Cyclic: the year ends where it starts, so the months together release the inflow.
        assert float(after["hydro"].iloc[-1]) == pytest.approx(float(before["hydro"].iloc[0]))
        # The store falls ahead of the wet month and rises into it.
        boundary = int(np.argmax(sns.month != sns[0].month))
        assert float(after["hydro"].iloc[boundary - 1]) < 84.0
        flat_before, flat_after = large_hydro_trajectory(inflow, pd.Series(1.0, index=sns), 84.0, 0.0, 168.0)
        np.testing.assert_allclose(flat_after["hydro"].to_numpy(), 84.0)
        np.testing.assert_allclose(flat_before["hydro"].to_numpy(), 84.0)

    def test_month_boundaries_agree_on_the_shared_network(self):
        """Month 1 must end where month 2 starts; each run reads its own slice of the built path."""
        n = _network()
        cfg = _hydro_config("availability", "trajectory", {"mean_level_hours": 84.0, "smoothing_days": 10.0})
        prepare_large_hydro_reservoir(n, cfg, LOG)
        months = [n.snapshots[n.snapshots.month == m] for m in (1, 2)]
        ends, starts = [], []
        for sns in months:
            run = n.copy()
            run.set_snapshots(sns)
            starts.append(float(initialise_large_hydro_storage_state(run, cfg)["hydro"]))
            net = large_hydro_net_inflow(run, cfg)
            ends.append(float(large_hydro_end_floor(net, None, sns[-1])["hydro"]))
        assert ends[0] == pytest.approx(starts[1])
        assert starts[0] == pytest.approx(ends[1])        # the built year is cyclic
        assert starts[0] != pytest.approx(84.0 * P_NOM)   # not a fixed start
        # Without the built series the path is flat at mean_level_hours.
        flat = _network()
        assert float(initialise_large_hydro_storage_state(flat, cfg)["hydro"]) == pytest.approx(84.0 * P_NOM)


# ── Solve ─────────────────────────────────────────────────────────────────

def _solve_rolling(n, cfg):
    """Daily windows with the reservoir carried, as the market stages run them."""
    config = {"renewables": {"hydro": cfg}}
    state = initialise_large_hydro_storage_state(n, cfg)
    start = state.copy()
    net = large_hydro_net_inflow(n, cfg)
    dispatch = []
    for day in range(len(n.snapshots) // 24):
        window = n.snapshots[day * 24:(day + 1) * 24]
        w = n.copy()
        w.set_snapshots(window)
        callback = build_hydro_constraints_callback(
            w, config, large_hydro_storage_state=state,
            large_hydro_end_storage=large_hydro_end_floor(net, start, window[-1]),
        )
        status, condition = w.optimize(solver_name="highs", extra_functionality=callback)
        assert status == "ok", condition
        dispatch.append(w.generators_t.p["hydro"].copy())
        state = update_large_hydro_storage_state(w, window, config, state)
    return pd.concat(dispatch), start, state


@pytest.mark.unit
class TestSolve:

    def test_availability_ceiling_moves_the_water_to_the_peak_and_keeps_the_energy(self):
        hours = 24 * 4
        old = _network(hours=hours)
        old_cfg = _hydro_config()
        old_p, _, _ = _solve_rolling(old, old_cfg)
        new = _network(hours=hours)
        new_cfg = _hydro_config("availability")
        separate_large_hydro_ceiling(new, new_cfg, LOG)
        new_p, _, _ = _solve_rolling(new, new_cfg)
        inflow = float(_inflow_pu(new.snapshots).sum() * P_NOM)
        # Gas is always dearer than hydro, so both use exactly their inflow.
        assert float(old_p.sum()) == pytest.approx(inflow, rel=1e-6)
        assert float(new_p.sum()) == pytest.approx(inflow, rel=1e-6)
        peak = new_p.index.hour.isin([17, 18, 19, 20])
        # Old ceiling 0.10 / 0.55 = 18.2 MW; the peak now takes 90.6 MW.
        assert float(old_p.max()) == pytest.approx(0.10 / FACTOR * P_NOM, rel=1e-6)
        assert float(new_p[peak].max()) == pytest.approx(AVAILABILITY * P_NOM, rel=1e-6)
        assert float(new_p[peak].sum()) > float(old_p[peak].sum()) + 100.0

    def test_trajectory_run_starts_and_ends_on_the_path(self):
        n = _network(hours=24 * 59)
        cfg = _hydro_config("availability", "trajectory", {"mean_level_hours": 84.0, "smoothing_days": 10.0})
        prepare_large_hydro_reservoir(n, cfg, LOG)
        run = n.copy()
        run.set_snapshots(n.snapshots[:24 * 4])   # four days of the dry month
        p, start, end_state = _solve_rolling(run, cfg)
        path_end = float(n.generators_t[LARGE_HYDRO_TRAJECTORY_END]["hydro"].iloc[24 * 4 - 1] * P_NOM)
        inflow = float(n.generators_t[LARGE_HYDRO_INFLOW_SERIES]["hydro"].iloc[:24 * 4].sum() * P_NOM)
        # Dear gas: the run spends down to the path's end level, and no further.
        assert float(end_state["hydro"]) == pytest.approx(path_end, rel=1e-6)
        assert float(p.sum()) == pytest.approx(float(start["hydro"]) + inflow - path_end, rel=1e-6)


# ── The annual pass (decision 33) aggregates the same inflow ──────────────

@pytest.mark.unit
def test_seasonal_pass_hydro_inflow_is_the_same_after_separation():
    from scripts.market.seasonal_storage import aggregate_for_seasonal_pass, resolve_seasonal_pass_config

    def inflow_of(net, cfg):
        agg, report, _ = aggregate_for_seasonal_pass(net, {"renewables": {"hydro": cfg}}, resolve_seasonal_pass_config({}), LOG)
        hydro = agg.storage_units.index[agg.storage_units.carrier == "large_hydro"]
        return agg.storage_units_t.inflow[hydro].sum(axis=1), agg.storage_units.loc[hydro, "p_max_pu"]

    old = _network(hours=48)
    old_inflow, old_ceiling = inflow_of(old, _hydro_config())
    new = _network(hours=48)
    cfg = _hydro_config("availability")
    separate_large_hydro_ceiling(new, cfg, LOG)
    new_inflow, _ = inflow_of(new, cfg)
    np.testing.assert_allclose(new_inflow.to_numpy(), old_inflow.to_numpy())
    np.testing.assert_allclose(old_inflow.to_numpy(), _inflow_pu(old.snapshots).to_numpy() * P_NOM)
