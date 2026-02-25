"""
Unit tests for renewable generator aggregation by (bus, carrier).

Tests verify:
  - Capacity conservation (p_nom summed exactly)
  - Energy conservation (capacity-weighted average profile)
  - Single-generator groups pass through unchanged
  - Mixed carriers at same bus aggregated independently
  - Non-renewable generators completely untouched
  - Generators without p_max_pu handled gracefully
  - Empty carrier list → no aggregation
  - Edge cases (zero capacity, empty network)
"""

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.generators.aggregate_renewable_generators import (
    aggregate_renewables_by_bus,
    DEFAULT_RENEWABLE_CARRIERS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def snapshots():
    """24-hour snapshot index."""
    return pd.date_range("2035-01-01", periods=24, freq="h")


@pytest.fixture
def network_with_renewables(snapshots):
    """
    Network with 3 buses and several renewable + thermal generators.

    Bus layout:
      BUS_A: 2 wind_onshore (different profiles), 1 solar_pv, 1 CCGT
      BUS_B: 1 wind_onshore
      BUS_C: 2 solar_pv (different profiles)
    """
    n = pypsa.Network()
    n.set_snapshots(snapshots)

    # Buses
    for bus in ["BUS_A", "BUS_B", "BUS_C"]:
        n.add("Bus", bus)

    # Carrier definitions
    for carrier in ["wind_onshore", "solar_pv", "CCGT"]:
        n.add("Carrier", carrier)

    # --- BUS_A generators ---
    # Wind 1: 100 MW, CF profile oscillates 0.2-0.8
    n.add("Generator", "wind_a1", bus="BUS_A", carrier="wind_onshore",
           p_nom=100.0, marginal_cost=0.0, lat=55.0, lon=-3.0)
    # Wind 2: 50 MW, CF profile is constant 0.4
    n.add("Generator", "wind_a2", bus="BUS_A", carrier="wind_onshore",
           p_nom=50.0, marginal_cost=0.0, lat=56.0, lon=-2.5)
    # Solar: 80 MW
    n.add("Generator", "solar_a1", bus="BUS_A", carrier="solar_pv",
           p_nom=80.0, marginal_cost=0.0, lat=55.5, lon=-2.8)
    # CCGT: 200 MW (thermal — should NOT be aggregated)
    n.add("Generator", "ccgt_a1", bus="BUS_A", carrier="CCGT",
           p_nom=200.0, marginal_cost=50.0)

    # --- BUS_B generators ---
    # Single wind — should pass through unchanged
    n.add("Generator", "wind_b1", bus="BUS_B", carrier="wind_onshore",
           p_nom=120.0, marginal_cost=0.0, lat=57.0, lon=-4.0)

    # --- BUS_C generators ---
    # Two solar at same bus
    n.add("Generator", "solar_c1", bus="BUS_C", carrier="solar_pv",
           p_nom=60.0, marginal_cost=0.0, lat=52.0, lon=-1.0)
    n.add("Generator", "solar_c2", bus="BUS_C", carrier="solar_pv",
           p_nom=40.0, marginal_cost=0.0, lat=52.5, lon=-0.5)

    # --- Time series ---
    # wind_a1: oscillating profile
    cf_wind_a1 = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(24) / 24)
    # wind_a2: constant profile
    cf_wind_a2 = np.full(24, 0.4)
    # solar_a1: daytime bell curve
    cf_solar_a1 = np.clip(np.sin(np.pi * np.arange(24) / 24), 0, 1)
    # wind_b1: constant
    cf_wind_b1 = np.full(24, 0.35)
    # solar_c1: daytime
    cf_solar_c1 = np.clip(0.8 * np.sin(np.pi * np.arange(24) / 24), 0, 1)
    # solar_c2: slightly different daytime
    cf_solar_c2 = np.clip(0.6 * np.sin(np.pi * (np.arange(24) - 1) / 24), 0, 1)

    n.generators_t.p_max_pu = pd.DataFrame({
        "wind_a1": cf_wind_a1,
        "wind_a2": cf_wind_a2,
        "solar_a1": cf_solar_a1,
        "wind_b1": cf_wind_b1,
        "solar_c1": cf_solar_c1,
        "solar_c2": cf_solar_c2,
    }, index=snapshots)

    return n


@pytest.fixture
def empty_network(snapshots):
    """Network with buses but no generators."""
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.add("Bus", "BUS_X")
    return n


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCapacityConservation:
    """Total p_nom per carrier must be identical before and after aggregation."""

    def test_wind_capacity_conserved(self, network_with_renewables):
        before = network_with_renewables.generators.loc[
            network_with_renewables.generators.carrier == "wind_onshore", "p_nom"
        ].sum()

        aggregate_renewables_by_bus(network_with_renewables)

        after = network_with_renewables.generators.loc[
            network_with_renewables.generators.carrier == "wind_onshore", "p_nom"
        ].sum()

        assert abs(before - after) < 0.01, f"Wind capacity changed: {before} → {after}"

    def test_solar_capacity_conserved(self, network_with_renewables):
        before = network_with_renewables.generators.loc[
            network_with_renewables.generators.carrier == "solar_pv", "p_nom"
        ].sum()

        aggregate_renewables_by_bus(network_with_renewables)

        after = network_with_renewables.generators.loc[
            network_with_renewables.generators.carrier == "solar_pv", "p_nom"
        ].sum()

        assert abs(before - after) < 0.01, f"Solar capacity changed: {before} → {after}"

    def test_total_capacity_conserved(self, network_with_renewables):
        """Total p_nom across ALL generators (including thermal) unchanged."""
        before = network_with_renewables.generators["p_nom"].sum()

        aggregate_renewables_by_bus(network_with_renewables)

        after = network_with_renewables.generators["p_nom"].sum()

        assert abs(before - after) < 0.01, f"Total capacity changed: {before} → {after}"


class TestWeightedAverageProfile:
    """Aggregated p_max_pu must equal capacity-weighted average."""

    def test_wind_bus_a_profile(self, network_with_renewables):
        """Two wind generators at BUS_A (100 MW + 50 MW) → weighted average."""
        n = network_with_renewables
        # Pre-compute expected weighted average
        p1, p2 = 100.0, 50.0
        cf1 = n.generators_t.p_max_pu["wind_a1"].values.copy()
        cf2 = n.generators_t.p_max_pu["wind_a2"].values.copy()
        expected = (p1 * cf1 + p2 * cf2) / (p1 + p2)

        aggregate_renewables_by_bus(n)

        # Find the aggregated generator
        agg_gens = n.generators[
            (n.generators.carrier == "wind_onshore") &
            (n.generators.bus == "BUS_A")
        ]
        assert len(agg_gens) == 1, f"Expected 1 aggregated wind at BUS_A, got {len(agg_gens)}"

        agg_name = agg_gens.index[0]
        actual = n.generators_t.p_max_pu[agg_name].values

        np.testing.assert_allclose(actual, expected, atol=1e-10,
                                   err_msg="Weighted average profile does not match")

    def test_solar_bus_c_profile(self, network_with_renewables):
        """Two solar generators at BUS_C (60 MW + 40 MW) → weighted average."""
        n = network_with_renewables
        p1, p2 = 60.0, 40.0
        cf1 = n.generators_t.p_max_pu["solar_c1"].values.copy()
        cf2 = n.generators_t.p_max_pu["solar_c2"].values.copy()
        expected = (p1 * cf1 + p2 * cf2) / (p1 + p2)

        aggregate_renewables_by_bus(n)

        agg_gens = n.generators[
            (n.generators.carrier == "solar_pv") &
            (n.generators.bus == "BUS_C")
        ]
        assert len(agg_gens) == 1
        actual = n.generators_t.p_max_pu[agg_gens.index[0]].values

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_energy_conservation(self, network_with_renewables):
        """Total expected energy must be conserved for every timestep.

        sum(p_nom_i * p_max_pu_i(t)) must equal p_nom_agg * p_max_pu_agg(t)
        for each timestep t.
        """
        n = network_with_renewables
        snapshots = n.snapshots

        # Compute total renewable energy per timestep BEFORE aggregation
        renewable_carriers = {"wind_onshore", "solar_pv"}
        ren_gens = n.generators[n.generators.carrier.isin(renewable_carriers)]
        energy_before = pd.Series(0.0, index=snapshots)
        for gen_name in ren_gens.index:
            p_nom = ren_gens.at[gen_name, "p_nom"]
            if gen_name in n.generators_t.p_max_pu.columns:
                energy_before += p_nom * n.generators_t.p_max_pu[gen_name]
            else:
                energy_before += p_nom  # default p_max_pu = 1.0

        aggregate_renewables_by_bus(n)

        # Compute total renewable energy per timestep AFTER aggregation
        ren_gens_after = n.generators[n.generators.carrier.isin(renewable_carriers)]
        energy_after = pd.Series(0.0, index=snapshots)
        for gen_name in ren_gens_after.index:
            p_nom = ren_gens_after.at[gen_name, "p_nom"]
            if gen_name in n.generators_t.p_max_pu.columns:
                energy_after += p_nom * n.generators_t.p_max_pu[gen_name]
            else:
                energy_after += p_nom

        np.testing.assert_allclose(
            energy_before.values, energy_after.values, atol=1e-8,
            err_msg="Total renewable energy not conserved after aggregation"
        )


class TestSingleGeneratorGroups:
    """Single-member groups should pass through unchanged."""

    def test_single_wind_at_bus_b_unchanged(self, network_with_renewables):
        n = network_with_renewables
        original_profile = n.generators_t.p_max_pu["wind_b1"].values.copy()
        original_p_nom = n.generators.at["wind_b1", "p_nom"]

        aggregate_renewables_by_bus(n)

        # Name should be preserved (no __agg suffix)
        assert "wind_b1" in n.generators.index, "Single generator was renamed or removed"
        assert n.generators.at["wind_b1", "p_nom"] == original_p_nom
        np.testing.assert_array_equal(
            n.generators_t.p_max_pu["wind_b1"].values, original_profile
        )

    def test_single_solar_at_bus_a_unchanged(self, network_with_renewables):
        n = network_with_renewables
        original_p_nom = n.generators.at["solar_a1", "p_nom"]

        aggregate_renewables_by_bus(n)

        # solar_a1 is the only solar at BUS_A → kept unchanged
        assert "solar_a1" in n.generators.index
        assert n.generators.at["solar_a1", "p_nom"] == original_p_nom


class TestNonRenewablesUntouched:
    """Thermal generators must not be affected by renewable aggregation."""

    def test_ccgt_preserved(self, network_with_renewables):
        n = network_with_renewables
        original_ccgt = n.generators.loc["ccgt_a1"].copy()

        aggregate_renewables_by_bus(n)

        assert "ccgt_a1" in n.generators.index, "CCGT generator was removed"
        assert n.generators.at["ccgt_a1", "p_nom"] == original_ccgt["p_nom"]
        assert n.generators.at["ccgt_a1", "marginal_cost"] == original_ccgt["marginal_cost"]
        assert n.generators.at["ccgt_a1", "carrier"] == "CCGT"

    def test_generator_count_reduction(self, network_with_renewables):
        """Should reduce from 7 to 5 generators.

        BUS_A: wind_a1+wind_a2 → 1, solar_a1 stays, ccgt_a1 stays = 3
        BUS_B: wind_b1 stays = 1
        BUS_C: solar_c1+solar_c2 → 1 = 1
        Total: 5
        """
        n = network_with_renewables
        assert len(n.generators) == 7

        _, removed = aggregate_renewables_by_bus(n)

        assert len(n.generators) == 5
        assert removed == 2


class TestGeneratorsWithoutTimeSeries:
    """Generators missing from p_max_pu should be treated as having constant 1.0."""

    def test_missing_profile_uses_default(self, snapshots):
        n = pypsa.Network()
        n.set_snapshots(snapshots)
        n.add("Bus", "BUS_X")
        n.add("Carrier", "wind_onshore")

        # Add two generators, only one has a profile
        n.add("Generator", "wind_x1", bus="BUS_X", carrier="wind_onshore", p_nom=100.0)
        n.add("Generator", "wind_x2", bus="BUS_X", carrier="wind_onshore", p_nom=100.0)

        # Only wind_x1 gets a profile
        n.generators_t.p_max_pu = pd.DataFrame({
            "wind_x1": np.full(24, 0.5),
        }, index=snapshots)

        aggregate_renewables_by_bus(n)

        agg_gens = n.generators[n.generators.carrier == "wind_onshore"]
        assert len(agg_gens) == 1
        agg_name = agg_gens.index[0]

        # Expected: (100 * 0.5 + 100 * 1.0) / 200 = 0.75
        expected = np.full(24, 0.75)
        actual = n.generators_t.p_max_pu[agg_name].values
        np.testing.assert_allclose(actual, expected, atol=1e-10)


class TestEmptyAndEdgeCases:
    """Edge cases: empty network, no matching carriers, empty carrier list."""

    def test_empty_network(self, empty_network):
        _, removed = aggregate_renewables_by_bus(empty_network)
        assert removed == 0

    def test_no_matching_carriers(self, network_with_renewables):
        """If only non-existent carriers specified, nothing happens."""
        n = network_with_renewables
        original_count = len(n.generators)

        _, removed = aggregate_renewables_by_bus(n, carriers=["geothermal", "nuclear"])

        assert removed == 0
        assert len(n.generators) == original_count

    def test_empty_carrier_list(self, network_with_renewables):
        """Empty carrier list means nothing is eligible."""
        n = network_with_renewables
        original_count = len(n.generators)

        _, removed = aggregate_renewables_by_bus(n, carriers=[])

        assert removed == 0
        assert len(n.generators) == original_count

    def test_selective_carriers(self, network_with_renewables):
        """Only aggregate wind, leave solar as individual sites."""
        n = network_with_renewables

        _, removed = aggregate_renewables_by_bus(n, carriers=["wind_onshore"])

        # Wind at BUS_A: 2 → 1 (removing 1)
        # Wind at BUS_B: 1 → 1 (no change)
        # Solar: untouched
        assert removed == 1

        # Solar generators should still be individual
        solar_gens = n.generators[n.generators.carrier == "solar_pv"]
        assert len(solar_gens) == 3  # solar_a1, solar_c1, solar_c2


class TestAggregatedNaming:
    """Aggregated generators should have informative names."""

    def test_aggregated_name_format(self, network_with_renewables):
        aggregate_renewables_by_bus(network_with_renewables)
        gen_names = list(network_with_renewables.generators.index)

        # Should contain __agg suffix for merged groups
        agg_names = [n for n in gen_names if "__agg" in n]
        assert len(agg_names) == 2  # wind at BUS_A, solar at BUS_C

        for name in agg_names:
            assert "__agg" in name
            # Format: {bus}_{carrier}__agg{count}
            parts = name.split("__agg")
            assert len(parts) == 2
            assert parts[1].isdigit()


class TestCoordinates:
    """Lat/lon should be capacity-weighted centroids."""

    def test_weighted_centroid(self, network_with_renewables):
        n = network_with_renewables
        # wind_a1: 100 MW at (55.0, -3.0), wind_a2: 50 MW at (56.0, -2.5)
        expected_lat = (100 * 55.0 + 50 * 56.0) / 150
        expected_lon = (100 * (-3.0) + 50 * (-2.5)) / 150

        aggregate_renewables_by_bus(n)

        agg_wind = n.generators[
            (n.generators.carrier == "wind_onshore") &
            (n.generators.bus == "BUS_A")
        ]
        assert len(agg_wind) == 1
        actual_lat = agg_wind.iloc[0]["lat"]
        actual_lon = agg_wind.iloc[0]["lon"]

        assert abs(actual_lat - expected_lat) < 1e-10
        assert abs(actual_lon - expected_lon) < 1e-10


class TestIdempotency:
    """Running aggregation twice should not change anything the second time."""

    def test_double_aggregation_is_noop(self, network_with_renewables):
        n = network_with_renewables

        _, removed_1 = aggregate_renewables_by_bus(n)
        count_after_first = len(n.generators)
        p_max_after_first = n.generators_t.p_max_pu.copy()

        _, removed_2 = aggregate_renewables_by_bus(n)
        count_after_second = len(n.generators)

        assert removed_1 > 0, "First aggregation should remove generators"
        assert removed_2 == 0, "Second aggregation should be a no-op"
        assert count_after_first == count_after_second

        # Profiles should be identical
        pd.testing.assert_frame_equal(
            p_max_after_first,
            n.generators_t.p_max_pu,
            check_names=False,
        )
