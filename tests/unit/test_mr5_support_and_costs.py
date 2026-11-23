"""Item MR5: CfD volume collars and by-product cost adjustments."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.market.revenue_tracking import compute_cfd_payments


def _network(dispatch_mw: float, hours: int = 24):
    snapshots = pd.date_range("2030-01-01", periods=hours, freq="h")
    generators = pd.DataFrame({
        "support_type": ["CfD", "CfD"], "carrier": ["biomass", "wind_offshore"],
        "bus": ["DRAX41", "X"], "cfd_round": ["DRAX_BRIDGE", "AR4"],
    }, index=["Drax", "Wind"])
    p = pd.DataFrame({"Drax": dispatch_mw, "Wind": 100.0}, index=snapshots)
    return SimpleNamespace(generators=generators, snapshots=snapshots,
                           generators_t=SimpleNamespace(p=p),
                           snapshot_weightings=pd.DataFrame({"generators": 1.0}, index=snapshots))


def test_collar_scales_only_the_collared_round_to_its_prorated_volume():
    network = _network(dispatch_mw=1000.0)
    strikes = {"DRAX_BRIDGE": 150.0, "AR4": 60.0}
    price = pd.Series(50.0, index=network.snapshots)
    uncapped = compute_cfd_payments(network, strikes, reference_price=price).set_index("generator")
    # 24 h at 1,000 MW is 24,000 MWh; a 4.38 TWh annual collar allows 12,000 MWh in 24 h.
    capped = compute_cfd_payments(network, strikes, reference_price=price,
                                  volume_caps={"DRAX_BRIDGE": 4.38e6}).set_index("generator")
    assert capped.loc["Drax", "volume_cap_factor"] == pytest.approx(0.5)
    assert capped.loc["Drax", "net_cfd_cost"] == pytest.approx(0.5 * uncapped.loc["Drax", "net_cfd_cost"])
    assert capped.loc["Wind", "net_cfd_cost"] == pytest.approx(uncapped.loc["Wind", "net_cfd_cost"])
    assert capped.loc["Wind", "volume_cap_factor"] == 1.0


def test_collar_does_not_bind_below_the_allowed_volume():
    network = _network(dispatch_mw=100.0)
    strikes = {"DRAX_BRIDGE": 150.0, "AR4": 60.0}
    price = pd.Series(50.0, index=network.snapshots)
    capped = compute_cfd_payments(network, strikes, reference_price=price,
                                  volume_caps={"DRAX_BRIDGE": 4.38e6}).set_index("generator")
    assert capped.loc["Drax", "volume_cap_factor"] == 1.0


def _config(adjustments):
    return {
        "modelled_year": 2030,
        "support": {
            "dispatch_costs": {"enabled": True, "cfd_dispatch_cost": 0.0,
                               "cfd_dispatch_cost_exempt_carriers": ["biomass"]},
        },
        "marginal_costs": {
            "byproduct_adjustments": adjustments,
            "future_costs": {
                "enabled": True, "case": "central",
                "efficiency_defaults": {"CHP": 0.42, "waste_to_energy": 0.25, "biomass": 0.35},
                "cases": {"central": {"fuel_prices": {"gas": 30.0, "waste": 15.0, "biomass": 45.0},
                                      "carbon": {"uk_ets": 50.0, "carbon_price_support": 18.0},
                                      "variable_om": {"waste_to_energy": 4.0}}},
            },
        },
    }


def _apply(config, carrier, support="merchant"):
    import pypsa
    from scripts.generators import apply_marginal_costs as amc
    n = pypsa.Network()
    n.add("Bus", "B")
    n.add("Generator", "G", bus="B", carrier=carrier, p_nom=100.0, efficiency=0.5)
    n.generators["support_type"] = support
    assumptions = amc.get_future_cost_assumptions(config)
    fuel = amc._expand_fuel_prices(assumptions["fuel_prices"])
    carbon = sum(amc._future_carbon_components(assumptions))
    df = amc.apply_marginal_costs_to_network(n, fuel, carbon, scenario_config=config)
    return float(df.set_index("generator").loc["G", "marginal_cost"]), fuel, carbon


def test_energy_from_waste_electricity_costs_vom_only():
    config = _config({"waste_to_energy": {"mode": "gate_fee_covers_fuel_and_carbon"}})
    mc, _, _ = _apply(config, "waste_to_energy")
    assert mc == pytest.approx(4.0)


def test_chp_heat_credit_is_heat_at_boiler_cost():
    plain, fuel, carbon = _apply(_config({}), "CHP")
    config = _config({"CHP": {"mode": "heat_credit", "heat_fuel": "gas",
                              "heat_to_power": 1.1, "boiler_efficiency": 0.85}})
    credited, _, _ = _apply(config, "CHP")
    expected_credit = 1.1 * (fuel["gas"] + 183.0 / 1000.0 * carbon) / 0.85
    assert plain - credited == pytest.approx(expected_credit)


def test_cfd_biomass_exempt_from_the_zero_dispatch_cost_keeps_its_fuel_cost():
    mc, _, _ = _apply(_config({}), "biomass", support="CfD")
    assert mc == pytest.approx(45.0 / 0.35)
