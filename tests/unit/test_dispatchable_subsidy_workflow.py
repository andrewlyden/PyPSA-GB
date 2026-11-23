import pandas as pd
import pypsa
import pytest
from pathlib import Path

from scripts.generators.apply_marginal_costs import apply_marginal_costs_to_network
from scripts.generators.integrate_renewable_generators import (
    _assign_subsidy_attributes,
    _apply_commissioning_date_overrides,
    _explicit_support_mask,
    _parse_operational_dates,
)
from scripts.generators.integrate_thermal_generators import apply_generator_support_overrides
from scripts.solve.solve_network import validate_network_costs


def _dispatchable_network():
    network = pypsa.Network()
    network.add("Bus", "bus")
    for name, support, banding in (
        ("cfd_biomass", "CfD", 0.0),
        ("roc_biomass", "ROC", 1.0),
        ("merchant_biomass", "merchant", 0.0),
    ):
        network.add(
            "Generator",
            name,
            bus="bus",
            carrier="biomass",
            p_nom=100.0,
            efficiency=0.5,
            support_type=support,
            ro_banding=banding,
        )
    return network


def test_explicit_support_mask_keeps_small_cfd_and_roc_sites():
    sites = pd.DataFrame(
        {
            "cfd_round": ["AR1", "", None, ""],
            "ro_banding": [None, 0.9, None, 0.0],
        }
    )
    assert _explicit_support_mask(sites).tolist() == [True, True, False, False]


def test_repd_zero_cfd_capacity_is_not_treated_as_an_award():
    sites = pd.DataFrame({
        "site_name": ["Unsuccessful bidder", "Awarded project"],
        "technology": ["Biomass (dedicated)", "Biomass (dedicated)"],
        "fuel_type": ["biomass", "biomass"],
        "capacity_mw": [1.5, 10.0],
        "operational_date": ["01/10/2014", "01/10/2014"],
        "cfd_round": ["AR2", "AR2"],
        "cfd_capacity_mw": ["0.0", "10.0"],
        "ro_banding": ["", ""],
    })
    result = _assign_subsidy_attributes(
        sites,
        modelled_year=2020,
        roc_eligibility_years=20,
        fes_cfd_fractions={},
        logger=__import__("logging").getLogger(__name__),
        default_roc_bandings={"biomass": 1.5},
    )
    assert result["support_type"].tolist() == ["ROC", "CfD"]
    assert _explicit_support_mask(sites).tolist() == [False, True]


def test_commissioning_override_is_applied_before_month_filtering():
    sites = pd.DataFrame({
        "site_name": ["East Anglia 1 (EA 1)", "Other"],
        "operational_date": ["28/07/2020", "01/06/2020"],
    })
    result = _apply_commissioning_date_overrides(
        sites, {"East Anglia 1 (EA 1)": "2020-01-01"}
    )
    assert result.loc[0, "operational_date"] == "2020-01-01"
    assert result.loc[0, "source_operational_date"] == "28/07/2020"
    assert pd.isna(result.loc[1, "source_operational_date"])


def test_operational_dates_accept_repd_and_iso_override_formats_without_guessing():
    dates = pd.Series(["28/07/2020", "2020-01-01", pd.Timestamp("2020-02-03"), ""])
    result = _parse_operational_dates(dates)
    assert result.iloc[0] == pd.Timestamp("2020-07-28")
    assert result.iloc[1] == pd.Timestamp("2020-01-01")
    assert result.iloc[2] == pd.Timestamp("2020-02-03")
    assert pd.isna(result.iloc[3])


def test_iso_commissioning_override_remains_eligible_for_roc_date_logic():
    sites = pd.DataFrame({
        "site_name": ["Overridden renewable"],
        "technology": ["wind_onshore"],
        "capacity_mw": [10.0],
        "operational_date": ["2015-01-01"],
        "cfd_round": [""],
        "ro_banding": [""],
    })
    result = _assign_subsidy_attributes(
        sites,
        modelled_year=2020,
        roc_eligibility_years=20,
        fes_cfd_fractions={},
        logger=__import__("logging").getLogger(__name__),
        default_roc_bandings={"wind_onshore": 0.9},
    )
    assert result.loc[0, "support_type"] == "ROC"
    assert result.loc[0, "ro_banding"] == pytest.approx(0.9)


def test_future_dispatchable_split_keeps_unique_station_names():
    sites = pd.DataFrame({
        "station_name": ["Future biomass"],
        "technology": ["biomass"],
        "capacity_mw": [100.0],
    })
    result = _assign_subsidy_attributes(
        sites,
        modelled_year=2030,
        roc_eligibility_years=20,
        fes_cfd_fractions={"biomass": 0.4},
        logger=__import__("logging").getLogger(__name__),
    )
    assert set(result["station_name"]) == {
        "Future biomass_CfD", "Future biomass_merchant"
    }
    assert result["capacity_mw"].sum() == pytest.approx(100.0)


def test_unit_support_overrides_are_exact_and_auditable():
    network = pypsa.Network()
    network.add("Bus", "bus")
    override_path = Path("data/market/support/generator_support_overrides_2020.csv")
    overrides = pd.read_csv(override_path)
    for name in overrides["generator_name"]:
        network.add(
            "Generator", name, bus="bus", carrier="biomass", p_nom=100.0
        )

    result = apply_generator_support_overrides(network, override_path, 2020)
    assert result.generators.at[
        "Drax Biomass Power Station - Unit 3", "support_type"
    ] == "CfD"
    assert result.generators.at[
        "Drax Biomass Power Station - Unit 1", "support_type"
    ] == "ROC"
    assert result.generators.at[
        "Drax Biomass Power Station - Unit 1", "ro_banding"
    ] == pytest.approx(1.0)


def test_dispatchable_support_labels_control_marginal_costs():
    network = _dispatchable_network()
    config = {
        "modelled_year": 2020,
        "support": {
            "dispatch_costs": {
                "enabled": True,
                "cfd_dispatch_cost": 0.0,
                "roc_buyout_prices": {2020: 50.05},
            },
        },
        "marginal_costs": {
            "biofuel_subsidy": {
                "enabled": True,
                "require_support_label": True,
                "mc_floor": 5.0,
                "default_roc_bandings": {"biomass": 1.5},
            },
        },
    }
    result = apply_marginal_costs_to_network(
        network,
        fuel_prices={"biomass": 20.0},
        carbon_price=0.0,
        scenario_config=config,
    ).set_index("generator")

    assert result.at["cfd_biomass", "source"] == "subsidy_CfD_dispatchable"
    assert result.at["cfd_biomass", "marginal_cost"] == pytest.approx(0.0)
    assert "ROC" in result.at["roc_biomass", "source"]
    assert result.at["roc_biomass", "marginal_cost"] == pytest.approx(5.0)
    assert "ROC" not in result.at["merchant_biomass", "source"]
    assert result.at["merchant_biomass", "marginal_cost"] == pytest.approx(40.0)


def test_cost_validation_allows_intentional_zero_cost_cfd_thermal():
    network = pypsa.Network()
    network.add("Bus", "bus")
    network.add(
        "Generator", "cfd_biomass", bus="bus", carrier="biomass",
        p_nom=1100.0, marginal_cost=0.0, support_type="CfD",
    )
    validate_network_costs(network, __import__("logging").getLogger(__name__))


def test_cost_validation_rejects_zero_cost_merchant_thermal():
    network = pypsa.Network()
    network.add("Bus", "bus")
    network.add(
        "Generator", "merchant_biomass", bus="bus", carrier="biomass",
        p_nom=100.0, marginal_cost=0.0, support_type="merchant",
    )
    with pytest.raises(ValueError, match="critical marginal cost issues"):
        validate_network_costs(network, __import__("logging").getLogger(__name__))
