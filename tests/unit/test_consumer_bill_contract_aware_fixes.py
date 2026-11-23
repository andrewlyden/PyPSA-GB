"""Consumer-bill fixes for the contract-aware wholesale order (26 Sep 2026).

* the economic-curtailment rule is refused on a contract-aware wholesale;
* the RO load factor comes from the ROC-labelled units themselves;
* the interconnector cap-and-floor pass-through is credited through TNUoS;
* the bill's CfD negative pricing rule must match the balancing bid audit;
* limitation text follows the run instead of asserting the as-built wholesale.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scripts.consumer_bill.project_future_bill as bill
from scripts.consumer_bill.build_ro_eligible_fleet import carrier_load_factors
from scripts.consumer_bill.extract_future_balancing_costs import (
    check_negative_price_rule, recorded_negative_price_rule,
)
from tests.unit.test_future_consumer_bill import ASSUMPTIONS_PATH, fixture_inputs

VALUE = "annual_gbp_per_3100kwh_ex_vat"


def _network(path, generators, dispatch=None, hours=3):
    """A minimal saved-network file with the variables the bill reads."""
    names = list(generators)
    snapshots = pd.date_range("2030-01-01", periods=hours, freq="h")
    data = {
        "snapshots_snapshot": (("snapshots",), snapshots.values),
        "snapshots_generators": (("snapshots",), np.ones(hours)),
    }
    for column in ("support_type", "carrier", "cfd_round"):
        data[f"generators_{column}"] = (("generators_i",), [generators[n].get(column, "") for n in names])
    for column in ("marginal_cost", "p_nom"):
        data[f"generators_{column}"] = (("generators_i",), [float(generators[n].get(column, 0.0)) for n in names])
    if dispatch is not None:
        data["generators_t_p"] = (("snapshots", "generators_t_p_i"),
                                  np.array([dispatch[n] for n in names]).T.astype(float))
    ds = xr.Dataset(data, coords={"snapshots": np.arange(hours), "generators_i": names,
                                  **({"generators_t_p_i": names} if dispatch is not None else {})})
    ds.to_netcdf(path)
    bill.clear_table_cache()
    return path


# --------------------------------------------------------------------------
# Fix 1: the economic-curtailment rule and the contract-aware order
# --------------------------------------------------------------------------

def test_saved_marginal_costs_reveal_the_curtailment_order(tmp_path):
    as_built = _network(tmp_path / "as_built.nc", {
        "off_ar3": {"support_type": "CfD", "carrier": "wind_offshore", "cfd_round": "AR3", "marginal_cost": 0.0},
        "roc": {"support_type": "ROC", "carrier": "wind_onshore", "marginal_cost": -60.0},
    })
    assert bill.wholesale_curtailment_order(as_built)["order"] == "as_built"
    aware = _network(tmp_path / "aware.nc", {
        "off_ar3": {"support_type": "CfD", "carrier": "wind_offshore", "cfd_round": "AR3", "marginal_cost": -56.7},
        "on_ar6": {"support_type": "CfD", "carrier": "wind_onshore", "cfd_round": "AR6", "marginal_cost": -1.0},
        "bio_ic": {"support_type": "CfD", "carrier": "biomass", "cfd_round": "Investment Contract", "marginal_cost": 80.0},
    })
    order = bill.wholesale_curtailment_order(aware)
    assert order == {"order": "contract_aware", "repriced_units": 2, "pre_ar4_units_below_zero": 1}


def test_economic_curtailment_rule_is_refused_on_a_contract_aware_wholesale(tmp_path, monkeypatch):
    network = _network(tmp_path / "aware.nc", {
        "off_ar3": {"support_type": "CfD", "carrier": "wind_offshore", "cfd_round": "AR3", "marginal_cost": -56.7},
    })
    index = pd.date_range("2030-01-01", "2031-01-01", freq="h", inclusive="left")
    frame = pd.DataFrame({"demand_mw": 1.0, "hours": 1.0, "price_gbp_per_mwh": 10.0}, index=index)
    monkeypatch.setattr(bill, "read_hourly_inputs", lambda *args: frame)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"scenario": "test",
        "accounting": {"reported_period_hours": 8760, "demand_MWh": 8760, "gross_redispatch_MWh": 0},
        "revenue": {"cfd_reference_price_source": "wholesale_system_price", "cfd_total_cost": 0,
                    "cfd_total_dispatch_MWh": 0}}))
    with pytest.raises(ValueError, match="contract-aware curtailment order"):
        bill.aggregate_inputs(["test"], [network], [tmp_path / "p.csv"], [summary], 2030,
                              strike_prices=object(),
                              economic_curtailment={"rule": "pay_pre_ar4_at_strike"})


def test_assumptions_turn_the_economic_curtailment_rule_off_with_a_decision():
    cfd = json.loads(Path(ASSUMPTIONS_PATH).read_text())["model_linked_rules"]["cfd"]
    assert cfd["economic_curtailment_rule"] == "none"
    assert "pay_pre_ar4_at_strike" in cfd["economic_curtailment_rule_options"]
    assert "contract-aware" in cfd["economic_curtailment_rule_decision"]
    # The HPC valuation simplification is recorded where the rule is described.
    assert "seasonal baseload reference" in cfd["economic_curtailment_rule_note"]


# --------------------------------------------------------------------------
# Fix 2: RO load factors from the ROC-labelled units
# --------------------------------------------------------------------------

def test_roc_labelled_units_give_their_own_load_factor(tmp_path):
    path = _network(tmp_path / "bal.nc", {
        "roc": {"support_type": "ROC", "carrier": "wind_onshore", "p_nom": 100.0},
        "cfd": {"support_type": "CfD", "carrier": "wind_onshore", "p_nom": 100.0},
    }, dispatch={"roc": [50.0, 50.0, 50.0], "cfd": [10.0, 10.0, 10.0]})
    average = carrier_load_factors([path], ["wind_onshore"]).set_index("carrier")
    labelled = carrier_load_factors([path], ["wind_onshore"], support_type="ROC").set_index("carrier")
    assert average.at["wind_onshore", "load_factor"] == pytest.approx(0.30)
    assert labelled.at["wind_onshore", "load_factor"] == pytest.approx(0.50)
    assert labelled.at["wind_onshore", "capacity_mw"] == pytest.approx(100.0)


# --------------------------------------------------------------------------
# Fix 3: interconnector cap-and-floor pass-through
# --------------------------------------------------------------------------

def _rule(**cables):
    return {"index": {"to_price_year": 2024, "to_index": 200.0}, "cables": cables}


def test_cap_and_floor_returns_revenue_above_the_cap_and_charges_the_floor_top_up():
    rule = _rule(
        high={"regime": "cap_and_floor", "levels_apply_to": "gb_share", "gb_share_of_rent": 0.5,
              "cap_gbp_m": 50.0, "floor_gbp_m": 25.0, "index_from": 100.0},     # indexed x2: cap 100m, floor 50m
        low={"regime": "cap_and_floor", "levels_apply_to": "gb_share", "gb_share_of_rent": 0.5,
             "cap_gbp_m": 50.0, "floor_gbp_m": 25.0, "index_from": 100.0},
        joint={"regime": "cap_and_floor", "levels_apply_to": "whole_link", "gb_share_of_rent": 0.5,
               "cap_gbp_m": 50.0, "floor_gbp_m": 25.0, "index_from": 100.0},   # GB half: cap 50m
        ifa={"regime": "use_of_revenues_share", "gb_share_of_rent": 0.5, "tariff_share_of_gb_revenue": 0.4},
        merchant={"regime": "exempt_merchant"},
    )
    rent = {"high": 300e6, "low": 60e6, "joint": 160e6, "ifa": 100e6, "merchant": 500e6}
    result = bill.interconnector_pass_through(rent, rule, 2024)
    by = {row["interconnector"]: row for row in result["by_cable"]}
    assert by["high"]["above_cap_to_consumers_gbp"] == pytest.approx(50e6)      # 150m - 100m
    assert by["low"]["floor_top_up_from_consumers_gbp"] == pytest.approx(20e6)  # 50m - 30m
    assert by["low"]["credit_to_consumers_gbp"] == pytest.approx(-20e6)
    assert by["joint"]["credit_to_consumers_gbp"] == pytest.approx(30e6)        # 80m - 50m
    assert by["ifa"]["credit_to_consumers_gbp"] == pytest.approx(20e6)          # 0.4 x 50m
    assert by["merchant"]["credit_to_consumers_gbp"] == 0.0
    assert result["credit_to_consumers_gbp"] == pytest.approx(80e6)


def test_a_cable_with_rent_but_no_declared_regime_is_refused():
    with pytest.raises(ValueError, match="no declared regime"):
        bill.interconnector_pass_through({"New Cable": 1e6}, _rule(), 2024)


def test_cap_and_floor_index_must_land_on_the_model_price_year():
    with pytest.raises(ValueError, match="model price year"):
        bill.interconnector_pass_through({}, _rule(), 2023)


def test_every_modelled_cable_is_declared_with_a_sourced_regime():
    rule = json.loads(Path(ASSUMPTIONS_PATH).read_text())["model_linked_rules"]["interconnector_cap_and_floor"]
    declared = rule["cables"]
    modelled = {"Auchencrosh (interconnector CCT)", "Britned", "East West Interconnector", "ElecLink",
                "Greenlink", "IFA", "IFA2", "Isle of Man Interconnector", "Moyle", "NS Link", "Nemo Link",
                "NeuConnect Interconnector", "Viking", "MaresConnect"}
    assert modelled <= set(declared)
    assert all(spec.get("source") for spec in declared.values())
    cap_floor = {name for name, spec in declared.items() if spec["regime"] == "cap_and_floor"}
    assert cap_floor == {"Nemo Link", "IFA2", "NS Link", "Viking", "Greenlink", "NeuConnect Interconnector",
                         "MaresConnect"}
    assert declared["ElecLink"]["regime"] == declared["Britned"]["regime"] == "exempt_merchant"


COMPOSITION = Path("data/interconnectors/cp30_interconnector_composition.csv")
ASSUMPTION_VARIANTS = (Path(ASSUMPTIONS_PATH),
                       Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions_cfd_rule_none.json"))


@pytest.mark.parametrize("path", ASSUMPTION_VARIANTS, ids=lambda p: p.name)
def test_every_cp30_composition_cable_has_a_regime_in_every_2030_assumptions_file(path):
    """15 Oct 2026: the CP30 2030 build carries the composition's cables; the bill and FTI stop on an undeclared one."""
    from scripts.CP30.storage_interconnectors import load_cp30_interconnector_composition

    comp = load_cp30_interconnector_composition(COMPOSITION)
    built = set(comp.loc[comp["action"].isin(["keep", "add"]), "name"])
    declared = json.loads(path.read_text(encoding="utf-8"))["model_linked_rules"]["interconnector_cap_and_floor"]["cables"]
    assert built <= set(declared)
    # The Isle of Man cable is out of the CP30 build but stays declared for builds that still carry it.
    assert "Isle of Man Interconnector" in declared and "not built" in declared["Isle of Man Interconnector"]["cp30_2030_build"]


def test_both_2030_assumptions_files_declare_the_same_regimes():
    rules = [json.loads(p.read_text(encoding="utf-8"))["model_linked_rules"]["interconnector_cap_and_floor"]
             for p in ASSUMPTION_VARIANTS]
    assert rules[0] == rules[1]


def test_maresconnect_levels_are_greenlinks_per_mw_and_price_through_the_bill():
    """MaresConnect (Ofgem OFG1164, Window 3 IPA, 12 Nov 2024): cap and floor in principle, levels not published."""
    rule = json.loads(Path(ASSUMPTIONS_PATH).read_text())["model_linked_rules"]["interconnector_cap_and_floor"]
    mares, green = rule["cables"]["MaresConnect"], rule["cables"]["Greenlink"]
    assert mares["regime"] == "cap_and_floor" and mares["levels_apply_to"] == "gb_share"
    assert mares["gb_share_of_rent"] == green["gb_share_of_rent"] == 0.5
    for key in ("cap_gbp_m", "floor_gbp_m"):
        assert mares[key] == pytest.approx(green[key] * 750.0 / 500.0, abs=0.005)
    assert mares["levels_price_year"] == green["levels_price_year"] and mares["index_from"] == green["index_from"]
    assert "OFG1164" in mares["source"] and "derived, not published" in mares["levels_status"]
    assert mares["public_ownership_treatment"] == "new_build_at_cost"
    # GB half 100 m of a 200 m rent against an indexed cap of 46.86 x 386.7 / 294.17 = 61.60 m.
    result = bill.interconnector_pass_through({"MaresConnect": 200e6}, rule, 2024)
    row = result["by_cable"][0]
    assert row["cap_gbp"] == pytest.approx(46.86e6 * 386.7 / 294.17)
    assert row["floor_gbp"] == pytest.approx(27.90e6 * 386.7 / 294.17)
    assert result["credit_to_consumers_gbp"] == pytest.approx(100e6 - 46.86e6 * 386.7 / 294.17)
    # A first year below the floor (Ofgem's IPA expects it in two of three scenarios) is topped up by consumers.
    low = bill.interconnector_pass_through({"MaresConnect": 40e6}, rule, 2024)
    assert low["credit_to_consumers_gbp"] == pytest.approx(20e6 - 27.90e6 * 386.7 / 294.17)


def test_fti_gb_half_shares_cover_maresconnect():
    from scripts.market.fti_settlement import gb_half_shares, interconnector_regime_pass_through, load_interconnector_rule

    rule = load_interconnector_rule(ASSUMPTIONS_PATH)
    shares = gb_half_shares(rule)
    assert shares["MaresConnect"] == 0.5 and shares["East West Interconnector"] == 0.0
    # A month of rent settles against a twelfth of the year's levels, without an undeclared-cable stop.
    month = interconnector_regime_pass_through({"MaresConnect": 20e6, "Greenlink": 10e6}, rule, 1.0 / 12.0)
    assert {row["interconnector"] for row in month["by_cable"]} == {"MaresConnect", "Greenlink"}


def test_the_credit_enters_the_bill_as_a_negative_network_line():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    annual["interconnector_congestion_rent_by_cable_gbp"] = {"Nemo Link": 300e6, "ElecLink": 200e6}
    rows, summary = bill.project_components(annual, assumptions, case, balancing, recovery, ro)
    line = rows.set_index("component_id").loc["interconnector_cap_and_floor_pass_through"]
    spec = assumptions["model_linked_rules"]["interconnector_cap_and_floor"]["cables"]["Nemo Link"]
    cap_gb = spec["cap_gbp_m"] * 1e6 * 386.7 / spec["index_from"] * 0.5
    # The calendar-2030 blend of the residual and its revenue (28 Sep 2026).
    blend = assumptions["sourced_externals"]["tnuos"]["derivation"]["calendar_blend"]
    share = blend["domestic_residual_gbp_per_year"] / (blend["demand_residual_revenue_gbp_m"] * 1e6)
    assert line["stack_group"] == "network"
    assert line[VALUE] == pytest.approx(-(0.5 * 300e6 - cap_gb) * share)   # fixture factor is 1
    assert summary["diagnostics"]["interconnector_cap_and_floor_pass_through"]["credit_to_consumers_gbp"] > 0


def test_the_credit_needs_the_per_cable_rent():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    del annual["interconnector_congestion_rent_by_cable_gbp"]
    with pytest.raises(ValueError, match="per-cable congestion rent"):
        bill.project_components(annual, assumptions, case, balancing, recovery, ro)


# --------------------------------------------------------------------------
# Fix 5: the bill's negative pricing rule matches the balancing bid audit
# --------------------------------------------------------------------------

def test_bid_audit_rule_is_read_only_when_recorded(tmp_path):
    old = tmp_path / "old.csv"
    pd.DataFrame({"generator": ["a"], "pricing": ["cfd_contract_value"]}).to_csv(old, index=False)
    assert recorded_negative_price_rule(old) is None
    assert recorded_negative_price_rule(tmp_path / "missing.csv") is None
    new = tmp_path / "new.csv"
    pd.DataFrame({"generator": ["a", "b"], "pricing": ["cfd_contract_value", "merchant"],
                  "negative_price_rule": ["gb_contract_terms", None]}).to_csv(new, index=False)
    assert recorded_negative_price_rule(new) == "gb_contract_terms"


def test_a_bill_rule_differing_from_the_bid_audit_is_refused():
    check_negative_price_rule("gb_contract_terms", "gb_contract_terms")
    check_negative_price_rule("gb_contract_terms", None)  # not recorded: not checked
    with pytest.raises(ValueError, match="balancing bids applied 'none'"):
        check_negative_price_rule("gb_contract_terms", "none", where="test")


# --------------------------------------------------------------------------
# Fix 4: limitation text follows the run
# --------------------------------------------------------------------------

def test_zero_price_limitation_describes_negative_hours_when_they_exist():
    as_built = {"cfd_negative_price_rule": "gb_contract_terms", "minimum_price_gbp_per_mwh": -2e-6,
                "cfd_economic_curtailment_rule": "none", "hours_price_at_or_below_1_gbp_per_mwh": 3764.0}
    assert "No modelled hour clears below zero" in bill._zero_price_limitation(as_built)
    aware = {**as_built, "minimum_price_gbp_per_mwh": -56.7, "hours_price_below_minus_0_01_gbp_per_mwh": 812.0,
             "wholesale_curtailment_order": "contract_aware"}
    text = bill._zero_price_limitation(aware)
    assert "No modelled hour" not in text and "812 modelled hours clear below zero" in text
    assert "six or more negative hours" in text


def test_bsuos_text_does_not_claim_the_compensation_was_added_when_it_was_reconciled():
    balancing = {"cfd_curtailment_compensation_gbp": 1e9, "cfd_curtailed_mwh": 1e7,
                 "cfd_curtailment_compensation_treatment": "reconcile_against_lp_bid"}
    assert "not added" in bill._compensation_text(balancing)
    assert "plus a post-hoc" not in bill._bsuos_limitation(balancing)
    assert "contract value" in bill._bsuos_limitation(balancing)
    balancing["cfd_curtailment_compensation_treatment"] = "add_post_hoc"
    assert bill._compensation_text(balancing).startswith("Plus")
    assert "plus a post-hoc" in bill._bsuos_limitation(balancing)


def test_remaining_ccs_no_longer_calls_net_zero_teesside_plant_outside_the_model():
    ccs = json.loads(Path(ASSUMPTIONS_PATH).read_text())["sourced_externals"]["remaining_ccs"]
    assert "plant outside the model" not in ccs["method"]
    assert "committed overlay" in ccs["method"]
