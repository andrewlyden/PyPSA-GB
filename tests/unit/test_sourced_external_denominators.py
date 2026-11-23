"""Item F4: a sourced external that is spread over demand takes the demand from one place.

The Capacity Market line was 1,086.9m GBP of agreements over a stored copy of final
demand, 311,772,038.58 MWh, the pre-reinforcement figure, while every other
recovery denominator came from ``2030_recovery_demand_assumptions.json`` (318.29
TWh since the rebuild). The line now names its denominator and the projector reads
it from that file, so a second copy cannot drift again.
"""

import json
from pathlib import Path

import pytest

from scripts.consumer_bill.project_future_bill import read_recovery_denominators, resolve_sourced_externals
from tests.helpers import requires_outputs

ASSUMPTIONS = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
RECOVERY = Path("data/market/consumer_bill_forecast/2030_recovery_demand_assumptions.json")


def _recovery():
    """The recovery volumes exactly as the projector reads them (F18: the two-volume basis, 6 Oct 2026).

    The tests below used to build the volume dictionary by hand from the D1 key names
    (``gsp_chargeable_volume_mwh``, ``eligible_supplier_sales_mwh``); F18 renamed the
    lines' denominators (``gsp_charging_volume_mwh``, ``ro_eligible_metered_supply_mwh``,
    ``cfd_eligible_gsp_volume_mwh``) and kept the old names as deprecated aliases, so the
    tests now read the file through the projector's own reader.
    """
    document = json.loads(RECOVERY.read_text(encoding="utf-8"))
    return read_recovery_denominators(RECOVERY, document["case_id"], document["system_demand_mwh"]["value"])


def _block(**derivation):
    return {"unit": "gbp_per_household_year", "consumption_basis_kwh": 3100,
            "derivation": {"agreement_cost_gbp": 1.0e9, **derivation}}


def test_the_value_is_the_cost_over_the_named_denominator():
    sourced = resolve_sourced_externals({"cm": _block(recovery_denominator="final_demand_mwh")},
                                        {"final_demand_mwh": 3.1e8}, consumption_kwh=3100)
    assert sourced["cm"]["value"] == pytest.approx(1.0e9 / 3.1e8 * 3.1)
    assert sourced["cm"]["derivation"]["recovery_denominator_mwh_used"] == 3.1e8


def test_a_stored_copy_of_a_demand_figure_is_refused():
    with pytest.raises(ValueError, match="recovery_denominator_mwh"):
        resolve_sourced_externals({"cm": _block(recovery_denominator_mwh=3.1e8)},
                                  {"final_demand_mwh": 3.1e8}, consumption_kwh=3100)


def test_a_derived_line_needs_the_recovery_file():
    with pytest.raises(ValueError, match="recovery"):
        resolve_sourced_externals({"cm": _block(recovery_denominator="final_demand_mwh")}, None,
                                  consumption_kwh=3100)


def test_an_unknown_denominator_is_refused():
    with pytest.raises(ValueError, match="peak_demand_mwh"):
        resolve_sourced_externals({"cm": _block(recovery_denominator="peak_demand_mwh")},
                                  {"final_demand_mwh": 3.1e8}, consumption_kwh=3100)


def test_lines_without_a_derivation_are_left_as_quoted():
    block = {"unit": "gbp_per_household_year", "value": 5.0}
    assert resolve_sourced_externals({"x": block}, None, consumption_kwh=3100)["x"]["value"] == 5.0


@requires_outputs("resources/public_ownership/cp30_2030_cp30_network_public_account_summary.json")
def test_the_input_files_carry_no_copy_of_an_annual_demand_figure():
    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY.read_text(encoding="utf-8"))
    for key, block in assumptions["sourced_externals"].items():
        assert "recovery_denominator_mwh" not in (block.get("derivation") or {}), key
    cm = assumptions["sourced_externals"]["capacity_market"]
    # Item D1: the CM household factor already carries losses, so the line divides
    # by the GSP volume, not by metered supply (F18: the GSP charging volume).
    assert cm["derivation"]["recovery_denominator"] == "gsp_charging_volume_mwh"
    assert "value" not in cm, "the value is derived at projection time"
    volumes = _recovery()
    gsp = volumes["gsp_charging_volume_mwh"]
    assert gsp == pytest.approx(recovery["metered_supply_basis"]["gsp_charging_volume_mwh"]["value"], rel=1e-12)
    resolved = resolve_sourced_externals(assumptions["sourced_externals"], volumes, consumption_kwh=3100,
                                         calendar_year=2030)
    # Item MR4: the calendar-2030 contract book and modelled auctions, in 2025 prices,
    # over the GSP volume at 3.1 MWh; no household loss factor on top (the allocation
    # factor carries the losses).
    derivation = resolved["capacity_market"]["derivation"]
    assert derivation["recovery_denominator_mwh_used"] == pytest.approx(gsp, rel=1e-12)
    assert derivation["household_loss_factor_used"] == 1.0
    assert resolved["capacity_market"]["value"] == pytest.approx(
        cm["derivation"]["agreement_cost_gbp"] / gsp * 3.1
        * cm["derivation"]["household_allocation_factor"], rel=1e-6)


def test_a_household_allocation_factor_scales_the_energy_share():
    block = {"unit": "gbp_per_household_year", "consumption_basis_kwh": 3100,
             "derivation": {"cost_gbp": 1.0e9, "recovery_denominator": "final_demand_mwh",
                            "household_allocation_factor": 1.25}}
    out = resolve_sourced_externals({"cm": block}, {"final_demand_mwh": 300e6}, consumption_kwh=3100)
    assert out["cm"]["value"] == pytest.approx(1.0e9 / 300e6 * 3.1 * 1.25)


# ── item F1: three of the five placeholders primary-sourced ──────────────────

def test_a_generic_cost_can_be_spread_over_eligible_supply():
    block = {"unit": "gbp_per_household_year", "consumption_basis_kwh": 3100,
             "derivation": {"cost_gbp": 1.7e9, "recovery_denominator": "cfd_eligible_supplier_sales_mwh"}}
    out = resolve_sourced_externals({"fit": block}, {"cfd_eligible_supplier_sales_mwh": 304.46e6},
                                    consumption_kwh=3100)
    assert out["fit"]["value"] == pytest.approx(1.7e9 / 304.46e6 * 3.1)


@requires_outputs("resources/public_ownership/cp30_2030_cp30_network_public_account_summary.json")
def test_the_five_placeholders_are_primary_sourced():
    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY.read_text(encoding="utf-8"))
    sourced = assumptions["sourced_externals"]
    # allocated_ccs was sourced (a nil) by F1 and is model-linked since item MR7.
    assert "allocated_ccs" in assumptions["retired_sourced_externals"]
    assert assumptions["model_linked_rules"]["cfd"]["ccs_levy_carriers"] == ["CCS Biomass"]
    for key in ("feed_in_tariffs", "warm_home_discount", "eco", "remaining_ccs"):
        block = sourced[key]
        assert block["source_class"] == "external_primary_source"
        assert Path(block["source_file"]).exists(), block["source_file"]
    assert assumptions["external_assumptions_gbp_per_3100kwh_ex_vat"] == {}
    for key in ("eco",):
        assert sourced[key]["value"] == 0.0 and sourced[key]["value_basis"] == "nil"
    volumes = _recovery()
    resolved = resolve_sourced_externals(sourced, volumes, consumption_kwh=3100, calendar_year=2030)
    # F18 (6 Oct 2026): each line divides by its own recovery volume, read from the file, never a
    # stored copy (the eligible-supply figure moves with every refresh of the solved months).
    fit_volume = volumes[sourced["feed_in_tariffs"]["derivation"]["recovery_denominator"]]
    ccs_volume = volumes[sourced["remaining_ccs"]["derivation"]["recovery_denominator"]]
    assert sourced["feed_in_tariffs"]["derivation"]["recovery_denominator"] == "ro_eligible_metered_supply_mwh"
    assert sourced["remaining_ccs"]["derivation"]["recovery_denominator"] == "cfd_eligible_gsp_volume_mwh"
    # DESNZ: 1.8bn (2029/30) and 1.7bn (2030/31), 2024/25 prices, blended to calendar 2030
    # (3/12 and 9/12, 28 Sep 2026), over obligation-eligible metered supply (no loss uplift), x 3.1 MWh.
    fit = resolved["feed_in_tariffs"]
    assert fit["derivation"]["household_loss_factor_used"] == 1.0
    assert fit["value"] == pytest.approx((0.25 * 1.8e9 + 0.75 * 1.7e9) / fit_volume * 3.1, rel=1e-6)
    # OBR 0.970bn, half on electricity, over Ofgem's 97.48 TWh of domestic supply.
    assert sourced["warm_home_discount"]["value"] == pytest.approx(0.970e9 * 0.5 / 97479142.979 * 3.1, rel=1e-3)
    # Net Zero Teesside's DPA: the ECC business case's 1.366bn over three years, an upper bound, over the
    # CfD-eligible GSP volume at the household loss factor (a GSP-rate line).
    ccs = resolved["remaining_ccs"]
    loss = ccs["derivation"]["household_loss_factor_used"]
    assert loss == pytest.approx(volumes["household_loss_factor"])
    assert ccs["value"] == pytest.approx(1.366e9 / 3 / ccs_volume * 3.1 * loss, rel=1e-6)
    assert sourced["remaining_ccs"]["derivation"]["value_basis"] == "upper_bound"
