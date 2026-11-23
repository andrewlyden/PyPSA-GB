"""F19 integration (7 Oct 2026): the public-ownership layer converts through each bill line's recovery entry.

F18 gave the bill two recovery volumes and one ``line_recovery`` entry per line. The ownership
layer and its reports now convert household and system pounds through those entries
(:mod:`scripts.public_ownership.bill_recovery`), never through the deprecated denominators on
the two-volume basis; a D1 (2050) bill keeps its own denominators, so 2050 does not move.
"""

import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.consumer_bill import line_recovery as lr
from scripts.public_ownership import bill_recovery as br
from tests.unit.test_line_recovery import _project, _two_volume_inputs
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]


def _two_volume_summary():
    lines, summary, assumptions, recovery = _project()
    return lines, summary, assumptions, recovery


class _Poisoned(dict):
    """A mapping that refuses the deprecated names, so a test proves no code path reads them."""

    DEPRECATED = {"final_demand_mwh", "gsp_chargeable_volume_mwh", "cfd_eligible_supplier_sales_mwh",
                  "cfd_recovery_denominator_mwh", "final_demand_recovery_denominator_mwh"}

    def __getitem__(self, key):
        if key in self.DEPRECATED:
            raise AssertionError(f"deprecated alias {key!r} read on the two-volume basis")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.DEPRECATED:
            raise AssertionError(f"deprecated alias {key!r} read on the two-volume basis")
        return super().get(key, default)


def _poison(summary):
    out = _Poisoned({k: v for k, v in summary.items() if k not in _Poisoned.DEPRECATED})
    return out


# ── the helper ───────────────────────────────────────────────────────────────────

def test_a_two_volume_summary_converts_through_each_lines_entry():
    lines, summary, _, _ = _two_volume_summary()
    assert br.is_two_volume(summary)
    for line in ("cfd", "bsuos_constraints", "capacity_market", "renewables_obligation", "direct_energy"):
        entry = summary["line_recovery"][line]
        assert br.household_per_system(summary, line) == pytest.approx(lr.system_to_household_gbp(entry, 1.0))
        assert br.system_from_household(summary, line, 10.0) == pytest.approx(lr.household_to_system_gbp(entry, 10.0))
    # A GSP-rate line converts ~4.7% higher per system pound than flat over metered supply would.
    flat = br.flat_household_per_system(summary)
    assert br.household_per_system(summary, "bsuos_constraints") / flat == pytest.approx(
        1.1313 * summary["exchequer_per_household_key"]["volume_mwh"]
        / summary["line_recovery"]["bsuos_constraints"]["volume_mwh"])
    with pytest.raises(KeyError, match="not_a_line"):
        br.household_per_system(summary, "not_a_line")


def test_the_two_volume_path_reads_no_deprecated_alias():
    _, summary, _, _ = _two_volume_summary()
    poisoned = _poison(summary)
    assert br.household_per_system(poisoned, "cfd") > 0
    assert br.exchequer_key_mwh(poisoned) == pytest.approx(summary["exchequer_per_household_key"]["volume_mwh"])
    with pytest.raises(ValueError, match="deprecated"):
        br.d1_denominators(summary)


def test_a_d1_summary_keeps_its_own_denominators():
    summary = {"model_rebasing_factor": 1.0384, "benchmark_kwh": 3100.0, "final_demand_recovery_denominator_mwh": 3.0e8,
               "cfd_recovery_denominator_mwh": 2.7e8}
    assert not br.is_two_volume(summary)
    assert br.household_per_system(summary, "cfd") == pytest.approx(1.0384 * 3.1 / 2.7e8)
    assert br.household_per_system(summary, "bsuos_constraints") == pytest.approx(1.0384 * 3.1 / 3.0e8)
    assert br.exchequer_key_mwh(summary) == 3.0e8
    assert br.flat_equivalent_mwh(summary, "cfd") == pytest.approx(2.7e8)


# ── the sites switched ───────────────────────────────────────────────────────────

def test_the_tariff_and_ledger_conversions_read_entries_on_two_volumes():
    from scripts.public_ownership import exchequer_ledger as el
    from scripts.public_ownership import nationalised_tariff as nt

    _, summary, assumptions, _ = _two_volume_summary()
    bill = {"cases": {"private": {"line_recovery": summary["line_recovery"]}}, "recovery_basis": summary["recovery_basis"],
            "exchequer_per_household_key": summary["exchequer_per_household_key"], "benchmark_kwh": 3100.0,
            "model_rebasing_factor": summary["model_rebasing_factor"]}
    recovery = _Poisoned({"recovery_basis": br.TWO_VOLUME_BASIS})
    conv = nt.conversion(assumptions, recovery, _poison(bill))
    assert conv["line_recovery"] is not None
    for rule, line in nt.RULE_ENTRY_LINES.items():
        if rule == "domestic_energy":
            continue
        assert nt._rule_rate(conv, rule) == pytest.approx(br.household_per_system(summary, line))
    assert nt._rule_rate(conv, "final_demand") == pytest.approx(br.flat_household_per_system(summary))
    ledger_conv = el.conversion(assumptions, recovery, _poison(bill))
    assert ledger_conv["final_demand_mwh"] == ledger_conv["exchequer_key_mwh"] == pytest.approx(
        summary["exchequer_per_household_key"]["volume_mwh"])


def test_balancing_costs_and_the_forecast_error_charge_convert_like_the_bsuos_line():
    from scripts.market.forecast_error_charge import gbp_per_household
    from scripts.public_ownership.merge_balancing_groups import _household_rates

    _, summary, _, _ = _two_volume_summary()
    bill = _poison({**summary, "benchmark_kwh": 3100.0})
    rate = br.household_per_system(summary, "bsuos_constraints")
    assert gbp_per_household(1.0e9, bill) == pytest.approx(1.0e9 * rate)
    rates = _household_rates(bill, 0.05)
    assert rates["final_demand"] == pytest.approx(rate * 1.05)
    assert rates["cfd_eligible_supply"] == pytest.approx(br.household_per_system(summary, "cfd") * 1.05)


def test_the_hpc_rows_use_the_supplied_line_rates():
    from scripts.public_ownership import hpc_schedule as hs

    shortfall = {"missing_mwh": 1.0e6, "replacement_price_gbp_per_mwh": 80.0}
    conv = {"factor": 1.0, "mwh": 3.1, "final_demand_mwh": 3.0e8}
    out = hs.market_bill_change(shortfall, cfd_payment_gbp=1.0e8, settled_mwh=2.0e6, strike_gbp_per_mwh=100.0,
                                conv=conv, line_rates={"cfd": 2.0e-8, "wholesale": 1.0e-8})
    assert out["cfd_line_gbp_per_3100kwh_ex_vat"] == pytest.approx(out["cfd_line_system_gbp"] * 2.0e-8)
    assert out["wholesale_line_gbp_per_3100kwh_ex_vat"] == pytest.approx(out["wholesale_line_system_gbp"] * 1.0e-8)


#: Read patterns of the deprecated names (F18): the summaries' two denominators and the recovery
#: file's aliases. Writing them (the ownership summary still states them for D1 readers) is not a read.
DEPRECATED_READ = re.compile(
    r'\[\s*"(?:cfd_recovery_denominator_mwh|final_demand_recovery_denominator_mwh)"\s*\]'
    r'|\.get\(\s*"(?:cfd_recovery_denominator_mwh|final_demand_recovery_denominator_mwh)"'
    r'|\[\s*"final_demand_mwh"\s*\]\s*\[\s*"value"\s*\]'
    r'|"gsp_chargeable_volume_mwh"|"cfd_eligible_supplier_sales_mwh"')
#: The one sanctioned reader (D1 only), and the 2050-only modules, which stay on the D1 basis.
SANCTIONED = {"scripts/public_ownership/bill_recovery.py"}
D1_ONLY = re.compile(r"_2050\.py$|pathway_2030_2050\.py$")


def test_no_public_ownership_caller_reads_a_deprecated_alias():
    files = sorted((ROOT / "scripts/public_ownership").glob("*.py")) + [
        ROOT / "scripts/market/forecast_error_charge.py",
        ROOT / "scripts/analysis/build_ownership_explorer_data.py",
        ROOT / "scripts/analysis/build_dispatch_pricing_explainer.py",
        ROOT / "scripts/analysis/build_cp30_bill_notebook.py",
        *sorted((ROOT / "scripts/analysis/public_ownership_explorer").glob("*.py")),
    ]
    offenders = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if rel in SANCTIONED or D1_ONLY.search(rel):
            continue
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if DEPRECATED_READ.search(text):
                offenders.append(f"{rel}:{number}: {text.strip()}")
    assert not offenders, "\n".join(offenders)


# ── the entries the ownership layer writes ───────────────────────────────────────

def test_the_levy_offset_and_the_levies_moved_to_taxation_declare_domestic_supply():
    from scripts.public_ownership import project_public_ownership_consumer_bill as po

    spec = po.domestic_recovery(9.748e7)
    assert spec["line_recovery"][0] == po.DOMESTIC_VOLUME_KEY and spec["line_recovery_volumes"] == {
        po.DOMESTIC_VOLUME_KEY: 9.748e7}
    assert po.domestic_recovery(None) == {}
    extra = [{"component_id": "public_surplus_offset_domestic_levies", "stack_group": "policy", "label": "x",
              "annual_gbp_per_3100kwh_ex_vat": -2.0, "method": "m", "source": "s", **spec},
             {"component_id": "pay_as_bid_ceiling", "stack_group": "hedged_wholesale", "label": "x",
              "annual_gbp_per_3100kwh_ex_vat": -1.0, "method": "m", "source": "s"}]
    _, summary, _, _ = _project(extra=extra)
    offset = summary["line_recovery"]["public_surplus_offset_domestic_levies"]
    assert offset["volume_key"] == po.DOMESTIC_VOLUME_KEY and offset["volume_mwh"] == pytest.approx(9.748e7)
    assert offset["household_loss_factor"] == 1.0
    # The pay-as-bid ceiling moves the wholesale line, so it is recovered like it (metered supply).
    assert summary["line_recovery"]["pay_as_bid_ceiling"]["volume_key"] == "metered_supply_mwh"
    assert lr.INHERITED_LINE_RECOVERY["public_cost_recovery_contracting"] == "direct_energy"


def test_the_capacity_market_booking_is_recovered_on_the_capacity_market_line():
    from scripts.public_ownership import project_public_ownership_consumer_bill as po

    _, summary, _, _ = _two_volume_summary()
    account = {"sensitivity_axes": {po.AXIS_ROUTE_BOOKING: po.ROUTE_BOOKING_BOOKED}, "capacity_market_booked_gbp": 1.0e8,
               "ccs_levy_sensitivity_gbp": 0.0}
    amounts = po.booked_route_amounts(account, factor=1.0384, final_demand_mwh=2.9e8, eligible_mwh=2.7e8, mwh=3.1,
                                      cm_line=None, ccs_line=None, cm_household_factor=1.348308,
                                      line_recovery=summary["line_recovery"])
    entry = summary["line_recovery"]["capacity_market"]
    assert amounts[po.LINE_CM_BOOKED] == pytest.approx(lr.system_to_household_gbp(entry, 1.0e8))
    options = po.cm_recovery_options(30.0, cm_factor=1.348308, final_demand_mwh=2.9e8, factor=1.0384, mwh=3.1,
                                     cm_entry=entry)
    assert options["system_cost_gbp_2024"] == pytest.approx(lr.household_to_system_gbp(entry, 30.0))


def test_every_tariff_line_carries_an_entry_that_reproduces_its_rate():
    from scripts.public_ownership import nationalised_tariff as nt

    _, summary, assumptions, _ = _two_volume_summary()
    entries = summary["line_recovery"]
    conv = {"factor": summary["model_rebasing_factor"], "mwh": 3.1, "final_demand_mwh": 2.9e8, "domestic_mwh": 9.7e7,
            "line_recovery": entries}
    lines = {nt.LINE_NEW_BUILD: 1.0e9, nt.LINE_CM_NOT_ACQUIRED: 5.0e8, nt.LINE_IC_RETRADING: 1.0e7,
             nt.LINE_LEGACY_FIT: -1.0e8}
    allocation = {"new_line_rates": {nt.LINE_NEW_BUILD: 1.2e-8, nt.LINE_CM_NOT_ACQUIRED: nt._rule_rate(conv, "capacity_market"),
                                     nt.LINE_IC_RETRADING: nt._rule_rate(conv, "gsp_rate"),
                                     nt.LINE_LEGACY_FIT: 3.1 * conv["factor"] / 9.7e7},
                  "new_line_rules": {nt.LINE_NEW_BUILD: nt.POOL, nt.LINE_CM_NOT_ACQUIRED: "capacity_market",
                                     nt.LINE_IC_RETRADING: "gsp_rate", nt.LINE_LEGACY_FIT: "domestic"}}
    out = nt.tariff_line_recovery(["tnuos", "duos"], lines, allocation, conv)
    assert out["tnuos"] == entries["tnuos"]
    for line in lines:
        assert lr.system_to_household_gbp(out[line], 1.0) == pytest.approx(allocation["new_line_rates"][line], rel=1e-12)
    assert out[nt.LINE_NEW_BUILD]["volume_key"] == "gsp_charging_volume_mwh"
    assert out[nt.LINE_CM_NOT_ACQUIRED]["household_allocation_factor"] == pytest.approx(1.348308)
    assert out[nt.LINE_LEGACY_FIT]["volume_key"] == "domestic_electricity_supply_mwh"


def test_the_gate_fee_credit_is_recovered_at_the_pool_rate_on_two_volumes():
    """F19 (27 Oct 2026): income of owned plant, spread as its cost is; a D1 bill keeps it flat."""
    from scripts.public_ownership import nationalised_tariff as nt
    from tests.unit.test_public_ownership_nationalised_tariff import CONV, _private_bill

    _, summary, _, _ = _two_volume_summary()
    entries = summary["line_recovery"]
    private = _private_bill()
    private = private.loc[private["component_id"].isin(set(entries))]
    conv = {"factor": summary["model_rebasing_factor"], "mwh": 3.1, "final_demand_mwh": 2.9e8, "domestic_mwh": 9.7e7,
            "line_recovery": entries, "vat_rate": 0.05}
    allocation = nt.recovery_rates(private, conv, {})
    assert allocation["pool_over_flat"] != pytest.approx(1.0)
    assert allocation["new_line_rules"][nt.LINE_GATE_FEE_INCOME] == nt.POOL
    assert nt.line_rate(allocation, conv, nt.LINE_GATE_FEE_INCOME) == pytest.approx(allocation["pool_rate"])
    lines = {nt.LINE_NEW_BUILD: 1.0e9, nt.LINE_GATE_FEE_INCOME: -1.5e9}
    out = nt.tariff_line_recovery([], lines, allocation, conv)
    gate = out[nt.LINE_GATE_FEE_INCOME]
    assert gate["volume_key"] == out[nt.LINE_NEW_BUILD]["volume_key"] == "gsp_charging_volume_mwh"
    assert gate["household_loss_factor"] == entries["bsuos_constraints"]["household_loss_factor"] > 1.0
    assert lr.system_to_household_gbp(gate, 1.0) == pytest.approx(allocation["pool_rate"], rel=1e-12)
    # The before/after: the flat comparison still prices it flat, so the move shows there.
    _, bill = nt.build_bill(private, lines, conv, allocation=allocation)
    moved = bill["new_lines_gbp_per_3100kwh_ex_vat"][nt.LINE_GATE_FEE_INCOME]
    assert moved == pytest.approx(-1.5e9 * allocation["pool_rate"])
    assert bill["tariff_allocation"]["like_for_like_less_flat_ex_vat_gbp"] == pytest.approx(
        (1.0e9 - 1.5e9) * (allocation["pool_rate"] - allocation["flat_rate"]))
    # A D1 bill (no entries) keeps the credit flat over final demand.
    d1 = nt.recovery_rates(_private_bill(), CONV, {})
    assert nt.LINE_GATE_FEE_INCOME not in d1["new_line_rates"]
    assert nt.line_rate(d1, CONV, nt.LINE_GATE_FEE_INCOME) == pytest.approx(d1["flat_rate"])


# ── leftovers (user decisions) ───────────────────────────────────────────────────

def test_a_wholly_private_store_under_developer_storage_private_floors_at_the_private_rate():
    from scripts.public_ownership import public_account as pa

    frame = pd.DataFrame({
        "case_id": ["public_ownership@developer_storage_private"] * 2,
        "new_build_storage": ["private", "private"], "ownership_share": [0.0, 1.0],
        "annual_capital_recovery_gbp": [100.0, 100.0], "wacc": [0.0883, 0.0883], "remaining_life_years": [30.0, 30.0],
        "annual_fixed_om_gbp": [0.0, 0.0], "annual_variable_om_gbp": [0.0, 0.0]})
    rates = {"private": 0.045, "public_ownership": None}
    floor = pa._floor_parts_original(frame, rates, "case")["row"]
    crf = pa.capital_recovery_factor
    assert floor.iloc[0] == pytest.approx(100.0 / crf(0.0883, 30.0) * crf(0.045, 30.0))
    assert floor.iloc[1] == pytest.approx(100.0)          # a state-held store keeps the case rule


def test_the_institution_comparison_flags_two_solve_runs():
    from scripts.public_ownership.nationalised_institutions import solve_vintage

    market = {"first_written_utc": "2026-10-07T01:00:00+00:00", "last_written_utc": "2026-10-07T08:00:00+00:00"}
    central = {"first_written_utc": "2026-10-03T01:00:00+00:00", "last_written_utc": "2026-10-04T03:00:00+00:00"}
    mixed = solve_vintage(market, central, 24.0)
    assert mixed["mixed_vintage"] is True and mixed["gap_between_runs_hours"] == pytest.approx(70.0)
    one_run = solve_vintage(market, {"first_written_utc": "2026-10-06T20:00:00+00:00",
                                     "last_written_utc": "2026-10-07T02:00:00+00:00"}, 24.0)
    assert one_run["mixed_vintage"] is False
    assert solve_vintage({}, central)["mixed_vintage"] is None


def test_the_mixed_vintage_gap_is_the_configured_one():
    import yaml

    from scripts.public_ownership.nationalised_institutions import MIXED_VINTAGE_GAP_HOURS_DEFAULT

    cfg = load_defaults()
    assert cfg["public_ownership"]["institution_comparison"]["mixed_vintage_gap_hours"] == MIXED_VINTAGE_GAP_HOURS_DEFAULT
