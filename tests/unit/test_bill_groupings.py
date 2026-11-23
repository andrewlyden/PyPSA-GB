"""Bill-line groupings as dated data (6 Oct 2026): coverage, Ofgem placements and the guard."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.consumer_bill import bill_groupings
from scripts.consumer_bill.bill_groupings import GroupingError

BENCHMARKS = Path("data/market/consumer_bill_benchmarks")


def _declared_line_ids() -> set[str]:
    """Every bill line id the code or the bill inputs can emit."""
    from scripts.consumer_bill import project_future_bill as projector
    from scripts.public_ownership import nationalised_tariff as tariff
    from scripts.public_ownership import project_public_ownership_consumer_bill as ownership

    ids = set(pd.read_csv(BENCHMARKS / "bill_line_sources.csv")["component_id"])
    for year in (2030, 2050):
        assumptions = json.loads((BENCHMARKS / f"{year}_pypsa_informed_assumptions.json").read_text(encoding="utf-8"))
        ids |= set(assumptions.get("sourced_externals", {}))
        ids |= set(assumptions.get("external_assumptions_gbp_per_3100kwh_ex_vat", {}))
    ids |= set(projector.EXTERNAL_COMPONENTS)
    ids |= {f"procurement_{key}" for key, _ in projector.PROCUREMENT_ALLOWANCES}
    ids |= {"direct_energy", "shaping_adjustment", "procurement_forward_premium", "cfd", "allocated_ccs", "dpa",
            "bsuos_constraints", "renewables_obligation", "interconnector_cap_and_floor_pass_through",
            "ldes_cap_and_floor"}
    ids |= set(tariff.NEW_LINES) | set(tariff.OPTIONAL_NEW_LINE_LABELS) | set(tariff.ALL_REPLACED_LINES)
    ids |= set(tariff.KEPT_DECLARED) | set(tariff.REGISTER_REPLACED_LINES)
    ids |= {value for name, value in vars(ownership).items() if name.startswith("LINE_") and isinstance(value, str)}
    ids |= set(ownership.NEW_LINES) | set(ownership.BACKGROUND_LINES) | set(ownership.DOMESTIC_LEVY_LINES.values())
    return ids


@pytest.mark.parametrize("view", bill_groupings.VIEWS)
def test_every_declared_line_has_a_group_in_every_current_view(view):
    groups = bill_groupings.group_of(view)
    missing = sorted(_declared_line_ids() - set(groups.index))
    assert not missing, f"lines with no {view!r} group: {missing}"


def test_every_grouped_line_is_a_real_line():
    stale = sorted(set(bill_groupings.load_groupings()["component_id"]) - _declared_line_ids())
    assert not stale, f"groupings for lines no code emits: {stale}"


def test_each_view_has_one_current_vintage_and_every_vintage_is_complete():
    views = bill_groupings.load_views()
    assert set(views["view"]) == set(bill_groupings.VIEWS)
    rows = bill_groupings.load_groupings()
    per_vintage = rows.groupby(["view", "vintage"])["component_id"].apply(frozenset)
    assert per_vintage.nunique() == 1, "every view and vintage must group the same set of lines"


@pytest.mark.parametrize("line, group", [
    # Ofgem's cap model v1.31 and Annex 2 v1.30 (Oct-Dec 2026): CfD and Capacity Market are wholesale,
    # losses sit inside direct fuel, BSUoS is a network cost, the policy levies are Annex 4.
    ("direct_energy", "Wholesale"), ("procurement_losses", "Wholesale"), ("cfd", "Wholesale"),
    ("capacity_market", "Wholesale"), ("allocated_ccs", "Wholesale"), ("remaining_ccs", "Wholesale"),
    ("bsuos_constraints", "Network costs"), ("bsuos_other_services", "Network costs"),
    ("bsuos_voltage_and_stability_constraints", "Network costs"), ("tnuos", "Network costs"),
    ("duos", "Network costs"), ("ldes_cap_and_floor", "Network costs"),
    ("renewables_obligation", "Policy costs"), ("feed_in_tariffs", "Policy costs"),
    ("warm_home_discount", "Policy costs"), ("sizewell_c_rab", "Policy costs"), ("bics_levy", "Policy costs"),
    ("aahedc", "Policy costs"), ("eco", "Policy costs"),
    ("supplier_costs_and_margin", "Operating costs, EBIT and headroom"),
    ("smart_meter_net_cost_change", "Operating costs, EBIT and headroom"),
])
def test_ofgem_2026q4_placements(line, group):
    assert bill_groupings.group_of("ofgem", "2026Q4")[line] == group


def test_ofgem_group_order_follows_the_cap_model():
    assert bill_groupings.group_order("ofgem", "2026Q4") == [
        "Wholesale", "Policy costs", "Network costs", "Operating costs, EBIT and headroom"]


def test_annotate_adds_both_views_and_keeps_the_source_columns():
    frame = pd.DataFrame({"component_id": ["cfd", "tnuos"], "stack_group": ["policy", "network"],
                          "category": ["Policy costs", "Network costs"], "annual_gbp_per_3100kwh_ex_vat": [1.0, 2.0]})
    out = bill_groupings.annotate(frame)
    assert out["ofgem_group"].tolist() == ["Wholesale", "Network costs"]
    assert out["functional_group"].tolist() == ["Low-carbon support schemes", "Networks"]
    assert (out["ofgem_group_vintage"] == bill_groupings.current_vintage("ofgem")).all()
    assert out["category"].tolist() == frame["category"].tolist()
    assert "ofgem_group" not in frame.columns


def test_annotate_refuses_a_line_with_no_group():
    frame = pd.DataFrame({"component_id": ["cfd", "a_new_levy"], "annual_gbp_per_3100kwh_ex_vat": [1.0, 2.0]})
    with pytest.raises(GroupingError, match="a_new_levy"):
        bill_groupings.annotate(frame)


def test_group_totals_add_up_to_the_bill():
    ids = sorted(bill_groupings.group_of("ofgem").index)
    frame = pd.DataFrame({"component_id": ids, "annual_gbp_per_3100kwh_ex_vat": range(len(ids))})
    for view in bill_groupings.VIEWS:
        totals = bill_groupings.totals_by_group(frame, view)
        assert list(totals.index) == bill_groupings.group_order(view)
        assert totals.sum() == pytest.approx(frame["annual_gbp_per_3100kwh_ex_vat"].sum())
    block = bill_groupings.summary_block(frame)
    assert block["ofgem"]["vintage"] == "2026Q4" and "Oct-Dec 2026" in block["ofgem"]["title"]


def test_a_new_vintage_must_be_declared(tmp_path):
    rows = bill_groupings.load_groupings()
    extra = rows.loc[rows["view"].eq("ofgem")].assign(vintage="2027Q1")
    path = tmp_path / "groupings.csv"
    pd.concat([rows, extra]).to_csv(path, index=False)
    with pytest.raises(GroupingError, match="not declared"):
        bill_groupings.load_groupings(path)
