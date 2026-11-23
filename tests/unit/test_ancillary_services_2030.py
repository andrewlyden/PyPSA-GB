"""The 2030 ancillary services on CP30's drivers (user decision, 6 Oct 2026).

One table (``ancillary_services_2030.csv``) sizes both the bill's two non-thermal BSUoS lines and
the register's ancillary income, so these tests pin: the table is the projection the script writes,
its CP30 inputs are the workbook's, the bill lines hold exactly the services the register and the
providers outside it are paid, in every case, and the held lines are still the outturn.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.public_ownership import ancillary_services_2030 as an
from scripts.public_ownership import support_lines_2050 as sl

TABLE = Path("data/public_ownership/financing/ancillary_services_2030.csv")
ASSUMPTIONS = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")


def test_the_table_is_the_cp30_projection():
    table = pd.read_csv(TABLE)
    pd.testing.assert_frame_equal(an.project(table), table, check_dtype=False)
    assert an.main(["--check"]) == 0
    assert table["projection_basis"].str.startswith("2030 (user decision, 6 Oct 2026)").all()
    assert set(table["model_year"]) == {2030}


def test_cp21_is_read_from_the_workbook():
    if not an.WORKBOOK.exists():
        pytest.skip("CP30 workbook not present")
    assert an.read_cp21() == an.CP21["regions"]
    totals = an.cp21_totals()
    assert totals["scl_mva"] == {"low": 23106.0, "central": 37053.0, "high": 51000.0}
    assert totals["reactive_mvar"] == {"low": 4611.0, "central": 6255.5, "high": 7900.0}
    assert an.band("1001-5000") == (1001.0, 5000.0) and an.band(0) == (0.0, 0.0) and an.band(None) == (0.0, 0.0)


def test_unit_costs_and_ratios_are_the_published_figures():
    u = an.unit_costs()
    assert u["scl_gbp_per_mva_year"] == pytest.approx((32.3 + 130.0) * 1e6 / 20270.0)
    assert u["reactive_gbp_per_mvar_year"] == pytest.approx(83e6 / 10 / 646)
    r = an.ratios()
    assert (r["response"], r["positive"], r["negative"]) == pytest.approx((4.8 / 4.2, 6.4 / 4.4, 5.2 / 2.4))
    assert an.stability_book_gbp_m() == pytest.approx(32.3 + 130.0 + 10.3)


def test_each_service_follows_its_driver():
    first = sl.load_ancillary(TABLE).groupby("service").first()
    spend = lambda s, c="central": float(first.loc[s, f"spend_2050_{c}_gbp_m_2024"])  # noqa: E731
    outturn = lambda s: float(first.loc[s, "outturn_spend_gbp_m"])  # noqa: E731
    r, t, u = an.ratios(), an.cp21_totals(), an.unit_costs()
    # CP30 holds the BM voltage and inertia actions (Annex 4 s.2.1.5); ORPS carries today's need.
    for held in ("voltage_bm_actions", "inertia_bm_actions", "obligatory_reactive_power_service",
                 "pumped_storage_bilateral_reserve"):
        assert spend(held) == pytest.approx(outturn(held))
    assert spend("dynamic_containment") == pytest.approx(outturn("dynamic_containment") * r["response"], abs=1e-4)
    assert spend("stor_and_slow_reserve") == pytest.approx(outturn("stor_and_slow_reserve") * r["positive"], abs=1e-4)
    assert spend("quick_reserve") == pytest.approx((8.95 + 103.04) * r["both"], abs=1e-4)
    assert spend("fast_reserve_optional") == spend("enhanced_frequency_response") == 0.0
    for case in ("low", "central", "high"):
        assert spend("stability_network_services", case) == pytest.approx(
            an.stability_book_gbp_m() + t["scl_mva"][case] * u["scl_gbp_per_mva_year"] / 1e6, abs=1e-4)
        assert spend("commercial_reactive_and_voltage_network_services", case) == pytest.approx(
            8.11 + t["reactive_mvar"][case] * u["reactive_gbp_per_mvar_year"] / 1e6, abs=1e-4)
    # The 2030 stability providers: no Phase 1 plant, grid-forming batteries take Phase 2's share.
    stability = sl.load_ancillary(TABLE).query("service == 'stability_network_services'")
    assert dict(zip(stability["provider_carrier"], stability["participation_share"])) == an.stability_providers()
    assert "pumped_hydro" not in set(stability["provider_carrier"])


def test_the_bill_lines_hold_what_the_register_and_outside_providers_are_paid():
    table = sl.load_ancillary(TABLE)
    se = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))["sourced_externals"]
    assets = pd.DataFrame({"asset_id": ["B", "C", "O", "P", "W", "N"],
                           "carrier": ["Battery", "CCGT", "OCGT", "pumped_hydro", "wind_offshore", "nuclear"],
                           "capacity_mw": [1000.0, 2000.0, 500.0, 1000.0, 3000.0, 1000.0]})
    for case in sl.CASES:
        held = sl.services_held_by_line(table, case)
        out, report = sl.ancillary_by_row(assets, table, case)
        assert report["allocated_gbp"] + sum(report["unallocated_gbp"].values()) == pytest.approx(sum(held.values()))
        assert held["bsuos_other_services"] + held["bsuos_voltage_and_stability_constraints"] == pytest.approx(
            float(out[sl.COLUMN_ANCILLARY].sum()) + sum(report["unallocated_gbp"].values()))
        for line in held:
            d = se[line]["derivation"]
            expected = d["cost_gbp"] if case == "central" else d["sensitivity_cost_gbp"][f"ancillary_{case}"]
            assert expected == pytest.approx(held[line])
    low, central, high = (sl.services_held_by_line(table, c) for c in ("low", "central", "high"))
    for line in central:
        assert low[line] <= central[line] <= high[line]
    assert central["bsuos_voltage_and_stability_constraints"] == pytest.approx(194.62e6)
    other = se["bsuos_other_services"]
    assert other["period_basis"] == "extrapolated_declared" and other["derivation"]["ancillary_case"] == "central"
    assert other["derivation"]["fy2024_25"]["services_outturn_gbp"] == pytest.approx(720.8e6)
