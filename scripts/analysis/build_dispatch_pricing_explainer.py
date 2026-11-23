"""Build the standalone dispatch and wholesale-pricing explainer.

The page deliberately keeps four questions separate: who owns assets, who
schedules them, how accepted energy is settled, and how fixed costs are
recovered.  Quantitative exhibits are read from the published CP30 outputs so
that the prose cannot silently drift away from the model results.

Audit 26 Sep 2026: the central-dispatch headline is the FTI-comparable *net*
saving (gross less the borrowed forecast-error charge), the market side carries
its interconnector re-trading, settlement money is labelled 2024 prices (PyPSA
money, as the bill treats it) and per-household figures use the bill's own
recovery denominator, never a count of households.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:   # run as python scripts/analysis/...: the repository root for scripts.*
    sys.path.insert(0, str(ROOT))
TEMPLATE = ROOT / "scripts" / "analysis" / "dispatch_pricing_explainer_template.html"
OUTPUT = ROOT / "resources" / "analysis" / "dispatch_and_wholesale_pricing_explainer.html"
FTI_SETTLEMENT = (
    ROOT
    / "resources"
    / "market"
    / "CP30_FFR_2030_year_cp30_network_central_dispatch_fti_settlement_copperplate.csv"
)
INSTITUTIONS = (
    ROOT
    / "resources"
    / "public_ownership"
    / "cp30_2030_cp30_network_nationalised_bill_by_institution.json"
)
#: The pay-as-bid bound and the household key, read from the ownership bill itself (30 Sep 2026;
#: previously through the retired explorer's case table).
BILL_BY_CASE = ROOT / "resources" / "public_ownership" / "cp30_2030_cp30_network_consumer_bill_by_ownership_case_summary.json"
PAY_AS_BID_CASE = "private@pay_as_bid_ceiling"
#: Item D4(b): each side's interconnector pass-through and the rent transfer line.
D4B_METRICS = (
    "market_ic_pass_through_credit_gbp_m", "central_ic_pass_through_credit_gbp_m",
    "ic_congestion_rent_transfer_info_gbp_m", "ic_congestion_rent_transfer_gb_half_info_gbp_m",
    "ic_congestion_rent_transfer_foreign_half_info_gbp_m",
)


def _number(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _read_settlement() -> dict[str, dict[str, float | str | None]]:
    with FTI_SETTLEMENT.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        result = {}
        for row in rows:
            metric = row.pop("metric")
            result[metric] = {
                key: (_number(value) if _number(value) is not None else value)
                for key, value in row.items()
            }
    return result


def _months_total(row: dict[str, float | str | None]) -> float:
    return sum(
        float(row[month])
        for month in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        if isinstance(row.get(month), (int, float))
    )


def build_data() -> dict:
    settlement = _read_settlement()
    institutions = json.loads(INSTITUTIONS.read_text(encoding="utf-8"))
    bill = json.loads(BILL_BY_CASE.read_text(encoding="utf-8"))
    ceiling, private = bill["cases"][PAY_AS_BID_CASE], bill["cases"]["private"]

    market_gen = _months_total(settlement["market_resource_cost_generators_only_info_gbp_m"])
    central_gen = _months_total(settlement["central_resource_cost_generators_only_info_gbp_m"])
    measured = institutions["audit"]["measured_deltas_gbp"]
    # The border-trade term: the link trade and (item D3) the rent owed to foreign halves.
    BORDER = ("nationalised_imports", "nationalised_interconnector_foreign_half_rent")
    interconnector_delta_bn = sum(float(measured.get(k, 0.0)) for k in BORDER) / 1e9
    institution_change_bn = institutions["institution_change_system_gbp"] / 1e9
    domestic_change_bn = institution_change_bn - interconnector_delta_bn
    total_household_change = institutions["institution_change_inc_vat_gbp"]
    # Per line, at the rate the tariff recovers it (like-for-like allocation,
    # 26 Sep 2026), rather than a pro-rata share of the system change.
    vat = float(institutions["i3_inc_vat_gbp"]) / float(institutions["i3_ex_vat_gbp"])
    lines_csv = INSTITUTIONS.with_suffix(".csv")
    if lines_csv.exists():
        with lines_csv.open(encoding="utf-8-sig", newline="") as handle:
            per_line = {r["component_id"]: float(r["change_gbp_per_3100kwh_ex_vat"]) for r in csv.DictReader(handle)}
        interconnector_household_change = sum(per_line.get(k, 0.0) for k in BORDER) * vat
    else:
        interconnector_household_change = total_household_change * (interconnector_delta_bn / institution_change_bn)

    def year(metric: str) -> float:
        value = settlement[metric].get("YEAR")
        if not isinstance(value, (int, float)):
            raise ValueError(f"No numeric YEAR value for {metric}")
        return float(value)

    if not {"consumer_saving_fti_comparable_net_gbp_m", "central_realtime_correction_replaced_gbp_m"} <= set(settlement):
        raise ValueError(f"{FTI_SETTLEMENT} predates the 26 Sep 2026 settlement (no re-trading line, no "
                         "forecast-error charge replacing the real-time correction); rerun rule "
                         "build_public_ownership_fti_settlement")
    if not set(D4B_METRICS) <= set(settlement):
        raise ValueError(f"{FTI_SETTLEMENT} predates item D4(b) (no interconnector pass-through on either side, "
                         "no congestion-rent transfer line); rerun rule build_public_ownership_fti_settlement")
    # House rule 8: system GBP to GBP per 3,100 kWh household through the bill's own
    # recovery of a balancing cost (the BSUoS constraint line's entry: a GSP rate on the
    # two-volume basis, F19), rebased to 2025.
    from scripts.public_ownership import bill_recovery

    per_household = bill_recovery.household_per_system(bill, "bsuos_constraints")
    settlement_basis = settlement.get("price_basis", {}).get("YEAR") or "GBP real 2024 (PyPSA model money)"

    return {
        "meta": {
            "generated": date.today().isoformat(),
            "model_year": 2030,
            "price_basis": bill["price_basis"],
            "settlement_price_basis": settlement_basis,
            "scope": "CP30 FFR 2030, twelve reinforced monthly runs",
        },
        "fti_style": {
            "demand_twh": year("gb_demand_twh"),
            "market_energy_bn": year("market_energy_gbp_m") / 1000,
            "market_constraint_bn": year("market_constraint_cost_gbp_m") / 1000,
            "market_retrading_bn": year("market_interconnector_retrading_gbp_m") / 1000,
            # Item D4(b): each side's cap-and-floor pass-through (a credit lowers the
            # consumer cost) and the rent difference, a transfer entering no total.
            "market_ic_pass_through_bn": year("market_ic_pass_through_credit_gbp_m") / 1000,
            "central_ic_pass_through_bn": year("central_ic_pass_through_credit_gbp_m") / 1000,
            "ic_rent_transfer_bn": year("ic_congestion_rent_transfer_info_gbp_m") / 1000,
            "ic_rent_transfer_gb_half_bn": year("ic_congestion_rent_transfer_gb_half_info_gbp_m") / 1000,
            "ic_rent_transfer_foreign_half_bn": year("ic_congestion_rent_transfer_foreign_half_info_gbp_m") / 1000,
            "market_consumer_bn": year("market_consumer_cost_gbp_m") / 1000,
            "market_redispatch_twh": year("market_redispatch_volume_twh"),
            "central_energy_bn": year("central_energy_gbp_m") / 1000,
            "central_cost_recovery_bn": year("central_cost_recovery_gbp_m") / 1000,
            "central_firm_access_bn": year("central_firm_access_gbp_m") / 1000,
            "central_realtime_bn": year("central_realtime_correction_gbp_m") / 1000,
            "central_consumer_bn": year("central_consumer_cost_gbp_m") / 1000,
            "central_realtime_twh": year("central_realtime_volume_twh"),
            "consumer_saving_gross_bn": year("consumer_saving_gross_gbp_m") / 1000,
            "forecast_error_charge_bn": year("forecast_error_charge_gbp_m") / 1000,
            # The charge replaces the model's own real-time correction (26 Sep 2026).
            "central_realtime_replaced_bn": year("central_realtime_correction_replaced_gbp_m") / 1000,
            # The headline (user decision, 26 Sep 2026): net of the borrowed charge.
            "consumer_saving_bn": year("consumer_saving_fti_comparable_net_gbp_m") / 1000,
            "consumer_saving_gbp_per_household_ex_vat": year("consumer_saving_fti_comparable_net_gbp_m") * 1e6 * per_household,
            "consumer_saving_gross_gbp_per_household_ex_vat": year("consumer_saving_gross_gbp_m") * 1e6 * per_household,
            "reported_resource_saving_bn": year("resource_saving_gbp_m") / 1000,
            "market_generator_resource_bn": market_gen / 1000,
            "central_generator_resource_bn": central_gen / 1000,
            "generator_resource_saving_bn": (market_gen - central_gen) / 1000,
            "generator_resource_saving_pct": (market_gen - central_gen) / market_gen * 100,
        },
        "nationalised": {
            "market_bill": institutions["i1_inc_vat_gbp"],
            "central_bill": institutions["i3_inc_vat_gbp"],
            "headline_change": institutions["institution_change_inc_vat_gbp"],
            "system_change_bn": institution_change_bn,
            "interconnector_change_bn": interconnector_delta_bn,
            "interconnector_household_change": interconnector_household_change,
            "domestic_household_change": total_household_change - interconnector_household_change,
            "generator_resource_change_pct": institutions["audit"]["generator_resource_cost_change_share"] * 100,
            "forecast_error_household_inc_vat": float(institutions.get("forecast_error_charge_ex_vat_gbp", 0.0))
            * (float(institutions["i3_inc_vat_gbp"]) / float(institutions["i3_ex_vat_gbp"])),
        },
        "pay_as_bid": {
            "private_bill": private["net_total_gbp_inc_vat"],
            "ceiling_bill": ceiling["net_total_gbp_inc_vat"],
            "ceiling_change": ceiling["net_total_gbp_inc_vat"] - private["net_total_gbp_inc_vat"],
            "ceiling_system_bn": ceiling["pay_as_bid_ceiling_gbp_2024"] / 1e9,
        },
    }


def main() -> None:
    import argparse

    global FTI_SETTLEMENT, INSTITUTIONS, BILL_BY_CASE
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--group", default="cp30_2030_cp30_network",
                        help="the run group whose institution comparison and bill the page reads (network bounds, "
                             "18-19 Oct 2026: public_ownership.report_group)")
    parser.add_argument("--fti-settlement", default=None,
                        help="the group's FTI settlement (its institution_comparison.fti_settlement)")
    args = parser.parse_args()
    public = ROOT / "resources" / "public_ownership"
    INSTITUTIONS = public / f"{args.group}_nationalised_bill_by_institution.json"
    BILL_BY_CASE = public / f"{args.group}_consumer_bill_by_ownership_case_summary.json"
    if args.fti_settlement:
        FTI_SETTLEMENT = Path(args.fti_settlement) if Path(args.fti_settlement).is_absolute() else ROOT / args.fti_settlement
    data = build_data()
    template = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = template.replace("/*__DISPATCH_DATA__*/", payload)
    if html == template:
        raise RuntimeError("Template data marker was not found")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
