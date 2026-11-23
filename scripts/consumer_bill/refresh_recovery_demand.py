"""Refresh the recovery-demand volumes of a bill case from its solved months (review 28 Sep 2026).

The recovery-demand file (``data/market/consumer_bill_forecast/2030_recovery_demand_assumptions.json``)
carries the volumes every levy and charge is recovered over, and the bill projector
refuses a run whose volumes differ from the file. The file was kept by hand, so every
re-solve left it describing the previous run. This script measures the volumes the
projector itself measures (:func:`scripts.consumer_bill.project_future_bill.read_hourly_inputs`
on each month's Stage 1 wholesale network, price and link schedule) and rewrites the
model-dependent fields, recomputing the derived ones on their declared identities:

    gsp_chargeable_volume = GB electrical load + electrolyser draw
    final_demand          = gsp_chargeable_volume / gsp_to_meter_loss_factor
    eligible_supplier_sales = final_demand x eligible_share_of_supplier_sales
    ro_obligation_eligible_supply = eligible_supplier_sales

A file on the two-volume basis (F18, the 2030 file: ``recovery_basis``
``loss_inclusive_two_volume``) recomputes its metered supply and GSP charging volume
from the load and the declared factors instead, with their eligible shares and the
deprecated aliases (``_refresh_two_volume``).

The sourced shares (loss factor, EII share) are kept. It also checks the Capacity
Market line's pinned cost against the builder's output when both are given. Run it
after the solve and before the bill:

    python -m scripts.consumer_bill.refresh_recovery_demand \\
        --recovery data/market/consumer_bill_forecast/2030_recovery_demand_assumptions.json \\
        --scenarios <the twelve month scenarios> \\
        [--assumptions data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json \\
         --capacity-market-summary resources/market/capacity_market/cm_calendar_2030_summary.json]
"""

import argparse
import copy
import json
from datetime import date
from pathlib import Path

import numpy as np

MARKET_DIR = Path("resources/market")
#: F18: a file on the two-volume basis (2030) refreshes its own volumes; others are on item D1.
TWO_VOLUME_BASIS = "loss_inclusive_two_volume"
VOLUME_KEYS = ("system_demand_mwh", "gb_electrical_load_mwh", "electrolyser_draw_mwh", "h2_offtake_mwh")


def measure_volumes(scenarios, market_dir=MARKET_DIR) -> dict:
    """The year's volumes, MWh, exactly as the bill projector measures them."""
    from scripts.consumer_bill.project_future_bill import read_hourly_inputs

    totals = dict.fromkeys(VOLUME_KEYS, 0.0)
    for scenario in scenarios:
        base = Path(market_dir) / scenario
        links = Path(f"{base}_wholesale_links.csv")
        hourly = read_hourly_inputs(f"{base}_wholesale.nc", f"{base}_wholesale_price.csv",
                                    links if links.exists() else None)
        hours = hourly["hours"]
        totals["system_demand_mwh"] += float((hourly["total_load_mw"] * hours).sum())
        totals["gb_electrical_load_mwh"] += float((hourly["gb_electrical_load_mw"] * hours).sum())
        totals["electrolyser_draw_mwh"] += float((hourly["electrolyser_draw_mw"] * hours).sum())
        totals["h2_offtake_mwh"] += float((hourly["h2_offtake_mw"] * hours).sum())
    return totals


def system_demand_source(scenarios, aggregate_study: str | None = None) -> str:
    """The ``system_demand_mwh.source`` label: the months measured and, when given, their annual aggregate.

    27 Oct 2026: the label was static and still named the ``reinforced`` aggregate after the value
    had moved to the calibrated months, so the writer now states what it measured.
    """
    scenarios = list(scenarios)
    text = (f"PyPSA annual total of the {len(scenarios)} monthly Stage 1 runs the bill reads ({scenarios[0]} .. "
            f"{scenarios[-1]}: resources/market/<month>_wholesale.nc, every Load's p_set summed over the hours, as "
            "read_hourly_inputs measures it)")
    if aggregate_study:
        text += (f"; the annual aggregate resources/analysis/{aggregate_study}_annual_summary.json "
                 "(aggregate.demand_MWh) carries the same total")
    return text


def refresh_document(document: dict, volumes: dict, source: str, demand_source: str | None = None) -> dict:
    """``document`` with its model volumes replaced and every derived field recomputed.

    ``demand_source`` (:func:`system_demand_source`) replaces ``system_demand_mwh.source`` when given.
    """
    doc = copy.deepcopy(document)
    stamp = f"refreshed {date.today().isoformat()} by scripts/consumer_bill/refresh_recovery_demand.py from {source}"
    load, draw, offtake = (float(volumes[k]) for k in ("gb_electrical_load_mwh", "electrolyser_draw_mwh", "h2_offtake_mwh"))
    system = float(volumes["system_demand_mwh"])
    if not np.isclose(load + offtake, system, rtol=1e-9, atol=1.0):
        raise ValueError(f"GB electrical load {load:,.0f} plus hydrogen offtake {offtake:,.0f} does not equal "
                         f"system demand {system:,.0f}; the months disagree with the projector's accounting")
    basis = doc["metered_supply_basis"]
    doc["system_demand_mwh"]["value"] = system
    doc["system_demand_mwh"]["refreshed"] = stamp
    if demand_source is not None:
        doc["system_demand_mwh"]["source"] = demand_source
    basis["gb_electrical_load_mwh"]["value"] = load
    basis["electrolyser_electricity_draw_mwh"]["value"] = draw
    basis["h2_offtake_mwh"]["value"] = offtake
    if doc.get("recovery_basis") == TWO_VOLUME_BASIS:
        return _refresh_two_volume(doc, volumes, stamp, load, draw)
    gsp = load + draw
    basis["gsp_chargeable_volume_mwh"]["value"] = gsp
    loss = float(basis["gsp_to_meter_loss_factor"]["value"])
    final = gsp / loss
    doc["final_demand_mwh"]["value"] = final
    share = float(doc["cfd_eligible_supply"]["eligible_share_of_supplier_sales"]["value"])
    eligible = final * share
    doc["cfd_eligible_supply"]["eligible_supplier_sales_mwh"]["value"] = eligible
    if "ro_obligation_eligible_supply_mwh" in doc:
        doc["ro_obligation_eligible_supply_mwh"]["value"] = eligible
    doc["refresh"] = {"stamp": stamp, "volumes_mwh": {k: float(volumes[k]) for k in VOLUME_KEYS},
                      "derived_mwh": {"gsp_chargeable_volume": gsp, "final_demand": final,
                                      "eligible_supplier_sales": eligible}}
    return doc


def _refresh_two_volume(doc, volumes, stamp, load, draw):
    """F18 (6 Oct 2026): the derived volumes of a two-volume file, and their deprecated aliases.

        metered_supply      = load / system_to_metered_factor + draw
        gsp_charging_volume = load / system_to_metered_factor x metered_to_gsp_factor + draw
        cfd_eligible_gsp_volume = gsp_charging_volume x eligible share
        ro_obligation_eligible_supply = metered_supply x eligible share
    """
    basis = doc["metered_supply_basis"]
    to_metered = float(basis["system_to_metered_factor"]["value"])
    to_gsp = float(basis["metered_to_gsp_factor"]["value"])
    metered = load / to_metered + draw
    gsp = load / to_metered * to_gsp + draw
    share = float(doc["cfd_eligible_supply"]["eligible_share_of_supplier_sales"]["value"])
    cfd_gsp, ro = gsp * share, metered * share
    basis["metered_supply_mwh"]["value"] = metered
    basis["gsp_charging_volume_mwh"]["value"] = gsp
    doc["cfd_eligible_supply"]["cfd_eligible_gsp_volume_mwh"]["value"] = cfd_gsp
    doc["ro_obligation_eligible_supply_mwh"]["value"] = ro
    # The deprecated aliases follow the volumes they alias.
    basis["gsp_chargeable_volume_mwh"]["value"] = gsp
    doc["final_demand_mwh"]["value"] = metered
    doc["cfd_eligible_supply"]["eligible_supplier_sales_mwh"]["value"] = ro
    doc["refresh"] = {"stamp": stamp, "volumes_mwh": {k: float(volumes[k]) for k in VOLUME_KEYS},
                      "derived_mwh": {"metered_supply": metered, "gsp_charging_volume": gsp,
                                      "cfd_eligible_gsp_volume": cfd_gsp, "ro_eligible_metered_supply": ro,
                                      "gsp_chargeable_volume": gsp, "final_demand": metered,
                                      "eligible_supplier_sales": ro}}
    return doc


def check_capacity_market_pin(assumptions: dict, builder_summary: dict, rtol: float = 1e-6) -> dict:
    """Compare the bill's pinned calendar-year CM cost with the builder's; raise on drift.

    The year is the bill's own ``benchmark.bill_calendar_year`` (item Y4; the default
    model year when the file does not declare one).
    """
    from scripts.public_ownership.model_year import DEFAULT_MODEL_YEAR

    year = int((assumptions.get("benchmark") or {}).get("bill_calendar_year") or DEFAULT_MODEL_YEAR)
    key = f"calendar_{year}_cost_gbp"
    pinned = float(assumptions["sourced_externals"]["capacity_market"]["derivation"]["agreement_cost_gbp"])
    built = builder_summary.get(key, builder_summary.get("agreement_cost_gbp"))
    if built is None:
        raise ValueError(f"the Capacity Market builder summary carries no {key}")
    built = float(built)
    if not np.isclose(pinned, built, rtol=rtol):
        raise ValueError(f"the bill pins a calendar-{year} Capacity Market cost of GBP{pinned:,.0f} but the builder now "
                         f"gives GBP{built:,.0f}; update sourced_externals.capacity_market.derivation.agreement_cost_gbp")
    return {"pinned_gbp": pinned, "builder_gbp": built, "agrees": True}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recovery", required=True)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--market-dir", default=str(MARKET_DIR))
    parser.add_argument("--out", default=None, help="Write here instead of overwriting --recovery")
    parser.add_argument("--assumptions", default=None)
    parser.add_argument("--capacity-market-summary", default=None)
    parser.add_argument("--aggregate-study", default=None,
                        help="the bill case's annual_aggregate_study (e.g. CP30_2030_cp30_network), named in the label")
    args = parser.parse_args(argv)
    path = Path(args.recovery)
    document = json.loads(path.read_text(encoding="utf-8"))
    volumes = measure_volumes(args.scenarios, args.market_dir)
    refreshed = refresh_document(document, volumes, f"{len(args.scenarios)} months: {args.scenarios[0]} .. {args.scenarios[-1]}",
                                 demand_source=system_demand_source(args.scenarios, args.aggregate_study))
    out = Path(args.out) if args.out else path
    out.write_text(json.dumps(refreshed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(refreshed["refresh"], indent=2))
    if args.assumptions and args.capacity_market_summary:
        result = check_capacity_market_pin(json.loads(Path(args.assumptions).read_text(encoding="utf-8")),
                                           json.loads(Path(args.capacity_market_summary).read_text(encoding="utf-8")))
        print(f"Capacity Market pin agrees with the builder: GBP{result['pinned_gbp']:,.0f}")


if __name__ == "__main__":
    main()
