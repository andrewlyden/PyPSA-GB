"""OW15: the onshore repowering share at 0.5, a report-only sensitivity on the acquisition tariff (user, 3 Oct 2026).

The central legacy valuation repowers 30% of each onshore cohort at 25 years and life-extends the rest to 35
(``public_ownership.acquisition.valuation.repowering_share_by_carrier``; FES 2050 decision 62). This report
revalues the legacy fleet with the share at 0.5 (``public_ownership.acquisition.repowering_sensitivity``)
and carries the change in the valued group's market value to the acquisition tariff the way the tariff prices a
market-value purchase (:func:`nationalised_tariff.legacy_capital_variants`): the change in price times the
capital recovery factor at the tariff's public rate and term, spread over final demand at the tariff's own flat
rate, plus VAT. Only the purchase price moves; the 2030 register (the CP30 network's fleet) and its operation do
not, so it is not a case and nothing else in the bill changes.

    python scripts/public_ownership/repowering_sensitivity.py REGISTER SNAPSHOT VALUATION TARIFF_SUMMARY OUT
        [--config config] [--run-group cp30_2030_cp30_network]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.public_ownership import legacy_valuation as lv  # noqa: E402
from scripts.public_ownership.finance import capital_recovery_factor  # noqa: E402
from config.config_loader import read_config

#: Defaults for ``public_ownership.acquisition.repowering_sensitivity`` (outside ``valuation``, which
#: accepts only :data:`legacy_valuation.DEFAULT_SETTINGS`).
DEFAULTS = {"enabled": False, "carrier": "wind_onshore", "share": 0.5}


def sensitivity_settings(acquisition_block: Mapping | None) -> dict:
    """The sensitivity's settings over :data:`DEFAULTS`; refuses a carrier with no central repowering rule."""
    acquisition_block = acquisition_block or {}
    merged = {**DEFAULTS, **(acquisition_block.get("repowering_sensitivity") or {})}
    unknown = sorted(set(merged) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"acquisition.repowering_sensitivity has unknown keys {unknown}")
    valuation_block = acquisition_block.get("valuation") or {}
    share = float(merged["share"])
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"repowering_sensitivity.share must be in [0, 1], not {share}")
    central = valuation_block.get("repowering_share_by_carrier") or {}
    if merged["enabled"] and merged["carrier"] not in central:
        raise ValueError(f"repowering_sensitivity names {merged['carrier']!r}, which has no central repowering rule "
                         "(valuation.repowering_share_by_carrier)")
    return {**merged, "share": share}


def valued_market_value(valuation: pd.DataFrame) -> pd.Series:
    """Per-asset market value of the valued group, floored at zero as the tariff prices it."""
    v = valuation.loc[valuation["scope_group"].astype(str).eq(lv.GROUP_VALUED)].set_index("asset_id")
    return pd.to_numeric(v["pv_market_gbp"], errors="coerce").fillna(0.0).clip(lower=0.0)


def tariff_effect(delta_price_gbp: float, tariff_summary: Mapping, public_rate: float) -> dict:
    """The change in the tariff's annual charge and household bill from a change in the purchase price."""
    term = float(tariff_summary["acquisition_terms"]["term_years"])
    crf = capital_recovery_factor(float(public_rate), term)
    # F19 (7 Oct 2026): on the two-volume basis the purchase annuity is recovered at the tariff's own
    # rate for it (the pool); a D1 (2050) tariff keeps the flat final-demand rate.
    allocation = tariff_summary["allocation"]
    two_volume = (tariff_summary.get("conversion") or {}).get("recovery_basis") == "loss_inclusive_two_volume"
    rate = float((allocation.get("new_line_rates") or {}).get("nationalised_legacy_acquisition_renewables_nuclear")
                 if two_volume else allocation["rule_rates"]["final_demand"])
    vat = float(tariff_summary["conversion"]["vat_rate"])
    annual = delta_price_gbp * crf
    ex_vat = annual * rate
    return {"term_years": term, "public_real_rate": float(public_rate), "capital_recovery_factor": crf,
            "annual_charge_change_gbp": annual, "bill_change_gbp_per_household_ex_vat": ex_vat,
            "bill_change_gbp_per_household_inc_vat": ex_vat * (1.0 + vat)}


def build(register: pd.DataFrame, central_valuation: pd.DataFrame | None, settings: Mapping, sensitivity: Mapping,
          stations: pd.DataFrame, public_rate: float, tariff_summary: Mapping) -> dict:
    """Revalue at the central and the sensitivity share and report the change.

    Both valuations are computed here from the same register and settings, so only the share differs.
    ``central_valuation`` (the workflow's valuation CSV) must match the central one computed here, or the
    file is stale against the config and the report is refused; ``None`` skips that check.
    """
    carrier = sensitivity["carrier"]
    alt_settings = {**settings, "repowering_share_by_carrier": {**settings["repowering_share_by_carrier"],
                                                                carrier: sensitivity["share"]}}
    central_frame, _, _ = lv.value_legacy_fleet(register, public_rate, settings, stations)
    alt_valuation, _, _ = lv.value_legacy_fleet(register, public_rate, alt_settings, stations)
    central, alt = valued_market_value(central_frame), valued_market_value(alt_valuation)
    if central_valuation is not None:
        on_file = valued_market_value(central_valuation)
        gap = (on_file.reindex(central.index) - central).abs()
        if not on_file.index.sort_values().equals(central.index.sort_values()) or float(gap.max()) > 1.0:
            raise ValueError("the legacy valuation file does not match a valuation on the current config and "
                             "register; rebuild it (rule build_public_ownership_legacy_valuation) first")
    if not central.index.equals(alt.index):
        raise ValueError("the sensitivity valuation's valued assets differ from the central valuation's")
    moved = (alt - central).abs() > 1.0
    other = central_frame.set_index("asset_id").reindex(central.index[moved])["carrier"].astype(str)
    if (other != carrier).any():
        raise ValueError(f"the repowering share moved assets other than {carrier}: {sorted(other[other != carrier].unique())}")
    delta = float(alt.sum() - central.sum())
    return {
        "item": "OW15 repowering-share sensitivity (report-only; user, 3 Oct 2026)",
        "carrier": carrier,
        "central_share": float(settings["repowering_share_by_carrier"][carrier]),
        "sensitivity_share": sensitivity["share"],
        "assets_moved": int(moved.sum()),
        "valued_pv_market_central_gbp": float(central.sum()),
        "valued_pv_market_sensitivity_gbp": float(alt.sum()),
        "purchase_price_change_gbp": delta,
        "tariff": tariff_effect(delta, tariff_summary, public_rate),
        "applies_to": "every acquisition tariff case priced at market value (P3 and its twins); the purchase price only",
        "method": __doc__.split("\n\n")[1].replace("\n", " "),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("finance_register", "snapshot", "valuation", "tariff_summary", "out"):
        parser.add_argument(name)
    parser.add_argument("--config", default="config")
    parser.add_argument("--run-group", default=None)
    parser.add_argument("--price-level-settlement", default=None,
                        help="the asset settlement at CP30's price level, as the central valuation read it (F19)")
    args = parser.parse_args(argv)
    po_block = (read_config(args.config) or {}).get("public_ownership") or {}
    acquisition_block = po_block.get("acquisition") or {}
    valuation_block = acquisition_block.get("valuation")
    sensitivity = sensitivity_settings(acquisition_block)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not sensitivity["enabled"]:
        out.write_text(json.dumps({"skipped": "public_ownership.acquisition.repowering_sensitivity is off"},
                                  indent=2), encoding="utf-8")
        return
    # The same settings the central valuation used (legacy_valuation.main).
    settings = lv.settings_from_config(valuation_block)
    settings = lv.group_repowering_settings(settings, po_block, args.run_group or lv._run_group_of(args.finance_register))
    snapshot = yaml.safe_load(Path(args.snapshot).read_text(encoding="utf-8"))
    public_rate = float(snapshot["assumptions"]["public_borrowing_real"]["value"])
    from scripts.public_ownership.register_io import read_register

    register = read_register(args.finance_register, cases=["private"])
    # F19 (8 Oct 2026): value on the price path the central valuation used, or the check below can never match.
    from scripts.public_ownership.levers import LEVER_LEGACY_PRICE_PATH, lever_default

    if args.price_level_settlement and lever_default(LEVER_LEGACY_PRICE_PATH) == lv.PRICE_PATH_CP30:
        register = lv.attach_price_path(register, pd.read_csv(args.price_level_settlement, low_memory=False))
    elif lever_default(LEVER_LEGACY_PRICE_PATH) == lv.PRICE_PATH_CP30 and not lv.is_vintaged_register(register):
        raise ValueError("the central valuation is on the cp30_level price path: pass --price-level-settlement")
    stations = pd.read_csv(settings["ro_register"], low_memory=False)
    report = build(register, pd.read_csv(args.valuation, low_memory=False), settings, sensitivity, stations,
                   public_rate, json.loads(Path(args.tariff_summary).read_text(encoding="utf-8")))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    t = report["tariff"]
    print(json.dumps({"purchase_price_change_gbp_bn": round(report["purchase_price_change_gbp"] / 1e9, 3),
                      "bill_change_inc_vat": round(t["bill_change_gbp_per_household_inc_vat"], 2)}))


if __name__ == "__main__":
    main()
