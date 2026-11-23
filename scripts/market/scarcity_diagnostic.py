"""Scarcity diagnostic on solved months (item MR3(a)).

For each solved wholesale month: the de-rated margin every hour (the MR3(b)
definition, ``model_margin_ex_post``), the scarcity curve's step price at that
margin, and what it is worth:

* hours by de-rated margin band and the time- and load-weighted uplift;
* the split between the shallow bands (thresholds 10 to 20 GW, "8 to 20 GW")
  and the deep bands (below 8 GW), so the shallow bands can be reported as the
  sensitivity the user asked for (23 Sep 2026);
* the uplift weighted by each gas class's output, and the extra energy margin
  it gives per kW.

On months solved without scarcity pricing this is the DDM-style ex-post
mark-up (MR3(a) as planned). On months solved with it, the step price is the
one the LP added to the energy price (the 2024 week runs showed the realised
uplift tracks the step price), so the table decomposes the price already paid.

    python scripts/market/scarcity_diagnostic.py OUT_PREFIX SCENARIO [SCENARIO ...]
        [--curve data/market/scarcity/scarcity_curve_2024.json] [--prices-include-scarcity]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.market.scarcity_pricing import (  # noqa: E402
    _foreign_buses, load_scarcity_curve, model_margin_ex_post, season_of,
)

GAS = ("CCGT", "OCGT", "gas_engine", "CHP")
SHALLOW_MIN_MW = 8000.0     # steps with a threshold above this are the 8 to 20 GW bands


def step_prices(margin: pd.Series, curve_json: dict, curve: dict) -> tuple[pd.Series, pd.Series]:
    """Price of the deepest step in use each hour, and the de-rated threshold of that step."""
    mapping = curve_json["drm_from_model_margin"]
    thresholds = curve["drm_thresholds_mw"]
    prices = curve["prices"]
    price = pd.Series(0.0, index=margin.index)
    band = pd.Series(np.nan, index=margin.index)
    for t, value in margin.items():
        season = season_of(t)
        if "model_thresholds_mw_by_season" in mapping:
            model = mapping["model_thresholds_mw_by_season"][season]
        else:
            b = (mapping.get("intercept_mw_by_season") or {}).get(season, mapping.get("intercept_mw", 0.0))
            model = [(d - float(b)) / float(mapping["slope"]) for d in thresholds]
        crossed = [k for k, m in enumerate(model) if value < m]
        if crossed:
            price[t] = prices[crossed[-1]]
            band[t] = thresholds[crossed[-1]]
    return price, band


def diagnose_month(scenario: str, curve_path: str, market_dir: Path = Path("resources/market")) -> tuple[pd.DataFrame, dict]:
    import pypsa

    base = market_dir / f"{scenario}_wholesale"
    n = pypsa.Network(str(base) + ".nc")
    links_csv = Path(str(base) + "_links.csv")
    links = pd.read_csv(links_csv, index_col=0, parse_dates=True).reindex(n.snapshots) if links_csv.exists() else n.links_t.p0
    curve_json = json.loads(Path(curve_path).read_text(encoding="utf-8"))
    curve = load_scarcity_curve(curve_path)
    dispatch = {"generators_p": n.generators_t.p,
                "storage_dispatch": (n.storage_units_t.p_dispatch if not n.storage_units_t.p_dispatch.empty
                                     else n.storage_units_t.p.clip(lower=0)),
                "links_p": links}
    margin = model_margin_ex_post(n, dispatch, curve["margin_definition"])
    uplift, band = step_prices(margin, curve_json, curve)
    foreign = _foreign_buses(n)
    gb_loads = n.loads.index[~n.loads.bus.isin(foreign)]
    hours = pd.DataFrame({"margin_mw": margin, "uplift_gbp_per_mwh": uplift, "band_threshold_mw": band,
                          "load_mw": n.loads_t.p_set.reindex(columns=gb_loads).fillna(0.0).sum(axis=1)})
    price_csv = Path(str(base) + "_price.csv")
    if price_csv.exists():
        hours["wholesale_price"] = pd.read_csv(price_csv, index_col=0, parse_dates=True)["wholesale_price"].reindex(n.snapshots)
    capacity = {}
    for carrier in GAS:
        gens = n.generators.index[n.generators.carrier.eq(carrier)]
        hours[f"{carrier}_mw"] = n.generators_t.p.reindex(columns=gens).fillna(0.0).sum(axis=1)
        capacity[carrier] = float(n.generators.loc[gens, "p_nom"].sum())
    hours["scenario"] = scenario
    return hours, capacity


def summarise(hours: pd.DataFrame, capacity: dict, prices_include_scarcity: bool) -> dict:
    up = hours["uplift_gbp_per_mwh"]
    shallow = hours["band_threshold_mw"] > SHALLOW_MIN_MW
    deep = hours["band_threshold_mw"] <= SHALLOW_MIN_MW
    load = hours["load_mw"]
    out = {
        "basis": ("decomposition of the scarcity uplift already in the Stage 1 price" if prices_include_scarcity
                  else "ex-post DDM-style mark-up on Stage 1 prices solved without scarcity pricing"),
        "hours": int(len(hours)),
        "hours_by_band_threshold_mw": {str(int(k)): int(v) for k, v in hours["band_threshold_mw"].value_counts().sort_index(ascending=False).items()},
        "hours_scarce": int(up.gt(0).sum()),
        "mean_uplift_gbp_per_mwh": float(up.mean()),
        "load_weighted_uplift_gbp_per_mwh": float((up * load).sum() / load.sum()),
        "shallow_bands_8_to_20gw": {"hours": int(shallow.sum()),
                                    "load_weighted_contribution_gbp_per_mwh": float((up * load)[shallow].sum() / load.sum())},
        "deep_bands_below_8gw": {"hours": int(deep.sum()),
                                 "load_weighted_contribution_gbp_per_mwh": float((up * load)[deep].sum() / load.sum())},
        "consumer_cost_gbp": float((up * load).sum()),
        "gas": {},
    }
    if "wholesale_price" in hours:
        price = hours["wholesale_price"]
        out["mean_price_gbp_per_mwh"] = float(price.mean())
        out["mean_price_with_uplift_gbp_per_mwh"] = float(price.mean() if prices_include_scarcity else (price + up).mean())
    for carrier, cap in capacity.items():
        mwh = hours[f"{carrier}_mw"]
        if cap <= 0 or mwh.sum() <= 0:
            continue
        value = float((up * mwh).sum())
        out["gas"][carrier] = {
            "capacity_mw": cap, "output_mwh": float(mwh.sum()),
            "output_weighted_uplift_gbp_per_mwh": value / float(mwh.sum()),
            "margin_from_uplift_gbp": value, "margin_from_uplift_gbp_per_kw_year": value / (cap * 1e3),
            "shallow_share_of_margin": float((up * mwh)[shallow].sum() / value) if value > 0 else 0.0,
        }
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_prefix")
    parser.add_argument("scenarios", nargs="+")
    parser.add_argument("--curve", default="data/market/scarcity/scarcity_curve_2024.json")
    parser.add_argument("--prices-include-scarcity", action="store_true")
    args = parser.parse_args(argv)
    frames, capacity = [], {}
    for scenario in args.scenarios:
        hours, cap = diagnose_month(scenario, args.curve)
        frames.append(hours)
        capacity = {k: max(capacity.get(k, 0.0), v) for k, v in cap.items()}
        print(scenario, "scarce hours", int(hours["uplift_gbp_per_mwh"].gt(0).sum()), flush=True)
    hours = pd.concat(frames).sort_index()
    summary = summarise(hours, capacity, args.prices_include_scarcity)
    summary["scenarios"] = args.scenarios
    summary["curve"] = args.curve
    summary["by_month"] = {s: summarise(h, capacity, args.prices_include_scarcity)["load_weighted_uplift_gbp_per_mwh"]
                           for s, h in hours.groupby("scenario")}
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    hours.to_csv(f"{out}_hours.csv")
    Path(f"{out}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_month", "scenarios")}, indent=1))


if __name__ == "__main__":
    main()
