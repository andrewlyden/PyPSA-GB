"""Calibrate the Stage 1 scarcity curve on 2024 (item MR3(b)).

Run by hand; writes ``data/market/scarcity/scarcity_curve_2024.json``, which
``scripts/market/scarcity_pricing.py`` reads.

1. NESO's de-rated margin and loss of load probability per settlement period
   (Elexon BMRS dataset LOLPDRM, cached by the fetch step at
   ``resources/market/elexon/lolpdrm_2024.csv``), taking the forecast nearest
   12 hours ahead for the margin (closest to the day-ahead auction) and nearest
   1 hour ahead for LoLP (the gate-closure value used in cash-out).
2. The gap between the actual N2EX day-ahead price (Elexon MID, APXMIDP) and
   the 2024 model's wholesale price, hour by hour.
3. Each step's price is the mean gap in its de-rated margin band. The band
   above the top threshold is the zero-uplift reference. The last step (margin
   below 1 GW) is the BSC Reserve Scarcity Price at NESO's own LoLP there,
   LoLP x VoLL (6,000 GBP/MWh), because 2024 has too few hours to average.
4. The margin definition: NESO's Capacity Market de-rating factors
   (``data/market/scarcity/neso_cm_derating_factors.csv``, NESO data portal)
   for the delivery year the curve is used in, applied to available capacity
   of conventional plant, storage by duration and each interconnector; wind
   and solar at available output, as the BSC margin uses the wind forecast.
5. The map to model units matches 2024 hours per season: each threshold's
   model-margin value is the model margin with as many 2024 hours below it
   (in that season) as NESO's de-rated margin had below the threshold. The
   model margin is ``model_margin_ex_post`` on the 2024 solved wholesale
   dispatch. Where NESO has fewer than ``MIN_MATCH_HOURS`` hours below a
   threshold in a season, the threshold is placed from the last matched one
   on the slope of a quantile-matched line (sorted pairs, seasonal intercepts).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.market.scarcity_pricing import model_margin_ex_post, parse_margin_definition  # noqa: E402

VOLL_GBP_PER_MWH = 6000.0
BANDS_MW = [20000, 15000, 12000, 10000, 8000, 6000, 4000, 2000, 1000]
OUTPUT = "data/market/scarcity/scarcity_curve_2024.json"
DERATING_FILE = "data/market/scarcity/neso_cm_derating_factors.csv"
DERATING_AUCTION = "2030-31 (T-4) Four Year Ahead Capacity Auction"
MIN_MATCH_HOURS = 20
WINTER_MONTHS = (10, 11, 12, 1, 2, 3)

# Model carrier -> CM Generating Technology Class.
CARRIER_TECHNOLOGY_CLASS = {
    "CCGT": "Combined Cycle Gas Turbine (CCGT)", "CHP": "Combined Heat and Power (CHP)",
    "OCGT": "Open Cycle Gas Turbine (OCGT)", "gas_engine": "Reciprocating engines",
    "oil": "Oil-fired steam generators", "coal": "Coal", "nuclear": "Nuclear",
    "biomass": "Biomass", "CCS Biomass": "Biomass", "advanced_biofuel": "Biomass", "biogas": "Biomass",
    "waste_to_energy": "Energy from Waste", "landfill_gas": "Energy from Waste", "sewage_gas": "Energy from Waste",
    "large_hydro": "Hydro", "small_hydro": "Hydro",
}
# Model link-name fragment -> CM interconnector class (longest fragment wins).
INTERCONNECTOR_CLASS = {
    "Britned": "BritNED (Netherlands)", "IFA2": "IFA2 (France)", "IFA": "IFA (France)",
    "Nemo": "NEMO (Belgium)", "NS Link": "NSL (Norway)", "Moyle": "Moyle (Northern Ireland)",
    "Auchencrosh": "Moyle (Northern Ireland)", "ElecLink": "Eleclink (France)", "Viking": "VikingLink (Denmark)",
    "East West": "EWIC (Republic of Ireland)", "Greenlink": "Greenlink (Republic of Ireland)",
    "NeuConnect": "NeuConnect (Germany)",
}


def build_margin_definition(derating_file=DERATING_FILE, auction=DERATING_AUCTION) -> dict:
    """The ``margin_definition`` block from NESO's Capacity Market de-rating factors.

    One auction for every year the curve is used in, so 2024 calibration and
    2030 use share one definition. Coal is 0 in auctions after closure; its
    factor is taken from the latest auction that priced it.
    """
    table = pd.read_csv(derating_file, encoding="utf-8-sig")
    rows = table[table["Auction Name"] == auction]
    if rows.empty:
        raise ValueError(f"no de-rating factors for auction {auction!r}")
    by_class = rows.set_index("Generating Technology Class")["De Rating Factor"].astype(float)
    notes = {}
    generator_factors = {}
    for carrier, cls in CARRIER_TECHNOLOGY_CLASS.items():
        factor = float(by_class.get(cls, np.nan))
        if not factor > 0:
            earlier = table[(table["Generating Technology Class"] == cls) & (table["De Rating Factor"] > 0)]
            if earlier.empty:
                raise ValueError(f"no de-rating factor for {cls}")
            latest = earlier.iloc[-1]
            factor = float(latest["De Rating Factor"])
            notes[carrier] = f"{cls} is 0 in {auction}; {latest['Auction Name']} value used"
        generator_factors[carrier] = round(factor, 4)
    storage = rows[rows["Primary Fuel Type"] == "Storage"]
    durations = storage["Generating Technology Class"].str.extract(r"Duration ([\d.]+)h")[0].astype(float)
    storage_factors = {f"{h:g}": round(float(f), 4) for h, f in sorted(zip(durations, storage["De Rating Factor"]))}
    ic_rows = rows[rows["Primary Fuel Type"] == "Interconnector"].set_index("Generating Technology Class")["De Rating Factor"]
    ic_factors = {fragment: round(float(ic_rows[cls]), 4) for fragment, cls in INTERCONNECTOR_CLASS.items()}
    return {
        "source": f"NESO Capacity Market de-rating factors, {auction} ({derating_file})",
        "generator_factors": generator_factors,
        "default_generator_factor": 1.0,
        "default_generator_note": ("wind, solar and other variable renewables at available output: the BSC de-rated "
                                   "margin adds the wind forecast rather than a de-rated capacity"),
        "generator_notes": notes,
        "storage": {"mode": "derated_capacity", "factors_by_duration_h": storage_factors,
                    "note": "the factor of the longest listed duration not above the unit's max_hours"},
        "interconnectors": {"mode": "derated_capacity", "factors": ic_factors,
                            "default_factor": round(float(ic_rows.median()), 4),
                            "default_note": "median of the auction's interconnector factors"},
    }


def match_thresholds(margin: pd.Series, drm: pd.Series, thresholds, min_hours=MIN_MATCH_HOURS) -> dict:
    """Model-margin value of each de-rated margin threshold, per season, by matching hours."""
    frame = pd.DataFrame({"m": margin, "drm": drm}).dropna()
    season = np.where(frame.index.month.isin(WINTER_MONTHS), "winter", "summer")
    # Quantile-matched line (sorted pairs, common slope, seasonal intercepts) for the thin tail.
    xs, ys, ws = [], [], []
    for s in ("winter", "summer"):
        part = frame[season == s]
        xs.append(np.sort(part["m"].to_numpy()))
        ys.append(np.sort(part["drm"].to_numpy()))
        ws.append(np.full(len(part), s == "winter"))
    x, y, w = map(np.concatenate, (xs, ys, ws))
    slope = float(np.linalg.lstsq(np.column_stack([x, w, ~w]).astype(float), y, rcond=None)[0][0])
    out, matched = {}, {}
    previous = None
    for s in ("winter", "summer"):
        part = frame[season == s]
        values, hours = [], []
        for d in thresholds:
            below = int((part["drm"] < d).sum())
            if below >= min_hours:
                values.append(float(np.quantile(part["m"], below / len(part))))
            elif not values:
                raise ValueError(f"{s}: fewer than {min_hours} hours below the top threshold; cannot match")
            else:
                values.append(values[-1] - (previous - d) / slope)
            previous = d
            hours.append(below)
        for k in range(1, len(values)):          # strictly decreasing
            values[k] = min(values[k], values[k - 1] - 1.0)
        out[s] = [round(v, 1) for v in values]
        matched[s] = hours
    return {"model_thresholds_mw_by_season": out, "neso_hours_below_by_season": matched,
            "tail_slope": round(slope, 4), "min_match_hours": min_hours}


def nearest_horizon(frame: pd.DataFrame, target_h: float) -> pd.DataFrame:
    x = frame.assign(err=(frame["horizon_h"] - target_h).abs()).sort_values("err")
    x = x.drop_duplicates("startTime")
    return x[x["err"] <= 1.0].set_index("startTime")


def calibrate(lolpdrm_path, mid_path, model_price_path, wholesale_nc, output=OUTPUT,
              derating_file=DERATING_FILE, auction=DERATING_AUCTION) -> dict:
    import pypsa

    lolp = pd.read_csv(lolpdrm_path, usecols=["startTime", "horizon_h", "lossOfLoadProbability", "deratedMargin"])
    lolp["startTime"] = pd.to_datetime(lolp["startTime"]).dt.tz_convert(None)
    drm = nearest_horizon(lolp, 12.0)["deratedMargin"].resample("h").mean()
    lolp1 = nearest_horizon(lolp, 1.0)
    lolp1_hourly = lolp1["lossOfLoadProbability"].resample("h").mean()

    mid = pd.read_csv(mid_path, parse_dates=["datetime"]).groupby("datetime")["mid_price"].mean()
    if mid.index.tz is not None:
        mid.index = mid.index.tz_convert(None)
    actual = mid.resample("h").mean()
    model = pd.read_csv(model_price_path, index_col=0, parse_dates=True)["wholesale_price"]

    n = pypsa.Network(wholesale_nc)
    # Link flows are written beside the network (the solved .nc may not carry them).
    links_csv = Path(str(wholesale_nc).replace(".nc", "_links.csv"))
    links_p = (pd.read_csv(links_csv, index_col=0, parse_dates=True).reindex(n.snapshots)
               if links_csv.exists() else n.links_t.p0)
    definition_block = build_margin_definition(derating_file, auction)
    definition = parse_margin_definition(definition_block)
    dispatch = {"generators_p": n.generators_t.p,
                "storage_p": n.storage_units_t.p if len(n.storage_units) else None,
                "links_p": links_p}
    margin = model_margin_ex_post(n, dispatch, definition)

    frame = pd.concat({"actual": actual, "model": model, "drm": drm, "lolp1": lolp1_hourly,
                       "model_margin": margin}, axis=1).dropna()
    frame["gap"] = frame["actual"] - frame["model"]
    corr = float(np.corrcoef(frame["model_margin"], frame["drm"])[0, 1])
    mapping = match_thresholds(frame["model_margin"], frame["drm"], BANDS_MW)

    steps = []
    edges = BANDS_MW + [-np.inf]
    for k, upper in enumerate(BANDS_MW):
        lower = edges[k + 1]
        band = frame[(frame["drm"] <= upper) & (frame["drm"] > lower)]
        if k == len(BANDS_MW) - 1:
            # Too few hours below 1 GW: price the step at LoLP x VoLL from NESO's
            # gate-closure LoLP in that band (all 2024 settlement periods).
            tail = lolp1[lolp1["deratedMargin"] <= upper]["lossOfLoadProbability"]
            price = float(tail.mean()) * VOLL_GBP_PER_MWH
            basis = f"BSC RSVP = LoLP x VoLL at NESO's mean gate-closure LoLP below {upper / 1e3:.0f} GW ({tail.mean():.4f}, {len(tail)} settlement periods)"
            hours = int(len(band))
        else:
            price = float(band["gap"].mean())
            basis = f"mean of actual N2EX less model price in the band ({len(band)} hours)"
            hours = int(len(band))
        steps.append({"drm_below_mw": float(upper), "price_gbp_per_mwh": round(price, 2),
                      "hours_2024": hours, "basis": basis})
    # Scarcity prices must not fall as the margin tightens.
    for k in range(1, len(steps)):
        steps[k]["price_gbp_per_mwh"] = max(steps[k]["price_gbp_per_mwh"], steps[k - 1]["price_gbp_per_mwh"])
    reference = frame[frame["drm"] > BANDS_MW[0]]["gap"].mean()

    curve = {
        "purpose": ("Item MR3(b): the Stage 1 wholesale scarcity curve, a stepped reserve shortage curve priced "
                    "from the 2024 gap between actual N2EX day-ahead prices and the model, binned on NESO's "
                    "published de-rated margin, with the BSC Reserve Scarcity Price below 1 GW."),
        "precedent": ("DESNZ Dynamic Dispatch Model: wholesale price = system marginal price plus a mark-up derived "
                      "from historic data at times of reduced capacity margins (DECC DDM, May 2012); the DDM peer "
                      "review asks that tightness be measured against capacity actually available. BSC Section T "
                      "3.13: RSVP = LoLP x VoLL."),
        "user_decision": ("23 Sep 2026: the 8 to 20 GW band (+7 to +13 GBP/MWh) counts as scarcity, following the DDM, "
                          "and is reported so it can be switched off as a sensitivity."),
        "margin_definition": definition_block,
        "drm_from_model_margin": {
            **mapping, "correlation": round(corr, 3),
            "basis": ("hours matched per season (October to March, April to September): each threshold's model margin "
                      "has as many 2024 hours below it as NESO's 12-hour-ahead de-rated margin had below the "
                      "threshold; the model margin is the de-rated margin of the 2024 solved wholesale dispatch"),
        },
        "reference_band_gap_gbp_per_mwh": round(float(reference), 2),
        "steps": steps,
        "sources": {
            "lolpdrm": "Elexon BMRS dataset LOLPDRM (https://bmrs.elexon.co.uk/api-documentation/endpoint/datasets/LOLPDRM), 2024",
            "mid": "Elexon Market Index Data, APXMIDP, 2024",
            "model": str(model_price_path), "wholesale_network": str(wholesale_nc),
            "voll_gbp_per_mwh": VOLL_GBP_PER_MWH,
        },
        "hours_used": int(len(frame)),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(curve, indent=2), encoding="utf-8")
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lolpdrm", default="resources/market/elexon/lolpdrm_2024.csv")
    parser.add_argument("--mid", default="resources/market/elexon/mid_prices_2024.csv")
    parser.add_argument("--model-price", default="resources/market/Historical_2024_ETYS_14zone_market_wholesale_price.csv")
    parser.add_argument("--wholesale", default="resources/market/Historical_2024_ETYS_14zone_market_wholesale.nc")
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--derating", default=DERATING_FILE)
    parser.add_argument("--auction", default=DERATING_AUCTION)
    args = parser.parse_args()
    curve = calibrate(args.lolpdrm, args.mid, args.model_price, args.wholesale, args.output,
                      args.derating, args.auction)
    print(json.dumps({k: curve[k] for k in ("drm_from_model_margin", "reference_band_gap_gbp_per_mwh", "hours_used")}, indent=1))
    for s in curve["steps"]:
        print(f"  DRM below {s['drm_below_mw'] / 1e3:5.1f} GW: {s['price_gbp_per_mwh']:8.2f} GBP/MWh ({s['hours_2024']} h)")


if __name__ == "__main__":
    main()
