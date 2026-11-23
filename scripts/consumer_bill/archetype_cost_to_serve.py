"""What a heat pump or an electric vehicle costs to serve (item G6 follow-up).

A single-rate tariff charges a heat-pump or an EV household the same unit rate as
anyone else, so its archetype bill (``household_archetypes.py``) does not show what
its load shape costs. This diagnostic does: the profile-weighted wholesale price of
the added load against the domestic profile's, per MWh and per household. The gap is
the cross-subsidy a flat rate hides; it is reported beside the bill, never in it.

Declared shapes, all on the weather year of the run (2013):

* **Heat pump.** Hourly heating demand is degree-hours below a base temperature,
  from the ERA5 cutout's 2 m temperature averaged over land cells; electricity is
  that over a COP falling linearly in the cold, anchored on the Electrification of
  Heat Demonstration's two published figures (median SPF 2.80 over the year, median
  COP 2.44 on cold days) and floored.
* **Electric vehicle.** Two declared daily charging windows: unmanaged evening
  charging on arrival home, and an overnight smart window.

Usage::

    python -m scripts.consumer_bill.archetype_cost_to_serve <hourly_inputs.csv> <cutout.nc>
        <bill_summary.json> <out.json>

Task Y13 (the 2050 bill, decision 12): ``--composite heat_pump_ev_2050 heat_pump_ev_2050_fes_per_unit``
prices each composed household (``household_archetypes.composition`` in the bill summary)
part by part: the base on the domestic profile price, the heat pump on its temperature
shape with the COP curve scaled by the declared SCOP gain, the EV on each charging window.
Without the flag the 2030 output is unchanged.
"""

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

DEFAULTS = {
    "heat_pump": {"base_temperature_c": 15.5, "spf": 2.80, "spf_mean_temperature_c": 7.0,
                  "cold_cop": 2.44, "cold_temperature_c": 0.0, "min_cop": 1.5},
    "ev_evening": {"hours": [17, 18, 19, 20, 21, 22]},
    "ev_overnight": {"hours": [0, 1, 2, 3, 4]},
}


def cop(temperature_c: pd.Series, params: Mapping) -> pd.Series:
    """A COP linear in temperature through (mean heating temperature, SPF) and (cold day, cold COP)."""
    slope = (params["spf"] - params["cold_cop"]) / (params["spf_mean_temperature_c"] - params["cold_temperature_c"])
    value = params["cold_cop"] + slope * (temperature_c - params["cold_temperature_c"])
    return value.clip(lower=params["min_cop"])


def heat_pump_profile(temperature_c: pd.Series, params: Mapping) -> pd.Series:
    """Hourly heat-pump electricity, normalised to sum to one."""
    degree_hours = (params["base_temperature_c"] - temperature_c).clip(lower=0.0)
    load = degree_hours / cop(temperature_c, params)
    total = float(load.sum())
    if total <= 0:
        raise ValueError("no heating degree-hours in the temperature series")
    return load / total


def ev_profile(index: pd.DatetimeIndex, params: Mapping) -> pd.Series:
    """Equal charging in the declared hours of every day, normalised to sum to one."""
    hours = set(int(h) for h in params["hours"])
    mask = pd.Series(pd.DatetimeIndex(index).hour.isin(hours).astype(float), index=index)
    if mask.sum() <= 0:
        raise ValueError("the EV charging window matches no hour")
    return mask / mask.sum()


def weighted_price(price: pd.Series, shape: pd.Series) -> float:
    shape = shape.reindex(price.index).fillna(0.0)
    return float((price * shape).sum() / shape.sum())


def cost_to_serve(price: pd.Series, shapes: Mapping[str, pd.Series], domestic_shape: pd.Series,
                  added_kwh: Mapping[str, float]) -> pd.DataFrame:
    """Per shape: weighted price, premium over the domestic shape per MWh and per household."""
    base = weighted_price(price, domestic_shape)
    rows = []
    for name, shape in shapes.items():
        p = weighted_price(price, shape)
        rows.append({"profile": name, "weighted_price_gbp_per_mwh": p, "domestic_price_gbp_per_mwh": base,
                     "premium_gbp_per_mwh": p - base,
                     "added_kwh": float(added_kwh.get(name, np.nan)),
                     "premium_gbp_per_household": (p - base) * float(added_kwh.get(name, np.nan)) / 1000.0})
    return pd.DataFrame(rows).set_index("profile")


def gb_land_temperature(cutout_path) -> pd.Series:
    """Hourly mean 2 m temperature over the cutout's land cells, in Celsius."""
    import xarray as xr

    with xr.open_dataset(cutout_path) as ds:
        land = ds["height"] > 0 if "height" in ds else xr.ones_like(ds["temperature"].isel(time=0), dtype=bool)
        temp = ds["temperature"].where(land).mean(dim=("x", "y")).to_pandas() - 273.15
    temp.index = pd.DatetimeIndex(temp.index)
    return temp


def _onto(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """A weather-year series on the model calendar, by month, day and hour."""
    keyed = pd.Series(series.to_numpy(), index=pd.MultiIndex.from_arrays(
        [series.index.month, series.index.day, series.index.hour]))
    keyed = keyed[~keyed.index.duplicated()]
    want = pd.MultiIndex.from_arrays([index.month, index.day, index.hour])
    return pd.Series(keyed.reindex(want).to_numpy(), index=index).ffill().bfill()


#: Task Y13: which load shape serves each part of a composed household.
PART_SHAPES = {"base": "domestic", "heat_pump": "heat_pump", "electric_vehicle": "ev"}


def heat_pump_parameters(composition: Mapping[str, list], base: Mapping = DEFAULTS["heat_pump"]) -> dict:
    """The COP curve for a composed household's heat pump (task Y13).

    The 2030 curve is anchored on the Electrification of Heat Demonstration (SPF 2.80,
    cold-day COP 2.44). A heat-pump part that declares a ``seasonal_performance_gain``
    (FES F.05) scales both anchors by it, so the whole curve moves with the gain; the
    first such part in ``composition`` sets the curve, which a part without an SPF (a
    per-unit average) then shares. Without one, the 2030 curve is returned.
    """
    for parts in composition.values():
        for part in parts:
            inputs = part.get("inputs") or {}
            if part.get("part") == "heat_pump" and "seasonal_performance_factor" in inputs:
                gain = 1.0 + float(inputs.get("seasonal_performance_gain", 0.0))
                return {**base, "spf": float(inputs["seasonal_performance_factor"]) * gain,
                        "cold_cop": float(base["cold_cop"]) * gain}
    return dict(base)


def composite_cost_to_serve(price: pd.Series, composition: Mapping[str, list], shapes: Mapping[str, pd.Series],
                            domestic_price: float, ev_variants: Mapping[str, str]) -> pd.DataFrame:
    """Time-of-use cost-to-serve of each composed household on its hourly profile (task Y13).

    Each part is priced on its own shape: the base on the bill's domestic profile price,
    the heat pump on its temperature shape, the EV on one of the charging windows. The
    household's profile price is the kWh-weighted mean of its parts; the premium is that
    less the domestic profile price, per MWh and per household. Diagnostic only.

    Args:
        price: the bill's hourly clearing price (GBP/MWh, model prices).
        composition: ``{archetype: parts}`` from the bill summary.
        shapes: ``{"heat_pump": shape, <ev variant>: shape, ...}``, each summing to one.
        domestic_price: the bill's domestic profile price (GBP/MWh).
        ev_variants: ``{variant label: shape key}`` for the EV part.

    Returns:
        One row per (archetype, EV variant).
    """
    rows = []
    for archetype, parts in composition.items():
        for variant, ev_key in ev_variants.items():
            kwh, cost, detail = 0.0, 0.0, {}
            for part in parts:
                role = PART_SHAPES.get(str(part["part"]))
                if role is None:
                    raise ValueError(f"{archetype}: part {part['part']!r} has no declared load shape")
                p = (domestic_price if role == "domestic"
                     else weighted_price(price, shapes["heat_pump" if role == "heat_pump" else ev_key]))
                kwh += float(part["kwh"])
                cost += float(part["kwh"]) * p / 1000.0
                detail[f"{part['part']}_kwh"] = float(part["kwh"])
                detail[f"{part['part']}_price_gbp_per_mwh"] = p
            household_price = cost / kwh * 1000.0
            rows.append({"archetype": archetype, "ev_charging": variant, "annual_kwh": kwh, **detail,
                         "household_profile_price_gbp_per_mwh": household_price,
                         "domestic_price_gbp_per_mwh": domestic_price,
                         "premium_gbp_per_mwh": household_price - domestic_price,
                         "energy_cost_on_profile_gbp": cost,
                         "energy_cost_at_domestic_price_gbp": kwh * domestic_price / 1000.0,
                         "premium_gbp_per_household": cost - kwh * domestic_price / 1000.0})
    return pd.DataFrame(rows)


def main_composite(price: pd.Series, summary: Mapping, temperature: pd.Series, archetypes: list[str]) -> dict:
    """The composed households' cost-to-serve (task Y13): ``--composite`` in :func:`main`."""
    block = summary["diagnostics"]["household_archetypes"]
    composition = block.get("composition")
    if not composition:
        raise ValueError("the bill summary carries no archetype composition (household_archetypes.report_composition)")
    missing = [a for a in archetypes if a not in composition]
    if missing:
        raise ValueError(f"archetypes {missing} are not in the bill's composition")
    chosen = {a: composition[a] for a in archetypes}
    hp = heat_pump_parameters(chosen)
    shapes = {"heat_pump": heat_pump_profile(temperature, hp),
              "ev_evening": ev_profile(price.index, DEFAULTS["ev_evening"]),
              "ev_overnight": ev_profile(price.index, DEFAULTS["ev_overnight"])}
    domestic_price = float(summary["domestic_profile_price_gbp_per_mwh"])
    result = composite_cost_to_serve(price, chosen, shapes, domestic_price,
                                     {"evening": "ev_evening", "overnight": "ev_overnight"})
    bills = {r["archetype"]: r for r in block["bills"]}
    result["single_rate_bill_ex_vat_gbp"] = result["archetype"].map(lambda a: bills[a]["ex_vat_gbp"])
    return {"parameters": {"heat_pump": hp, "ev_evening": DEFAULTS["ev_evening"],
                           "ev_overnight": DEFAULTS["ev_overnight"]},
            "headline_archetype": block.get("headline_archetype"),
            "composite_results": result.round(4).to_dict(orient="records"),
            "note": ("Diagnostic only: the single-rate bill is the headline; this is what the household's load "
                     "shape costs at the model's hourly prices, beside it (decision 12). Model price basis.")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hourly_inputs")
    ap.add_argument("cutout")
    ap.add_argument("bill_summary")
    ap.add_argument("out")
    ap.add_argument("--composite", nargs="+", default=None,
                    help="Task Y13: composed archetypes (from the bill's composition) to price on their profiles")
    a = ap.parse_args(argv)
    hourly = pd.read_csv(a.hourly_inputs, index_col=0, parse_dates=True)
    price = hourly["price_gbp_per_mwh"]
    summary = json.loads(Path(a.bill_summary).read_text(encoding="utf-8"))
    if a.composite:
        temperature = _onto(gb_land_temperature(a.cutout), price.index)
        out = {"hourly_inputs": a.hourly_inputs, "cutout": a.cutout,
               **main_composite(price, summary, temperature, a.composite)}
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(pd.DataFrame(out["composite_results"]).round(2).to_string())
        return
    archetypes = {r["archetype"]: r for r in summary["diagnostics"]["household_archetypes"]["bills"]}
    medium = archetypes["ofgem_medium"]["annual_consumption_kwh"]
    added = {"heat_pump": archetypes["heat_pump"]["annual_consumption_kwh"] - medium,
             "ev_evening": archetypes["electric_vehicle"]["annual_consumption_kwh"] - medium,
             "ev_overnight": archetypes["electric_vehicle"]["annual_consumption_kwh"] - medium}
    temperature = _onto(gb_land_temperature(a.cutout), price.index)
    shapes = {"heat_pump": heat_pump_profile(temperature, DEFAULTS["heat_pump"]),
              "ev_evening": ev_profile(price.index, DEFAULTS["ev_evening"]),
              "ev_overnight": ev_profile(price.index, DEFAULTS["ev_overnight"])}
    domestic_price = float(summary["domestic_profile_price_gbp_per_mwh"])
    result = cost_to_serve(price, shapes, pd.Series(1.0, index=price.index), added)
    # The bill's own domestic profile price is the comparator, not a flat year.
    result["domestic_price_gbp_per_mwh"] = domestic_price
    result["premium_gbp_per_mwh"] = result["weighted_price_gbp_per_mwh"] - domestic_price
    result["premium_gbp_per_household"] = result["premium_gbp_per_mwh"] * result["added_kwh"] / 1000.0
    out = {"parameters": DEFAULTS, "hourly_inputs": a.hourly_inputs, "cutout": a.cutout,
           "results": result.round(4).reset_index().to_dict(orient="records"),
           "note": "Diagnostic only: a single-rate tariff charges these households the average unit rate."}
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(result.round(2).to_string())


if __name__ == "__main__":
    main()
