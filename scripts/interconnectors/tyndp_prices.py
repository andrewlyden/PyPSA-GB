"""European foreign prices of a TYNDP year from ENTSO-E / ENTSOG TYNDP.

Year-generic: the TYNDP year is the modelled
year, and its files are ``tyndp2024_{scenario}{year}CY{climate year}_hourly.csv`` and
``tyndp2026_NTplus{year}_mc.csv`` (``tyndp_input_files``). Only the 2050 files are extracted
today; another year needs its own (``scripts/interconnectors/extract_tyndp_hourly.py``). The
text below describes the 2050 build.

ENTSO-E day-ahead prices stop at the present, so a 2050 run cannot borrow a
historical shape the way the 2030 IC1 route does (``price_shape.py``). TYNDP 2024
publishes hourly 2050 marginal prices by bidding zone for Distributed Energy (DE)
and Global Ambition (GA) on climate years 1995, 2008 and 2009; the TYNDP 2026 draft
publishes National Trends+ (NT+) 2050 on three synthetic climate-projection years.
Data and licence
(CC-BY 4.0): ``data/interconnectors/tyndp2024/README.md``.

**Route (plan Y3, decisions 7, 8, 9 and 27).**

1. *Zones.* One model price column per counterparty column of the IC1 ENTSO-E
   file: France FR00, Belgium BE00, Netherlands NL00, Germany DE00, Denmark DKW1,
   Norway NOS0 (NT+: NOS2, the zone of North Sea Link's Kvilldal end), Ireland =
   the SEM, the GB-cable-weighted mean of IE00 and UKNI (declared weights below).
   The continent is combined inside ``add_to_network`` exactly as IC1 does it
   (GB-capacity-weighted, ``single_price_market_shapes``), so the file keeps one
   column per continental zone.
2. *Shape.* The shape climate year's hourly series (CY2009, the largest weight and
   TYNDP's expansion year; NT+: WS092, weight 0.57), laid on the 2013 calendar by
   date (TYNDP hour ``k`` = 1 January 00:00 + ``k`` hours), then conditioned on
   2013 GB offshore wind with :func:`price_shape.condition_prices_on_wind`. The
   shape-year regressor is TYNDP's own UK00 offshore wind for that climate year.
3. *Level.* TYNDP's climate-weighted mean (1995 23%, 2008 37%, 2009 40%; NT+ 0.23 /
   0.57 / 0.20), converted from EUR at the TYNDP price base to GBP 2024, the price
   base of the model's money (IC1's ENTSO-E prices are 2024 EUR converted at the
   configured ``eur_to_gbp`` 0.86 with no deflator; TYNDP 2024 is in 2022 EUR, so
   it is first brought to 2024 EUR by the euro-area HICP). The conditioned shape
   is first held at the source's own price floor (0 in every zone: TYNDP's market
   model has no negative bids, and the linear wind swap would otherwise push windy
   zero-price hours negative), then scaled to that level by a ratio
   (``LEVEL_METHOD``), which keeps TYNDP's zero-price hours at zero; ``shift`` is
   the additive alternative.
4. *Decision 8 sensitivity* (``route: ic1_rescale``): the IC1 2024 ENTSO-E shape
   (conditioned on 2013 wind) rescaled per column to the same TYNDP level. It
   separates the effect of the shape from that of the level.

**Calendar (declared).** The TYNDP 2024 year is 52 weeks, 8,736 hours, from
``01JAN00:00`` to ``30DEC23:00``, with no year stamp. It is mapped onto the weather
year (2013) by date, so its seasons line up with 2013 wind; a leap weather year takes
28 February's hours for 29 February. The weekday is NOT re-aligned (TYNDP's
weekly demand cycle is reported by :func:`weekly_low_day`, and a mismatch with the
2050 model calendar is declared, not corrected, since re-aligning weekdays would
move the season by up to three days against the wind driver). 31 December is filled
with 24 December (the same weekday one week earlier). NT+ has 8,760 hours from
1 January to 31 December and maps one to one. TYNDP hours are taken as published;
no time-zone shift is applied.

**Declared double count (decision 28).** TYNDP's prices come from a model with a GB
zone, so they already reflect GB's trade; the IC1 volume tranches, kept at
elasticity 1.0 with the 1.6 high case, may count GB's effect on foreign prices a
second time. The low elasticity is the bound.
"""

import calendar
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.interconnectors.price_shape import align_by_calendar, condition_prices_on_wind, wind_index
from scripts.utilities.calendar_year import hours_in_year, on_calendar

_LOG = logging.getLogger(__name__)

HOURS_TYNDP2024 = 8736
#: Hours of a common (non-leap) year: the NT+ tables and the calendar the TYNDP hours are laid on.
HOURS_YEAR = 8760
#: A common year the TYNDP hours are first laid on when the weather year is a leap year.
COMMON_REFERENCE_YEAR = 2013

#: TYNDP 2024 climate-year weights (Methodology Report, section 4.1 and Table 4).
TYNDP2024_CLIMATE_WEIGHTS = {1995: 0.23, 2008: 0.37, 2009: 0.40}
#: TYNDP 2026 draft weather-scenario weights by target year. 2050: Methodology Report Annex III, Tables 33 and
#: 36. 2040 (10 Nov 2026, provisional until Annex III's 2040 table is checked): the draft TYNDP 2026
#: implementation guidelines' 2040 selection, WS065 0.20, WS071 0.40, WS077 0.40.
TYNDP2026_WEATHER_WEIGHTS_BY_YEAR = {
    2050: {"091": 0.23, "092": 0.57, "106": 0.20},
    2040: {"065": 0.20, "071": 0.40, "077": 0.40},
}
TYNDP2026_WEATHER_WEIGHTS = TYNDP2026_WEATHER_WEIGHTS_BY_YEAR[2050]
#: NT+ shape weather scenario by target year: the largest weight; a tie goes to the lower id (2040: WS071).
NTPLUS_SHAPE_SCENARIO_BY_YEAR = {
    year: min(w, key=lambda ws: (-w[ws], ws)) for year, w in TYNDP2026_WEATHER_WEIGHTS_BY_YEAR.items()}


def ntplus_bracket(tyndp_year: int) -> tuple[int, int, float] | None:
    """The NT+ years either side of a year NT+ does not publish, and the later one's weight (user, 11 Nov 2026).

    NT+ 2026 publishes 2030, 2035, 2040 and 2050 (the years with declared weather weights here). A modelled year
    between two of them (2045) blends their hourly shapes, each built and conditioned as usual, in proportion to
    distance: 2045 is half 2040 and half 2050. None for a year with its own weights; a year outside the declared
    range fails.
    """
    year = int(tyndp_year)
    years = sorted(TYNDP2026_WEATHER_WEIGHTS_BY_YEAR)
    if year in years:
        return None
    lower, upper = [y for y in years if y < year], [y for y in years if y > year]
    if not lower or not upper:
        raise ValueError(f"no TYNDP 2026 NT+ weather weights declared for {year}, and it lies outside the declared "
                         f"years {years}, so it cannot be interpolated")
    y0, y1 = max(lower), min(upper)
    return y0, y1, (year - y0) / (y1 - y0)


def blend_ntplus_shapes(lo_shape, hi_shape, lo_audit, hi_audit, out_file, audit_file, tyndp_year: int,
                        variant: str, logger=_LOG) -> pd.DataFrame:
    """Write a bracketed year's shape: its neighbours' shapes blended hour by hour (user decision, 11 Nov 2026).

    Both neighbours are on the same weather-year calendar and conditioned on the same GB wind, so the blend keeps
    that conditioning, and its mean is the distance-weighted mean of the two levels exactly.
    """
    y0, y1, share = ntplus_bracket(tyndp_year)
    lo = pd.read_csv(lo_shape, index_col=0, parse_dates=True)
    hi = pd.read_csv(hi_shape, index_col=0, parse_dates=True)
    if not lo.index.equals(hi.index) or list(lo.columns) != list(hi.columns):
        raise ValueError(f"the {y0} and {y1} shapes are not on the same calendar and columns; rebuild both on the "
                         "same weather year")
    shaped = lo * (1.0 - share) + hi * share
    audits = {str(y0): json.loads(Path(lo_audit).read_text()), str(y1): json.loads(Path(hi_audit).read_text())}
    audit = {
        "task": "TYNDP NT+ for a year NT+ does not publish (user decision, 11 Nov 2026)",
        "variant": variant,
        "tyndp_year": int(tyndp_year),
        "method": f"hourly blend of the {y0} and {y1} shapes, {1.0 - share:.3f} x {y0} + {share:.3f} x {y1}",
        "bracket": {"lower": y0, "upper": y1, "upper_share": share},
        "sources": {y: {"shape_source": a.get("shape_source"), "climate_weights": a.get("climate_weights"),
                        "mean_gbp": a.get("mean_gbp")} for y, a in audits.items()},
        "mean_gbp": shaped.mean().round(3).to_dict(),
        "std_gbp": shaped.std().round(3).to_dict(),
        "hours_below_zero": {c: int((shaped[c] < 0).sum()) for c in shaped},
        "hours": int(len(shaped)),
        "licence": "TYNDP data CC-BY 4.0, ENTSO-E and ENTSOG (see data/interconnectors/tyndp2024/README.md)",
        "declared": [f"NT+ 2026 publishes no {int(tyndp_year)}; its prices are interpolated between {y0} and {y1}"],
    }
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    out = shaped.copy()
    out.index.name = "datetime"
    out.to_csv(out_file)
    Path(audit_file).write_text(json.dumps(audit, indent=2, default=str))
    logger.info("%s %s: blended %s", variant, tyndp_year, audit["method"])
    return shaped


def ntplus_weights(tyndp_year: int) -> dict:
    """The NT+ weather-scenario weights of a TYNDP target year; a year without declared weights fails."""
    try:
        return TYNDP2026_WEATHER_WEIGHTS_BY_YEAR[int(tyndp_year)]
    except KeyError:
        raise ValueError(f"no TYNDP 2026 NT+ weather weights declared for {tyndp_year}; "
                         f"declared: {sorted(TYNDP2026_WEATHER_WEIGHTS_BY_YEAR)}") from None


def shape_year_of(scenario: str, tyndp_year: int):
    """The shape climate year (TYNDP 2024) or weather scenario (NT+) of a variant and target year."""
    if scenario == "NTplus2026":
        ntplus_weights(tyndp_year)
        return NTPLUS_SHAPE_SCENARIO_BY_YEAR[int(tyndp_year)]
    return SHAPE_YEAR[scenario]

#: Euro-area (EA20) HICP, all items, annual average index 2015 = 100. Eurostat
#: ``prc_hicp_aind`` (unit INX_A_AVG), retrieved 30 September 2026 from the
#: Eurostat dissemination API. Brings TYNDP 2024's 2022 euros to 2024 euros.
EA_HICP_ANNUAL_INDEX = {2022: 116.83, 2023: 123.17, 2024: 126.08, 2025: 128.76}
TYNDP_PRICE_BASE_YEAR = {"DE": 2022, "GA": 2022, "NTplus2026": 2024}
MODEL_PRICE_BASE_YEAR = 2024

#: SEM weights from today's register cables (GB cable capacity to each side of the island):
#: IE00 = East West 505 MW + Greenlink 504 MW, UKNI = Moyle 500 MW (the Isle of Man link is
#: priced on the SEM but not weighted). The 2050 TYNDP route uses these; the CP30 2030 routes
#: derive theirs from CP30's cable set (:func:`sem_node_weights_from_composition`, F25.3).
SEM_NODE_WEIGHTS_REGISTER = {"IE00": 1009.0, "UKNI": 500.0}
#: TYNDP SEM node of each GB-SEM cable a cable list names without a counterparty country.
SEM_CABLE_NODES = {"Moyle": "UKNI", "East West Interconnector": "IE00", "Greenlink": "IE00",
                   "MaresConnect": "IE00"}
#: TYNDP SEM node of a cable list's counterparty country.
SEM_COUNTERPARTY_NODES = {"ireland": "IE00", "northern ireland": "UKNI"}


def _normalised(weights: dict) -> dict:
    total = sum(float(w) for w in weights.values())
    if total <= 0:
        raise ValueError(f"zone weights sum to zero: {weights}")
    return {node: float(w) / total for node, w in weights.items()}


#: Model price column (the IC1 ENTSO-E file's counterparty columns) -> {TYNDP node: weight}.
ZONE_WEIGHTS_TYNDP2024 = {
    "France": {"FR00": 1.0},
    "Belgium": {"BE00": 1.0},
    "Netherlands": {"NL00": 1.0},
    "Germany": {"DE00": 1.0},
    "Denmark": {"DKW1": 1.0},
    "Norway": {"NOS0": 1.0},
    "Ireland": _normalised(SEM_NODE_WEIGHTS_REGISTER),
}


def sem_node_weights_from_composition(composition, year: int) -> dict:
    """GB cable MW to each TYNDP SEM node (IE00, UKNI) in ``year`` from the CP30 cable composition (F25.3).

    ``composition`` is the composition file's path (``data/interconnectors/cp30_interconnector_composition.csv``)
    or its table. A ``keep`` or ``add`` row counts from its ``first_year`` (blank = always) at its ``cp30_mw``;
    ``exclude`` rows never count. A row's node is its ``counterparty_country`` (Ireland -> IE00, Northern
    Ireland -> UKNI) where given, else :data:`SEM_CABLE_NODES` by name; other cables are not SEM cables.
    2030: IE00 = EWIC 500 + Greenlink 500 + MaresConnect 750 = 1,750 MW, UKNI = Moyle 500 MW.
    """
    if isinstance(composition, pd.DataFrame):
        table = composition.copy()
    else:
        from scripts.CP30.storage_interconnectors import load_cp30_interconnector_composition
        table = load_cp30_interconnector_composition(composition)
    table["action"] = table["action"].astype(str).str.strip().str.lower()
    first_year = pd.to_numeric(table["first_year"], errors="coerce").fillna(-np.inf)
    counted = table["action"].isin(["keep", "add"]) & (first_year <= int(year))
    country = (table["counterparty_country"] if "counterparty_country" in table else pd.Series(index=table.index,
                                                                                                    dtype=object))
    weights = {"IE00": 0.0, "UKNI": 0.0}
    for idx, row in table[counted].iterrows():
        cp = country.get(idx)
        if isinstance(cp, str) and cp.strip():
            node = SEM_COUNTERPARTY_NODES.get(cp.strip().lower())
        else:
            node = SEM_CABLE_NODES.get(str(row["name"]).strip())
        if node is not None:
            weights[node] += float(pd.to_numeric(row["cp30_mw"], errors="coerce"))
    if sum(weights.values()) <= 0:
        raise ValueError(f"CP30 cable composition has no GB-SEM cable in {year}")
    return weights


def with_sem_weights(zone_weights: dict, sem_weights: dict | None) -> dict:
    """``zone_weights`` with its Ireland (SEM) column re-weighted by GB cable MW per node; None keeps it."""
    if sem_weights is None:
        return dict(zone_weights)
    return dict(zone_weights, Ireland=_normalised(sem_weights))
ZONE_WEIGHTS_NTPLUS2026 = dict(ZONE_WEIGHTS_TYNDP2024, Norway={"NOS2": 1.0})

VARIANTS = ("DE", "GA", "NTplus2026")
ROUTES = ("tyndp_shape", "ic1_rescale")
#: Shape climate year per variant: the largest-weight year (TYNDP 2024 CY2009, NT+ WS092).
SHAPE_YEAR = {"DE": 2009, "GA": 2009, "NTplus2026": "092"}
LEVEL_METHOD = "ratio"


# --------------------------------------------------------------------------- config

def is_tyndp_pricing(pricing) -> bool:
    """True when a scenario's ``interconnectors.pricing.foreign_price_source`` is ``tyndp``.

    The key is absent in every scenario today; absent (or ``entsoe``) is the
    existing ENTSO-E / CP.29 behaviour.
    """
    return str((pricing or {}).get("foreign_price_source", "entsoe")).strip().lower() == "tyndp"


def tyndp_settings(pricing) -> dict:
    """The ``interconnectors.pricing.tyndp`` block with its declared defaults filled in.

    Keys: ``scenario`` (``DE`` central, ``GA`` or ``NTplus2026``; decision 27) and
    ``route`` (``tyndp_shape`` central, ``ic1_rescale`` the decision 8 sensitivity).
    The shape year (CY2009, NT+ WS092) and the ratio re-level are fixed and declared
    (``SHAPE_YEAR``, ``LEVEL_METHOD``), so the file name identifies the method.
    """
    block = dict((pricing or {}).get("tyndp") or {})
    settings = {
        "scenario": str(block.get("scenario", "DE")),
        "route": str(block.get("route", "tyndp_shape")),
    }
    if settings["scenario"] not in VARIANTS:
        raise ValueError(f"interconnectors.pricing.tyndp.scenario must be one of {VARIANTS}, got {settings['scenario']!r}")
    if settings["route"] not in ROUTES:
        raise ValueError(f"interconnectors.pricing.tyndp.route must be one of {ROUTES}, got {settings['route']!r}")
    unknown = sorted(set(block) - set(settings))
    if unknown:
        raise ValueError(f"interconnectors.pricing.tyndp: unknown keys {unknown}")
    return settings


def tyndp_variant(pricing) -> str:
    """The variant token in the shape file name: ``DE``, ``GA``, ``NTplus2026``, plus ``_ic1rescale``."""
    s = tyndp_settings(pricing)
    return s["scenario"] + ("_ic1rescale" if s["route"] == "ic1_rescale" else "")


def parse_variant(variant: str) -> tuple[str, str]:
    """(scenario, route) from a file-name variant token."""
    if variant.endswith("_ic1rescale"):
        scenario, route = variant[: -len("_ic1rescale")], "ic1_rescale"
    else:
        scenario, route = variant, "tyndp_shape"
    if scenario not in VARIANTS:
        raise ValueError(f"unknown TYNDP variant {variant!r}")
    return scenario, route


def eur_to_model_gbp(eur_to_gbp: float, price_base_year: int, model_base_year: int = MODEL_PRICE_BASE_YEAR) -> float:
    """GBP (model base year) per EUR (TYNDP price base year): HICP rebase then the IC1 rate."""
    deflator = EA_HICP_ANNUAL_INDEX[model_base_year] / EA_HICP_ANNUAL_INDEX[price_base_year]
    return float(eur_to_gbp) * deflator


# --------------------------------------------------------------------------- calendar

def _on_leap_year(common: pd.DataFrame | pd.Series, year: int):
    """A common-year series put on leap ``year`` by date (29 February takes 28 February's hours)."""
    return on_calendar(common, pd.date_range(f"{year}-01-01 00:00", periods=hours_in_year(year), freq="h"))


def tyndp2024_to_calendar(values: pd.DataFrame | pd.Series, year: int = 2013):
    """Lay 8,736 TYNDP hours on ``year``'s calendar by date; fill 31 December with 24 December.

    Hour ``k`` of TYNDP maps to 1 January 00:00 + ``k`` hours, so 1 January to
    30 December line up by date. The 24 missing hours of 31 December repeat the
    hours one week earlier (24 December), keeping the weekly cycle. In a leap year
    the hours are laid on a common year first and 29 February takes 28 February's.
    """
    if len(values) != HOURS_TYNDP2024:
        raise ValueError(f"expected {HOURS_TYNDP2024} TYNDP hours, got {len(values)}")
    if calendar.isleap(int(year)):
        return _on_leap_year(tyndp2024_to_calendar(values, COMMON_REFERENCE_YEAR), year)
    index = pd.date_range(f"{year}-01-01 00:00", periods=HOURS_YEAR, freq="h")
    arr = np.asarray(values.to_numpy(dtype=float))
    # The last 168 TYNDP hours are 24 to 30 December; their first 24 are 24 December.
    fill = arr[-168:-144]
    full = np.concatenate([arr, fill], axis=0)
    if isinstance(values, pd.Series):
        return pd.Series(full, index=index, name=values.name)
    return pd.DataFrame(full, index=index, columns=values.columns)


def ntplus_to_calendar(values: pd.DataFrame | pd.Series, year: int = 2013):
    """Lay 8,760 NT+ hours (1 January to 31 December) on ``year``'s calendar one to one.

    In a leap year 29 February takes 28 February's hours.
    """
    if len(values) != HOURS_YEAR:
        raise ValueError(f"expected {HOURS_YEAR} NT+ hours, got {len(values)}")
    if calendar.isleap(int(year)):
        return _on_leap_year(ntplus_to_calendar(values, COMMON_REFERENCE_YEAR), year)
    index = pd.date_range(f"{year}-01-01 00:00", periods=HOURS_YEAR, freq="h")
    out = values.copy()
    out.index = index
    return out


def weekly_low_day(demand_mw: pd.Series) -> int:
    """Day position 0..6 (0 = TYNDP day 1, 1 January) with the lowest mean daily demand.

    Used to report the TYNDP calendar's weekday: the low day is the Sunday.
    """
    daily = pd.Series(np.asarray(demand_mw, dtype=float)[: (len(demand_mw) // 24) * 24]).groupby(
        np.arange((len(demand_mw) // 24) * 24) // 24).mean()
    by_position = daily.groupby(daily.index % 7).mean()
    return int(by_position.idxmin())


def _first_day_if_sunday_at(position: int) -> str:
    """Weekday of day 0 when day ``position`` (mod 7) is a Sunday."""
    return ["Sunday", "Saturday", "Friday", "Thursday", "Wednesday", "Tuesday", "Monday"][int(position) % 7]


# --------------------------------------------------------------------------- zones and levels

def zone_prices(hourly: pd.DataFrame, zone_weights: dict, column_format: str = "mc_{node}") -> pd.DataFrame:
    """One price series per model column from the TYNDP node columns (weighted where declared)."""
    out = {}
    for column, weights in zone_weights.items():
        total = sum(float(w) for w in weights.values())
        if total <= 0:
            raise ValueError(f"{column}: zone weights sum to zero")
        series = None
        for node, weight in weights.items():
            name = column_format.format(node=node)
            if name not in hourly:
                raise ValueError(f"{column}: TYNDP column {name!r} not found")
            part = pd.to_numeric(hourly[name], errors="coerce").astype(float) * (float(weight) / total)
            series = part if series is None else series + part
        out[column] = series.to_numpy()
    return pd.DataFrame(out, index=hourly.index)


def climate_weighted_level(means_by_year: dict, weights: dict) -> pd.Series:
    """Weighted mean of per-climate-year means (each a Series by column)."""
    missing = [y for y in weights if y not in means_by_year]
    if missing:
        raise ValueError(f"climate years without data: {missing}")
    total = sum(weights.values())
    return sum(means_by_year[y] * (w / total) for y, w in weights.items())


def relevel(prices: pd.DataFrame, level: pd.Series, method: str = "ratio") -> pd.DataFrame:
    """Move each column's annual mean to ``level``: by ratio (zeros stay zero) or by shift."""
    out = prices.copy()
    for column in prices.columns:
        mean = float(prices[column].mean())
        target = float(level[column])
        if method == "ratio":
            if abs(mean) < 1e-9:
                raise ValueError(f"{column}: shape mean is zero; a ratio re-level is undefined")
            out[column] = prices[column] * (target / mean)
        elif method == "shift":
            out[column] = prices[column] + (target - mean)
        else:
            raise ValueError(f"unknown level method {method!r}")
    return out


# --------------------------------------------------------------------------- loaders

def tyndp_input_files(variant: str, tyndp_year: int, data_dir: str) -> dict:
    """The TYNDP files of a variant and TYNDP year, keyed as rule build_tyndp_price_shape names them.

    TYNDP 2024 DE/GA: ``tyndp2024_cy_<climate year>`` -> ``tyndp2024_{scenario}{year}CY{cy}_hourly.csv``;
    NT+ 2026: ``ntplus_prices`` -> ``tyndp2026_NTplus{year}_mc.csv`` and ``ntplus_wind`` -> the shape
    weather scenario's UK00 wind. A year whose files are not extracted fails as a missing input.
    """
    scenario, _ = parse_variant(variant)
    year = int(tyndp_year)
    if scenario == "NTplus2026" and ntplus_bracket(year) is not None:
        # A year NT+ does not publish reads its neighbours' files (user decision, 11 Nov 2026).
        y0, y1, _ = ntplus_bracket(year)
        return {f"{key}_{y}": path for y in (y0, y1) for key, path in tyndp_input_files(variant, y, data_dir).items()}
    if scenario == "NTplus2026":
        return {"ntplus_prices": f"{data_dir}/tyndp2026_NTplus{year}_mc.csv",
                "ntplus_wind": f"{data_dir}/tyndp2026_NTplus{year}_WS{shape_year_of(scenario, year)}_UK00_wind.csv"}
    return {f"tyndp2024_cy_{cy}": f"{data_dir}/tyndp2024_{scenario}{year}CY{cy}_hourly.csv"
            for cy in TYNDP2024_CLIMATE_WEIGHTS}


def load_tyndp2024_climate_years(files: dict) -> dict:
    """{climate year: hourly frame} from ``tyndp2024_{scen}{year}CY{climate year}_hourly.csv`` files."""
    frames = {}
    for year, path in files.items():
        frame = pd.read_csv(path)
        if len(frame) != HOURS_TYNDP2024:
            raise ValueError(f"{path}: expected {HOURS_TYNDP2024} hours, got {len(frame)}")
        frames[int(year)] = frame
    return frames


def load_ntplus_prices(path, tyndp_year: int = 2050) -> dict:
    """{weather scenario: frame with ``mc_{node}`` columns} from an NT+ price table (``tyndp2026_NTplus{year}_mc.csv``).

    TYNDP publishes 8,760 hours in every target year (a leap target year such as 2040 included); the hours are
    laid on the weather year's calendar, and the model year's calendar is handled downstream.
    """
    table = pd.read_csv(path)
    if len(table) != HOURS_YEAR:
        raise ValueError(f"{path}: expected {HOURS_YEAR} hours, got {len(table)}")
    out = {}
    for ws in ntplus_weights(tyndp_year):
        if not any(c.endswith(f"_WS{ws}") for c in table.columns):
            raise ValueError(f"{path}: no columns for weather scenario {ws} of {tyndp_year}")
        columns = {c: "mc_" + c.rsplit("_WS", 1)[0] for c in table.columns if c.endswith(f"_WS{ws}")}
        out[ws] = table[list(columns)].rename(columns=columns)
    return out


def ntplus_wind_regressor(frame: pd.DataFrame) -> pd.Series:
    """NT+ UK00 offshore wind proxy: dispatched offshore generation plus RES curtailment (MW).

    The NT+ dashboards publish dispatched generation only, not the potential that
    TYNDP 2024 publishes; adding total RES curtailment restores most of the
    potential because GB curtailment in 2050 is mostly wind (declared).
    """
    return (pd.to_numeric(frame["UK00_wind_offshore_generation_mw"], errors="coerce").fillna(0.0)
            + pd.to_numeric(frame["UK00_res_curtailment_mw"], errors="coerce").fillna(0.0))


# --------------------------------------------------------------------------- build

def build_tyndp_shape(
    variant: str,
    wind_weather_file,
    out_file,
    audit_file,
    tyndp2024_files: dict | None = None,
    ntplus_price_file=None,
    ntplus_wind_file=None,
    ic1_shape_file=None,
    eur_to_gbp: float = 0.86,
    level_method: str = LEVEL_METHOD,
    shape_climate_year=None,
    calendar_year: int = 2013,
    logger=_LOG,
    tyndp_year: int = 2050,
) -> pd.DataFrame:
    """Build a TYNDP year's hourly foreign-price shape of one variant and write it with an audit.

    Args:
        variant: ``DE``, ``GA`` or ``NTplus2026``, optionally with ``_ic1rescale``.
        wind_weather_file: the model's weather-year offshore wind profile (2013).
        out_file, audit_file: the shape CSV (GBP 2024/MWh, one column per model
            counterparty column, indexed on ``calendar_year``) and its JSON audit.
        tyndp2024_files: {climate year: hourly CSV} for DE or GA.
        ntplus_price_file, ntplus_wind_file: the NT+ price table and the UK00 wind
            extract of the shape weather scenario.
        ic1_shape_file: the IC1 conditioned ENTSO-E shape (``ic1_rescale`` only).
        eur_to_gbp: the model's EUR->GBP rate (``interconnectors.pricing.eur_to_gbp``).
        level_method: ``ratio`` or ``shift``.
        shape_climate_year: CY for DE/GA (default 2009), WS for NT+ (default ``092``).
        calendar_year: the weather year whose calendar the shape is laid on.
        tyndp_year: the TYNDP year the files are for (the modelled year).

    Returns:
        The shape frame that was written.
    """
    scenario, route = parse_variant(variant)
    price_base = TYNDP_PRICE_BASE_YEAR[scenario]
    fx = eur_to_model_gbp(eur_to_gbp, price_base)
    wind_weather_raw = pd.read_csv(wind_weather_file, index_col=0, parse_dates=True)
    z_weather = wind_index(wind_weather_raw)

    if scenario == "NTplus2026":
        weights = ntplus_weights(tyndp_year)
        shape_key = str(shape_climate_year or shape_year_of(scenario, tyndp_year))
        by_ws = load_ntplus_prices(ntplus_price_file, tyndp_year)
        per_year = {ws: zone_prices(frame, ZONE_WEIGHTS_NTPLUS2026) for ws, frame in by_ws.items()}
        shape_raw = ntplus_to_calendar(per_year[shape_key], calendar_year)
        wind_shape = ntplus_to_calendar(ntplus_wind_regressor(pd.read_csv(ntplus_wind_file)), calendar_year)
        zone_note = dict(ZONE_WEIGHTS_NTPLUS2026)
        calendar_rule = "8,760 NT+ hours laid one to one on the calendar"
        weekday_note = "not checked for NT+ (synthetic climate-projection years)"
    else:
        weights = TYNDP2024_CLIMATE_WEIGHTS
        shape_key = int(shape_climate_year or SHAPE_YEAR[scenario])
        frames = load_tyndp2024_climate_years(tyndp2024_files)
        per_year = {cy: zone_prices(frame, ZONE_WEIGHTS_TYNDP2024) for cy, frame in frames.items()}
        shape_raw = tyndp2024_to_calendar(per_year[shape_key], calendar_year)
        wind_shape = tyndp2024_to_calendar(
            frames[shape_key]["UK00_wind_offshore_potential_mw"].astype(float), calendar_year)
        zone_note = dict(ZONE_WEIGHTS_TYNDP2024)
        calendar_rule = ("TYNDP hour k -> 1 Jan 00:00 + k h (1 Jan to 30 Dec by date); "
                         "31 Dec filled with 24 Dec (same weekday one week earlier)")
        low = {node: weekly_low_day(frames[shape_key][f"demand_{node}_mw"])
               for node in ("UK00", "FR00", "DE00") if f"demand_{node}_mw" in frames[shape_key]}
        weekday_note = {
            "lowest_demand_day_position_0_is_1_january": low,
            "tyndp_1_january_weekday": sorted({_first_day_if_sunday_at(p) for p in low.values()}),
            "calendar_1_january": {str(calendar_year): pd.Timestamp(f"{calendar_year}-01-01").day_name(),
                                   str(tyndp_year): pd.Timestamp(f"{tyndp_year}-01-01").day_name()},
            "treatment": "weekdays not re-aligned (declared)",
        }

    means_eur = {k: v.mean() for k, v in per_year.items()}
    level_eur = climate_weighted_level(means_eur, weights)
    level_gbp = level_eur * fx

    if route == "ic1_rescale":
        base = pd.read_csv(ic1_shape_file, index_col=0, parse_dates=True)
        if base.index.tz is not None:
            base.index = base.index.tz_convert(None)
        base = base[list(level_gbp.index)]
        shaped = relevel(base, level_gbp, level_method)
        betas = pd.Series(dtype=float)
        conditioned_before_level = base
        source_floor, floored_hours, in_year_corr = pd.Series(dtype=float), {}, {}
        shape_source = f"IC1 shape {ic1_shape_file} rescaled to the TYNDP {scenario} level (decision 8)"
    else:
        prices_gbp = shape_raw * fx
        z_shape = wind_index(wind_shape.to_frame("uk00"))
        conditioned_before_level, betas = condition_prices_on_wind(prices_gbp, z_shape, z_weather)
        # TYNDP's market model has no negative bids, so its prices never fall below
        # the zone's own minimum (0 in every zone). The linear wind swap would push
        # windy zero-price hours below it; they are held at the source floor, and
        # the ratio re-level then restores the weighted mean exactly (declared).
        source_floor = prices_gbp.min()
        floored_hours = {c: int((conditioned_before_level[c] < source_floor[c]).sum()) for c in prices_gbp}
        conditioned_before_level = conditioned_before_level.clip(lower=source_floor, axis=1)
        in_year_corr = {c: round(float(prices_gbp[c].corr(z_shape.set_axis(prices_gbp.index))), 4) for c in prices_gbp}
        shaped = relevel(conditioned_before_level, level_gbp, level_method)
        shape_source = (f"TYNDP {scenario} {tyndp_year} shape year {shape_key}, "
                        f"conditioned on {calendar_year} GB offshore wind")

    z_on_shape = align_by_calendar(z_weather, shaped.index)
    audit = {
        "task": "Y3",
        "variant": variant,
        "tyndp_scenario": scenario,
        "route": route,
        "shape_source": shape_source,
        "shape_year": shape_key,
        "climate_weights": {str(k): v for k, v in weights.items()},
        "zone_weights": zone_note,
        "calendar": {"year": calendar_year, "rule": calendar_rule, "weekday": weekday_note},
        "price_base": {
            "tyndp_eur_year": price_base,
            "model_gbp_year": MODEL_PRICE_BASE_YEAR,
            "hicp_ea20_index": {str(price_base): EA_HICP_ANNUAL_INDEX[price_base],
                                str(MODEL_PRICE_BASE_YEAR): EA_HICP_ANNUAL_INDEX[MODEL_PRICE_BASE_YEAR]},
            "hicp_deflator": round(EA_HICP_ANNUAL_INDEX[MODEL_PRICE_BASE_YEAR] / EA_HICP_ANNUAL_INDEX[price_base], 6),
            "eur_to_gbp": float(eur_to_gbp),
            "gbp_per_tyndp_eur": round(fx, 6),
            "source": "HICP: Eurostat prc_hicp_aind EA20 INX_A_AVG (retrieved 30 Sep 2026); "
                      "FX: interconnectors.pricing.eur_to_gbp, the rate IC1's ENTSO-E prices use",
        },
        "climate_year_means_eur": {str(k): v.round(3).to_dict() for k, v in means_eur.items()},
        "weighted_level_eur": level_eur.round(3).to_dict(),
        "weighted_level_gbp": level_gbp.round(3).to_dict(),
        "level_method": level_method,
        "slope_gbp_per_mwh_per_sd": betas.round(4).to_dict(),
        "corr_with_shape_year_own_uk00_wind": in_year_corr,
        "source_floor_gbp": source_floor.round(4).to_dict(),
        "hours_held_at_source_floor": floored_hours,
        "corr_with_weather_year_wind_before": (
            {c: round(float(pd.Series((shape_raw * fx)[c].to_numpy()).corr(pd.Series(z_on_shape.to_numpy()))), 4)
             for c in shaped} if route == "tyndp_shape" else {}),
        "corr_with_weather_year_wind_after": {
            c: round(float(pd.Series(shaped[c].to_numpy()).corr(pd.Series(z_on_shape.to_numpy()))), 4) for c in shaped},
        "mean_gbp": shaped.mean().round(3).to_dict(),
        "std_gbp": shaped.std().round(3).to_dict(),
        "hours_below_zero": {c: int((shaped[c] < 0).sum()) for c in shaped},
        "hours": int(len(shaped)),
        "licence": "TYNDP data CC-BY 4.0, ENTSO-E and ENTSOG (see data/interconnectors/tyndp2024/README.md)",
        "declared": [
            "decision 28: TYNDP already models a GB zone, so the IC1 elasticity may double-count GB's price effect",
            f"weekdays of the TYNDP calendar are not re-aligned to the {tyndp_year} model calendar",
            "no time-zone shift applied to TYNDP hours",
            "the conditioned shape is held at the source's own price floor (0) before the ratio re-level",
        ],
    }
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    out = shaped.copy()
    out.index.name = "datetime"
    out.to_csv(out_file)
    Path(audit_file).write_text(json.dumps(audit, indent=2, default=str))
    for column in shaped.columns:
        logger.info("%s %s: level %.2f EUR%d = %.2f GBP2024/MWh, slope %s, corr with %d wind %.3f",
                    variant, column, level_eur[column], price_base, level_gbp[column],
                    f"{betas[column]:.2f}" if column in betas else "n/a", calendar_year,
                    audit["corr_with_weather_year_wind_after"][column])
    return shaped


# --------------------------------------------------------------------------- network hook

def tyndp_market_reference(hourly, country, pricing, price_floor, price_cap, apply_price_case):
    """One TYNDP hourly price at its own level (no CP.29 re-centring) and its annual target.

    The configured price case still scales the annual level (``apply_price_case``,
    the ``add_to_network`` helper); at ``central`` the series is the file's own.
    Returns ``(reference, target)``.
    """
    hourly = pd.to_numeric(hourly, errors="coerce").astype(float)
    if hourly.notna().sum() == 0:
        raise ValueError(f"TYNDP foreign price: no hourly observations for {country}")
    hourly = hourly.interpolate(limit_direction="both").ffill().bfill()
    own = float(hourly.mean())
    target = float(apply_price_case(own, country, pricing))
    reference = (hourly * (target / own) if abs(own) > 1e-9 else hourly + target).clip(price_floor, price_cap)
    return reference, float(reference.mean())


# --------------------------------------------------------------------------- tranches

def tranche_steps(capacity_mw_by_link: pd.DataFrame, level_by_country: dict, tranche_config: dict,
                  demand_twh: pd.Series) -> pd.DataFrame:
    """IC1 tranche steps for a set of cables priced at one level per counterparty.

    ``capacity_mw_by_link`` carries ``capacity_mw`` and ``counterparty_country`` per
    cable; ``level_by_country`` maps the ENTSO-E/TYNDP column (``France``, ...,
    ``Ireland``) to the annual level (GBP/MWh). On the TYNDP basis import and export
    level are the same single price. Uses ``volume_tranches.derive_market_tranches``
    unchanged, so the 2050 steps follow the 2030 formula exactly.
    """
    from scripts.interconnectors.volume_tranches import derive_market_tranches

    column_of = {"northern ireland": "Ireland", "isle of man": "Ireland"}
    links = capacity_mw_by_link.copy()
    col = links["counterparty_country"].astype(str).map(lambda c: column_of.get(c.lower(), c))
    links["p_nom"] = pd.to_numeric(links["capacity_mw"], errors="coerce").astype(float)
    links["market_import_level"] = col.map(level_by_country).astype(float)
    links["market_export_level"] = links["market_import_level"]
    if links["market_import_level"].isna().any():
        raise ValueError(f"no level for {sorted(set(col[links['market_import_level'].isna()]))}")
    return derive_market_tranches(
        links[["p_nom", "counterparty_country", "market_import_level", "market_export_level"]],
        tranche_config["markets"], demand_twh,
        elasticity=float(tranche_config["elasticity"]), n_tranches=int(tranche_config["tranches"]),
    )
