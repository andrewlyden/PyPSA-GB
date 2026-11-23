"""Contract-aware CfD strike prices, indexed from published 2012-price results.

The inline ``support.labels.cfd_strike_prices`` mapping in ``config/defaults/``
mixed auction clearing prices, administrative strike prices and round numbers
that never cleared, all without a declared price year. Three of its entries
were wrong: ``AR5_offshore`` recorded the AR4 offshore clearing price (AR5
attracted no offshore bids at all), while ``AR6_offshore`` (73.0),
``AR6_onshore`` (53.0) and ``AR6_solar`` (47.0) recorded *administrative*
strike prices rather than the 58.87 / 50.90 / 50.07 that actually cleared.

This module replaces that mapping for forecast cases with a table of published
clearing prices in 2012 prices, and indexes them to the model price year with
CPI. DESNZ publishes AR7 and AR7a strikes in both 2012 and 2024 prices, and
those pairs reproduce the ONS CPI all-items annual-average ratio
(133.9 / 96.1 = 1.393340) to five decimal places, so the indexation basis here
is the department's own.

A round that has several delivery-year clearing prices (AR1 onshore, AR2 and
AR3 offshore) is collapsed to a capacity-weighted mean, because a PyPSA
generator carries a round label but no delivery-year vintage.

Contract-level technology (rebuild batch, 14 Oct 2026). A generator may carry
``cfd_technology`` (``CFD_TECHNOLOGY_COLUMN``), the strike-table technology of
its own contract where it is known: ``floating_offshore`` (Green Volt, Erebus,
Pentland), ``offshore_permitted_reduction`` (the AR6 permitted-reduction
contracts), ``offshore_scotland`` (the AR7 Scottish pot) or
``remote_island_wind`` (Viking). The lookup takes that row first and otherwise
keys on the carrier as before, so the anonymous increment keeps the
(round, carrier) lookup. Every consumer passes the same column
(:func:`contract_technology`): the strike table, the contract-value bids, the
bill's CfD line, the settlement, CfD re-pricing and revenue tracking agree.
"""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "cfd_round", "technology", "delivery_year",
    "strike_price_gbp_per_mwh_2012", "price_basis", "source",
)

#: The 2030 chain's modelled round (user decision 27 Oct 2026, F18): CfD capacity the vintage
#: schedule cannot place on a published round wins a contract in Allocation Round 8 at a modelled
#: clearing price (``scripts/market/build_ar8_modelled_strikes.py``). It replaces the old declared
#: ``future`` round, a copy of the latest round's clear. Its rows are also the carrier fallback.
MODELLED_ROUND = "AR8_modelled"

#: Generator column naming the strike-table technology of the unit's own contract.
CFD_TECHNOLOGY_COLUMN = "cfd_technology"


def strike_lookup(table, carrier, cfd_round, technology=""):
    """``table.lookup`` with the contract technology when there is one.

    Older strike objects (a legacy mapping wrapper, test stubs) take only
    (carrier, round); they are called that way when no technology is recorded.
    """
    if technology is not None and str(technology).strip() and str(technology) != "nan":
        return table.lookup(carrier, cfd_round, technology)
    return table.lookup(carrier, cfd_round)


def contract_technology(frame, index=None) -> pd.Series:
    """Each row's contract technology (``cfd_technology``), "" where none is recorded."""
    if index is None:
        index = frame.index
    if CFD_TECHNOLOGY_COLUMN not in getattr(frame, "columns", ()):
        return pd.Series("", index=index, dtype=object)
    values = frame[CFD_TECHNOLOGY_COLUMN].reindex(index)
    return values.where(values.notna(), "").astype(str).str.strip().replace({"nan": "", "None": ""})

# PyPSA carrier to the technology names used by the DESNZ auction results.
CARRIER_TO_TECHNOLOGY = {
    "wind_offshore": "offshore",
    "wind_onshore": "onshore",
    "solar_pv": "solar",
    "nuclear": "nuclear",
    "tidal_stream": "tidal",
    "marine": "tidal",
    "shoreline_wave": "wave",
    "biomass": "dedicated_biomass",
    "waste_to_energy": "energy_from_waste",
    "CCS Biomass": "beccs",
    "geothermal": "geothermal",
}


# Rounds where a carrier's usual technology name was not on offer, so the
# labelled capacity can only be the round's neighbouring technology.
#
# AR2 and AR3 were Pot 2 rounds: onshore wind was excluded from both, and AR3
# was the first round to admit remote island wind, which cleared at the Pot 2
# price. Onshore capacity carrying an AR3 label is therefore remote island
# wind - in the CP30 fleet, Muaitheabhal on Lewis and Costa Head and Hesta Head
# on Orkney - and must settle at the published AR3 remote island wind strike
# rather than fall back to the modelled AR8 onshore strike.
ROUND_TECHNOLOGY_ALIASES = {
    ("AR3", "onshore"): "remote_island_wind",
}


def carrier_to_technology(carrier) -> str:
    """Map a model carrier to an auction technology name, defaulting to itself."""
    if carrier is None or (isinstance(carrier, float) and np.isnan(carrier)):
        return ""
    return CARRIER_TO_TECHNOLOGY.get(str(carrier), str(carrier))


class StrikeTable:
    """Round- and technology-keyed strike prices on one declared price basis.

    ``lookup`` returns ``(strike, key, is_fallback)``. ``is_fallback`` is true
    whenever the generator's own round could not be matched and the modelled
    AR8 strike (:data:`MODELLED_ROUND`) was used instead, which must stay visible
    in the report. Capacity labelled with the AR8 round itself is not a fallback.
    """

    def __init__(self, strikes: dict, price_year: int, indexation: dict, sources: dict):
        self.strikes = strikes
        self.price_year = int(price_year)
        self.indexation = indexation
        self.sources = sources

    def __bool__(self) -> bool:
        return bool(self.strikes)

    @property
    def rounds(self) -> list:
        return sorted({round_name for round_name, _ in self.strikes})

    def resolve(self, carrier, cfd_round, technology=None):
        """Return ``((round, technology), is_fallback)`` for the table entry a contract settles at.

        ``technology`` is the contract's own strike-table technology
        (``cfd_technology``); when the table has that row for the round it wins,
        and otherwise the carrier's technology is used. ``None`` when no entry
        matches, not even the modelled AR8 strike.
        """
        contract = str(technology).strip() if technology is not None and pd.notna(technology) else ""
        if contract and pd.notna(cfd_round) and str(cfd_round).strip():
            key = (str(cfd_round).strip(), contract)
            if key in self.strikes:
                return key, False
        technology = carrier_to_technology(carrier)
        if pd.notna(cfd_round) and str(cfd_round).strip():
            round_name = str(cfd_round).strip()
            key = (round_name, technology)
            if key in self.strikes:
                return key, False
            if aliased := ROUND_TECHNOLOGY_ALIASES.get(key):
                alias_key = (round_name, aliased)
                if alias_key in self.strikes:
                    return alias_key, False
        # A long-range (vintaged) table carries PAR7_<year> rounds and, since 13 Nov 2026, the 2030 cohort's own AR8
        # strikes: there a labelled round the table lacks is an error to find, never the AR8 strike (only an
        # unlabelled 2030 unit falls back to it).
        if pd.notna(cfd_round) and str(cfd_round).strip() and any(
                str(r).startswith("PAR7_") for r, _ in self.strikes):
            return None, True
        fallback = (MODELLED_ROUND, technology)
        if fallback in self.strikes:
            return fallback, True
        return None, True

    def lookup(self, carrier, cfd_round, technology=None):
        key, is_fallback = self.resolve(carrier, cfd_round, technology)
        if key is None:
            return None, None, True
        return self.strikes[key], f"{key[0]}_{key[1]}", is_fallback

    def adjusted(self, deltas: dict, label: str) -> "StrikeTable":
        """Return a copy with ``deltas`` (keyed by ``(round, technology)``) subtracted.

        Used by the public-ownership bill link: a re-priced contract settles at
        ``strike - delta`` through the same calculator as the private case, so a
        zero delta reproduces the private bill exactly and a negative payment
        (two-way contract) is kept.
        """
        unknown = sorted(set(deltas) - set(self.strikes))
        if unknown:
            raise KeyError(f"strike adjustments name entries absent from the table: {unknown}")
        strikes = {key: value - float(deltas.get(key, 0.0)) for key, value in self.strikes.items()}
        table = StrikeTable(strikes, self.price_year, self.indexation, self.sources)
        table.adjustment_label = label
        table.adjustments = dict(deltas)
        return table

    def as_records(self) -> list:
        return [
            {"cfd_round": round_name, "technology": technology,
             "strike_price_gbp_per_mwh": strike, "price_year": self.price_year}
            for (round_name, technology), strike in sorted(self.strikes.items())
        ]


def indexation_factor(indexation: dict) -> float:
    """Return the CPI factor carrying 2012-price strikes to the model basis."""
    required = {"from_price_year", "to_price_year", "from_index", "to_index", "factor", "index"}
    if missing := required.difference(indexation):
        raise ValueError(f"cfd_strike_indexation is missing: {sorted(missing)}")
    if int(indexation["from_price_year"]) != 2012:
        raise ValueError("Published CfD strike prices are quoted in 2012 prices")
    factor = float(indexation["factor"])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("CfD strike indexation factor must be finite and positive")
    implied = float(indexation["to_index"]) / float(indexation["from_index"])
    if not np.isclose(factor, implied, rtol=1e-9):
        raise ValueError(
            f"CfD strike indexation factor {factor} does not reconcile to its indices {implied}"
        )
    return factor


#: Keys a strike override must declare (``model_linked_rules.cfd.strike_overrides``).
OVERRIDE_COLUMNS = ("cfd_round", "technology", "strike_price_gbp_per_mwh_2012", "price_basis", "source", "replaces")


def strike_overrides(assumptions: dict | None) -> list:
    """The bill's declared strike overrides (``model_linked_rules.cfd.strike_overrides``), empty when none."""
    return list((((assumptions or {}).get("model_linked_rules") or {}).get("cfd") or {}).get("strike_overrides") or [])


def apply_strike_overrides(frame: pd.DataFrame, overrides) -> pd.DataFrame:
    """Replace a published row's strike with a declared later contract term (29 Sep 2026).

    Each override names the (round, technology) it changes, the published value
    it replaces (checked against the table, so a stale override fails) and its
    source.  Used only where the bill and the ownership layer read the table:
    the table file itself is a declared input of the market solves, which keep
    the published value (see the override's ``solve_side`` note).
    """
    frame = frame.copy()
    for override in overrides or ():
        missing = [k for k in OVERRIDE_COLUMNS if k not in override]
        if missing:
            raise ValueError(f"CfD strike override lacks {missing}: {override}")
        if str(override["price_basis"]).strip() != "2012 prices":
            raise ValueError("A CfD strike override must be quoted in 2012 prices")
        mask = (frame["cfd_round"].astype(str).str.strip().eq(str(override["cfd_round"]))
                & frame["technology"].astype(str).str.strip().eq(str(override["technology"])))
        if not mask.any():
            raise ValueError(f"CfD strike override names no table row: {override['cfd_round']} {override['technology']}")
        published = pd.to_numeric(frame.loc[mask, "strike_price_gbp_per_mwh_2012"], errors="raise")
        if not np.allclose(published, float(override["replaces"])):
            raise ValueError(f"CfD strike override for {override['cfd_round']} {override['technology']} replaces "
                             f"{override['replaces']}, but the table has {sorted(set(published))}")
        frame.loc[mask, "strike_price_gbp_per_mwh_2012"] = float(override["strike_price_gbp_per_mwh_2012"])
        frame.loc[mask, "source"] = str(override["source"])
    return frame


def load_strike_table(path, indexation: dict, model_price_year: int, overrides=None) -> StrikeTable:
    """Read the published strike table and index it to the model price year.

    ``overrides`` (:func:`strike_overrides`) replace published rows with declared
    later contract terms; the market solves pass none.
    """
    frame = pd.read_csv(path)
    if missing := set(REQUIRED_COLUMNS).difference(frame.columns):
        raise ValueError(f"CfD strike table is missing columns: {sorted(missing)}")
    frame = apply_strike_overrides(frame, overrides)
    if (frame["price_basis"].astype(str).str.strip() != "2012 prices").any():
        raise ValueError("Every strike-table row must be quoted in 2012 prices")
    if frame["source"].isna().any() or frame["source"].astype(str).str.strip().eq("").any():
        raise ValueError("Every strike-table row needs a source")

    if int(indexation["to_price_year"]) != int(model_price_year):
        raise ValueError(
            f"cfd_strike_indexation lands on {indexation['to_price_year']}, but the model prices "
            f"are {model_price_year}; CfD payments would mix bases against the reference price."
        )
    factor = indexation_factor(indexation)

    frame["strike_price_gbp_per_mwh_2012"] = pd.to_numeric(
        frame["strike_price_gbp_per_mwh_2012"], errors="raise"
    )
    if not np.isfinite(frame["strike_price_gbp_per_mwh_2012"]).all():
        raise ValueError("CfD strike table contains non-finite strike prices")
    # A round with several delivery-year prices collapses to a capacity-weighted
    # mean: a PyPSA generator carries a round label but no delivery-year vintage,
    # so a plain mean would over-weight small tranches.
    weights = pd.to_numeric(frame.get("capacity_mw"), errors="coerce")
    frame["weight"] = weights.where(weights.gt(0), 1.0).fillna(1.0)
    frame["weighted"] = frame["strike_price_gbp_per_mwh_2012"] * frame["weight"]
    grouped = frame.groupby(["cfd_round", "technology"], dropna=False)[["weighted", "weight"]].sum()
    strikes = {
        (str(round_name).strip(), str(technology).strip()):
            float(row.weighted / row.weight) * factor
        for (round_name, technology), row in grouped.iterrows()
    }
    sources = {
        f"{str(round_name).strip()}_{str(technology).strip()}": sorted(
            set(group["source"].astype(str))
        )
        for (round_name, technology), group in frame.groupby(["cfd_round", "technology"], dropna=False)
    }
    return StrikeTable(strikes, int(indexation["to_price_year"]), indexation, sources)


def bm_strike_table_path(assumptions: dict | None):
    """The table the solved BM bid on (``model_linked_rules.cfd.bm_strike_table``), or None (every 2030 bill).

    FES 2050 decision 41: the 2050 contract strikes (``strike_table``) are set in the bill and
    settlement layer only, after the months were solved on decision 6's strikes. The CfD
    curtailment compensation reconciles against the solved BM's accepted bids, so it is valued
    at the strikes those bids were priced on; the difference payments at the contract strikes.
    """
    return (((assumptions or {}).get("model_linked_rules") or {}).get("cfd") or {}).get("bm_strike_table")


def load_bm_strike_table(assumptions: dict, model_price_year: int):
    """:func:`bm_strike_table_path` loaded on the bill's indexation (no overrides), or None."""
    path = bm_strike_table_path(assumptions)
    if not path:
        return None
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    full = Path(path) if Path(path).is_absolute() else root / path
    return load_strike_table(full, assumptions["cfd_strike_indexation"], int(model_price_year))
