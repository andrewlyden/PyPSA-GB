"""Volume tranches on the foreign price: GB's trade moves the price it trades at (item IC1).

A flat CP.29 price let every exported MWh earn the same price however much GB
sold. Here the foreign price responds to GB's aggregate trade with each coupled
market. GB's capacity into a market is split into equal tranches; the first trades
at the shaped CP.29 price, and each later tranche's import price is one step higher
and its export price one step lower, with

    step = elasticity x level / market demand (GW) x tranche width (GW),

a constant-elasticity supply curve (dP/dQ = elasticity x P / Q) linearised over
each tranche. ``level`` is the CP.29 import price for the import side and the CP.29
export price for the export side, so the asymmetry follows from CP.29 itself.

**Markets, not cables.** France, Belgium, the Netherlands, Denmark and Germany clear
inside one coupled continental market, so a GW into Belgium moves the same price as a
GW into France; the tranche is on GB's summed flow into each market. The hourly
shape stays per bidding zone.

**Sources, as approved on 22 September 2026.** Market demand: NESO FES 2025 ES2
European supply table, 2030. Elasticity 1.0: Swinand and O'Mahoney (2014, MPRA
56246), SEM price elasticity to demand of about 1 on 2008-12 half-hourly data;
Cludius et al. (2014, Energy Economics 44) imply about 1.6 for Germany, the high case.

The steps are stored as link attributes (``market_tranche_*``), which persist in the
NetCDF, so the solve and every post-solve valuation read the same curve.
"""

import logging

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8760.0
TRANCHE_COLUMNS = (
    "market_tranche_group", "market_tranche_width_mw", "market_tranche_count",
    "market_tranche_import_step", "market_tranche_export_step",
)


def load_market_demand_twh(path, eu_scenario: str, pathway_alignment: str, year: int) -> pd.Series:
    """Annual demand (TWh) per country for one ES2 EU scenario and FES pathway alignment."""
    frame = pd.read_csv(path)
    rows = frame[
        frame["Variable"].astype(str).eq("Demand (TWh)")
        & frame["EU Scenario"].astype(str).eq(eu_scenario)
        & frame["FES Pathway Alignment"].astype(str).eq(pathway_alignment)
    ]
    if rows.empty:
        raise ValueError(f"no ES2 demand rows for {eu_scenario!r} / {pathway_alignment!r} in {path}")
    demand = pd.to_numeric(rows.set_index("Country")[str(year)], errors="coerce")
    if demand.index.duplicated().any():
        raise ValueError(f"ES2 demand has duplicate countries for {eu_scenario!r} / {pathway_alignment!r}")
    return demand.dropna().astype(float)


def _market_of(markets: dict) -> dict[str, str]:
    lookup = {}
    for market, spec in markets.items():
        for country in spec.get("counterparties", []):
            key = str(country).strip().lower()
            if key in lookup:
                raise ValueError(f"counterparty {country!r} is assigned to both {lookup[key]!r} and {market!r}")
            lookup[key] = market
    return lookup


def derive_market_tranches(
    ic_links: pd.DataFrame, markets: dict, demand_twh: pd.Series, elasticity: float, n_tranches: int
) -> pd.DataFrame:
    """One row per market with cables: capacity, tranche width and the two steps.

    ``ic_links`` is indexed by link name and carries ``p_nom``,
    ``counterparty_country``, ``market_import_level`` and ``market_export_level``.
    Every cable must belong to a market; one that does not is refused, not dropped.
    """
    if n_tranches < 1:
        raise ValueError("n_tranches must be at least 1")
    if elasticity < 0:
        raise ValueError("elasticity must be non-negative")
    market_of = _market_of(markets)
    country = ic_links["counterparty_country"].astype(str).str.strip().str.lower()
    unassigned = sorted(set(ic_links.loc[~country.isin(market_of), "counterparty_country"].astype(str)))
    if unassigned:
        raise ValueError(f"interconnector counterparties with no tranche market: {unassigned}")
    market = country.map(market_of)
    rows = []
    for name, spec in markets.items():
        members = ic_links[market.eq(name)]
        if members.empty:
            continue
        missing = [c for c in spec.get("demand_countries", []) if c not in demand_twh.index]
        if missing or not spec.get("demand_countries"):
            raise ValueError(f"market {name!r} has no ES2 demand for {missing or 'any country'}")
        capacity = float(members["p_nom"].sum())
        weights = members["p_nom"] / capacity
        import_level = float((members["market_import_level"] * weights).sum())
        export_level = float((members["market_export_level"] * weights).sum())
        demand = float(demand_twh[list(spec["demand_countries"])].sum())
        demand_gw = demand * 1000.0 / HOURS_PER_YEAR
        width = capacity / n_tranches
        rows.append({
            "market": name,
            "links": ";".join(members.index.astype(str)),
            "counterparties": ";".join(sorted(set(members["counterparty_country"].astype(str)))),
            "capacity_mw": capacity,
            "tranches": int(n_tranches),
            "width_mw": width,
            "demand_countries": ";".join(spec["demand_countries"]),
            "demand_twh": demand,
            "demand_gw": demand_gw,
            "elasticity": float(elasticity),
            "import_level": import_level,
            "export_level": export_level,
            "import_slope_per_gw": elasticity * import_level / demand_gw,
            "export_slope_per_gw": elasticity * export_level / demand_gw,
            "import_step": elasticity * import_level / demand_gw * width / 1000.0,
            "export_step": elasticity * export_level / demand_gw * width / 1000.0,
        })
    return pd.DataFrame(rows).set_index("market") if rows else pd.DataFrame(
        columns=["links", "width_mw", "tranches", "import_step", "export_step"])


def apply_volume_tranches(network, tranche_config: dict, demand_twh: pd.Series, logger=None) -> pd.DataFrame:
    """Derive the tranches for the signed interconnectors and write them onto the links."""
    logger = logger or logging.getLogger(__name__)
    links = network.links
    signed = links[links.get("foreign_market_pricing", pd.Series("", index=links.index))
                   .astype(str).str.lower().eq("piecewise_bidirectional")]
    if signed.empty:
        raise ValueError("volume tranches are enabled but the network has no signed interconnectors")
    table = derive_market_tranches(
        signed[["p_nom", "counterparty_country", "market_import_level", "market_export_level"]],
        tranche_config["markets"], demand_twh,
        elasticity=float(tranche_config["elasticity"]), n_tranches=int(tranche_config["tranches"]),
    )
    for column, default in zip(TRANCHE_COLUMNS, ("", np.nan, 0, np.nan, np.nan)):
        if column not in network.links.columns:
            network.links[column] = default
    network.links["market_tranche_group"] = network.links["market_tranche_group"].fillna("").astype(str)
    for market, row in table.iterrows():
        members = row["links"].split(";")
        network.links.loc[members, "market_tranche_group"] = market
        network.links.loc[members, "market_tranche_width_mw"] = float(row["width_mw"])
        network.links.loc[members, "market_tranche_count"] = int(row["tranches"])
        network.links.loc[members, "market_tranche_import_step"] = float(row["import_step"])
        network.links.loc[members, "market_tranche_export_step"] = float(row["export_step"])
        steps_in = " / ".join(f"+{k * row['import_step']:.2f}" for k in range(int(row["tranches"])))
        steps_out = " / ".join(f"-{k * row['export_step']:.2f}" for k in range(int(row["tranches"])))
        logger.info(
            "Volume tranches %s: %.0f MW in %d x %.0f MW, demand %.1f TWh, import %s, export %s GBP/MWh",
            market, row["capacity_mw"], int(row["tranches"]), row["width_mw"], row["demand_twh"],
            steps_in, steps_out,
        )
    for column in TRANCHE_COLUMNS[1:]:
        network.links[column] = pd.to_numeric(network.links[column], errors="coerce").astype(float)
    return table
