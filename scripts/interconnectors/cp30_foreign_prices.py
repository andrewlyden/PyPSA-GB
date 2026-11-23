"""Foreign prices on CP30's fuel basis for the 2030 interconnectors (item F13, 7 Oct 2026).

**Why.** IC1 priced GB's trade on 2024 ENTSO-E day-ahead prices plus a fitted offset, so foreign
plant ran on 2024 gas and carbon while GB plant ran on CP30's (FES 2024: gas about 100 p/therm,
UK carbon about GBP 129/t in the model). CP30 itself priced trade with an endogenous European
PLEXOS dispatch on its own commodities, so its Stage 1 price (CP.32, mean 69.9) sat far above ours
(45.7). This route takes a published 2030 European dispatch and rebases it to CP30's fuel.

**Source.** ENTSO-E and ENTSOG, TYNDP 2024 National Trends 2030, climate year 2009, PLEXOS market
model standard output (CC-BY 4.0): hourly marginal cost per bidding zone. Extract:
``data/interconnectors/tyndp2024/tyndp2024_NT2030CY2009_hourly.csv`` (see that folder's README).

**Method.**

1. *Zones.* As ``tyndp_prices.ZONE_WEIGHTS_TYNDP2024``: France FR00, Belgium BE00, Netherlands NL00,
   Germany DE00, Denmark DKW1, Norway NOS0, Ireland = the SEM (IE00 and UKNI by GB cable capacity:
   from CP30's cable set in 2030 when the rule passes the composition file, F25.3, so MaresConnect
   counts on IE00; else today's register cables, ``tyndp_prices.SEM_NODE_WEIGHTS_REGISTER``).
2. *Fuel rebase.* One ratio per TYNDP euro: TYNDP's new-CCGT price plateau
   (``tyndp_new_ccgt_eur``, EUR 82/MWh, the most efficient gas tier on the DE/NL/BE/FR plateau of
   82-89) maps to a foreign new CCGT's short-run cost on CP30's basis (``foreign_new_ccgt_gbp``):
   the GB model's most efficient CCGT (103.0) less the UK-EU carbon gap (CP30 FES 2024 carbon, UK
   129 vs EU 122 GBP/t, x 0.34 t/MWh_e), 100.6. Every hour is scaled by the same ratio, so zero-price
   hours stay zero and hydro water values, which track gas, move with it. Declared simplification:
   hours set by coal or nuclear are scaled too (both are a small share of 2030 price-setting).
3. *Calendar and weather.* TYNDP hour k -> 1 January of the weather year + k hours
   (``tyndp_prices.tyndp2024_to_calendar``, 31 December from 24 December), then the GB-wind term of
   climate year 2009 is swapped for the weather year's (``price_shape.condition_prices_on_wind``),
   with TYNDP's UK00 offshore generation plus UK00 dumped energy as the 2009 regressor. The result is
   held at the source floor (0: TYNDP has no negative bids) and returned to each column's mean by
   ratio, so the rebased annual level is kept exactly.

``add_to_network`` uses the file at its own level (no CP.29 re-centring) when
``interconnectors.pricing.foreign_price_source`` is ``"cp30_basis"``, as it does for the 2050 TYNDP
route; the trading hurdle, export-bid floor and IC1 volume tranches are unchanged.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from scripts.interconnectors.price_shape import condition_prices_on_wind, wind_index
from scripts.interconnectors.tyndp_prices import (
    SEM_NODE_WEIGHTS_REGISTER,
    ZONE_WEIGHTS_TYNDP2024,
    sem_node_weights_from_composition,
    tyndp2024_to_calendar,
    with_sem_weights,
    zone_prices,
)

_LOG = logging.getLogger(__name__)

#: Must match config/defaults/ interconnectors.pricing.cp30_basis.
DEFAULT_TYNDP_NEW_CCGT_EUR = 82.0
DEFAULT_FOREIGN_NEW_CCGT_GBP = 100.62
DEFAULT_METHOD = "rebase_cy2009"
#: The year of GB's cable set the SEM weights describe: TYNDP 2024 National Trends 2030 is a 2030 market.
CP30_BASIS_CABLE_YEAR = 2030


def sem_node_weights(composition_file=None, year: int = CP30_BASIS_CABLE_YEAR) -> dict:
    """GB cable MW per TYNDP SEM node: CP30's cable set in ``year`` if a composition file is given, else
    today's register cables (EWIC + Greenlink 1,009 MW on IE00, Moyle 500 MW on UKNI)."""
    if composition_file is None:
        return dict(SEM_NODE_WEIGHTS_REGISTER)
    return sem_node_weights_from_composition(composition_file, year)


def rebase_ratio(tyndp_new_ccgt_eur: float, foreign_new_ccgt_gbp: float) -> float:
    """GBP on CP30's fuel basis per TYNDP euro."""
    if tyndp_new_ccgt_eur <= 0 or foreign_new_ccgt_gbp <= 0:
        raise ValueError("cp30_basis: both CCGT reference prices must be positive")
    return float(foreign_new_ccgt_gbp) / float(tyndp_new_ccgt_eur)


def tyndp_wind_regressor(hourly: pd.DataFrame) -> pd.Series:
    """TYNDP UK00 offshore wind (generation + UK00 dumped energy), the climate year's GB-wind index."""
    for column in ("UK00_wind_offshore_mw", "UK00_dumped_mw"):
        if column not in hourly:
            raise ValueError(f"cp30_basis: TYNDP extract has no {column!r} column")
    off = pd.to_numeric(hourly["UK00_wind_offshore_mw"], errors="coerce").fillna(0.0)
    dumped = pd.to_numeric(hourly["UK00_dumped_mw"], errors="coerce").fillna(0.0)
    return off + dumped


def build_cp30_basis_prices(tyndp_file, wind_weather_file, out_file, audit_file,
                            tyndp_new_ccgt_eur: float = DEFAULT_TYNDP_NEW_CCGT_EUR,
                            foreign_new_ccgt_gbp: float = DEFAULT_FOREIGN_NEW_CCGT_GBP,
                            calendar_year: int = 2013, composition_file=None, logger=_LOG) -> pd.DataFrame:
    """Write the rebased, weather-conditioned hourly price per counterparty column and an audit.

    ``composition_file``: the CP30 cable composition; when given, the SEM column weights IE00 and UKNI by
    CP30's 2030 cables (:func:`sem_node_weights`), else by today's register cables.
    """
    hourly = pd.read_csv(tyndp_file)
    ratio = rebase_ratio(tyndp_new_ccgt_eur, foreign_new_ccgt_gbp)
    sem_weights = sem_node_weights(composition_file)
    zones = with_sem_weights(ZONE_WEIGHTS_TYNDP2024, sem_weights)
    eur = zone_prices(hourly, zones)
    gbp = tyndp2024_to_calendar(eur * ratio, calendar_year)
    z_shape = wind_index(tyndp2024_to_calendar(tyndp_wind_regressor(hourly), calendar_year).to_frame("uk00"))
    z_weather = wind_index(pd.read_csv(wind_weather_file, index_col=0, parse_dates=True))
    conditioned, betas = condition_prices_on_wind(gbp, z_shape, z_weather)
    floor = gbp.min()
    conditioned = conditioned.clip(lower=floor, axis=1)
    for column in conditioned.columns:
        mean = float(conditioned[column].mean())
        if abs(mean) > 1e-9:
            conditioned[column] *= float(gbp[column].mean()) / mean
    audit = {
        "item": "F13",
        "source": "TYNDP 2024 NT2030 CY2009 PLEXOS standard output (ENTSO-E/ENTSOG, CC-BY 4.0)",
        "zones": {k: dict(v) for k, v in zones.items()},
        "sem_node_weights_mw": sem_weights,
        "sem_weights_source": ("cp30_composition" if composition_file is not None else "register_2025"),
        "tyndp_new_ccgt_eur": float(tyndp_new_ccgt_eur),
        "foreign_new_ccgt_gbp": float(foreign_new_ccgt_gbp),
        "gbp_per_tyndp_eur": round(ratio, 6),
        "tyndp_mean_eur": eur.mean().round(3).to_dict(),
        "mean_gbp": conditioned.mean().round(3).to_dict(),
        "p10_gbp": conditioned.quantile(0.1).round(3).to_dict(),
        "p90_gbp": conditioned.quantile(0.9).round(3).to_dict(),
        "share_at_or_below_1": (conditioned <= 1.0).mean().round(4).to_dict(),
        "wind_slope_gbp_per_sd": betas.round(4).to_dict(),
        "calendar_year": calendar_year,
        "hours": int(len(conditioned)),
    }
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    out = conditioned.copy()
    out.index.name = "datetime"
    out.to_csv(out_file)
    Path(audit_file).write_text(json.dumps(audit, indent=2))
    for column in out.columns:
        logger.info("%s: mean %.2f GBP/MWh (TYNDP %.2f EUR), wind slope %.2f", column,
                    audit["mean_gbp"][column], audit["tyndp_mean_eur"][column], betas[column])
    return out


# =============================================================================
# Method "refit_2013" (7 Oct 2026, user decision): TYNDP prices refitted to the weather year
# =============================================================================
# The CY2009 rebase above keeps TYNDP's 2009 weather and only swaps a linear GB-wind term, so continental
# prices barely co-move with GB's 2013 shortages (correlation -0.02 to -0.15 against -0.14 to -0.27 in real
# 2024 data). The refit learns TYNDP's own price response to residual load in CY2009 (a gradient-boosted
# regression per output zone on own-zone and regional residual load, their 24 h and 168 h means, and the
# calendar), rebuilds those features on the weather year's wind and solar (renewables.ninja 2013 shapes on
# TYNDP's 2009 annual energy, TYNDP's own 2030 demand laid on the calendar), predicts the weather year's
# hourly price, then rebases it to CP30's fuel with the same ratio. Out-of-month skill R2 0.81-0.89 on the
# continent, 0.79 SEM, 0.40 Norway (hydro). The SEM column defaults to 2024 ENTSO-E (+7.5, as IC1) because
# TYNDP's SEM price is set against TYNDP's own cheap GB zone (declared).

#: Zones of the 2030 European model (TYNDP 2024 nodes summed), as in the F13 Option B dispatch.
REFIT_ZONES = {
    "AT": ["AT00"], "BE": ["BE00", "BEOF"], "CZ": ["CZ00"], "DKW": ["DKW1", "DKNS"], "DKE": ["DKE1", "DKBH", "DKKF"],
    "FI": ["FI00"], "FR": ["FR00"], "DE": ["DE00", "DEKF"],
    "IT": ["ITCA", "ITCN", "ITCS", "ITN1", "ITS1", "ITSA", "ITSI"],
    "NL": ["NL00", "NLLL", "NL60", "NL6H", "NLA0", "NLBH"], "NO_S": ["NOS0"], "NO_MN": ["NOM1", "NON1"],
    "PL": ["PL00"], "PT": ["PT00"], "SEM": ["IE00", "UKNI"], "SI": ["SI00"], "ES": ["ES00"],
    "SE1": ["SE01"], "SE2": ["SE02"], "SE3": ["SE03"], "SE4": ["SE04"], "CH": ["CH00"],
}
#: Output columns (the IC1 counterparty columns) -> TYNDP price nodes (weighted by GB cable capacity).
PRICE_NODES = {"France": ["FR00"], "Belgium": ["BE00"], "Netherlands": ["NL00"], "Germany": ["DE00"],
               "Denmark": ["DKW1"], "Norway": ["NOS0"], "Ireland": ["IE00", "UKNI"]}
OUTPUT_ZONE = {"France": "FR", "Belgium": "BE", "Netherlands": "NL", "Germany": "DE", "Denmark": "DKW",
               "Norway": "NO_S", "Ireland": "SEM"}
#: Fallback SEM node weights (today's register cables); the refit takes CP30's 2030 cables when the rule
#: passes the composition file (it does when ``ireland_source`` is ``refit``; under ``entsoe_2024`` the
#: refitted SEM column is replaced, so its weights reach only the audit's out-of-month skill).
NODE_WEIGHTS = dict(SEM_NODE_WEIGHTS_REGISTER)
REFIT_REGIONS = {
    "core": ["FR", "BE", "NL", "DE", "DKW", "DKE", "AT", "CH", "CZ", "PL"],
    "nordic": ["NO_S", "NO_MN", "SE1", "SE2", "SE3", "SE4", "FI", "DKE"],
    "south": ["IT", "ES", "PT", "SI"],
}
REFIT_TECHS = ("onwind", "offwind", "solar")
#: Must match config/defaults/ interconnectors.pricing.cp30_basis (refit keys).
DEFAULT_TARGET_PRICE_CAP_EUR = 300.0
DEFAULT_IRELAND_SOURCE = "entsoe_2024"
DEFAULT_IRELAND_OFFSET_GBP = 7.5


def refit_features(demand: pd.DataFrame, potential: pd.DataFrame, index) -> pd.DataFrame:
    """Residual-load and calendar features, one row per hour (``index`` gives the calendar)."""
    import numpy as np
    zones = list(REFIT_ZONES)
    vre = pd.DataFrame({z: sum((potential[f"{z}|{t}"] for t in REFIT_TECHS if f"{z}|{t}" in potential),
                               pd.Series(0.0, index=potential.index)) for z in zones})
    rl = demand[zones] - vre[zones]
    mean_d = demand[zones].mean()
    f = pd.DataFrame(index=range(len(demand)))
    for z in OUTPUT_ZONE.values():
        f[f"rl_{z}"] = (rl[z] / mean_d[z]).to_numpy()
    for name, members in REFIT_REGIONS.items():
        r = rl[members].sum(axis=1) / mean_d[members].sum()
        f[f"rl_{name}"] = r.to_numpy()
        f[f"rl_{name}_24"] = r.rolling(24, min_periods=1).mean().to_numpy()
        f[f"rl_{name}_168"] = r.rolling(168, min_periods=1).mean().to_numpy()
    idx = pd.DatetimeIndex(index)
    f["doy_s"], f["doy_c"] = np.sin(2 * np.pi * idx.dayofyear / 365.0), np.cos(2 * np.pi * idx.dayofyear / 365.0)
    f["hour_s"], f["hour_c"] = np.sin(2 * np.pi * idx.hour / 24.0), np.cos(2 * np.pi * idx.hour / 24.0)
    f["weekend"] = (idx.dayofweek >= 5).astype(float)
    return f


def _regressor():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0)


def build_cp30_refit_prices(zonal_file, cf_file, out_file, audit_file, ireland_shape_file=None,
                            tyndp_new_ccgt_eur: float = DEFAULT_TYNDP_NEW_CCGT_EUR,
                            foreign_new_ccgt_gbp: float = DEFAULT_FOREIGN_NEW_CCGT_GBP,
                            target_price_cap_eur: float = DEFAULT_TARGET_PRICE_CAP_EUR,
                            ireland_source: str = DEFAULT_IRELAND_SOURCE,
                            ireland_offset_gbp: float = DEFAULT_IRELAND_OFFSET_GBP,
                            calendar_year: int = 2013, cv_folds: int = 12, composition_file=None,
                            logger=_LOG) -> pd.DataFrame:
    """Refit TYNDP's CY2009 prices to the weather year (see the block comment); write file and audit."""
    import numpy as np
    from sklearn.model_selection import GroupKFold

    ratio = rebase_ratio(tyndp_new_ccgt_eur, foreign_new_ccgt_gbp)
    zonal = pd.read_csv(zonal_file, index_col=0)
    demand09 = pd.DataFrame({z: zonal[f"{z}|demand"].to_numpy() for z in REFIT_ZONES})
    pot09 = pd.DataFrame({c: zonal[c].to_numpy() for c in zonal.columns
                          if "|" in c and c.split("|")[1] in REFIT_TECHS})
    idx09 = pd.date_range(f"{calendar_year}-01-01", periods=len(zonal), freq="h")
    f09 = refit_features(demand09, pot09, idx09)

    cf = pd.read_csv(cf_file, index_col=0, parse_dates=True)
    idx = pd.date_range(f"{calendar_year}-01-01", periods=8760, freq="h")
    if len(cf) != len(idx):
        raise ValueError(f"{cf_file}: expected 8,760 hours, got {len(cf)}")
    pot = pd.DataFrame(index=range(8760))
    for c in pot09.columns:
        mean09 = float(pot09[c].mean())
        if c in cf.columns and float(cf[c].mean()) > 0:
            pot[c] = cf[c].to_numpy() / float(cf[c].mean()) * mean09      # weather-year timing, TYNDP energy
        else:
            pot[c] = mean09
    demand = tyndp2024_to_calendar(demand09, calendar_year).reset_index(drop=True)
    f = refit_features(demand, pot, idx)

    out = {}
    audit = {"item": "F13 refit_2013", "method": "refit_2013",
             "gbp_per_tyndp_eur": round(ratio, 6), "target_price_cap_eur": target_price_cap_eur,
             "cv_out_of_month": {}}
    X09, groups = f09.to_numpy(), idx09.month
    node_weights = dict(NODE_WEIGHTS, **sem_node_weights(composition_file))
    for country, nodes in PRICE_NODES.items():
        w = [node_weights.get(n, 1.0) for n in nodes]
        y = sum(zonal[f"price|{n}"].to_numpy() * wi for n, wi in zip(nodes, w)) / sum(w)
        y = np.minimum(y, target_price_cap_eur)
        if cv_folds:   # out-of-month skill (month blocks), reported in the audit
            pred = np.empty_like(y)
            for tr, te in GroupKFold(n_splits=cv_folds).split(X09, y, groups):
                pred[te] = _regressor().fit(X09[tr], y[tr]).predict(X09[te])
            audit["cv_out_of_month"][country] = {
                "r2": round(float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)), 3),
                "mae_eur": round(float(np.mean(np.abs(y - pred))), 2)}
        out[country] = np.clip(_regressor().fit(X09, y).predict(f.to_numpy()), 0.0, None) * ratio
    frame = pd.DataFrame(out, index=idx)
    if ireland_source == "entsoe_2024":
        if ireland_shape_file is None:
            raise ValueError("refit_2013: ireland_source entsoe_2024 needs the IC1 2024 shape file")
        from scripts.interconnectors.price_shape import align_by_calendar
        shape = pd.read_csv(ireland_shape_file, index_col=0)
        shape.index = pd.DatetimeIndex(pd.to_datetime(shape.index, utc=True)).tz_convert(None)
        frame["Ireland"] = align_by_calendar(shape["Ireland"], idx).to_numpy() + float(ireland_offset_gbp)
    elif ireland_source != "refit":
        raise ValueError(f"refit_2013: unknown ireland_source {ireland_source!r} (entsoe_2024 | refit)")
    audit.update({"ireland_source": ireland_source, "ireland_offset_gbp": ireland_offset_gbp,
                  "sem_node_weights_mw": {n: node_weights[n] for n in PRICE_NODES["Ireland"]},
                  "sem_weights_source": ("cp30_composition" if composition_file is not None else "register_2025"),
                  "mean_gbp": frame.mean().round(3).to_dict(),
                  "share_at_or_below_1": (frame <= 1.0).mean().round(4).to_dict(),
                  "calendar_year": calendar_year, "hours": int(len(frame))})
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    frame.index.name = "datetime"
    frame.to_csv(out_file)
    Path(audit_file).write_text(json.dumps(audit, indent=2))
    for c in frame.columns:
        logger.info("%s: mean %.2f GBP/MWh (out-of-month R2 %s)", c, audit["mean_gbp"][c],
                    audit["cv_out_of_month"].get(c, {}).get("r2"))
    return frame
