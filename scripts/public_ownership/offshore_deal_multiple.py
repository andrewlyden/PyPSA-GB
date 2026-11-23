"""Legacy offshore wind bought at observed deal multiples (lever ``legacy_offshore_price``, user decision 24 Oct 2026).

The central P3 purchase prices every legacy row at its owners' value from the model's
cash flows (``legacy_valuation``): legacy offshore wind comes to about 2.31 GBP m/MW on
the CENTRAL network (14.7 GW), against recent UK operating offshore wind stake sales at
about 4-8 GBP m/MW.  The named case
``public_ownership@fully_nationalised_market_value_offshore_deal_multiples`` (lever value
``transaction_multiple``) prices legacy offshore wind at the deals instead; every other row
keeps its owners' value.

* **The deals.** ``data/public_ownership/financing/legacy_offshore_transaction_multiples.csv``:
  UK operating offshore wind stake sales 2020-2025, each as GBP per MW of the
  100%-equivalent capacity the stake buys (enterprise value: equity plus the stake's share of
  project debt, or a debt-free consideration; net of transmission assets sold to an OFTO
  where the deal says so), rebased to 2024 money (the register's) by ONS CPIH annual averages
  (series L522).  ``in_central`` marks the deals the multiple is drawn from (a deal with no
  disclosed consideration is reported, not used).
* **The multiple.** One central multiple, not by cohort (six deals cannot support a split):
  ``statistic`` of the ``in_central`` deals, ``median`` by default; ``low`` and ``high``
  (their minimum and maximum) are reported beside it.
* **The price.** Each legacy row of the carrier in the valued group:
  ``multiple x 1e6 x capacity_mw`` (the row's own MW, a cohort row its legacy share), in place
  of its owners' value, before any control premium.  The streams the state takes over are
  unchanged: the deals price the same farms on their buyers' forecasts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Lever values (scripts/public_ownership/levers.py declares the lever).
OWNERS_VALUE, TRANSACTION_MULTIPLE = "owners_value", "transaction_multiple"
#: ``public_ownership.lever_parameters`` key; calculate_financing_scenarios.LEVER_PARAMETER_DEFAULTS holds these.
PARAMETER_KEY = "legacy_offshore_transaction_multiple"
DEFAULT_PARAMETERS: dict = {
    "file": "data/public_ownership/financing/legacy_offshore_transaction_multiples.csv",
    "carrier": "wind_offshore",
    "statistic": "median",
}
STATISTICS = ("median", "mean", "low", "high")
#: The acquisition basis a priced row records.
BASIS = "offshore_transaction_multiple"
REQUIRED_COLUMNS = {"deal", "year", "gbp_m_per_mw_2024", "in_central", "source"}


def parameters(block: Mapping | None) -> dict:
    """``lever_parameters.legacy_offshore_transaction_multiple`` over the defaults, checked."""
    block = dict(block or {})
    unknown = sorted(set(block) - set(DEFAULT_PARAMETERS))
    if unknown:
        raise ValueError(f"public_ownership.lever_parameters.{PARAMETER_KEY} takes {sorted(DEFAULT_PARAMETERS)}; "
                         f"unknown {unknown}")
    out = {**DEFAULT_PARAMETERS, **block}
    if out["statistic"] not in STATISTICS:
        raise ValueError(f"public_ownership.lever_parameters.{PARAMETER_KEY}.statistic={out['statistic']!r}; "
                         f"supported: {list(STATISTICS)}")
    return {"file": str(out["file"]), "carrier": str(out["carrier"]), "statistic": str(out["statistic"])}


def _resolve(path) -> Path:
    path = Path(path)
    return path if path.exists() or path.is_absolute() else PROJECT_ROOT / path


def read_deals(source) -> pd.DataFrame:
    """The deals table, with ``in_central`` as a boolean; refuses a table without its columns or central deals."""
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(_resolve(source))
    if missing := REQUIRED_COLUMNS - set(frame.columns):
        raise ValueError(f"the offshore deals table lacks {sorted(missing)}")
    frame["in_central"] = frame["in_central"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    frame["gbp_m_per_mw_2024"] = pd.to_numeric(frame["gbp_m_per_mw_2024"], errors="raise")
    if not frame["in_central"].any():
        raise ValueError("the offshore deals table marks no deal in_central")
    if (frame["gbp_m_per_mw_2024"] <= 0).any():
        raise ValueError("every offshore deal needs a positive GBP m/MW")
    return frame


def multiple(params: Mapping | None, deals: pd.DataFrame | None = None) -> dict:
    """The central multiple (GBP m/MW, 2024 money) with its low, high, median, mean and the deals it is drawn from."""
    params = parameters(params)
    table = read_deals(params["file"] if deals is None else deals)
    used = table.loc[table["in_central"]]
    values = used["gbp_m_per_mw_2024"].to_numpy(dtype=float)
    stats = {"median": float(np.median(values)), "mean": float(np.mean(values)),
             "low": float(values.min()), "high": float(values.max())}
    return {"gbp_m_per_mw": stats[params["statistic"]], **{f"{k}_gbp_m_per_mw": v for k, v in stats.items()},
            "statistic": params["statistic"], "carrier": params["carrier"], "file": params["file"],
            "deals_used": [f"{d} ({int(y)})" for d, y in zip(used["deal"], used["year"])],
            "deals_reported_only": [f"{d} ({int(y)})" for d, y in
                                    zip(table.loc[~table["in_central"], "deal"], table.loc[~table["in_central"], "year"])],
            "price_year": 2024}


def price_rows(capacity_mw, gbp_m_per_mw: float) -> np.ndarray:
    """``multiple x 1e6 x capacity`` per row (GBP)."""
    return np.asarray(capacity_mw, dtype=float) * float(gbp_m_per_mw) * 1.0e6
