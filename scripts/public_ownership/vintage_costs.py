"""Loader for the 2025-2050 vintage technology cost table (FES 2050 plan, task Y2).

The table (``data/public_ownership/financing/technology_costs_vintage.csv``) is
built by ``build_vintage_technology_costs.py``.  It is on the full DESNZ basis
(decision 26): its 2030 row minus the ``basis_adjustment_*`` columns reproduces
``technology_costs.csv``, which stays the 2030 table.  Nothing here is read by
any 2030 rule.

Typical use::

    from scripts.public_ownership.vintage_costs import costs_for, vintage_cost_table
    row = costs_for("Offshore wind", 2042)
    table_2040 = vintage_cost_table(2040)   # technology_costs.csv-shaped, one build year

Two cautions for the finance layer (Y4):

* ``fixed_om_gbp_per_kw_year`` INCLUDES DESNZ insurance (decision 26), reported in
  ``insurance_gbp_per_kw_year``.  The 2030 finance layer adds its own insurance
  line (item D4(a), 0.4% of capex a year); using both double-counts.  Use
  :func:`vintage_cost_table` with ``insurance="exclude"`` when the D4(a) line is on.
* Connection and use-of-system charges are not in the table: TNUoS is its own
  line (item D4(a)), as in 2030.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINANCING = ROOT / "data" / "public_ownership" / "financing"
DEFAULT_VINTAGE_COSTS = FINANCING / "technology_costs_vintage.csv"
DEFAULT_HURDLE_RATES = FINANCING / "fes2025_hurdle_rates.csv"
DEFAULT_TECHNOLOGY_COSTS = FINANCING / "technology_costs.csv"

FIRST_BUILD_YEAR = 2025
LAST_BUILD_YEAR = 2050
ANCHOR_YEAR = 2030

#: The columns of ``technology_costs.csv`` (the finance layer's cost schema).
TECHNOLOGY_COST_COLUMNS = (
    "technology", "capex_power_gbp_per_kw", "capex_energy_gbp_per_kwh", "duration_hours",
    "fixed_om_gbp_per_kw_year", "variable_om_gbp_per_mwh", "economic_life_years", "price_year",
    "real_or_nominal", "source", "source_page", "status",
)
REQUIRED_COLUMNS = TECHNOLOGY_COST_COLUMNS + (
    "build_year", "insurance_gbp_per_kw_year", "fixed_om_gbp_per_kwh_year",
    "basis_adjustment_capex_power_gbp_per_kw", "basis_adjustment_capex_energy_gbp_per_kwh",
    "basis_adjustment_fixed_om_gbp_per_kw_year", "basis_adjustment_variable_om_gbp_per_mwh",
    "learning_capex_gbp_per_kw", "wacc_technology", "year_basis",
)
#: Value column -> its declared basis adjustment (vintage basis minus technology_costs.csv basis).
ADJUSTED_COLUMNS = {
    "capex_power_gbp_per_kw": "basis_adjustment_capex_power_gbp_per_kw",
    "capex_energy_gbp_per_kwh": "basis_adjustment_capex_energy_gbp_per_kwh",
    "fixed_om_gbp_per_kw_year": "basis_adjustment_fixed_om_gbp_per_kw_year",
    "variable_om_gbp_per_mwh": "basis_adjustment_variable_om_gbp_per_mwh",
}
PROVENANCE_COLUMNS = ("source", "source_page", "status")
INSURANCE_MODES = ("include", "exclude")

_CACHE: dict = {}


def _key(value: object) -> str:
    """Technology names compared as the finance layer compares them (dashes and spaces normalised)."""
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"[‐-―−]", "-", text)
    return re.sub(r"\s+", " ", text).casefold()


def load_vintage_costs(path: str | Path | None = None, *, use_cache: bool = True) -> pd.DataFrame:
    """Read and validate the vintage table.

    Args:
        path: The CSV; defaults to ``technology_costs_vintage.csv``.
        use_cache: Reuse an earlier read of the same path.

    Returns:
        The table, one row per technology and build year.

    Raises:
        ValueError: A required column is missing, a (technology, build year) repeats,
            a technology lacks a build year in 2025-2050, a row lacks provenance, or a
            row is not in real 2024 prices.
    """
    path = Path(path) if path is not None else DEFAULT_VINTAGE_COSTS
    cache_key = ("vintage", str(path.resolve()))
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key].copy()
    table = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"vintage cost table lacks columns: {sorted(missing)}")
    table["build_year"] = table["build_year"].astype(int)
    if table.duplicated(["technology", "build_year"]).any():
        raise ValueError("vintage cost table repeats a (technology, build_year)")
    expected = set(range(FIRST_BUILD_YEAR, LAST_BUILD_YEAR + 1))
    for technology, years in table.groupby("technology")["build_year"]:
        if set(years) != expected:
            raise ValueError(f"{technology}: build years are not exactly {FIRST_BUILD_YEAR}-{LAST_BUILD_YEAR}")
    for column in PROVENANCE_COLUMNS:
        blank = table[column].isna() | table[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(f"rows without {column}: {sorted(set(table.loc[blank, 'technology']))}")
    if not (table["price_year"].astype(int).eq(2024).all() and table["real_or_nominal"].eq("real").all()):
        raise ValueError("vintage cost table must be real 2024 GBP throughout")
    table["_key"] = table["technology"].map(_key)
    _CACHE[cache_key] = table
    return table.copy()


def technologies(table: pd.DataFrame | None = None) -> list[str]:
    """The technologies the table covers."""
    table = load_vintage_costs() if table is None else table
    return sorted(table["technology"].unique())


def _row(technology: str, build_year: int, table: pd.DataFrame) -> tuple[pd.Series, bool]:
    if "_key" not in table.columns:
        table = table.assign(_key=table["technology"].map(_key))
    rows = table[table["_key"].eq(_key(technology))]
    if rows.empty:
        raise KeyError(f"no vintage costs for technology {technology!r}; known: {technologies(table)}")
    year = int(build_year)
    if year < FIRST_BUILD_YEAR:
        raise ValueError(f"build year {year} is before {FIRST_BUILD_YEAR}, the table's first vintage")
    held = year > LAST_BUILD_YEAR
    lookup = min(year, LAST_BUILD_YEAR)  # decision 11: flat after the last source year
    return rows.loc[rows["build_year"].eq(lookup)].iloc[0], held


def costs_for(technology: str, build_year: int, *, learning: bool = False,
              table: pd.DataFrame | None = None) -> dict:
    """The cost row of one technology and build year.

    Args:
        technology: A vintage-table technology name (compared ignoring dash style and case).
        build_year: Commissioning year; later than 2050 holds the 2050 value (decision 11).
        learning: Take capex from the CCC learning-rate sensitivity column.
        table: A table already loaded with :func:`load_vintage_costs`.

    Returns:
        The row as a dict, with ``build_year`` the requested year, ``held_flat`` True when
        the year is beyond the table, and ``capex_case`` "central" or "learning".

    Raises:
        KeyError: Unknown technology.
        ValueError: Build year before 2025.
    """
    table = load_vintage_costs() if table is None else table
    row, held = _row(technology, build_year, table)
    out = row.drop(labels=["_key"], errors="ignore").to_dict()
    out["build_year"] = int(build_year)
    out["held_flat"] = bool(held)
    out["capex_case"] = "learning" if learning else "central"
    if learning:
        out["capex_power_gbp_per_kw"] = float(row["learning_capex_gbp_per_kw"])
    return out


def vintage_cost_table(build_year: int, *, learning: bool = False, insurance: str = "include",
                       table: pd.DataFrame | None = None) -> pd.DataFrame:
    """A ``technology_costs.csv``-shaped table for one build year.

    Args:
        build_year: Commissioning year (later than 2050 holds 2050).
        learning: Take capex from the learning-rate sensitivity column.
        insurance: "include" keeps DESNZ insurance in fixed O&M (decision 26); "exclude"
            removes it, for use beside the finance layer's own insurance line (item D4(a)).
        table: A table already loaded with :func:`load_vintage_costs`.

    Returns:
        One row per technology with the ``technology_costs.csv`` columns first and the
        vintage columns after them.
    """
    if insurance not in INSURANCE_MODES:
        raise ValueError(f"insurance must be one of {INSURANCE_MODES}, got {insurance!r}")
    table = load_vintage_costs() if table is None else table
    year = int(build_year)
    if year < FIRST_BUILD_YEAR:
        raise ValueError(f"build year {year} is before {FIRST_BUILD_YEAR}, the table's first vintage")
    out = table[table["build_year"].eq(min(year, LAST_BUILD_YEAR))].drop(columns=["_key"], errors="ignore").copy()
    out["build_year"] = year
    out["held_flat"] = year > LAST_BUILD_YEAR
    if learning:
        out["capex_power_gbp_per_kw"] = out["learning_capex_gbp_per_kw"]
    if insurance == "exclude":
        out["fixed_om_gbp_per_kw_year"] = out["fixed_om_gbp_per_kw_year"] - out["insurance_gbp_per_kw_year"]
    ordered = list(TECHNOLOGY_COST_COLUMNS) + [c for c in out.columns if c not in TECHNOLOGY_COST_COLUMNS]
    return out[ordered].reset_index(drop=True)


def technology_costs_basis(table: pd.DataFrame | None = None, build_year: int = ANCHOR_YEAR) -> pd.DataFrame:
    """The vintage rows of one build year with the declared basis adjustment taken off.

    For 2030 and the technologies in ``technology_costs.csv`` this reproduces that file
    (the Y2 parity test); for other years it is the 2030 table's basis carried forward on
    the same source ratios.

    Args:
        table: A loaded vintage table.
        build_year: The build year.

    Returns:
        ``technology_costs.csv``-shaped rows.
    """
    rows = vintage_cost_table(build_year, table=table)
    for value, adjustment in ADJUSTED_COLUMNS.items():
        rows[value] = rows[value] - rows[adjustment]
    return rows[list(TECHNOLOGY_COST_COLUMNS)]


def basis_adjustments(table: pd.DataFrame | None = None, build_year: int = ANCHOR_YEAR) -> pd.DataFrame:
    """The declared adjustment by technology (vintage basis minus ``technology_costs.csv`` basis).

    Args:
        table: A loaded vintage table.
        build_year: The build year.

    Returns:
        technology, the four adjustment columns and ``basis_adjustment_note``.
    """
    rows = vintage_cost_table(build_year, table=table)
    return rows[["technology", *ADJUSTED_COLUMNS.values(), "basis_adjustment_note"]]


def load_hurdle_rates(path: str | Path | None = None) -> pd.DataFrame:
    """Read the FES 2025 / DESNZ hurdle-rate table (wide, one ``hurdle_<year>`` column per year).

    Args:
        path: The CSV; defaults to ``fes2025_hurdle_rates.csv``.

    Returns:
        The table.
    """
    path = Path(path) if path is not None else DEFAULT_HURDLE_RATES
    table = pd.read_csv(path)
    for column in ("source", "source_page", "status"):
        if table[column].isna().any():
            raise ValueError(f"hurdle-rate rows without {column}")
    return table


def hurdle_rate(technology: str, build_year: int, *, source: str = "FES", case: str = "central",
                table: pd.DataFrame | None = None) -> float:
    """A pre-tax real hurdle rate for one technology and build year.

    DESNZ publishes commissioning years 2030, 2035, ..., 2050 only: a year between two
    takes the earlier published year (Annex A's FOAK/NOAK steps are at 2035); a year
    before 2030 takes 2030.  FES is annual.  After 2050 both hold 2050.

    Args:
        technology: A vintage-table technology name.
        build_year: Commissioning year.
        source: "FES" (FES 2025 E.03 / ER.01) or "DESNZ" (Annex A).
        case: "central", or "low"/"high" for DESNZ.
        table: A table already loaded with :func:`load_hurdle_rates`.

    Returns:
        The rate as a fraction.

    Raises:
        KeyError: No such technology, source and case.
    """
    table = load_hurdle_rates() if table is None else table
    source_mask = table["source"].str.startswith("NESO FES" if source.upper() == "FES" else "DESNZ")
    rows = table[source_mask & table["technology"].map(_key).eq(_key(technology)) & table["case"].eq(case)]
    if rows.empty:
        raise KeyError(f"no {source} {case} hurdle rate for {technology!r}")
    row = rows.iloc[0]
    year = min(max(int(build_year), FIRST_BUILD_YEAR), LAST_BUILD_YEAR)
    published = [y for y in range(FIRST_BUILD_YEAR, LAST_BUILD_YEAR + 1) if pd.notna(row.get(f"hurdle_{y}"))]
    earlier = [y for y in published if y <= year]
    pick = max(earlier) if earlier else min(published)
    return float(row[f"hurdle_{pick}"])


def is_finite_table(table: pd.DataFrame) -> bool:
    """True when every numeric cost column is finite (blank duration aside)."""
    numeric = table.select_dtypes(include=[np.number]).drop(columns=["duration_hours", "learning_rate_per_doubling"], errors="ignore")
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())
