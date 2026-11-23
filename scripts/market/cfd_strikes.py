"""CfD strikes and live contracts in a model year (FES 2050 plan, task Y11; decision 6; year-generic stage 5).

Nothing here is read by a 2030 rule. The 2030 CfD line settles published round
strikes (``cfd_strike_table.py``); a long-range model year (2040, 2050, ...) needs two
things that table cannot give. The model year is always the caller's (the scenario's
``modelled_year``); ``*_2050`` names are kept as 2050 aliases for older callers.

* **Post-AR7 strikes (decision 6).** Every private low-carbon plant built after the
  last held round holds a CfD whose strike is the build year's levelised cost at the
  private hurdle rate:

      strike = (capex x CRF(r, n) + fixed O&M + use of system) x 1000 / (8760 x LF)
               + variable O&M + fuel

  - capex, fixed and variable O&M and life ``n`` are Y2's vintage row
    (``technology_costs_vintage.csv``, real 2024 GBP, full DESNZ basis);
  - ``r`` is the workbook's private rate for the row's ``wacc_technology`` (snapshot
    ``wacc_by_basis.notional_gearing_shielded.private``, the register's own rate),
    held flat by build year (decision 19). ``rate_path="neso_wind_ramp"`` adds the
    FES 2025 hurdle change since 2030 (offshore -2.0, onshore -1.0, floating -2.0
    points by 2035, a linear ramp over 2031-2035), the declared sensitivity;
  - ``LF`` is the expected net load factor a bidder prices (DESNZ Annex A,
    ``cfd_strike_inputs_2050.csv``), not the solved year's output: the strike is
    fixed at award. :func:`levelised_strike` takes ``load_factor=`` to price on a
    solved year's available capacity factor instead (the sensitivity hook);
  - **insurance** is counted once, on the register's basis (task Y14-prep, 1 Oct 2026):
    the vintage table is read with ``insurance="exclude"`` (DESNZ insurance taken out of
    fixed O&M, decision 26) and the finance layer's item D4(a) insurance is added back
    (:func:`finance_insurance_gbp_per_kw_year`: a sourced GBP/kW/yr where one exists,
    offshore wind 11.22, else 0.4% of capex a year; none for the technologies whose cost
    source does not separate it), exactly as the vintaged register costs the same row
    (``calculate_financing_scenarios.VINTAGE_INSURANCE_RULE`` and ``generator_charges``).
    So an at-cost strike equals the register's required revenue per MWh.
    ``insurance="include"`` (DESNZ insurance, no D4(a) line) is kept as a sensitivity;
    both at once is refused;
  - **use of system** (TNUoS and connection) is Annex A's charge, because the
    generator pays it out of the strike. The register books TNUoS on its own D4(a)
    line; the strike's value is declared, not the register's zonal tariff;
  - ``amortisation="economic_life"`` (central) levelises over the plant's life: the
    CfD pays 20 years and the merchant tail is taken to be worth the same levelised
    cost. ``"contract_term"`` recovers capital in the 20 years (no tail value), the
    upper bound. The 20-year term decides whether a contract is live in the model year;
  - **AR7 offshore calibration**: the offshore wind capex is scaled by one factor
    so the 2030 build year prices at AR7's cleared 91.20, as the 2030 auction
    calibration trusts the auction (``private_cost_calibration.csv``). The factor is
    solved on the central basis from the anchor in ``cfd_strike_cost_calibration.csv``
    (:func:`cost_calibration_factor`) and scales capex in every variant and build year.
    An anchor applies to the model years in its ``model_years`` column (``all`` for the
    offshore anchor; the AR7a onshore, solar and tidal anchors are ``2030`` only, for the
    2030 chain's modelled AR8 round, F18): a caller names its model year, and one that
    names none reads the ``all`` anchors only, so the 2050 strikes are unchanged.

* **Known contracts in the model year.** One table of every round
  (``cfd_contracts.csv``: AR1 to AR7a, the Investment Contracts, MGT, the Drax
  bridge, the declared Power BECCS CfD, HPC and Sizewell C's RAB), each with its
  delivery year, term and capacity; the live share is computed for the model year
  (:func:`known_contracts`). AR7 and AR7a are unit by unit (phases split equally, one
  delivery year apart); the earlier rounds one row per round, technology and first
  delivery year; every start is spread evenly through its delivery year. In 2050 every
  round before AR7 has expired; in 2040 AR3 to AR6 are partly live.

Settlement (:func:`cfd_settlement`) takes a solved year's hourly dispatch and
reference price (the hook): intermittent contracts against the hourly day-ahead
price, baseload (nuclear, BECCS) against the seasonal demand-weighted baseload price,
as the 2030 bill does; post-AR7 contracts keep the AR4 negative-pricing rule.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from scripts.market.capacity_market_2050 import live_fraction
from scripts.market.cfd_strike_table import MODELLED_ROUND
from scripts.market.revenue_tracking import negative_price_eligibility
from scripts.public_ownership import vintage_costs as vc
from scripts.public_ownership.finance import capital_recovery_factor

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "data" / "market" / "support"
STRIKE_INPUTS = SUPPORT / "cfd_strike_inputs_2050.csv"
#: Every CfD round's contracts (and Sizewell C's RAB), with delivery years, terms and capacities.
CONTRACTS = SUPPORT / "cfd_contracts.csv"
#: Kept for older callers (``vintage_capital_2050.known_contract_terms``).
KNOWN_CONTRACTS = CONTRACTS
PUBLISHED_STRIKES = ROOT / "data" / "market" / "lccc" / "cfd_strike_prices_by_round.csv"
#: The AR7 offshore anchor the strike path's offshore capex is calibrated to (user, 27 Oct 2026).
STRIKE_COST_CALIBRATION = SUPPORT / "cfd_strike_cost_calibration.csv"
#: Switch for :data:`STRIKE_COST_CALIBRATION` when a caller passes nothing (a guard rebuild turns it off).
COST_CALIBRATION_ENABLED = True
DEFAULT_SNAPSHOT = ROOT / "resources" / "public_ownership" / "workbook_v5_assumptions.yaml"
#: The model year of the ``*_2050`` aliases only; every other function takes the year from its caller.
LEGACY_MODEL_YEAR = 2050
PRICE_YEAR = 2024
#: 2012 to 2024 prices: ONS CPI annual average 133.9 / 96.1, DESNZ's own conversion
#: (the 2030 bill's ``cfd_strike_indexation``; AR7 65.45 -> 91.20 reproduces it).
CPI_2012_TO_2024 = 133.9 / 96.1
#: The first build year on a modelled (post-AR7) contract: AR7's last phase-1 year is 2030/31.
FIRST_MODELLED_BUILD_YEAR = 2031
RATE_PATHS = ("flat", "neso_wind_ramp")
AMORTISATIONS = ("economic_life", "contract_term")
#: The technologies a private long-range fleet holds post-AR7 CfDs for (the report's main table).
MAIN_TECHNOLOGIES = ("Offshore wind", "Floating offshore wind", "Onshore wind", "Solar PV", "Tidal stream",
                     "Small modular reactors (SMRs)", "Hydropower")
#: Round labels of the modelled contracts: ``PAR7_<build year>`` (post-AR7, one per vintage).
MODELLED_ROUND_PREFIX = "PAR7_"
#: Carrier -> technology name the 2030 strike table uses (``cfd_strike_table.CARRIER_TO_TECHNOLOGY``).
TECHNOLOGY_TO_TABLE_NAME = {
    "Offshore wind": "offshore", "Floating offshore wind": "floating_offshore", "Onshore wind": "onshore",
    "Solar PV": "solar", "Tidal stream": "tidal", "Small modular reactors (SMRs)": "nuclear",
    "Large-scale nuclear": "nuclear", "Hydropower": "hydro", "Biomass - with CCUS - mature": "beccs",
}
#: The strike variants: decision 6's economic-life central and its sensitivities (``levelised_strike`` kwargs).
STRIKE_VARIANTS = {"central": {}, "neso_wind_ramp": {"rate_path": "neso_wind_ramp"}, "learning": {"learning": True},
                   "contract_term": {"amortisation": "contract_term"}}
#: Negative-pricing terms a contract row may carry, and the round ``revenue_tracking`` reads each as.
NEGATIVE_PRICE_RULES = {"ar4_plus": "AR4", "six_hour": "AR3", "none": "HPC"}
#: Published rounds whose contracts the table lists another way (HPC by unit) or that are not contracts.
UNCHECKED_PUBLISHED_ROUNDS = ("HPC", MODELLED_ROUND)
#: The insurance rule strikes and DPA rates are priced on: the vintaged register's
#: (``calculate_financing_scenarios.VINTAGE_INSURANCE_RULE``; a test keeps the two equal).
INSURANCE_RULE = "exclude"
#: Where the finance layer's D4(a) insurance settings live (read only): the 2050 study's
#: resolved config (decision 31: ``config/defaults/`` with the ``FES25_HT_2050`` scenario
#: entry's ``public_ownership`` block merged over it, ``study_config.py``), so a study
#: override of the insurance rule reaches the strikes as it reaches the register (task
#: SETTLE2050; plan Y14 D31 block). ``DEFAULTS_CONFIG`` is kept as the merge base.
DEFAULTS_CONFIG = ROOT / "config"
DEFAULT_STUDY = "FES25_HT_2050"


# ── insurance on the register's basis (item D4(a)) ───────────────────────────

def _insurance_of(config) -> dict:
    return dict(((config.get("public_ownership") or {}).get("generator_costs") or {}).get("insurance") or {})


@lru_cache(maxsize=4)
def _finance_insurance_block(path: str) -> dict:
    from config.config_loader import read_config

    return _insurance_of(read_config(path) or {})


@lru_cache(maxsize=4)
def _study_insurance_block(study: str, config_dir: str) -> dict:
    from scripts.public_ownership.study_config import load_studies

    studies = load_studies(config_dir, validate=False)
    if study not in studies.configs:
        raise KeyError(f"no finance study {study!r} in {config_dir}/scenarios.yaml; studies: {sorted(studies.configs)}")
    return _insurance_of(studies.configs[study])


def finance_insurance_config(path: str | Path | None = None, *, study: str = DEFAULT_STUDY,
                             defaults_without_study: bool = False) -> dict:
    """The finance layer's item D4(a) insurance block (``public_ownership.generator_costs.insurance``).

    Args:
        path: A resolved config file (a study's ``{study}_finance_config.yaml``, as the rules
            pass ``--config``). None reads the study's resolved config from ``config/``
            (decision 31), never ``config/defaults/`` alone (plan Y14: a study override
            of the insurance rule must reach the strikes).
        study: The study whose config is resolved when ``path`` is None.
        defaults_without_study: A scenario that is not a finance study (a private-only year,
            e.g. a first 2040 run) reads ``config/defaults/``'s block instead of failing.

    Returns:
        ``enabled``, ``share_of_capex_per_year``, ``gbp_per_kw_year_by_technology`` and
        ``excluded_technologies``, as the register reads them.
    """
    if path is not None:
        return dict(_finance_insurance_block(str(path)))
    try:
        return dict(_study_insurance_block(str(study), str(ROOT / "config")))
    except KeyError:
        if not defaults_without_study:
            raise
        return dict(_finance_insurance_block(str(DEFAULTS_CONFIG)))


def finance_insurance_gbp_per_kw_year(finance_technology: str, capex_gbp_per_kw: float,
                                      config: Mapping | None = None) -> float:
    """Item D4(a) insurance per kW a year, as ``calculate_financing_scenarios.generator_charges`` books it.

    Args:
        finance_technology: The register's finance (workbook) technology of the row, which is
            the vintage row's ``wacc_technology``.
        capex_gbp_per_kw: As-new capex per kW (power plus energy x duration), the register's basis.
        config: The D4(a) block; :func:`finance_insurance_config` if None.

    Returns:
        GBP/kW/yr: 0 when the line is off or the technology is excluded, the sourced rate where
        one exists, else ``share_of_capex_per_year`` x capex.
    """
    block = finance_insurance_config() if config is None else dict(config)
    if not block.get("enabled", False):
        return 0.0
    key = vc._key(finance_technology)
    if key in {vc._key(t) for t in (block.get("excluded_technologies") or [])}:
        return 0.0
    by_technology = {vc._key(k): float(v) for k, v in (block.get("gbp_per_kw_year_by_technology") or {}).items()}
    if key in by_technology:
        return by_technology[key]
    share = block.get("share_of_capex_per_year")
    return float(capex_gbp_per_kw) * float(share) if share is not None else 0.0


def insured_fixed_cost(cost: Mapping, capex_gbp_per_kw: float, *, insurance: str = INSURANCE_RULE,
                       finance_insurance: Optional[bool] = None,
                       insurance_config: Mapping | None = None) -> dict:
    """Fixed O&M per kW a year with insurance counted once, and the parts.

    Args:
        cost: A vintage row (``vintage_costs.costs_for``).
        capex_gbp_per_kw: The capex the D4(a) share applies to.
        insurance: ``exclude`` (the register's rule) or ``include`` (DESNZ insurance kept).
        finance_insurance: Add the D4(a) line; None follows ``insurance`` (on with ``exclude``,
            off with ``include``). ``include`` with the line on is refused (a double count).
        insurance_config: The D4(a) block (default: the config's).

    Returns:
        ``fixed_om_gbp_per_kw_year`` (without insurance), ``finance_insurance_gbp_per_kw_year``
        (the D4(a) line), ``fixed_total_gbp_per_kw_year`` (their sum) and ``insurance_rule``.
    """
    if insurance not in vc.INSURANCE_MODES:
        raise ValueError(f"insurance must be one of {vc.INSURANCE_MODES}")
    add_line = (insurance == "exclude") if finance_insurance is None else bool(finance_insurance)
    if insurance == "include" and add_line:
        raise ValueError("insurance='include' keeps DESNZ insurance in fixed O&M; adding the D4(a) line too counts it twice")
    fixed = float(cost["fixed_om_gbp_per_kw_year"])
    if insurance == "exclude":
        fixed -= float(cost["insurance_gbp_per_kw_year"])
    wacc_technology = cost.get("wacc_technology")
    finance_technology = wacc_technology if isinstance(wacc_technology, str) and wacc_technology.strip() else cost["technology"]
    line = finance_insurance_gbp_per_kw_year(str(finance_technology), capex_gbp_per_kw, insurance_config) if add_line else 0.0
    rule = f"vintage_insurance_{insurance}" + ("_plus_d4a" if add_line else "")
    return {"fixed_om_gbp_per_kw_year": fixed, "finance_insurance_gbp_per_kw_year": line,
            "fixed_total_gbp_per_kw_year": fixed + line, "insurance_rule": rule}


# ── inputs ───────────────────────────────────────────────────────────────────

def load_strike_inputs(path: str | Path = STRIKE_INPUTS) -> pd.DataFrame:
    """The declared load factors, use-of-system charges, terms and settlement bases.

    Args:
        path: ``cfd_strike_inputs_2050.csv``.

    Returns:
        The table, provenance checked.
    """
    table = pd.read_csv(path)
    for column in ("source", "source_page", "status"):
        blank = table[column].isna() | table[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(f"{path}: rows without {column}: {sorted(table.loc[blank, 'technology'])}")
    if not table["expected_load_factor"].between(0, 1, inclusive="right").all():
        raise ValueError(f"{path}: load factors must be in (0, 1]")
    return table


def strike_inputs_for(technology: str, build_year: int, table: pd.DataFrame | None = None) -> dict:
    """The input row in force for a build year (the latest ``from_build_year`` at or before it).

    Args:
        technology: A vintage-table technology.
        build_year: Commissioning year.
        table: :func:`load_strike_inputs` output.

    Returns:
        The row as a dict.
    """
    table = load_strike_inputs() if table is None else table
    rows = table[table["technology"].map(vc._key).eq(vc._key(technology))]
    if rows.empty:
        raise KeyError(f"no CfD strike inputs for {technology!r}")
    earlier = rows[rows["from_build_year"] <= int(build_year)]
    row = (earlier if not earlier.empty else rows).sort_values("from_build_year").iloc[-1 if not earlier.empty else 0]
    return row.to_dict()


def load_snapshot(path: str | Path = DEFAULT_SNAPSHOT) -> dict:
    """The workbook assumptions snapshot (``extract_workbook_v5.py``, schema 4)."""
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def private_rate(technology: str, build_year: int, *, snapshot: Mapping, rate_path: str = "flat",
                 basis: str = "notional_gearing_shielded", vintage_table: pd.DataFrame | None = None,
                 hurdle_table: pd.DataFrame | None = None) -> dict:
    """The private rate a vintage is financed at, and where it comes from.

    Args:
        technology: A vintage-table technology.
        build_year: Commissioning year.
        snapshot: The workbook snapshot.
        rate_path: ``flat`` (decision 19) or ``neso_wind_ramp`` (FES 2025 hurdle change since 2030 added).
        basis: The snapshot WACC block (the ``private_wacc_basis`` lever's value).
        vintage_table: A loaded vintage table.
        hurdle_table: A loaded FES/DESNZ hurdle-rate table.

    Returns:
        ``rate``, ``base_rate``, ``shift``, ``wacc_technology`` and ``source``.
    """
    if rate_path not in RATE_PATHS:
        raise ValueError(f"rate_path must be one of {RATE_PATHS}")
    row = vc.costs_for(technology, min(max(int(build_year), vc.FIRST_BUILD_YEAR), vc.LAST_BUILD_YEAR),
                       table=vintage_table)
    wacc_technology = row.get("wacc_technology")
    if not isinstance(wacc_technology, str) or not wacc_technology.strip():
        raise KeyError(f"{technology!r} has no wacc_technology in the vintage table")
    match = [t for t in snapshot["technologies"] if vc._key(t["technology"]) == vc._key(wacc_technology)]
    if len(match) != 1:
        raise KeyError(f"snapshot has no single technology row for {wacc_technology!r}")
    trow = match[0]
    if basis == "notional_gearing_shielded" and "wacc" in trow:
        base = float(trow["wacc"]["private"])
        cell = trow.get("wacc_source_cells", {}).get("private", "WACC Calculator!Y")
    else:
        base = float(trow["wacc_by_basis"][basis]["private"])
        cell = trow.get("wacc_by_basis_source_cells", {}).get(basis, {}).get("private", basis)
    shift = 0.0
    if rate_path == "neso_wind_ramp":
        try:
            shift = (vc.hurdle_rate(technology, build_year, source="FES", table=hurdle_table)
                     - vc.hurdle_rate(technology, vc.ANCHOR_YEAR, source="FES", table=hurdle_table))
        except KeyError:
            shift = 0.0
    return {"rate": base + shift, "base_rate": base, "shift": shift, "wacc_technology": trow["technology"],
            "source": f"workbook snapshot {cell} ({basis}, private)"
                      + (f" + FES 2025 E.03 hurdle change {build_year} vs 2030" if shift else "")}


# ── the AR7 offshore calibration (user decision, 27 Oct 2026) ────────────────

def load_cost_calibration(path: str | Path = STRIKE_COST_CALIBRATION) -> pd.DataFrame:
    """The strike path's cost anchors (``cfd_strike_cost_calibration.csv``), provenance checked.

    Returns:
        One row per anchor: the technology, the build year and cleared strike (2024 prices) its
        capex is calibrated to, and the model years it applies to (``model_years``: ``all`` or
        years separated by ``;``; a table without the column is ``all`` throughout).

    Raises:
        ValueError: a row without provenance, a bad ``model_years`` entry, or two anchors of one
            technology that apply to a common model year.
    """
    table = pd.read_csv(path)
    for column in ("source", "source_page", "status"):
        blank = table[column].isna() | table[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(f"{path}: rows without {column}: {sorted(table.loc[blank, 'technology'])}")
    if "model_years" not in table.columns:
        table["model_years"] = ALL_MODEL_YEARS
    table["model_years"] = table["model_years"].fillna("").astype(str).str.strip()
    scopes = [_model_years(value, path) for value in table["model_years"]]
    for key, positions in pd.Series(range(len(table))).groupby(table["technology"].map(vc._key).to_numpy()):
        years = [scopes[i] for i in positions]
        if len(years) > 1 and (any(y is None for y in years)
                               or len(frozenset().union(*years)) < sum(len(y) for y in years)):
            raise ValueError(f"{path}: technology {key!r} has two anchors for one model year")
    if not (pd.to_numeric(table["anchor_strike_gbp_per_mwh_2024"], errors="coerce") > 0).all():
        raise ValueError(f"{path}: every anchor strike must be a positive number")
    return table


#: ``model_years`` value of an anchor that applies to every model year.
ALL_MODEL_YEARS = "all"


def _model_years(value: str, path=None) -> Optional[frozenset]:
    """``all`` -> None (every year); ``2030`` or ``2030;2035`` -> the years."""
    text = str(value).strip()
    if text.lower() == ALL_MODEL_YEARS:
        return None
    try:
        years = frozenset(int(float(part)) for part in text.split(";") if part.strip())
    except ValueError:
        years = frozenset()
    if not years:
        raise ValueError(f"{path}: model_years must be 'all' or years separated by ';', got {value!r}")
    return years


def anchor_applies(model_years: str, model_year: Optional[int]) -> bool:
    """Whether an anchor's ``model_years`` covers a caller's model year (None: ``all`` anchors only)."""
    years = _model_years(model_years)
    return years is None or (model_year is not None and int(model_year) in years)


def _calibration_table(cost_calibration) -> pd.DataFrame | None:
    """``True``: the default table (unless :data:`COST_CALIBRATION_ENABLED` is off); ``False``/``None``: none."""
    if cost_calibration is None or cost_calibration is False:
        return None
    if cost_calibration is True:
        return load_cost_calibration() if COST_CALIBRATION_ENABLED else None
    return cost_calibration


def cost_calibration_factor(technology: str, *, snapshot: Mapping, cost_calibration=True,
                            vintage_table: pd.DataFrame | None = None, inputs: pd.DataFrame | None = None,
                            hurdle_table: pd.DataFrame | None = None,
                            insurance_config: Mapping | None = None,
                            model_year: Optional[int] = None) -> Optional[float]:
    """The capex factor ``k`` that puts a technology's anchor build year on its cleared strike, or None.

    Trust the auction, as the 2030 calibration does (``derive_cost_calibration.py``): the
    cleared strike is revealed required revenue at the private rate and capex is the
    unobserved part, so ``k`` solves ``strike(k) = anchor`` on the central basis
    (economic-life amortisation, flat rate, no learning, the register's insurance rule;
    the strike is linear in ``k``, so two evaluations solve it exactly).

    Args:
        technology: A vintage-table technology.
        snapshot: The workbook snapshot (private rates).
        cost_calibration: ``True`` (the default table), a :func:`load_cost_calibration` frame,
            or ``False``/``None`` (no calibration: returns None).
        vintage_table, inputs, hurdle_table: Loaded tables (as :func:`levelised_strike`).
        insurance_config: The D4(a) block the caller prices on.
        model_year: The caller's model year; only anchors whose ``model_years`` cover it apply.
            None reads the ``all`` anchors only (the AR7 offshore one).

    Returns:
        ``k``, or None when the technology has no anchor for the model year.

    Raises:
        ValueError: ``k`` outside [0.05, 2] (the 2030 table's bounds).
    """
    table = _calibration_table(cost_calibration)
    if table is None:
        return None
    rows = table[table["technology"].map(vc._key).eq(vc._key(technology))]
    if "model_years" in rows.columns:
        rows = rows[[anchor_applies(value, model_year) for value in rows["model_years"].fillna(ALL_MODEL_YEARS)]]
    if rows.empty:
        return None
    anchor = rows.iloc[0]
    year, target = int(anchor["anchor_build_year"]), float(anchor["anchor_strike_gbp_per_mwh_2024"])
    common = dict(snapshot=snapshot, vintage_table=vintage_table, inputs=inputs, hurdle_table=hurdle_table,
                  insurance_config=insurance_config)
    at_one = levelised_strike(technology, year, capex_factor=1.0, **common)["strike_gbp_per_mwh_2024"]
    at_zero = levelised_strike(technology, year, capex_factor=0.0, **common)["strike_gbp_per_mwh_2024"]
    factor = (target - at_zero) / (at_one - at_zero)
    if not 0.05 <= factor <= 2.0:
        raise ValueError(f"{technology}: the capex factor {factor:.4f} that lands {year} on {target} is outside [0.05, 2]")
    return float(factor)


def vintage_capex_factors(technologies: Iterable[str], *, snapshot: Mapping, insurance_config: Mapping | None = None,
                          cost_calibration=True, model_year: Optional[int] = None) -> np.ndarray:
    """Per row, the capex factor the strike path puts on its vintage technology (1 where none).

    For the vintaged register, which costs the same vintage rows: scaling their capex by the
    same factor keeps an at-cost strike equal to the register's required revenue per MWh.
    """
    names = [str(t) for t in technologies]
    factors = {}
    for name in sorted(set(names)):
        try:
            k = cost_calibration_factor(name, snapshot=snapshot, cost_calibration=cost_calibration,
                                        insurance_config=insurance_config, model_year=model_year)
        except KeyError:
            k = None                                   # not a vintage-table technology with strike inputs
        factors[name] = 1.0 if k is None else k
    return np.array([factors[n] for n in names], dtype=float)


# ── strikes ──────────────────────────────────────────────────────────────────

def levelised_strike(technology: str, build_year: int, *, snapshot: Mapping, rate_path: str = "flat",
                     learning: bool = False, insurance: str = INSURANCE_RULE, amortisation: str = "economic_life",
                     load_factor: Optional[float] = None, use_of_system_gbp_per_kw_year: Optional[float] = None,
                     fuel_gbp_per_mwh: float = 0.0, other_revenue_gbp_per_mwh: float = 0.0,
                     vintage_table: pd.DataFrame | None = None, inputs: pd.DataFrame | None = None,
                     hurdle_table: pd.DataFrame | None = None, finance_insurance: Optional[bool] = None,
                     insurance_config: Mapping | None = None, cost_calibration=True,
                     capex_factor: Optional[float] = None, model_year: Optional[int] = None) -> dict:
    """The post-AR7 strike of one technology and build year, with every component.

    Args:
        technology: A vintage-table technology.
        build_year: Commissioning year (later than 2050 holds 2050, decision 11).
        snapshot: The workbook snapshot (private rates).
        rate_path: ``flat`` or ``neso_wind_ramp``.
        learning: Capex from the CCC learning-rate column (decision 11 sensitivity).
        insurance: ``exclude`` (the register's rule: DESNZ insurance out of fixed O&M, the
            D4(a) line added) or ``include`` (DESNZ insurance kept, no D4(a) line).
        finance_insurance: Override whether the D4(a) line is added (:func:`insured_fixed_cost`).
        insurance_config: The D4(a) block (default: the config's).
        amortisation: ``economic_life`` (central) or ``contract_term``.
        load_factor: Override the declared expected load factor (e.g. a solved year's).
        use_of_system_gbp_per_kw_year: Override the declared use-of-system charge.
        fuel_gbp_per_mwh: Fuel (and carbon) per MWh electrical, for fuelled plant.
        other_revenue_gbp_per_mwh: Revenue outside the CfD per MWh (BECCS removals), netted off.
        vintage_table: A loaded vintage table.
        inputs: A loaded strike-input table.
        hurdle_table: A loaded hurdle-rate table.
        cost_calibration: The AR7 offshore calibration (:func:`cost_calibration_factor`):
            ``True`` the default anchors, a frame, or ``False``/``None`` for DESNZ capex as is.
        capex_factor: A capex factor to use as given (skips ``cost_calibration``).
        model_year: The model year whose anchors calibrate capex (:func:`cost_calibration_factor`);
            None: the ``all`` anchors only.

    Returns:
        The strike (real 2024 GBP/MWh) and its parts, rate, term and provenance;
        ``capex_calibration_factor`` is 1 for a technology without an anchor.
    """
    if amortisation not in AMORTISATIONS:
        raise ValueError(f"amortisation must be one of {AMORTISATIONS}")
    if capex_factor is None:
        capex_factor = cost_calibration_factor(technology, snapshot=snapshot, cost_calibration=cost_calibration,
                                               vintage_table=vintage_table, inputs=inputs,
                                               hurdle_table=hurdle_table, insurance_config=insurance_config,
                                               model_year=model_year)
    capex_factor = 1.0 if capex_factor is None else float(capex_factor)
    cost = vc.costs_for(technology, build_year, learning=learning, table=vintage_table)
    spec = strike_inputs_for(technology, build_year, inputs)
    rate = private_rate(technology, build_year, snapshot=snapshot, rate_path=rate_path,
                        vintage_table=vintage_table, hurdle_table=hurdle_table)
    lf = float(spec["expected_load_factor"] if load_factor is None else load_factor)
    if not 0.0 < lf <= 1.0:
        raise ValueError(f"load factor must be in (0, 1], got {lf}")
    life = float(cost["economic_life_years"])
    term = int(spec["term_years"])
    years = life if amortisation == "economic_life" else min(float(term), life)
    crf = capital_recovery_factor(rate["rate"], years)
    energy_capex = float(np.nan_to_num(float(cost.get("capex_energy_gbp_per_kwh") or 0.0)))
    duration = float(np.nan_to_num(float(cost.get("duration_hours") or 0.0)))
    capex_desnz = float(cost["capex_power_gbp_per_kw"]) + energy_capex * duration
    # The calibrated capex; insurance that is a share of capex follows it, as in the 2030 register.
    capex = capex_desnz * capex_factor
    insured = insured_fixed_cost(cost, capex, insurance=insurance, finance_insurance=finance_insurance,
                                 insurance_config=insurance_config)
    fixed = insured["fixed_om_gbp_per_kw_year"]
    uos = float(spec["use_of_system_gbp_per_kw_year"] if use_of_system_gbp_per_kw_year is None
                else use_of_system_gbp_per_kw_year)
    mwh_per_kw = 8.760 * lf
    parts = {"capital_gbp_per_mwh": capex * crf / mwh_per_kw, "fixed_om_gbp_per_mwh": fixed / mwh_per_kw,
             "finance_insurance_gbp_per_mwh": insured["finance_insurance_gbp_per_kw_year"] / mwh_per_kw,
             "use_of_system_gbp_per_mwh": uos / mwh_per_kw,
             "variable_om_gbp_per_mwh": float(cost["variable_om_gbp_per_mwh"]),
             "fuel_gbp_per_mwh": float(fuel_gbp_per_mwh),
             "other_revenue_gbp_per_mwh": -float(other_revenue_gbp_per_mwh)}
    strike = float(sum(parts.values()))
    return {"technology": technology, "build_year": int(build_year), "strike_gbp_per_mwh_2024": strike,
            **parts, "private_rate": rate["rate"], "base_rate": rate["base_rate"], "rate_shift": rate["shift"],
            "rate_path": rate_path, "capex_gbp_per_kw": capex, "capex_case": cost["capex_case"],
            "capex_desnz_gbp_per_kw": capex_desnz, "capex_calibration_factor": capex_factor,
            "fixed_om_gbp_per_kw_year": fixed, "insurance_gbp_per_kw_year": float(cost["insurance_gbp_per_kw_year"]),
            "insurance": insurance, "insurance_rule": insured["insurance_rule"],
            "finance_insurance_gbp_per_kw_year": insured["finance_insurance_gbp_per_kw_year"],
            "use_of_system_gbp_per_kw_year": uos, "load_factor": lf,
            "load_factor_source": "declared" if load_factor is None else "override",
            "economic_life_years": life, "amortisation": amortisation, "amortisation_years": years,
            "term_years": term, "contract_end_year": int(build_year) + term,
            "reference_price": spec["reference_price"], "negative_price_rule": spec["negative_price_rule"],
            "cost_source": f"{cost['source']} | {cost['source_page']}", "rate_source": rate["source"],
            "inputs_source": f"{spec['source']} | {spec['source_page']}", "price_year": PRICE_YEAR}


def modelled_strikes(technologies: Iterable[str] = MAIN_TECHNOLOGIES, build_years: Iterable[int] | None = None, *,
                     model_year: int, snapshot: Mapping | None = None,
                     variants: Mapping[str, Mapping] | None = None, cost_calibration=True) -> pd.DataFrame:
    """Post-AR7 strikes by technology, build year and variant (central plus sensitivities).

    Args:
        technologies: Vintage-table technologies.
        build_years: Build years; default every post-AR7 year up to the model year.
        model_year: The calendar year the live shares are for (the scenario's ``modelled_year``).
        snapshot: The workbook snapshot.
        variants: ``{label: levelised_strike kwargs}``; default :data:`STRIKE_VARIANTS` (central,
            NESO wind ramp, learning and contract-term amortisation).
        cost_calibration: As :func:`levelised_strike` (the factor is solved once per technology).

    Returns:
        One row per technology, build year and variant, with its live share in the model year
        (``live_share``; ``model_year`` names the year).
    """
    snapshot = load_snapshot() if snapshot is None else snapshot
    variants = variants or STRIKE_VARIANTS
    if build_years is None:
        build_years = range(FIRST_MODELLED_BUILD_YEAR, int(model_year) + 1)
    vintage, inputs, hurdles = vc.load_vintage_costs(), load_strike_inputs(), vc.load_hurdle_rates()
    calibration = _calibration_table(cost_calibration)
    rows = []
    for technology in technologies:
        factors = {}
        for year in build_years:
            for label, kwargs in variants.items():
                kwargs = dict(kwargs)
                # The factor depends only on the D4(a) block a variant prices on: solve it once per block.
                key = id(kwargs.get("insurance_config"))
                if key not in factors:
                    factors[key] = cost_calibration_factor(
                        technology, snapshot=snapshot, cost_calibration=calibration, vintage_table=vintage,
                        inputs=inputs, hurdle_table=hurdles, insurance_config=kwargs.get("insurance_config"),
                        model_year=int(model_year))
                # No anchor (or no calibration): DESNZ capex as is (factor 1).
                factor = 1.0 if factors[key] is None else factors[key]
                row = levelised_strike(technology, int(year), snapshot=snapshot, vintage_table=vintage,
                                       inputs=inputs, hurdle_table=hurdles, capex_factor=factor, **kwargs)
                row["variant"] = label
                row["cfd_round"] = f"{MODELLED_ROUND_PREFIX}{int(year)}"
                row["model_year"] = int(model_year)
                row["live_share"] = live_fraction(int(year), row["term_years"], int(model_year))
                rows.append(row)
    return pd.DataFrame(rows)


def strike_table_2050(technologies: Iterable[str] = MAIN_TECHNOLOGIES, build_years: Iterable[int] | None = None, *,
                      snapshot: Mapping | None = None, variants: Mapping[str, Mapping] | None = None,
                      model_year: int = LEGACY_MODEL_YEAR, cost_calibration=True) -> pd.DataFrame:
    """:func:`modelled_strikes` for 2050 (older callers); ``live_share_2050`` kept beside ``live_share``."""
    out = modelled_strikes(technologies, build_years, model_year=model_year, snapshot=snapshot, variants=variants,
                           cost_calibration=cost_calibration)
    out["live_share_2050"] = out["live_share"]
    return out


# ── known contracts ──────────────────────────────────────────────────────────

def delivery_year_start(label: str) -> float:
    """'2030/31' -> 2030.25: a delivery year starts on 1 April."""
    return float(str(label).split("/")[0]) + 0.25


#: Columns every row of :data:`CONTRACTS` carries.
CONTRACT_COLUMNS = ("contract_id", "cfd_round", "project", "technology", "carrier", "region", "capacity_mw", "phases",
                    "phase_1_delivery_year", "term_years", "strike_gbp_per_mwh_2012", "published_strike_gbp_per_mwh_2024",
                    "reference_price", "negative_price_rule", "scheme", "source", "source_page", "status", "note",
                    "strike_table_technology")


def load_contracts(path: str | Path = CONTRACTS, published: str | Path | None = PUBLISHED_STRIKES) -> pd.DataFrame:
    """Every CfD round's contracts (``cfd_contracts.csv``), checked.

    Checks: provenance on every row; unique ids; a known negative-pricing term; and, when
    ``published`` is given, that every published round (``cfd_strike_prices_by_round.csv``,
    bar HPC, listed by unit, and the 2030 chain's modelled AR8 strikes) is in the table with the
    same capacity and strikes per round and strike-table technology, so the two tables
    cannot drift apart.

    Returns:
        The table as read.
    """
    table = pd.read_csv(path)
    missing = set(CONTRACT_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    for column in ("source", "source_page", "status"):
        if table[column].isna().any():
            raise ValueError(f"{path}: rows without {column}")
    if table["contract_id"].duplicated().any():
        raise ValueError(f"{path} repeats contract ids {sorted(table.loc[table['contract_id'].duplicated(), 'contract_id'])}")
    rules = table["negative_price_rule"].fillna("none").astype(str)
    if not rules.isin(list(NEGATIVE_PRICE_RULES)).all():
        raise ValueError(f"{path}: negative_price_rule must be one of {sorted(NEGATIVE_PRICE_RULES)}")
    if published is not None:
        pub = pd.read_csv(published)
        pub = pub[~pub["cfd_round"].isin(UNCHECKED_PUBLISHED_ROUNDS)]
        mine = table[table["scheme"].eq("CfD") & ~table["cfd_round"].isin(UNCHECKED_PUBLISHED_ROUNDS)]
        for (cfd_round, technology), group in pub.groupby(["cfd_round", "technology"]):
            rows = mine[mine["cfd_round"].eq(cfd_round) & mine["strike_table_technology"].eq(technology)]
            if rows.empty:
                raise ValueError(f"{path}: published round {cfd_round} {technology} is not in the contract table")
            published_mw = float(pd.to_numeric(group["capacity_mw"], errors="coerce").sum())
            if abs(float(rows["capacity_mw"].sum()) - published_mw) > 0.01:
                raise ValueError(f"{path}: {cfd_round} {technology} holds {rows['capacity_mw'].sum():.2f} MW against "
                                 f"the published {published_mw:.2f}")
            if set(rows["strike_gbp_per_mwh_2012"].round(3)) != set(group["strike_price_gbp_per_mwh_2012"].round(3)):
                raise ValueError(f"{path}: {cfd_round} {technology} strikes differ from the published table")
    return table


#: User decisions (13 Nov 2026, results review finding 9 and the contracts audit): every allocation round's contracts
#: follow LCCC's own unit records (the CfD contract portfolio), as the 2030 chain does: one phase per LCCC unit on its
#: own start date (``Expected_Start_Date``; for an operating unit without one, its first settlement date from
#: ``legacy_cfd_contract_starts.csv``; else the contract row's delivery-year start), terminated units hold nothing,
#: and AR1 to AR6 carry LCCC's live MW (terminations and permitted reductions netted, so AR4 capacity relinquished and
#: re-contracted in AR6 counts once). AR7 and AR7a keep DESNZ's awarded MW (the 2030 chain's), on LCCC's dates.
LCCC_UNIT_ROUNDS = {"AR1": "Allocation Round 1", "AR2": "Allocation Round 2", "AR3": "Allocation Round 3",
                    "AR4": "Allocation Round 4", "AR5": "Allocation Round 5", "AR6": "Allocation Round 6",
                    "AR7": "Allocation Round 7", "AR7a": "Allocation Round 7"}
#: Rounds that keep the contract table's (DESNZ's) MW and take only LCCC's dates.
LCCC_DATES_ONLY_ROUNDS = ("AR7", "AR7a")
LCCC_PORTFOLIO = ROOT / "data" / "market" / "lccc" / "cfd_contract_portfolio_status.csv"
LCCC_SETTLEMENT_STARTS = ROOT / "data" / "public_ownership" / "financing" / "legacy_cfd_contract_starts.csv"
CHECKED_CONTRACTS = ROOT / "data" / "market" / "lccc" / "checked_cfd_contracts.csv"
#: LCCC technology types -> the contract table's strike-table technologies they cover.
LCCC_TECHNOLOGY = {
    "Offshore Wind": ("offshore", "offshore_scotland", "offshore_permitted_reduction"),
    "Floating Offshore Wind": ("floating_offshore",), "Onshore Wind": ("onshore",),
    "Remote Island Wind": ("remote_island_wind",), "Solar PV": ("solar",), "Tidal Stream": ("tidal",),
    "Advanced Conversion Technology": ("advanced_conversion",), "Energy from Waste": ("energy_from_waste",),
    "Dedicated Biomass": ("dedicated_biomass",), "Geothermal": ("geothermal",)}
PERMITTED_REDUCTION = "offshore_permitted_reduction"


def _decimal_year(value) -> float:
    stamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(stamp):
        return np.nan
    return float(stamp.year + (stamp.dayofyear - 1) / 365.25)


def lccc_units(portfolio: str | Path = LCCC_PORTFOLIO, settlement: str | Path = LCCC_SETTLEMENT_STARTS,
               checked: str | Path = CHECKED_CONTRACTS) -> pd.DataFrame:
    """LCCC's CfD units with a decimal start year, live flag and the strike-table technology they settle at.

    Start: ``Expected_Start_Date``, else the unit's first settlement date (``legacy_cfd_contract_starts.csv``,
    the 2030 valuation's), else NaN (the caller falls back to the contract row's delivery year). Offshore units are
    ``offshore_permitted_reduction`` where ``checked_cfd_contracts.csv`` lists their CfD id under it (AR6's permitted
    reduction capacity), else plain offshore.
    """
    units = pd.read_csv(portfolio)
    units["mw"] = pd.to_numeric(units["Maximum_Contract_Capacity_MW"], errors="coerce").fillna(0.0)
    units["live"] = ~units["Status"].astype(str).str.strip().str.lower().eq("terminated")
    units["start_year"] = units["Expected_Start_Date"].map(_decimal_year)
    settled = pd.read_csv(settlement).drop_duplicates("cfd_id").set_index("cfd_id")["contract_start"].map(_decimal_year)
    missing = units["start_year"].isna()
    units.loc[missing, "start_year"] = units.loc[missing, "CFD_ID"].map(settled)
    table = pd.read_csv(checked)
    pr_ids = {i.strip() for ids in table.loc[table["cfd_technology"].astype(str).eq(PERMITTED_REDUCTION), "cfd_ids"]
              .dropna().astype(str) for i in ids.split(";")}
    units["is_permitted_reduction"] = units["CFD_ID"].astype(str).isin(pr_ids)
    return units


def lccc_unit_phases(table: pd.DataFrame, units: pd.DataFrame | None = None) -> dict[str, list[tuple[float, float]]]:
    """``contract_id -> [(start_year, mw), ...]``: the contract rows of the LCCC rounds as LCCC's live units.

    Rows are grouped by round and LCCC technology (AR6's permitted-reduction row apart, on its own units); each
    row takes its share of the group's MW on every live unit of the group (rows of one group differ in delivery
    year and strike, which LCCC does not record per unit). The group's MW is LCCC's live MW for AR1 to AR6 (at most
    the table's), the table's for AR7 and AR7a. A start the portfolio lacks is NaN (the row's own start applies).
    A group LCCC does not list keeps the table (absent from the result); a group with no live unit holds nothing.
    """
    units = lccc_units() if units is None else units
    out: dict[str, list[tuple[float, float]]] = {}
    for cfd_round, lccc_name in LCCC_UNIT_ROUNDS.items():
        mine = table[table["cfd_round"].eq(cfd_round)]
        for lccc_tech, techs in LCCC_TECHNOLOGY.items():
            for pr in (False, True):
                sub_techs = [t for t in techs if (t == PERMITTED_REDUCTION) == pr]
                rows = mine[mine["strike_table_technology"].isin(sub_techs)]
                if rows.empty:
                    continue
                listed = units[units["Allocation_Round"].eq(lccc_name) & units["Technology_Type"].eq(lccc_tech)]
                if lccc_tech == "Offshore Wind":
                    listed = listed[listed["is_permitted_reduction"].eq(pr)]
                if listed.empty:
                    continue
                live = listed[listed["live"] & listed["mw"].gt(0)]
                table_mw = float(rows["capacity_mw"].sum())
                live_mw = float(live["mw"].sum())
                group_mw = table_mw if cfd_round in LCCC_DATES_ONLY_ROUNDS else min(live_mw, table_mw)
                for _, r in rows.iterrows():
                    share = float(r["capacity_mw"]) / table_mw if table_mw > 0 else 0.0
                    scale = group_mw / live_mw if live_mw > 0 else 0.0
                    out[str(r["contract_id"])] = [(float(u["start_year"]), float(u["mw"]) * scale * share)
                                                  for _, u in live.iterrows()]
    return out


def known_contracts(model_year: int, path: str | Path = CONTRACTS,
                    published: str | Path | None = PUBLISHED_STRIKES) -> pd.DataFrame:
    """Every contract phase in the table and whether it is live in the model year.

    A row splits into ``phases`` (equal MW, phase k starting k-1 delivery years after
    phase 1), each on its own term from a start spread evenly through its delivery year
    (``capacity_market_2050.live_fraction``). The table holds every round (AR1 to AR7a,
    the Investment Contracts, MGT, the Drax bridge, the declared Power BECCS CfD, HPC and
    Sizewell C's RAB), so nothing is inferred per year: in 2050 every round before AR7 has
    expired; in 2040 AR3 to AR6 are partly live.

    Args:
        model_year: The calendar year (the scenario's ``modelled_year``).
        path: ``cfd_contracts.csv``.
        published: ``cfd_strike_prices_by_round.csv`` for :func:`load_contracts`'s check (None: skip).

    Returns:
        One row per contract phase: MW, start, expiry, ``live_share``, ``live_mw``,
        strike in 2012 and 2024 prices (checked against the published 2024 value), status.
    """
    table = load_contracts(path, published)
    lccc = lccc_unit_phases(table)
    rows = []
    for _, c in table.iterrows():
        phases = int(c["phases"])
        start1 = delivery_year_start(c["phase_1_delivery_year"])
        # (start, MW) per phase: LCCC's units for the rounds it lists (contracts audit, 13 Nov 2026), else the table's.
        unit_phases = lccc.get(str(c["contract_id"]))
        if unit_phases is None:
            plan = [(start1 + k, float(c["capacity_mw"]) / phases) for k in range(phases)]
        else:
            plan = [(start if np.isfinite(start) else start1, mw) for start, mw in unit_phases]
        term = int(c["term_years"])
        strike_2012 = float(c["strike_gbp_per_mwh_2012"]) if pd.notna(c["strike_gbp_per_mwh_2012"]) else np.nan
        strike_2024 = strike_2012 * CPI_2012_TO_2024 if strike_2012 == strike_2012 else np.nan
        published_2024 = c["published_strike_gbp_per_mwh_2024"]
        if pd.notna(published_2024) and abs(strike_2024 - float(published_2024)) > 0.02:
            raise ValueError(f"{c['contract_id']}: 2012 strike x CPI ({strike_2024:.2f}) does not reproduce the "
                             f"published 2024 value ({published_2024})")
        if not plan:
            # Every LCCC unit of the contract is terminated: it holds nothing (one zero row keeps it visible).
            plan = [(start1, 0.0)]
        for k, (start, mw) in enumerate(plan):
            share = live_fraction(start, term, int(model_year)) if mw > 0 else 0.0
            rows.append({"contract_id": c["contract_id"], "phase": k + 1, "phases": len(plan), "cfd_round": c["cfd_round"],
                         "project": c["project"], "technology": c["technology"], "carrier": c["carrier"],
                         "region": c["region"], "scheme": c["scheme"], "capacity_mw": mw,
                         "start_year": start, "term_years": term, "expiry_year": start + term,
                         "live_share": share, "live_mw": mw * share,
                         "strike_gbp_per_mwh_2012": strike_2012, "strike_gbp_per_mwh_2024": strike_2024,
                         "reference_price": c["reference_price"], "negative_price_rule": c["negative_price_rule"],
                         "status": "live" if share >= 1.0 - 1e-9 else ("expiring" if share > 0 else "expired"),
                         "source": c["source"], "source_page": c["source_page"], "note": c.get("note"),
                         "model_year": int(model_year)})
    return pd.DataFrame(rows)


def known_contracts_2050(model_year: int = LEGACY_MODEL_YEAR, path: str | Path = CONTRACTS,
                         published: str | Path | None = PUBLISHED_STRIKES) -> pd.DataFrame:
    """:func:`known_contracts` (older callers; 2050 unless a year is given)."""
    return known_contracts(model_year, path, published)


# ── settlement (the hook for a solved model year) ────────────────────────────

def seasonal_baseload_reference(price: pd.Series, demand: pd.Series, hours: pd.Series | None = None) -> pd.Series:
    """Season-ahead baseload reference: the season's demand-weighted mean price (Apr-Sep, Oct-Mar).

    The 2030 bill's rule (``project_future_bill.seasonal_demand_weighted_reference``),
    restated here so a 2050 caller needs no bill-layer import.
    """
    index = pd.DatetimeIndex(price.index)
    hours = pd.Series(1.0, index=index) if hours is None else hours.reindex(index)
    season = pd.Series(np.where(index.month.isin([4, 5, 6, 7, 8, 9]), "summer", "winter"), index=index)
    energy = demand.reindex(index) * hours
    by_season = (price * energy).groupby(season).sum() / energy.groupby(season).sum()
    return season.map(by_season).astype(float)


def cfd_settlement(dispatch_mwh: pd.DataFrame, contracts: pd.DataFrame, reference_price: pd.Series, *,
                   demand_mw: pd.Series | None = None) -> pd.DataFrame:
    """Difference payments of a set of CfD contracts on a solved year.

    The payment is ``revenue_tracking.compute_cfd_payments``'s, the calculator the bill, the
    asset settlement and the strike adjustments use (task SETTLE2050: one CfD settlement;
    this function used to restate the formula). Each contract becomes one unit with its own
    strike entry; ``ar4_plus`` contracts take a post-AR7 round (``PAR7_<year>``: no payment
    in a negative hour), ``none`` the HPC round (paid throughout), under
    ``gb_contract_terms``.

    Args:
        dispatch_mwh: MWh per snapshot (rows) and contract unit (columns: ``contracts.unit``).
        contracts: One row per unit with ``unit``, ``strike_gbp_per_mwh_2024``,
            ``reference_price`` (``intermittent_hourly`` or ``baseload_seasonal``),
            ``negative_price_rule`` (``ar4_plus``, ``six_hour`` or ``none``) and ``live_share`` (the
            share of the year the contract is live; applied pro rata to energy,
            declared, since expiry dates within the year are spread evenly).
        reference_price: The GB day-ahead (Stage 1 wholesale) price by snapshot, 2024 prices.
        demand_mw: GB demand by snapshot (needed for baseload contracts).

    Returns:
        Per unit: settled MWh, unpaid MWh (negative-price hours), reference captured,
        payment (positive = cost to consumers).
    """
    from types import SimpleNamespace

    from scripts.market.cfd_strike_table import StrikeTable
    from scripts.market.revenue_tracking import compute_cfd_payments

    index = pd.DatetimeIndex(dispatch_mwh.index)
    if not index.equals(pd.DatetimeIndex(reference_price.index)):
        raise ValueError("dispatch and reference price snapshots differ")
    missing = sorted(set(contracts["unit"]) - set(dispatch_mwh.columns))
    if missing:
        raise ValueError(f"no dispatch for contract units {missing[:5]}")
    baseload = None
    if contracts["reference_price"].eq("baseload_seasonal").any():
        if demand_mw is None:
            raise ValueError("baseload contracts need demand_mw for the seasonal reference")
        baseload = seasonal_baseload_reference(reference_price, demand_mw)
    # One pseudo-carrier per unit, so each unit has its own strike entry and reference.
    carriers = [f"cfd_unit_{i}" for i in range(len(contracts))]
    # Each contract's negative-pricing term as the round revenue_tracking reads it (AR4 onward:
    # a post-AR7 round; AR2/AR3: six hours; none: HPC's terms, paid throughout).
    rounds = [f"{MODELLED_ROUND_PREFIX}{FIRST_MODELLED_BUILD_YEAR}" if rule == "ar4_plus"
              else NEGATIVE_PRICE_RULES.get(str(rule), "HPC") for rule in contracts["negative_price_rule"]]
    strikes = {(r, c): float(s) for r, c, s in zip(rounds, carriers, contracts["strike_gbp_per_mwh_2024"])}
    table = StrikeTable(strikes, PRICE_YEAR, {}, {})
    units = pd.DataFrame({"support_type": "CfD", "carrier": carriers, "bus": "", "cfd_round": rounds},
                         index=pd.Index(carriers))
    live = pd.to_numeric(contracts.get("live_share", pd.Series(1.0, index=contracts.index)),
                         errors="coerce").fillna(1.0).to_numpy()
    energy = pd.DataFrame(dispatch_mwh[contracts["unit"].tolist()].to_numpy(dtype=float) * live[None, :],
                          index=index, columns=carriers)
    overrides = ({c: baseload for c, b in zip(carriers, contracts["reference_price"]) if b == "baseload_seasonal"}
                 if baseload is not None else None)
    network = SimpleNamespace(generators=units, snapshots=index, generators_t=SimpleNamespace(p=energy),
                              snapshot_weightings=pd.DataFrame({"generators": 1.0}, index=index))
    details = compute_cfd_payments(network, table, reference_price=reference_price,
                                   negative_price_rule="gb_contract_terms", reference_price_overrides=overrides)
    payment = details.set_index("generator")["net_cfd_cost"].reindex(carriers).fillna(0.0)
    rows = []
    hourly = reference_price.astype(float).to_numpy()
    for i, (_, c) in enumerate(contracts.iterrows()):
        ref = baseload.to_numpy() if c["reference_price"] == "baseload_seasonal" else hourly
        eligible = negative_price_eligibility(ref, [rounds[i]], "gb_contract_terms")[:, 0]
        e = energy[carriers[i]].to_numpy()
        paid = np.where(eligible, e, 0.0)
        rows.append({"unit": c["unit"], "settled_mwh": float(paid.sum()), "unpaid_mwh": float((e - paid).sum()),
                     "reference_captured_gbp_per_mwh": float((ref * paid).sum() / paid.sum()) if paid.sum() else np.nan,
                     "strike_gbp_per_mwh_2024": float(c["strike_gbp_per_mwh_2024"]),
                     "payment_gbp": float(payment[carriers[i]])})
    return pd.DataFrame(rows)


def strike_table_for_bill(strikes: pd.DataFrame, known: pd.DataFrame | None = None, variant: str = "central",
                          beccs: pd.DataFrame | None = None):
    """A ``cfd_strike_table.StrikeTable`` for a model year's bill projector (the hook for Y13/Y14).

    Keys are ``(PAR7_<build year>, technology)`` for the modelled vintages and the AR7,
    AR7a and HPC keys for the live known contracts, all in real 2024 GBP, so a CfD unit
    labelled ``cfd_round = PAR7_<year>`` settles through
    ``revenue_tracking.compute_cfd_payments`` unchanged. There is no fallback key (the
    2030 chain's modelled AR8 round): an unlabelled 2050 CfD unit is an error to find,
    not a strike to guess.

    Task SETTLE2050: built from ``contract_book_2050.strike_table_frame`` (the file the
    2050 bill reads) and indexed exactly as ``load_strike_table`` indexes it, so the table
    in memory and the table on disk are one table. A round's known contracts collapse to
    their live-MW-weighted strike per ``(round, technology)`` as the 2030 loader collapses
    delivery years: AR7's English, Scottish and floating units are one ``(AR7, offshore)``
    entry, because a network unit's carrier cannot tell them apart.

    Args:
        strikes: :func:`modelled_strikes` output.
        known: :func:`known_contracts` output (none: the modelled vintages only).
        variant: Which strike variant.
        beccs: BECCS electricity-leg strikes (``support_regime_2050.beccs_strike`` rows).

    Returns:
        The table.
    """
    from scripts.public_ownership.contract_book_2050 import strike_table_frame

    empty = pd.DataFrame(columns=["contract_id", "phase", "cfd_round", "technology", "region", "scheme",
                                  "live_share", "live_mw", "strike_gbp_per_mwh_2012", "source"])
    frame = strike_table_frame(strikes, known if known is not None else empty, variant=variant, beccs=beccs)
    return strike_table_from_frame(frame)


def strike_table_from_frame(frame: pd.DataFrame):
    """A ``StrikeTable`` from a strike table frame (``contract_book_2050.strike_table_frame`` columns).

    Indexed exactly as ``load_strike_table`` indexes the file: a ``(round, technology)``'s rows
    collapse to their capacity-weighted strike, in real 2024 GBP. Shared by
    :func:`strike_table_for_bill` and the calibrated support labels (FES 2050 decision 49 (b),
    ``contract_book_2050.basis_strike_frame``), so both index one way.
    """
    from scripts.market.cfd_strike_table import StrikeTable

    frame = frame.copy()
    weight = pd.to_numeric(frame["capacity_mw"], errors="coerce")
    frame["weight"] = weight.where(weight.gt(0), 1.0).fillna(1.0)
    frame["weighted"] = frame["strike_price_gbp_per_mwh_2012"] * frame["weight"]
    grouped = frame.groupby(["cfd_round", "technology"])[["weighted", "weight"]].sum()
    table = {(str(r), str(tech)): float(g.weighted / g.weight) * CPI_2012_TO_2024 for (r, tech), g in grouped.iterrows()}
    sources = {f"{r}_{tech}": sorted(set(g["source"].astype(str))) for (r, tech), g in frame.groupby(["cfd_round", "technology"])}
    indexation = {"from_price_year": 2012, "to_price_year": PRICE_YEAR, "from_index": 96.1, "to_index": 133.9,
                  "factor": CPI_2012_TO_2024, "index": "ONS CPI D7BT annual average"}
    return StrikeTable(table, PRICE_YEAR, indexation, sources)
