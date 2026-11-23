"""Pathway-lite 2030 to 2050: the public ownership benefits year by year, without solving a network.

FES 2050 plan, task Y15 (user, 3 October 2026). The two solved snapshots each measure one
calendar year: 2030 (run group ``cp30_2030_cp30_network``, CP30) and 2050 (``fes_ht_2050``, FES 2025
Holistic Transition). The policy case is the cumulative benefit over 2030 to 2050, and the
two years show different mechanisms (P3's legacy-rent capture fades as the RO and old CfDs
expire; P2's financing saving grows with the stock of state-financed capital), so neither
year stands for the twenty-one.

**Three mechanisms, each with its own year rule.**

1. **Financing (exact from the vintages).** Every new-build row of the 2050 vintaged
   register (``fes_ht_2050_asset_finance_register.parquet``) carries its build year, its
   capital, its contract and, per case, the rate and the state share the finance layer gave
   it (``calculate_financing_scenarios``, ``levers.py``; nothing is re-derived here). For a
   year ``y`` its capital charge is the level annuity its recovery rule pays, weighted by the
   share of ``y`` it is paid:

   * a row on a live contract in 2050 (decision 41: ``post_ar7_modelled``, ``dpa_modelled``,
     ``known_contract``): the register's own 2050 annuity divided by its 2050 live share, paid
     in ``y`` at ``capacity_market_2050.live_fraction(build year, term, y)`` (the register's
     rule); in 2050 at the register's live share, so the 2050 row is the register's;
   * a row whose contract has ended by 2050 (``contract_expired_merchant``): capital less the
     merchant tail's value, recovered over the term at the row's rate (decision 41 (a)), the
     term being 15 years for CfD technologies built before 2031 (AR1 to AR6) and the modelled
     term otherwise; zero in 2050, as in the register;
   * every other row (merchant storage, hydro, biomass, energy from waste, Sizewell C's RAB):
     the capital the register annuitises, recovered as a level annuity over the economic life.
     The register's 2050 snapshot instead annuitises it over the life remaining in 2050 (the
     convention decision 43 questions), so these rows differ from the register in 2050; the
     difference is reported, not forced;
   * 2025 to 2050 cohorts that retire before 2050 (25.2 GW of lithium batteries built 2025 to
     2035), which no 2050 register row holds: from the stock-flow
     (``fes_ht_2050_fleet_vintage_flows.csv`` less ``fes_ht_2050_fleet_vintages.csv``), costed
     from the vintage table and the case's battery rates, share and overhead in the register.

   Corporation tax forgone is the register's per row on live contracts and
   ``finance.embedded_corporation_tax`` on the same wedges elsewhere; the state's overhead
   (OH1) is the register's per row, paid every year the row runs. The financing saving is
   private less case, by technology group.

2. **Legacy rent and the acquisition (P3).** The acquired fleet's cash by year is the 2030
   legacy valuation's profile (``cp30_2030_cp30_network_legacy_cash_flow_profile.csv``: net operating cash,
   CfD top-up and ROC income on their own expiry dates); the purchase is the 2030 annuity
   carried forward (``fes_ht_2050_carried_acquisition.json``, each asset over its remaining life, to the record's
   ``last_payment_year``: 2087 since the per-asset terms of F28); the state's
   overhead on the acquired fleet follows the survival schedule there; the corporation tax the
   former owners no longer pay is interpolated between the snapshots (it is on the ledger's
   as-new base, not the cash).

3. **Market and other.** Everything else each snapshot ledger books (wholesale, balancing,
   Capacity Market, capture-driven revenue, interconnectors, the dispatch institution, the
   state's deficits on non-self-financing plant, ...) is interpolated linearly between the two
   snapshots, on the private-versus-case difference, never the levels. Declared.

**P2 is on its own bid stack** (decision 46 (a), user, 3 October 2026): P2 and its L4 shares
are the state-bids cases (``public_ownership@deployment_share_<pct>_state_bids``: the state's
plant bids at cost in the balancing mechanism, decision 14), anchored in 2030 on the CP30 P2
bid-stack case (``public_ownership@state_bids_at_cost``, run group ``cp30_2030_cp30_network_public_bids_p2``).

**Beyond 2050** (decision 46 (b)): a separate line per case, never added to the 2030-2050
headline. It is the financing saving still to come on plant built by 2050, each row on its own
recovery rule after 2050 (live contracts to the end of their term, merchant and RAB rows to the
end of their life; expired contracts have none left), and P3's purchase annuity from 2051 to its
last payment (the carried record's ``last_payment_year``). Both are held at 2050 levels (the 2050 fleet, rates and shares; no build
after 2050; per household on the 2050 key) and discounted to 2027 like the headline. The
acquired fleet's cash after 2050 is counted beside it (user, 13 Nov 2026: both sides or neither).

**More solved years (user, 10 Nov 2026).** ``--snapshot YEAR=LEDGER`` adds a solved year between 2030
and 2050 (2040 when it is run): its ledger is one more anchor, so every interpolated line runs
piecewise-linear through it and the anchor lines close it to GBP 1 m as they close 2030 and 2050.
A middle year's cases are the 2050 case ids (a FES study). The 2030 anchor is the report group's
ledger (CENTRAL network, ``cp30_2030_no_acceleration``, since 10 Nov 2026; the LOWER network before).

**Anchoring.** Each snapshot ledger's case is split into the same three mechanisms by its own
part labels (:func:`snapshot_mechanisms`). Where the modelled financing or legacy line differs
from the snapshot's in a snapshot year, the difference is an explicit *anchor* line,
interpolated linearly between the two years, so the 2030 and 2050 rows reproduce each ledger to
GBP 1 m while the gap stays visible (the 2030 snapshot is CP30, not FES HT; it has no vintaged
register; its new build is on economic-life levelising). Cases without a 2030 snapshot (P2 at
the L4 deployment shares) take ``share x P2`` in 2030 (decision 17: the same share of every
vintage), declared.

**Parties.** Consumers, the Exchequer, investors and the resource saving, closing every year as
the ledgers do: consumers + Exchequer + investors = resource saving. The financing line's
capital and overhead go to consumers (bill-payers first, the base of every case since
30 September 2026), its corporation tax to the Exchequer; the legacy line's margin and annuity
are between consumers and the former owners (cash basis).

**Money.** System GBP real 2024 (the model's price year, both ledgers); per household on the
ledgers' own key (GBP real 2025 per 3,100 kWh, the system figure x 1.0384 x 3.1 MWh / final
demand, final demand interpolated between the snapshots) and per domestic customer account
(FES 2025 HT ``Dem_BB001a`` by year). NPVs at the HM Treasury Green Book STPR (3.5% real) with
3.0% and 5.0% sensitivities, base year 2027 (2027 undiscounted).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config_loader import read_config

FIRST_YEAR, LAST_YEAR = 2030, 2050
#: User decision (10 Nov 2026): the cumulative view starts in 2027, before the first solved year (2030). Before
#: 2030 the financing line runs year by year from the build years (state plant committed after the law and open
#: by then), and the purchase and every market-dependent line are zero: they start with the 2030 transfer.
START_YEAR = 2027
YEARS = tuple(range(START_YEAR, LAST_YEAR + 1))
SNAPSHOT_YEARS = (FIRST_YEAR, LAST_YEAR)
#: Present values are taken to the view's first year.
BASE_YEAR = START_YEAR
#: HM Treasury Green Book STPR (real), with the sensitivities the brief names.
DISCOUNT_RATES = {"stpr_3p5": 0.035, "low_3p0": 0.03, "high_5p0": 0.05}
CENTRAL_RATE = "stpr_3p5"
REPORT_YEARS = (2027, 2030, 2035, 2040, 2045, 2050)
ANCHOR_TOLERANCE_GBP = 1e6

PARTIES = ("consumers", "exchequer", "investors", "resource")
MECHANISMS = ("financing", "legacy", "market_and_other")
#: Line -> mechanism. The anchors carry the snapshot gap of the modelled lines.
LINES = {"financing_vintage": "financing", "financing_anchor": "financing",
         "legacy_modelled": "legacy", "legacy_anchor": "legacy",
         "market_and_other_interpolated": "market_and_other"}

#: Decision 46 (a): the 2030 P2 bid-stack case (run group ``cp30_2030_cp30_network_public_bids_p2``) anchors P2 in 2030.
P2_2030_STATE_BIDS = "public_ownership@state_bids_at_cost"
#: The cases reported: id -> (label, the 2030 source: a 2030 case id, or (P2 id, share)). P2 and its
#: L4 shares are on their own balancing bid stacks (decision 46 (a)).
CASES = {
    "public_control": ("P1", "public_control"),
    "public_ownership@deployment_share_100_state_bids": ("P2 (100%)", P2_2030_STATE_BIDS),
    "public_ownership@deployment_share_25_state_bids": ("P2 25%", (P2_2030_STATE_BIDS, 0.25)),
    "public_ownership@deployment_share_50_state_bids": ("P2 50%", (P2_2030_STATE_BIDS, 0.50)),
    "public_ownership@deployment_share_75_state_bids": ("P2 75%", (P2_2030_STATE_BIDS, 0.75)),
    "public_ownership@fully_nationalised_market_value": ("P3 I1", "public_ownership@fully_nationalised_market_value"),
    "public_ownership@fully_nationalised_market_value_central_dispatch":
        ("P3 I3", "public_ownership@fully_nationalised_market_value_central_dispatch"),
}
PRIVATE = "private"
#: The legacy valuation scope groups the acquisition buys ('excluded' is never bought).
ACQUIRED_SCOPE_GROUPS = ("valued", "thermal", "storage")
#: The P3 case whose purchase the pathway carries (its settings decide which carriers the purchase leaves out).
P3_CARRIED_CASE = "public_ownership@fully_nationalised_market_value"
#: Carriers the 2030 valuation profiles inside a bought scope group although the purchase may leave them private
#: (user decision 10 Nov 2026: energy-from-waste stays with its owners unless ``efw_acquisition: included``).
EFW_CARRIERS = ("waste_to_energy",)


def purchase_excluded_carriers(lever: Mapping) -> tuple[str, ...]:
    """Carriers in a bought scope group the case's purchase leaves private (review finding 9, 13 Nov 2026: the
    pathway counted energy-from-waste's cash and overhead, about GBP 1.6 bn a year in 2040, which S3 never buys)."""
    return () if str(lever.get("efw_acquisition", "excluded")) == "included" else EFW_CARRIERS

TECH_GROUPS = {
    "Offshore wind": "offshore_wind", "Floating offshore wind": "offshore_wind", "Onshore wind": "onshore_wind",
    "Solar PV": "solar", "Large-scale nuclear": "nuclear", "Small modular reactors (SMRs)": "nuclear",
    "Gas generation - CCUS - mature": "dispatchable_low_carbon", "Hydrogen CCHT": "dispatchable_low_carbon",
    "Hydrogen OCHT": "dispatchable_low_carbon", "Biomass - with CCUS - mature": "dispatchable_low_carbon",
    "Lithium batteries": "storage", "Liquid air energy storage": "storage", "Compressed air energy storage": "storage",
    "Pumped hydro energy storage (PHES)": "storage",
}
OTHER_GROUP = "other"
#: AR1 to AR6 CfDs run 15 years; a CfD-technology row built before 2031 holds one of them.
PRE_AR7_CFD_TERM_YEARS = 15

# Ledger part labels -> mechanism (matched on the label's start; everything else is market and other).
FINANCING_RESOURCE_PARTS = (
    "capital charges at the financing party's rate", "O&M and the state's overhead",
    "less the corporation tax inside the private rate", "new-build capital: private WACC to the public rate",
    "state operating overhead on owned plant",
)
LEGACY_RESOURCE_PARTS = ("acquisition advisory and transaction costs",)
LEGACY_INVESTOR_PARTS = (
    "legacy operating margin the owners give up", "legacy acquisition annuity",
    "interconnector GB halves: acquisition annuity", "corporation tax the former owners no longer pay on legacy plant",
    "acquisition annuity paid from general taxation", "cancelled contracts: residual",
    # 10 Nov 2026 (user): the acquired fleet's other income, which the ledgers book under their own headings
    # but the 2030 valuation's cash profile (the modelled line) holds inside the fleet's margin. Left in
    # market-and-other they made the legacy anchor a constant -GBP 5.3-5.5 bn a year (2030, 2050) and the
    # existing-fleet line negative from 2034. Interconnector rent stays market: the GB halves' rent is not in
    # the fleet's cash profile (their annuity is).
    "Capacity Market agreements of acquired plant", "EfW gate fees the former owners no longer receive",
    "ancillary-service revenue the former owners no longer receive",
    "life-extension top-up no longer paid to the owners of acquired plant",
    "carbon-removal CfD the former owners of acquired BECCS no longer receive",
    "battery revenue top-up no longer paid to the owners of acquired batteries",
    "LDES floor top-up no longer paid to the stores' owners",
    "battery sub-hourly revenue the former owners no longer receive",
)
OVERHEAD_RESOURCE_PART = "state operating overhead on owned plant"

LEDGER_TOTALS = {"consumers": "consumer_change_vs_private_gbp", "exchequer": "exchequer_net_vs_private_gbp",
                 "investors": "investor_change_vs_private_gbp", "resource": "resource_saving_vs_private_gbp"}


class PathwayError(ValueError):
    """An input the pathway cannot use."""


# ── pure arithmetic ──────────────────────────────────────────────────────────

def crf(rate, years):
    """Capital recovery factor over arrays (``finance.capital_recovery_factor``; 0 where years <= 0)."""
    rate = np.asarray(rate, dtype=float)
    years = np.asarray(years, dtype=float)
    out = np.zeros(np.broadcast(rate, years).shape)
    ok = years > 0
    r = np.broadcast_to(rate, out.shape)[ok]
    n = np.broadcast_to(years, out.shape)[ok]
    small = np.abs(r) < 1e-12
    growth = np.power(1.0 + r, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = np.where(small, 1.0 / n, r * growth / (growth - 1.0))
    return out


def live_fraction_matrix(build_year, term_years, years: Sequence[int]) -> np.ndarray:
    """``capacity_market_2050.live_fraction`` over rows x years, and zero before the build year.

    The register's rule: a contract starts with the plant (the start year counts in full)
    and expires evenly through the year beginning ``build year + term``.
    """
    b = np.asarray(build_year, dtype=float)[:, None]
    t = np.asarray(term_years, dtype=float)[:, None]
    y = np.asarray(list(years), dtype=float)[None, :]

    def integral(x):
        return np.where(x <= 0, 0.0, np.where(x <= 1, x * x / 2.0, x - 0.5))

    end = b + t - y
    share = np.clip(integral(end + 1.0) - integral(end), 0.0, 1.0)
    return np.where((y >= b) & (t > 0), share, 0.0)


def alive_matrix(build_year, life_years, years: Sequence[int]) -> np.ndarray:
    """1 in the years ``build year <= y < build year + life``, else 0 (rows x years)."""
    b = np.asarray(build_year, dtype=float)[:, None]
    life = np.asarray(life_years, dtype=float)[:, None]
    y = np.asarray(list(years), dtype=float)[None, :]
    return ((y >= b) & (y < b + life)).astype(float)


def interpolate(values: Mapping[int, float], years: Sequence[int] = YEARS, *,
                zero_before_first: bool = False) -> pd.Series:
    """Linear in the year between the snapshot values (flat outside them; zero before the first snapshot when
    ``zero_before_first``: a line that starts with the 2030 transfer)."""
    known = sorted((int(k), float(v)) for k, v in values.items())
    if not known:
        raise PathwayError("nothing to interpolate")
    xs, ys = zip(*known)
    out = pd.Series(np.interp(list(years), xs, ys), index=list(years), dtype=float)
    if zero_before_first:
        out[out.index < xs[0]] = 0.0
    return out


def npv(by_year: Mapping[int, float] | pd.Series, rate: float, base_year: int = BASE_YEAR) -> float:
    """Present value at ``base_year`` of a real annual stream (the base year undiscounted)."""
    series = pd.Series(by_year, dtype=float)
    years = np.asarray(series.index, dtype=float)
    return float((series.to_numpy() / np.power(1.0 + float(rate), years - float(base_year))).sum())


def technology_group(technology) -> str:
    """The reporting group of a stock-flow technology."""
    return TECH_GROUPS.get(str(technology), OTHER_GROUP)


# ── financing from the vintages ──────────────────────────────────────────────

FINANCE_COLUMNS = ("annual_capital_case_gbp", "annual_capital_private_gbp", "tax_forgone_gbp", "overhead_gbp")


def schedule_financing(rows: pd.DataFrame, years: Sequence[int] = YEARS) -> pd.DataFrame:
    """The financing lines by year and technology group from per-row level values and their weights.

    Args:
        rows: One row per cohort row: ``tech_group``, ``build_year``, ``kind`` (``term``,
            ``expired`` or ``level``), ``term_years``, ``life_years``, ``live_share_2050``
            (term rows) and the level (fully live) annual values ``annual_capital_case_gbp``,
            ``annual_capital_private_gbp``, ``tax_forgone_gbp`` and ``overhead_gbp``.
        years: The years to schedule.

    Returns:
        Long frame: ``year``, ``tech_group``, ``capital_saving_gbp`` (private less case),
        ``tax_forgone_gbp``, ``overhead_gbp``.
    """
    years = list(years)
    if rows.empty:
        return pd.DataFrame(columns=["year", "tech_group", "capital_saving_gbp", "tax_forgone_gbp", "overhead_gbp"])
    kind = rows["kind"].astype(str).to_numpy()
    b = rows["build_year"].to_numpy(dtype=float)
    contract = live_fraction_matrix(b, rows["term_years"].fillna(0).to_numpy(dtype=float), years)
    level = alive_matrix(b, rows["life_years"].fillna(0).to_numpy(dtype=float), years)
    weight = np.where(np.isin(kind, ["term", "expired"])[:, None], contract, level)
    if LAST_YEAR in years and "live_share_2050" in rows.columns:
        # A live contract pays its register share in 2050, so that year is the register's.
        j = years.index(LAST_YEAR)
        term = kind == "term"
        weight[term, j] = rows["live_share_2050"].to_numpy(dtype=float)[term]
    runs = (b[:, None] <= np.asarray(years, dtype=float)[None, :]) & (
        np.asarray(years, dtype=float)[None, :] < b[:, None] + rows["life_years"].fillna(np.inf).to_numpy(dtype=float)[:, None])
    saving = (rows["annual_capital_private_gbp"].to_numpy(float) - rows["annual_capital_case_gbp"].to_numpy(float))
    parts = {"capital_saving_gbp": weight * saving[:, None],
             "tax_forgone_gbp": weight * rows["tax_forgone_gbp"].to_numpy(float)[:, None],
             "overhead_gbp": runs * rows["overhead_gbp"].to_numpy(float)[:, None]}
    groups = rows["tech_group"].astype(str).to_numpy()
    out = []
    for g in sorted(set(groups)):
        mask = groups == g
        for j, y in enumerate(years):
            out.append({"year": y, "tech_group": g, **{k: float(v[mask, j].sum()) for k, v in parts.items()}})
    return pd.DataFrame(out)


def financing_parties(capital_saving: float, tax_forgone: float, overhead: float) -> dict:
    """The financing line by party: the saving and the overhead to consumers, the tax to the Exchequer."""
    consumers = float(capital_saving) - float(overhead)
    exchequer = -float(tax_forgone)
    return {"consumers": consumers, "exchequer": exchequer, "investors": 0.0, "resource": consumers + exchequer}


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def embedded_tax(capex, rate, wedge, years):
    """``finance.embedded_corporation_tax`` over arrays: capex x [CRF(r) - CRF(r - wedge)]."""
    capex = np.asarray(capex, dtype=float)
    return np.where(np.asarray(years, float) > 0, capex * (crf(rate, years) - crf(np.asarray(rate, float)
                                                                                 - np.asarray(wedge, float), years)), 0.0)


REGISTER_COLUMNS = ("case_id", "asset_id", "component_type", "finance_vintage", "stock_flow_technology",
                    "finance_technology", "build_year", "capacity_mw", "energy_capacity_mwh", "support_type",
                    "support_basis", "cfd_live_share", "cfd_contract_id", "state_owned", "ownership_share", "wacc",
                    "remaining_life_years", "effective_capex_power_gbp_per_kw", "annual_capital_recovery_gbp",
                    "annual_state_overhead_gbp", "corporation_tax_forgone_gbp", "annual_required_revenue_gbp",
                    # Decision 66: the merchant level-annuity rows need each row's age and economic life
                    # (vintage_capital_2050.merchant_level_rows), the life costed as the register costs it.
                    "model_year", "commissioning_year", "cost_technology", "cost_build_year")


def with_economic_life(rows: pd.DataFrame) -> pd.DataFrame:
    """``rows`` with ``economic_life_years`` from the vintage cost table, keyed as the register costs a row.

    The register does not store the life it costed a row on; ``calculate_financing_scenarios`` merges
    the vintage table on the row's cost technology (else its finance technology) and cost build year.
    A row whose technology the table does not hold gets NaN, as the register's left merge gives it.
    """
    if "economic_life_years" in rows.columns:
        return rows
    from scripts.public_ownership import vintage_costs as vc

    names = (rows["cost_technology"].where(rows["cost_technology"].notna(), rows["finance_technology"])
             if "cost_technology" in rows.columns else rows["finance_technology"])
    years = pd.to_numeric(rows["cost_build_year"] if "cost_build_year" in rows.columns else rows["build_year"],
                          errors="coerce")
    cache: dict = {}
    lives = []
    for name, year in zip(names.astype(object), years):
        if pd.isna(name) or pd.isna(year):
            lives.append(np.nan)
            continue
        key = (str(name), int(min(max(float(year), vc.FIRST_BUILD_YEAR), vc.LAST_BUILD_YEAR)))
        if key not in cache:
            try:
                cache[key] = float(vc.costs_for(*key)["economic_life_years"])
            except KeyError:
                cache[key] = np.nan
        lives.append(cache[key])
    return rows.assign(economic_life_years=lives)


def register_rows(case_rows: pd.DataFrame, private_rows: pd.DataFrame, *, basis: Mapping, wedges: Mapping,
                  cepa_jv: bool) -> pd.DataFrame:
    """Per new-build register row of one case: its recovery rule and level annual values.

    Args:
        case_rows, private_rows: The case's and the private case's register rows
            (:data:`REGISTER_COLUMNS`).
        basis: The study's capital basis (``calculate_financing_scenarios.vintage_capital_options``).
        wedges: ``{canonical finance technology: {"private": w, "public_coordination": w}}``
            (``finance.corporation_tax_wedges``).
        cepa_jv: The case's state rows are a taxable joint venture (P1's ``cepa_decomposition``),
            so only the change in tax is forgone.

    Returns:
        One row per new-build asset row: ``asset_id``, ``stock_flow_technology``,
        ``tech_group``, ``build_year``, ``capacity_mw``, ``kind``, ``term_years``,
        ``life_years``, ``live_share_2050``, the level values of :data:`FINANCE_COLUMNS`, and the
        register's own 2050 values (``register_capital_case_2050_gbp``, ...) for the check.
    """
    from scripts.public_ownership.calculate_financing_scenarios import canonical_technology
    from scripts.public_ownership.fleet_vintages import CFD_TECHNOLOGIES, POST_AR7_FIRST_BUILD_YEAR
    from scripts.public_ownership.vintage_capital_2050 import (DPA_TECHNOLOGY_SCHEME, SUPPORT_BASIS_EXPIRED,
                                                                register_capital_terms, tail_value_vector)

    keep = ["asset_id", "wacc", "annual_capital_recovery_gbp", "corporation_tax_forgone_gbp"]
    c = case_rows.loc[case_rows["finance_vintage"].astype(str).eq("new_build")].copy()
    # The private twin is the same asset privately owned, whatever its private vintage (15 Nov 2026: S3's own new
    # capacity built after the purchase is legacy or in construction in the private case).
    p = private_rows[keep]
    if c["asset_id"].duplicated().any():
        raise PathwayError("a case repeats a register row")
    c["asset_id"] = c["asset_id"].astype(str)
    p = p.assign(asset_id=p["asset_id"].astype(str)).rename(
        columns={"wacc": "wacc_private", "annual_capital_recovery_gbp": "capital_private_2050",
                 "corporation_tax_forgone_gbp": "_tax_private"})
    m = c.merge(p, on="asset_id", how="left", validate="one_to_one")
    if m["wacc_private"].isna().any() and m["wacc"].notna().any():
        missing = m.loc[m["wacc_private"].isna() & m["wacc"].notna(), "asset_id"].head(3).tolist()
        if missing:
            raise PathwayError(f"case rows without a private twin, e.g. {missing}")
    m = with_economic_life(m.loc[pd.to_numeric(m["wacc"], errors="coerce").notna()].reset_index(drop=True))
    n = len(m)
    b = pd.to_numeric(m["build_year"], errors="coerce").to_numpy(dtype=float)
    mw = _num(m, "capacity_mw").to_numpy()
    r_c = pd.to_numeric(m["wacc"], errors="coerce").to_numpy(dtype=float)
    r_p = pd.to_numeric(m["wacc_private"], errors="coerce").to_numpy(dtype=float)
    cap_c = _num(m, "annual_capital_recovery_gbp").to_numpy()
    cap_p = _num(m, "capital_private_2050").to_numpy()
    tax50 = _num(m, "corporation_tax_forgone_gbp").to_numpy()
    remaining = _num(m, "remaining_life_years").to_numpy()
    life = remaining + (float(LAST_YEAR) - b)
    technology = m["stock_flow_technology"].astype(str).to_numpy()
    support_basis = m["support_basis"].astype(object).fillna("").astype(str).to_numpy()

    terms = register_capital_terms(m, basis, insurance_config=basis.get("insurance_config"),
                                   learning=bool(basis.get("learning", False)))
    term_row = terms["term_row"]
    expired = support_basis == SUPPORT_BASIS_EXPIRED
    # The expired rows' terms and tails: the same machinery on the rows relabelled as live.
    relabel = m.copy()
    relabel["support_basis"] = np.where(expired, "post_ar7_modelled", relabel["support_basis"].astype(object))
    relabel["support_type"] = np.where(expired, np.where(np.isin(technology, list(DPA_TECHNOLOGY_SCHEME)), "DPA", "CfD"),
                                       relabel["support_type"].astype(object))
    relabel["cfd_live_share"] = np.where(expired, 1.0, pd.to_numeric(relabel["cfd_live_share"], errors="coerce"))
    relabel["cfd_contract_id"] = np.where(expired, None, relabel["cfd_contract_id"].astype(object))
    eterms = register_capital_terms(relabel, basis, insurance_config=basis.get("insurance_config"),
                                    learning=bool(basis.get("learning", False)))
    pre_ar7 = expired & np.isin(technology, list(CFD_TECHNOLOGIES)) & (b < POST_AR7_FIRST_BUILD_YEAR)
    eterm = np.where(pre_ar7, np.minimum(float(PRE_AR7_CFD_TERM_YEARS), eterms["life_years"]), eterms["term_years"])

    kind = np.where(term_row, "term", np.where(expired, "expired", "level"))
    term_years = np.where(term_row, terms["term_years"], np.where(expired, eterm, np.nan))
    live50 = np.where(term_row, terms["live_share"], np.nan)
    if (expired & ~np.isfinite(eterm)).any():
        raise PathwayError("an expired row has no contract term")
    if expired.any():
        live_check = live_fraction_matrix(b[expired], eterm[expired], [LAST_YEAR])[:, 0]
        if np.abs(live_check).max() > 1e-12:
            raise PathwayError("an expired row would still be live in 2050 on its term")

    capex_power = _num(m, "effective_capex_power_gbp_per_kw").to_numpy()
    gross = mw * 1000.0 * capex_power
    # Level values.
    a_c, a_p = np.zeros(n), np.zeros(n)
    # Term rows: the register's 2050 annuity per unit of live share.
    with np.errstate(divide="ignore", invalid="ignore"):
        a_c[term_row] = np.where(live50[term_row] > 0, cap_c[term_row] / live50[term_row], 0.0)
        a_p[term_row] = np.where(live50[term_row] > 0, cap_p[term_row] / live50[term_row], 0.0)
    # Expired rows: capex less the tail at each party's rate over the term.
    if expired.any():
        e = np.flatnonzero(expired)
        margin, elife, et = eterms["tail_margin_gbp_per_kw_year"][e], eterms["life_years"][e], eterm[e]
        tail_c = tail_value_vector(margin, elife, et, r_c[e]) * mw[e] * 1000.0
        tail_p = tail_value_vector(margin, elife, et, r_p[e]) * mw[e] * 1000.0
        a_c[e] = np.maximum(gross[e] - tail_c, 0.0) * crf(r_c[e], et)
        a_p[e] = np.maximum(gross[e] - tail_p, 0.0) * crf(r_p[e], et)
    # Level rows: the register's capital (backed out of the annuity it charged) over the life. Decision 66
    # (merchant_capital: level_annuity): a merchant row's annuity runs over its economic life, as
    # calculate_financing_scenarios charges it; any other level row over its remaining life.
    lvl = ~term_row & ~expired
    annuity_years = (np.where(terms["level_row"], terms["level_life_years"], remaining) if "level_row" in terms
                     else remaining)
    capex_level = np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = crf(r_p[lvl], annuity_years[lvl])
        capex_level[lvl] = np.where(f > 0, cap_p[lvl] / f, 0.0)
        f_c = crf(r_c[lvl], annuity_years[lvl])
        capex_case = np.where(f_c > 0, cap_c[lvl] / f_c, 0.0)
    if lvl.any() and np.nanmax(np.abs(capex_case - capex_level[lvl]) / np.maximum(capex_level[lvl], 1.0)) > 1e-6:
        raise PathwayError("a level row's capital differs between the case and private (a capex lever?)")
    a_c[lvl] = capex_level[lvl] * crf(r_c[lvl], life[lvl])
    a_p[lvl] = capex_level[lvl] * crf(r_p[lvl], life[lvl])

    # Corporation tax forgone, level values.
    owned = m["state_owned"].astype(str).str.lower().isin(["true", "1"]).to_numpy() | (_num(m, "ownership_share").to_numpy() > 0)
    tech_key = m["finance_technology"].map(lambda v: canonical_technology(v) if pd.notna(v) else "").to_numpy()
    w_p = np.array([float((wedges.get(t) or {}).get("private", np.nan)) for t in tech_key])
    w_pc = np.array([float((wedges.get(t) or {}).get("public_coordination", np.nan)) for t in tech_key])
    tax = np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        tax[term_row] = np.where(live50[term_row] > 0, tax50[term_row] / live50[term_row], 0.0)
    for mask, base, years in ((expired, None, eterm), (lvl, capex_level, life)):
        idx = np.flatnonzero(mask & owned & np.isfinite(w_p))
        if not len(idx):
            continue
        if base is None:
            margin = eterms["tail_margin_gbp_per_kw_year"][idx]
            tail_p = tail_value_vector(margin, eterms["life_years"][idx], eterm[idx], r_p[idx]) * mw[idx] * 1000.0
            capex_net = np.maximum(gross[idx] - tail_p, 0.0)
        else:
            capex_net = base[idx]
        t = embedded_tax(capex_net, r_p[idx], w_p[idx], years[idx])
        if cepa_jv:
            t = t - embedded_tax(capex_net, r_c[idx], np.nan_to_num(w_pc[idx]), years[idx])
        # S1/S2 audit (13 Nov 2026): the tax forgone is on the state's share of the row only (the register's own
        # term-row figure already is); a deployment share or a stake below one forgoes that much less.
        share = np.clip(np.nan_to_num(_num(m, "ownership_share").to_numpy()[idx]), 0.0, 1.0)
        tax[idx] = t * np.where(share > 0, share, 1.0)
    return pd.DataFrame({
        "asset_id": m["asset_id"].to_numpy(), "stock_flow_technology": technology,
        "tech_group": [technology_group(t) for t in technology], "build_year": b, "capacity_mw": mw,
        "ownership_share": _num(m, "ownership_share").to_numpy(), "kind": kind, "term_years": term_years,
        "life_years": life, "live_share_2050": live50, "rate_case": r_c, "rate_private": r_p,
        "annual_capital_case_gbp": a_c, "annual_capital_private_gbp": a_p, "tax_forgone_gbp": tax,
        "overhead_gbp": _num(m, "annual_state_overhead_gbp").to_numpy(),
        "register_capital_case_2050_gbp": cap_c, "register_capital_private_2050_gbp": cap_p,
        "register_tax_2050_gbp": tax50,
    })


def retired_cohort_rows(flows: pd.DataFrame, survivors: pd.DataFrame, register: pd.DataFrame, *,
                        wedges: Mapping, cepa_jv: bool, first_build_year: int = 2025,
                        technologies: Iterable[str] = ("Lithium batteries",),
                        legislated_year: int | None = None,
                        lead_years: Mapping[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-2025 cohorts that retire before 2050, which the 2050 register cannot hold.

    Gross additions (``fes_ht_2050_fleet_vintage_flows.csv``) less the 2050 survivors
    (``fes_ht_2050_fleet_vintages.csv``) by technology and build year. Each is costed from the
    vintage table (``vintage_costs.costs_for``) at the case's and private's rates for the same
    technology in the register, with the case's capacity-weighted state share and overhead per
    owned kW, as a level annuity over its economic life.

    Returns:
        ``(rows, omitted)``: rows as :func:`register_rows` returns them, and the retired cohorts
        of other technologies, left out and declared.
    """
    from scripts.public_ownership import vintage_costs as vc
    from scripts.public_ownership.calculate_financing_scenarios import canonical_technology

    adds = flows.loc[flows["year"] >= first_build_year].groupby(["technology", "year"])["gross_addition_mw"].sum()
    surv = survivors.loc[survivors["build_year"] >= first_build_year].groupby(["technology", "build_year"])["capacity_mw"].sum()
    surv.index = surv.index.set_names(["technology", "year"])
    gone = (adds - surv.reindex(adds.index).fillna(0.0)).clip(lower=0.0)
    gone = gone.loc[gone > 1e-6].reset_index(name="retired_mw")
    technologies = set(technologies)
    omitted = gone.loc[~gone["technology"].isin(technologies)].copy()
    out = []
    nb = register.loc[register["finance_vintage"].astype(str).eq("new_build")]
    for tech, cohort in gone.loc[gone["technology"].isin(technologies)].groupby("technology"):
        rows = nb.loc[nb["stock_flow_technology"].astype(str).eq(tech)]
        if rows.empty:
            omitted = pd.concat([omitted, cohort])
            continue
        mw = _num(rows, "capacity_mw")
        r_c = float(np.average(pd.to_numeric(rows["wacc"], errors="coerce"), weights=mw))
        r_p = float(np.average(pd.to_numeric(rows["wacc_private"], errors="coerce"), weights=mw))
        share = float((_num(rows, "ownership_share") * mw).sum() / mw.sum())
        owned_kw = (_num(rows, "ownership_share") * mw).sum() * 1000.0
        overhead_rate = float(_num(rows, "annual_state_overhead_gbp").sum() / owned_kw) if owned_kw > 0 else 0.0
        tkey = canonical_technology(rows["finance_technology"].dropna().astype(str).iloc[0])
        w = wedges.get(tkey) or {}
        # S1/S2 audit (13 Nov 2026): a cohort committed before the legislation year (build year less the carrier's
        # investment lead) is pre-law plant, private in every case, as the registers hold it.
        carrier = str(rows["carrier"].dropna().astype(str).iloc[0]) if "carrier" in rows.columns else ""
        lead = float((lead_years or {}).get(carrier, 0.0))
        for rec in cohort.itertuples(index=False):
            share_y = share if legislated_year is None or float(rec.year) - lead >= float(legislated_year) else 0.0
            cost = vc.costs_for(tech, int(min(max(rec.year, vc.FIRST_BUILD_YEAR), vc.LAST_BUILD_YEAR)))
            capex = float(rec.retired_mw) * 1000.0 * (float(cost["capex_power_gbp_per_kw"]) + float(
                np.nan_to_num(float(cost.get("capex_energy_gbp_per_kwh") or 0.0))) * float(
                np.nan_to_num(float(cost.get("duration_hours") or 0.0))))
            life = float(cost["economic_life_years"])
            tax = 0.0
            if share_y > 0 and "private" in w:
                tax = float(embedded_tax(capex, r_p, float(w["private"]), life))
                if cepa_jv:
                    tax -= float(embedded_tax(capex, r_c, float(w.get("public_coordination", 0.0)), life))
                tax *= share_y                     # the state's share only (S1/S2 audit, 13 Nov 2026)
            # A pre-law cohort is private: both parties pay the private rate, so it moves nothing.
            rate_case = r_c if share_y > 0 else r_p
            out.append({"asset_id": f"retired_cohort::{tech}::b{int(rec.year)}", "stock_flow_technology": tech,
                        "tech_group": technology_group(tech), "build_year": float(rec.year),
                        "capacity_mw": float(rec.retired_mw), "ownership_share": share_y, "kind": "level",
                        "term_years": np.nan, "life_years": life, "live_share_2050": np.nan, "rate_case": rate_case,
                        "rate_private": r_p, "annual_capital_case_gbp": capex * float(crf(rate_case, life)),
                        "annual_capital_private_gbp": capex * float(crf(r_p, life)), "tax_forgone_gbp": tax,
                        "overhead_gbp": float(rec.retired_mw) * 1000.0 * share_y * overhead_rate,
                        "register_capital_case_2050_gbp": 0.0, "register_capital_private_2050_gbp": 0.0,
                        "register_tax_2050_gbp": 0.0})
    return pd.DataFrame(out), omitted


def beyond_2050_years(rows: pd.DataFrame, last_payment_year: int | None = None) -> list[int]:
    """The years after 2050 in which a row built by 2050 still recovers capital, or an annuity is paid."""
    ends = []
    if len(rows):
        b = rows["build_year"].to_numpy(dtype=float)
        # The last year a row pays: a contract's expiry year (part-live), a level row's last year of life.
        life_end = b + rows["life_years"].fillna(0).to_numpy(dtype=float) - 1.0
        term_end = b + rows["term_years"].fillna(0).to_numpy(dtype=float)
        ends.append(float(np.nanmax(np.where(rows["kind"].astype(str).isin(["term", "expired"]), term_end, life_end))))
    if last_payment_year is not None:
        ends.append(float(last_payment_year))
    last = int(np.ceil(max(ends))) if ends else LAST_YEAR
    return list(range(LAST_YEAR + 1, max(last, LAST_YEAR) + 1))


def beyond_2050_financing(rows: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    """Decision 46 (b): the financing lines after 2050 on plant built by 2050, by year and technology group.

    :func:`schedule_financing` on the same rows past 2050: live contracts pay on the register's
    live-fraction rule to the end of their term, merchant and RAB rows to the end of their
    life, expired contracts nothing; the overhead runs while the row runs.
    """
    return schedule_financing(rows if len(years) else rows.iloc[0:0], years)


def beyond_2050_case(rows: pd.DataFrame, carried_case: Mapping | None, profile_cash: pd.DataFrame | None,
                     key_2050: float) -> tuple[dict, list[dict]]:
    """One case's beyond-2050 line: totals by line and party, and its long rows.

    Args:
        rows: The case's financing rows (:func:`register_rows` and the retired cohorts).
        carried_case: P3's carried purchase (``fes_ht_2050_carried_acquisition.json``), or None.
        profile_cash: :func:`legacy_cash_by_year` over the profile's own years (the memo), or None.
        key_2050: System GBP to GBP 2025 per 3,100 kWh household, held at its 2050 value.

    Returns:
        ``(entry, long rows)``; ``entry["parties"][party]["total"]`` is :func:`totals` of the line.
    """
    last_payment = int(carried_case["last_payment_year"]) if carried_case else None
    years_after = beyond_2050_years(rows, last_payment)
    fin = beyond_2050_financing(rows, years_after)
    by_year = fin.groupby("year")[["capital_saving_gbp", "tax_forgone_gbp", "overhead_gbp"]].sum().reindex(
        years_after).fillna(0.0) if len(fin) else pd.DataFrame(
        0.0, index=years_after, columns=["capital_saving_gbp", "tax_forgone_gbp", "overhead_gbp"])
    annuity = (annuity_by_year(carried_case, years_after) if carried_case
               else pd.Series(0.0, index=years_after, dtype=float))
    # User, 13 Nov 2026 (results review, finding 2): count both sides after 2050. The acquired fleet's own cash
    # (the 2030 valuation's profile, to the last annuity payment) is counted beside the annuity, on the legacy
    # line's cash basis: the former owners give it up, so it reaches consumers (it was a memo, not counted).
    fleet_cash = pd.Series(0.0, index=years_after, dtype=float)
    if carried_case and profile_cash is not None and years_after:
        fleet_cash = profile_cash["total"].reindex(years_after).fillna(0.0).astype(float)
        fleet_cash[fleet_cash.index > last_payment] = 0.0
    series = {line: {party: pd.Series(0.0, index=years_after, dtype=float) for party in PARTIES}
              for line in ("financing", "p3_annuity", "p3_fleet_cash")}
    long_rows = []
    for y in years_after:
        parts = {"financing": financing_parties(by_year.at[y, "capital_saving_gbp"], by_year.at[y, "tax_forgone_gbp"],
                                                by_year.at[y, "overhead_gbp"]),
                 "p3_annuity": legacy_parties(0.0, float(annuity[y]), 0.0, 0.0),
                 "p3_fleet_cash": legacy_parties(float(fleet_cash[y]), 0.0, 0.0, 0.0)}
        for line, parties in parts.items():
            for party in PARTIES:
                series[line][party][y] = float(parties[party])
                long_rows.append({"year": y, "line": f"beyond_2050_{line}", "mechanism": "beyond_2050",
                                  "party": party, "gbp": float(parties[party])})
    empty = pd.Series({LAST_YEAR + 1: 0.0})
    entry = {"years": [years_after[0], years_after[-1]] if years_after else [], "by_line": {}, "parties": {}}
    for party in PARTIES:
        both = series["financing"][party] + series["p3_annuity"][party] + series["p3_fleet_cash"][party]
        for line in series:
            entry["by_line"].setdefault(line, {})[party] = totals(series[line][party] if years_after else empty)
        entry["parties"][party] = {"total": totals(both if years_after else empty),
                                   "per_household_3100kwh_gbp2025": totals((both if years_after else empty) * key_2050)}
    if carried_case and profile_cash is not None:
        after = profile_cash.loc[(profile_cash.index > LAST_YEAR) & (profile_cash.index <= last_payment), "total"]
        entry["acquired_fleet_cash_after_2050"] = {
            "years": [int(after.index.min()), int(after.index.max())] if len(after) else [],
            "gbp": totals(after) if len(after) else None,
            "counted": True,
            "source": "the 2030 legacy valuation's profile (the acquired fleet's cash on that valuation's lives); "
                      "counted since 13 Nov 2026 (line p3_fleet_cash)"}
    return entry, long_rows


# ── legacy ───────────────────────────────────────────────────────────────────

def legacy_cash_by_year(profile: pd.DataFrame, years: Sequence[int] = YEARS,
                        groups: Sequence[str] = ACQUIRED_SCOPE_GROUPS,
                        exclude_carriers: Sequence[str] = ()) -> pd.DataFrame:
    """The acquired fleet's cash by year and stream (``legacy_valuation``'s profile), zero where it has none;
    ``exclude_carriers`` drops carriers the purchase leaves private (:func:`purchase_excluded_carriers`)."""
    rows = profile.loc[profile["scope_group"].isin(list(groups)) & profile["year"].isin(list(years))
                       & ~profile["carrier"].astype(str).isin(list(exclude_carriers))]
    table = rows.pivot_table(index="year", columns="stream", values="gbp", aggfunc="sum").reindex(list(years)).fillna(0.0)
    table["total"] = table.sum(axis=1)
    return table


def annuity_by_year(carried_case: Mapping, years: Sequence[int] = YEARS) -> pd.Series:
    """The 2030 purchase's annuity by year (Y12): flat real from the acquisition year to its last payment,
    or, with each asset repaid over its own remaining life (F28 (e)), each line running off on its schedule."""
    first, last = int(carried_case["acquisition_year"]), int(carried_case["last_payment_year"])
    records = carried_case.get("annuity_schedules") or {}
    if records and carried_case.get("gross_lines_acquisition_year_gbp"):
        from scripts.public_ownership.acquisition_carry_forward import LINE_SCHEDULE, schedule_live_share

        gross = carried_case["gross_lines_acquisition_year_gbp"]
        return pd.Series([0.0 if int(y) < first else sum(
            float(amount) * schedule_live_share(records.get(LINE_SCHEDULE[line], records["legacy"]), int(y) - first)
            for line, amount in gross.items()) for y in years], index=list(years), dtype=float)
    total = float(carried_case["total_model_year_gbp"])
    return pd.Series([total if first <= y <= last else 0.0 for y in years], index=list(years), dtype=float)


def operating_mw_by_year(schedule: Sequence[Mapping], years: Sequence[int] = YEARS,
                         groups: Sequence[str] = ACQUIRED_SCOPE_GROUPS,
                         exclude_carriers: Sequence[str] = ()) -> pd.Series:
    """Acquired capacity still operating, linear between the schedule's years (Y12's survival rules)."""
    frame = pd.DataFrame(list(schedule))
    frame = frame.loc[frame["scope_group"].isin(list(groups))]
    if exclude_carriers and "carrier" in frame.columns:
        frame = frame.loc[~frame["carrier"].astype(str).isin(list(exclude_carriers))]
    points = frame.groupby("year")["operating_mw"].sum()
    return interpolate(points.to_dict(), years)


def legacy_parties(margin: float, annuity: float, legacy_tax: float, overhead: float) -> dict:
    """The legacy line by party (cash basis): owners give up the margin and receive the annuity."""
    investors = float(annuity) - float(margin) + float(legacy_tax)
    exchequer = -float(legacy_tax)
    resource = -float(overhead)
    return {"consumers": resource - investors - exchequer, "exchequer": exchequer, "investors": investors,
            "resource": resource}


# ── the snapshots ────────────────────────────────────────────────────────────

def _starts(label: str, prefixes: Sequence[str]) -> bool:
    return any(str(label).startswith(p) for p in prefixes)


def snapshot_mechanisms(record: Mapping, *, legacy_overhead_gbp: float = 0.0) -> dict:
    """One ledger case split into financing, legacy and market-and-other, by party.

    The split reads the ledger's own part labels: resource parts and investor parts are
    assigned by label (:data:`FINANCING_RESOURCE_PARTS`, :data:`LEGACY_RESOURCE_PARTS`,
    :data:`LEGACY_INVESTOR_PARTS`), the Exchequer's corporation tax forgone by its new-build and
    legacy columns (and an acquisition annuity paid from taxation to legacy), and consumers are
    the rest of each mechanism (resource - Exchequer - investors). ``legacy_overhead_gbp``
    moves the state's overhead on the acquired fleet (inside the ledger's single overhead
    part) from financing to legacy.

    Returns:
        ``{mechanism: {party: GBP}}``; the three mechanisms add to the ledger's totals.
    """
    resource_parts = dict(record.get("resource_saving_vs_private_parts_gbp") or {})
    investor_parts = dict(record.get("investor_change_vs_private_parts_gbp") or {})
    out = {m: dict.fromkeys(PARTIES, 0.0) for m in MECHANISMS}
    for label, value in resource_parts.items():
        mech = ("financing" if _starts(label, FINANCING_RESOURCE_PARTS)
                else "legacy" if _starts(label, LEGACY_RESOURCE_PARTS) else None)
        if mech:
            out[mech]["resource"] += float(value)
    for label, value in investor_parts.items():
        if _starts(label, LEGACY_INVESTOR_PARTS):
            out["legacy"]["investors"] += float(value)
    out["financing"]["exchequer"] = -float(record.get("corporation_tax_forgone_new_build_gbp") or 0.0)
    out["legacy"]["exchequer"] = -(float(record.get("corporation_tax_forgone_legacy_gbp") or 0.0)
                                   + float(record.get("acquisition_annuity_exchequer_gbp") or 0.0))
    if legacy_overhead_gbp:
        out["financing"]["resource"] += float(legacy_overhead_gbp)
        out["legacy"]["resource"] -= float(legacy_overhead_gbp)
    for mech in ("financing", "legacy"):
        d = out[mech]
        d["consumers"] = d["resource"] - d["exchequer"] - d["investors"]
    totals = {party: float(record.get(column) or 0.0) for party, column in LEDGER_TOTALS.items()}
    out["market_and_other"] = {p: totals[p] - out["financing"][p] - out["legacy"][p] for p in PARTIES}
    return out


def scaled(mechanisms: Mapping, share: float) -> dict:
    """A snapshot's mechanisms at ``share`` (P2 at an L4 share in 2030, decision 17), declared."""
    return {m: {p: float(share) * float(v) for p, v in parties.items()} for m, parties in mechanisms.items()}


def check_identity(parties: Mapping, what: str, tolerance: float = 1.0) -> None:
    """Consumers + Exchequer + investors = resource saving, as every ledger closes."""
    residual = parties["consumers"] + parties["exchequer"] + parties["investors"] - parties["resource"]
    if abs(residual) > tolerance:
        raise PathwayError(f"{what}: the three parties miss the resource saving by GBP {residual:,.0f}")


# ── assembly ─────────────────────────────────────────────────────────────────

def assemble_case(case_id: str, *, snapshots: Mapping[int, Mapping], financing: pd.DataFrame,
                  legacy: pd.DataFrame | None = None, years: Sequence[int] = YEARS) -> pd.DataFrame:
    """One case's pathway: every line and party by year (long frame).

    Args:
        case_id: The case.
        snapshots: ``{2030: mechanisms, 2050: mechanisms}`` (:func:`snapshot_mechanisms`).
        financing: :func:`schedule_financing` output for the case.
        legacy: Per year ``margin``, ``annuity``, ``legacy_tax``, ``overhead`` (tariff cases), or None.
        years: The pathway's years.

    Returns:
        ``case_id``, ``year``, ``line``, ``mechanism``, ``party``, ``gbp``. In each snapshot year
        the lines of a mechanism add to the snapshot's mechanism exactly.
    """
    years = list(years)
    fin = financing.groupby("year")[["capital_saving_gbp", "tax_forgone_gbp", "overhead_gbp"]].sum().reindex(years).fillna(0.0)
    modelled_fin = {y: financing_parties(fin.at[y, "capital_saving_gbp"], fin.at[y, "tax_forgone_gbp"],
                                         fin.at[y, "overhead_gbp"]) for y in years}
    if legacy is not None:
        leg = legacy.reindex(years).fillna(0.0)
        modelled_leg = {y: legacy_parties(leg.at[y, "margin"], leg.at[y, "annuity"], leg.at[y, "legacy_tax"],
                                          leg.at[y, "overhead"]) for y in years}
    else:
        modelled_leg = {y: dict.fromkeys(PARTIES, 0.0) for y in years}
    rows = []
    for party in PARTIES:
        fin_anchor = interpolate({s: snapshots[s]["financing"][party] - modelled_fin[s][party] for s in snapshots}, years,
                                 zero_before_first=True)
        leg_anchor = interpolate({s: snapshots[s]["legacy"][party] - modelled_leg[s][party] for s in snapshots}, years,
                                 zero_before_first=True)
        market = interpolate({s: snapshots[s]["market_and_other"][party] for s in snapshots}, years,
                             zero_before_first=True)
        for y in years:
            for line, value in (("financing_vintage", modelled_fin[y][party]), ("financing_anchor", fin_anchor[y]),
                                ("legacy_modelled", modelled_leg[y][party]), ("legacy_anchor", leg_anchor[y]),
                                ("market_and_other_interpolated", market[y])):
                rows.append({"case_id": case_id, "year": y, "line": line, "mechanism": LINES[line], "party": party,
                             "gbp": float(value)})
    return pd.DataFrame(rows)


def anchor_check(pathway: pd.DataFrame, snapshots: Mapping[str, Mapping[int, Mapping]],
                 tolerance: float = ANCHOR_TOLERANCE_GBP) -> pd.DataFrame:
    """Each snapshot year's pathway against its ledger, by case, mechanism and party."""
    out = []
    for case_id, by_year in snapshots.items():
        sub = pathway.loc[pathway["case_id"].eq(case_id)]
        for year, mech in by_year.items():
            for m in MECHANISMS:
                for party in PARTIES:
                    got = float(sub.loc[sub["year"].eq(year) & sub["mechanism"].eq(m) & sub["party"].eq(party), "gbp"].sum())
                    want = float(mech[m][party])
                    out.append({"case_id": case_id, "year": year, "mechanism": m, "party": party, "pathway_gbp": got,
                                "ledger_gbp": want, "difference_gbp": got - want,
                                "passes": abs(got - want) <= tolerance})
    return pd.DataFrame(out)


def households_by_year(fes: pd.DataFrame, pathway: str = "Holistic Transition",
                       years: Sequence[int] = YEARS, block: str = "Dem_BB001a") -> pd.Series:
    """Domestic customer accounts by year (FES 2025 ``Dem_BB001a``, summed over GSPs)."""
    rows = fes.loc[fes["Building Block ID Number"].eq(block) & fes["FES Pathway"].eq(pathway)]
    if rows.empty:
        raise PathwayError(f"no {block} rows for {pathway}")
    return pd.Series([float(pd.to_numeric(rows[str(y)], errors="coerce").sum()) for y in years], index=list(years))


def totals(series_by_year: pd.Series) -> dict:
    """Undiscounted total and NPVs at each discount rate (base year 2027, BASE_YEAR)."""
    out = {"undiscounted": float(series_by_year.sum())}
    for name, rate in DISCOUNT_RATES.items():
        out[f"npv_{name}"] = npv(series_by_year, rate)
    return out


# ── the run ──────────────────────────────────────────────────────────────────

def parse_snapshots(items: Sequence[str]) -> dict[int, str]:
    """``YEAR=LEDGER`` pairs for solved years strictly between the first and last year."""
    out = {}
    for item in items:
        year, _, path = str(item).partition("=")
        if not path or not year.strip().isdigit():
            raise PathwayError(f"--snapshot takes YEAR=LEDGER, got {item!r}")
        y = int(year)
        if not FIRST_YEAR < y < LAST_YEAR:
            raise PathwayError(f"--snapshot {y} is not between {FIRST_YEAR} and {LAST_YEAR}")
        out[y] = path
    return out


def snapshot_paths(args) -> dict[int, str]:
    """Every anchor year's ledger path, in year order."""
    paths = {FIRST_YEAR: args.ledger_2030, LAST_YEAR: args.ledger_2050,
             **parse_snapshots(getattr(args, "snapshot", None) or [])}
    return dict(sorted(paths.items()))


def _ledger(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _case_record(ledger: Mapping, case_id: str, what: str) -> dict:
    cases = ledger.get("cases") or {}
    if case_id not in cases:
        raise PathwayError(f"{what} has no case {case_id!r}")
    return cases[case_id]


def run(args) -> dict:
    """Build the pathway from the files the CLI names; returns the outputs in memory."""
    import yaml

    from scripts.public_ownership.calculate_financing_scenarios import canonical_technology, vintage_capital_options
    from scripts.public_ownership.finance import corporation_tax_wedges
    from scripts.public_ownership.register_io import read_register

    ledgers = {FIRST_YEAR: _ledger(args.ledger_2030), LAST_YEAR: _ledger(args.ledger_2050)}
    for year, path in parse_snapshots(getattr(args, "snapshot", None) or []).items():
        ledgers[year] = _ledger(path)
    ledgers = dict(sorted(ledgers.items()))
    snapshot_years = tuple(ledgers)
    # Price years: every ledger states its basis and conversion; they must agree.
    bases = {y: (l.get("price_basis"), float((l.get("conversion") or {}).get("factor", np.nan))) for y, l in ledgers.items()}
    if any(b[0] != bases[FIRST_YEAR][0] or not np.isclose(b[1], bases[FIRST_YEAR][1]) for b in bases.values()):
        raise PathwayError(f"the ledgers' price bases differ: {bases}")
    conversion = {y: dict(l["conversion"]) for y, l in ledgers.items()}

    config = read_config(args.config_2050) or {}
    block = dict(config.get("public_ownership") or {})
    basis = vintage_capital_options(block, ROOT)
    if not basis:
        raise PathwayError("the 2050 study config carries no vintage_capital_2050 block")
    snapshot = yaml.safe_load((ROOT / block["assumptions_snapshot"]).read_text(encoding="utf-8"))
    wedges = {canonical_technology(k): v for k, v in corporation_tax_wedges(snapshot).items()}
    settings = pd.read_csv(args.case_settings).set_index("case_id")

    flows = pd.read_csv(args.fleet_flows)
    survivors = pd.read_csv(args.fleet_vintages)
    profile = pd.read_csv(args.legacy_profile_2030)
    carried = _ledger(args.carried_acquisition)
    overhead_rate = float((block.get("state_operating_overhead") or {}).get("central_gbp_per_kw_year", 0.0))
    # S1/S2 audit (13 Nov 2026): the legislation year and lead times decide which retired cohorts the state built.
    from scripts.public_ownership.legislation_year import load_lead_times

    legislated_year = int((block.get("legislation") or {}).get("year", 2028))
    lead_years = load_lead_times()["investment_lead_years"].astype(float).to_dict()
    excluded = (purchase_excluded_carriers(settings.loc[P3_CARRIED_CASE]) if P3_CARRIED_CASE in settings.index
                else EFW_CARRIERS)
    legacy_mw = operating_mw_by_year(carried["survival_schedule_all_groups"], exclude_carriers=excluded)
    cash = legacy_cash_by_year(profile, exclude_carriers=excluded)

    private = read_register(args.register_2050, columns=list(REGISTER_COLUMNS), cases=[PRIVATE])
    pieces, financing_rows, checks, snaps, omitted_all, case_meta, all_rows = [], [], [], {}, [], {}, {}
    for case_id, (label, source_2030) in CASES.items():
        if case_id not in settings.index:
            raise PathwayError(f"the 2050 register has no case {case_id!r}")
        lever = settings.loc[case_id]
        cepa_jv = str(lever.get("public_return")) == "cepa_decomposition"
        case_rows = read_register(args.register_2050, columns=list(REGISTER_COLUMNS), cases=[case_id])
        rows = register_rows(case_rows, private, basis=basis, wedges=wedges, cepa_jv=cepa_jv)
        new_build = case_rows.loc[case_rows["finance_vintage"].astype(str).eq("new_build")]
        merged = new_build.assign(asset_id=new_build["asset_id"].astype(str)).merge(
            private[["asset_id", "wacc"]].assign(asset_id=private["asset_id"].astype(str)).rename(
                columns={"wacc": "wacc_private"}), on="asset_id", validate="one_to_one")
        extra, omitted = retired_cohort_rows(flows, survivors, merged, wedges=wedges, cepa_jv=cepa_jv,
                                             legislated_year=legislated_year, lead_years=lead_years)
        rows = pd.concat([rows, extra], ignore_index=True)
        omitted_all.append(omitted.assign(case_id=case_id))
        financing = schedule_financing(rows)
        financing_rows.append(financing.assign(case_id=case_id))
        all_rows[case_id] = rows
        # The 2050 check of the vintage reconstruction against the register, row by row.
        term = rows["kind"].eq("term")
        lvl = rows["kind"].eq("level") & ~rows["asset_id"].str.startswith("retired_cohort::")
        checks.append({
            "case_id": case_id,
            "register_capital_saving_2050_gbp": float((rows["register_capital_private_2050_gbp"] - rows["register_capital_case_2050_gbp"]).sum()),
            "pathway_capital_saving_2050_gbp": float(financing.loc[financing["year"].eq(LAST_YEAR), "capital_saving_gbp"].sum()),
            "term_rows_register_gbp": float((rows.loc[term, "register_capital_private_2050_gbp"] - rows.loc[term, "register_capital_case_2050_gbp"]).sum()),
            "term_rows_pathway_gbp": float(((rows.loc[term, "annual_capital_private_gbp"] - rows.loc[term, "annual_capital_case_gbp"]) * rows.loc[term, "live_share_2050"]).sum()),
            "level_rows_register_remaining_life_gbp": float((rows.loc[lvl, "register_capital_private_2050_gbp"] - rows.loc[lvl, "register_capital_case_2050_gbp"]).sum()),
            "level_rows_pathway_whole_life_gbp": float((rows.loc[lvl, "annual_capital_private_gbp"] - rows.loc[lvl, "annual_capital_case_gbp"]).sum()),
            "register_tax_2050_gbp": float(rows["register_tax_2050_gbp"].sum()),
            "pathway_tax_2050_gbp": float(financing.loc[financing["year"].eq(LAST_YEAR), "tax_forgone_gbp"].sum()),
            "register_overhead_2050_gbp": float(_num(case_rows.loc[case_rows["finance_vintage"].astype(str).eq("new_build")], "annual_state_overhead_gbp").sum()),
            "pathway_overhead_2050_gbp": float(financing.loc[financing["year"].eq(LAST_YEAR), "overhead_gbp"].sum()),
            "retired_cohort_mw": float(extra["capacity_mw"].sum()) if len(extra) else 0.0,
        })
        tariff = str(lever.get("legacy_fleet")) != "private"
        legacy = None
        snapshot_legacy_overhead = dict.fromkeys(snapshot_years, 0.0)
        rec50 = _case_record(ledgers[LAST_YEAR], case_id, "the 2050 ledger")
        # A middle year (2040) carries the 2050 case ids (a FES study, the L4 shares included).
        recs_mid = {y: _case_record(ledgers[y], case_id, f"the {y} ledger") for y in snapshot_years
                    if y not in (FIRST_YEAR, LAST_YEAR)}
        if isinstance(source_2030, tuple):
            rec30_source, share = source_2030
            rec30 = _case_record(ledgers[FIRST_YEAR], rec30_source, "the 2030 ledger")
        else:
            rec30, share = _case_record(ledgers[FIRST_YEAR], source_2030, "the 2030 ledger"), None
        if tariff:
            carried_case = (carried.get("cases") or {}).get(P3_CARRIED_CASE)
            if carried_case is None:
                raise PathwayError("the carried acquisition has no P3 case")
            overhead = legacy_mw * 1000.0 * overhead_rate
            # The acquired fleet's overhead starts with the 2030 transfer (the view runs from 2027).
            overhead = overhead.where(overhead.index >= FIRST_YEAR, 0.0)
            snapshot_legacy_overhead = {y: float(overhead[y]) for y in snapshot_years}
            tax = interpolate({FIRST_YEAR: float(rec30.get("corporation_tax_forgone_legacy_gbp") or 0.0),
                               **{y: float(r.get("corporation_tax_forgone_legacy_gbp") or 0.0) for y, r in recs_mid.items()},
                               LAST_YEAR: float(rec50.get("corporation_tax_forgone_legacy_gbp") or 0.0)},
                              zero_before_first=True)
            legacy = pd.DataFrame({"margin": cash["total"], "annuity": annuity_by_year(carried_case),
                                   "legacy_tax": tax, "overhead": overhead})
            for stream in [c for c in cash.columns if c != "total"]:
                legacy[f"margin_{stream}"] = cash[stream]
        m30 = snapshot_mechanisms(rec30, legacy_overhead_gbp=snapshot_legacy_overhead[FIRST_YEAR])
        if share is not None:
            m30 = scaled(m30, share)
        m50 = snapshot_mechanisms(rec50, legacy_overhead_gbp=snapshot_legacy_overhead[LAST_YEAR])
        snaps[case_id] = {FIRST_YEAR: m30,
                          **{y: snapshot_mechanisms(r, legacy_overhead_gbp=snapshot_legacy_overhead[y])
                             for y, r in recs_mid.items()},
                          LAST_YEAR: m50}
        piece = assemble_case(case_id, snapshots=snaps[case_id], financing=financing, legacy=legacy)
        pieces.append(piece)
        case_meta[case_id] = {"label": label, "source_2030": (f"{source_2030[1]:.2f} x {source_2030[0]}"
                                                              if isinstance(source_2030, tuple) else source_2030),
                              "tariff_case": tariff, "public_return": str(lever.get("public_return")),
                              "legacy": None if legacy is None else legacy.round(0).to_dict(orient="index"),
                              "carried_case": carried_case if tariff else None}
        del case_rows
    pathway = pd.concat(pieces, ignore_index=True)
    for (case_id, year), g in pathway.groupby(["case_id", "year"]):
        check_identity({p: float(g.loc[g["party"].eq(p), "gbp"].sum()) for p in PARTIES}, f"{case_id} {year}", tolerance=10.0)
    anchors = anchor_check(pathway, snaps)
    if not anchors["passes"].all():
        bad = anchors.loc[~anchors["passes"]].head(5)
        raise PathwayError(f"a snapshot year misses its ledger:\n{bad}")
    financing_all = pd.concat(financing_rows, ignore_index=True)

    # Households: the ledgers' key (final demand interpolated) and FES domestic accounts.
    final_demand = interpolate({y: float(conversion[y]["final_demand_mwh"]) for y in snapshot_years})
    factor = float(conversion[LAST_YEAR]["factor"])
    mwh = float(conversion[LAST_YEAR]["mwh"])
    fes = pd.read_csv(args.fes_data, usecols=["Building Block ID Number", "FES Pathway", *[str(y) for y in YEARS]])
    accounts = households_by_year(fes)
    per_household_key = factor * mwh / final_demand          # GBP 2025 per 3,100 kWh household per system GBP
    # 13 Nov 2026 (review finding 10): no per-account figure. Dividing the whole system amount by domestic accounts put
    # the non-domestic and electrolyser shares on households (about three times too high); the accounts stay as a memo.

    # The long CSV: lines, plus the financing by technology group and the legacy streams.
    tech = financing_all.melt(id_vars=["case_id", "year", "tech_group"],
                              value_vars=["capital_saving_gbp", "tax_forgone_gbp", "overhead_gbp"],
                              var_name="component", value_name="gbp")
    tech["line"] = "financing_vintage_by_technology:" + tech["tech_group"] + ":" + tech["component"].str.replace("_gbp", "")
    tech["mechanism"], tech["party"] = "financing", "memo"
    out = pd.concat([pathway, tech[["case_id", "year", "line", "mechanism", "party", "gbp"]]], ignore_index=True)
    out["label"] = out["case_id"].map(lambda c: CASES[c][0])
    out["gbp_per_household_3100kwh_2025"] = out["gbp"] * out["year"].map(per_household_key)
    out["interpolated"] = out["line"].isin(["financing_anchor", "legacy_anchor", "market_and_other_interpolated"])

    # Decision 46 (b): beyond 2050, a separate line per case (never in the 2030-2050 totals).
    key_2050 = float(per_household_key[LAST_YEAR])
    profile_cash = legacy_cash_by_year(profile, years=sorted(set(profile["year"].astype(int))),
                                       exclude_carriers=excluded)
    beyond, beyond_rows = {}, []
    for case_id, meta in case_meta.items():
        entry, long_rows = beyond_2050_case(all_rows[case_id], meta.get("carried_case"), profile_cash, key_2050)
        beyond[case_id] = entry
        beyond_rows.extend({"case_id": case_id, **r} for r in long_rows)
    if beyond_rows:
        frame = pd.DataFrame(beyond_rows)
        frame["label"] = frame["case_id"].map(lambda c: CASES[c][0])
        frame["gbp_per_household_3100kwh_2025"] = frame["gbp"] * key_2050
        frame["interpolated"] = False
        out = pd.concat([out, frame[out.columns]], ignore_index=True)

    summary = build_summary(pathway, financing_all, anchors, checks, case_meta, per_household_key, accounts,
                            final_demand, omitted_all, args, ledgers)
    for case_id, entry in beyond.items():
        summary["cases"][case_id]["beyond_2050"] = entry
    last_payment = max((int(m["carried_case"]["last_payment_year"]) for m in case_meta.values()
                        if m.get("carried_case") and m["carried_case"].get("last_payment_year")), default=None)
    summary["beyond_2050_rule"] = (
        "Decision 46 (b): a separate line, never in the 2030-2050 totals. The financing saving still to come on "
        "plant built by 2050 (live contracts to the end of their term, merchant and RAB rows to the end of their "
        "life; expired contracts none), and P3's purchase annuity "
        + (f"2051-{last_payment} (each asset over its remaining life)" if last_payment else "after 2050")
        + ", both held at 2050 levels (fleet, "
        "rates, shares; per household on the 2050 key) and discounted to 2027 as the headline. The acquired fleet's "
        "own cash after 2050 (the 2030 valuation's profile, to the last annuity payment) is counted beside the "
        "annuity (user, 13 Nov 2026: both sides or neither).")
    return {"pathway": out, "summary": summary, "lines": pathway}


def _by(pathway: pd.DataFrame, case_id: str, party: str, mechanism: str | None = None) -> pd.Series:
    sub = pathway.loc[pathway["case_id"].eq(case_id) & pathway["party"].eq(party)]
    if mechanism:
        sub = sub.loc[sub["mechanism"].eq(mechanism)]
    return sub.groupby("year")["gbp"].sum().reindex(list(YEARS)).fillna(0.0)


def build_summary(pathway, financing, anchors, checks, case_meta, per_household_key, accounts, final_demand,
                  omitted, args, ledgers) -> dict:
    """Totals, NPVs, the snapshot tables, the anchoring, assumptions and provenance."""
    cases = {}
    for case_id, meta in case_meta.items():
        entry = {"label": meta["label"], "source_2030": meta["source_2030"], "parties": {}, "annual": {}}
        for party in PARTIES:
            total = _by(pathway, case_id, party)
            per_mech = {m: _by(pathway, case_id, party, m) for m in MECHANISMS}
            entry["parties"][party] = {
                "total": totals(total),
                "by_mechanism": {m: totals(s) for m, s in per_mech.items()},
                "per_household_3100kwh_gbp2025": totals(total * per_household_key),
            }
            # 10 Nov 2026: the anchor lines apart (the gap each solved year closes), so a page can show them.
            anchor_lines = {line: (pathway.loc[pathway["case_id"].eq(case_id) & pathway["party"].eq(party)
                                          & pathway["line"].eq(line)].groupby("year")["gbp"].sum()
                              .reindex(YEARS).fillna(0.0)) for line in ("financing_anchor", "legacy_anchor")}
            entry["annual"][party] = {str(y): {"total": float(total[y]), **{m: float(per_mech[m][y]) for m in MECHANISMS},
                                               **{line: float(a[y]) for line, a in anchor_lines.items()}}
                                      for y in YEARS}
        public = _by(pathway, case_id, "consumers") + _by(pathway, case_id, "exchequer")
        entry["public_gain_consumers_plus_exchequer"] = {
            "total": totals(public), "per_household_3100kwh_gbp2025": totals(public * per_household_key),
            "by_mechanism": {m: totals(_by(pathway, case_id, "consumers", m) + _by(pathway, case_id, "exchequer", m))
                             for m in MECHANISMS}}
        fin = financing.loc[financing["case_id"].eq(case_id)]
        groups = fin.assign(saving=fin["capital_saving_gbp"] - fin["tax_forgone_gbp"] - fin["overhead_gbp"]).pivot_table(
            index="year", columns="tech_group", values="saving", aggfunc="sum")
        entry["financing_resource_saving_by_technology_group"] = {
            str(y): {g: float(groups.at[y, g]) for g in groups.columns} for y in REPORT_YEARS}
        if meta["legacy"] is not None:
            entry["legacy_inputs_by_year"] = {str(k): v for k, v in meta["legacy"].items()}
        entry["headline"] = f"{YEARS[0]}-{YEARS[-1]} totals, anchored to every snapshot ledger (decision 46 (d))"
        cases[case_id] = entry
    anchor_rows = anchors.copy()
    gaps = pathway.loc[pathway["line"].isin(["financing_anchor", "legacy_anchor"]) & pathway["year"].isin(list(ledgers))]
    gap_table = gaps.pivot_table(index=["case_id", "year", "party"], columns="line", values="gbp", aggfunc="sum").reset_index()
    omitted = pd.concat(omitted, ignore_index=True) if omitted else pd.DataFrame()
    return {
        "task": "FES 2050 plan Y15: pathway-lite 2030-2050",
        "snapshot_years": sorted(int(y) for y in ledgers),
        "snapshot_sources": {str(y): str(p) for y, p in snapshot_paths(args).items()},
        "price_basis": ledgers[LAST_YEAR].get("price_basis"),
        "base_year": BASE_YEAR, "years": [YEARS[0], YEARS[-1]],
        "discount_rates": DISCOUNT_RATES, "central_discount_rate": CENTRAL_RATE,
        "cases": cases,
        "anchoring": {
            "rule": "each snapshot year reproduces its ledger by case, mechanism and party to GBP 1 m",
            "tolerance_gbp": ANCHOR_TOLERANCE_GBP, "all_pass": bool(anchor_rows["passes"].all()),
            "worst_difference_gbp": float(anchor_rows["difference_gbp"].abs().max()),
            "snapshot_gaps_of_the_modelled_lines": gap_table.round(0).to_dict(orient="records"),
            "register_reconstruction_2050": checks,
        },
        "households": {
            "domestic_accounts": {str(y): float(accounts[y]) for y in YEARS},
            "final_demand_mwh": {str(y): float(final_demand[y]) for y in YEARS},
            "per_household_key": "system GBP x 1.0384 x 3.1 MWh / final demand (the ledgers' own key), GBP real 2025",
        },
        "omitted_retired_cohorts": omitted.round(3).to_dict(orient="records") if len(omitted) else [],
        "assumptions": ASSUMPTIONS,
        "interpolated_lines": {
            "market_and_other_interpolated": ("linear between the snapshots on the private-vs-case difference: "
                                              "wholesale and balancing, Capacity Market, capture-driven revenue, "
                                              "interconnectors (re-trading, cross-border energy, GB halves' rent), the "
                                              "dispatch institution (I3), the state's deficits and rents on "
                                              "non-self-financing plant, LDES floor, hedging allowances, VAT"),
            "financing_anchor": "the snapshot gap of the vintage financing line, linear between the snapshots",
            "legacy_anchor": "the snapshot gap of the modelled legacy line, linear between the snapshots",
        },
        "sources": {k: str(v) for k, v in vars(args).items()},
    }


ASSUMPTIONS = [
    "No network is solved: market-dependent lines are piecewise-linear between the solved snapshots (2030 CP30 on the CENTRAL network, any middle year given, 2050 FES HT).",
    "Interpolation is on each case's difference from private, never on levels; the CP30-to-FES-HT basis change sits inside it.",
    "Financing is exact from the 2050 vintaged register's rows; years before 2050 reuse each row's rate, share and capital.",
    "Contracted rows recover capital over the contract term (decision 41); live share by year is the register's live_fraction rule.",
    "CfD-technology rows built before 2031 hold AR1-AR6 contracts: 15-year terms, less the decision-6 merchant-tail credit (decision 46 (c)); AR7 and AR7a known contracts keep their 20 years.",
    "Merchant and RAB rows recover capital as a level annuity over the economic life (the 2050 register annuitises over the remaining life; decision 43).",
    "Post-2025 cohorts retired before 2050: lithium batteries from the stock-flow; 1.27 GW of unabated gas (2026-27) and 16 MW of other plant are left out.",
    "P2 and its shares are the state-bids cases (decision 46 (a)); 2030 is the CP30 P2 bid-stack case, and share x that case at the L4 shares (decision 17).",
    "Legacy (P3): the 2030 valuation's cash profile (flows held at 2030 prices, contracts on their expiry dates) and the carried 2030 annuity, each acquired asset repaid over its own remaining life (to its last payment year).",
    "Legacy overhead follows the acquisition's survival schedule (linear between its years); former owners' corporation tax is linear between snapshots.",
    "Networks are private in every reported case; nothing is added for them.",
    "Parties: financing and legacy savings reach consumers (bill-payers first); corporation tax forgone is the Exchequer's.",
    "Per household: the ledgers' key (3.1 MWh x 1.0384 over final demand), final demand interpolated between the snapshot years' own figures; no per-account figure (review finding 10, 13 Nov 2026).",
    "NPV base year 2027 (undiscounted), real GBP 2024, STPR 3.5% with 3.0% and 5.0% sensitivities; 2027-2050 is the headline.",
    "2027-2029 (before the first solved year, user 10 Nov 2026): financing year by year from the build years (state plant committed after the law and open by then); the purchase, the market-dependent lines and the anchors are zero, starting with the 2030 transfer.",
    "Beyond 2050 (decision 46 (b)), reported separately: the remaining financing saving on plant built by 2050 and P3's annuity to its last payment, held at 2050 levels and discounted to 2027; the acquired fleet's own cash after 2050 is counted beside the annuity.",
]


# ── figure ───────────────────────────────────────────────────────────────────

def write_figure(lines: pd.DataFrame, path: Path) -> None:
    """Annual benefit by case and mechanism: consumers + Exchequer stacked by mechanism, resource saving as a line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = ["public_control", "public_ownership@deployment_share_100_state_bids",
             "public_ownership@deployment_share_50_state_bids",
             "public_ownership@fully_nationalised_market_value",
             "public_ownership@fully_nationalised_market_value_central_dispatch"]
    colours = {"financing": "#2a6f97", "legacy": "#c97c1d", "market_and_other": "#8a8f98"}
    names = {"financing": "Financing (vintages)", "legacy": "Legacy rent less purchase",
             "market_and_other": "Market and other (interpolated)"}
    fig, axes = plt.subplots(1, len(shown), figsize=(18, 4.8), sharey=True)
    years = np.array(YEARS)
    for ax, case_id in zip(axes, shown):
        pos, neg = np.zeros(len(years)), np.zeros(len(years))
        for mech in MECHANISMS:
            v = (_by(lines, case_id, "consumers", mech) + _by(lines, case_id, "exchequer", mech)).to_numpy() / 1e9
            bottom = np.where(v >= 0, pos, neg)
            ax.bar(years, v, bottom=bottom, color=colours[mech], width=0.8, label=names[mech], linewidth=0)
            pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
        res = _by(lines, case_id, "resource").to_numpy() / 1e9
        net = (_by(lines, case_id, "consumers") + _by(lines, case_id, "exchequer")).to_numpy() / 1e9
        ax.plot(years, res, color="#111111", linewidth=1.6, label="Resource saving")
        ax.plot(years, net, color="#b0272d", linewidth=1.4, linestyle="--", label="Consumers + Exchequer, net")
        ax.axhline(0, color="#444444", linewidth=0.6)
        ax.set_title(CASES[case_id][0], fontsize=11)
        ax.set_xticks([2030, 2035, 2040, 2045, 2050])
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("GBP bn a year vs private (real 2024)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Public ownership benefit 2030-2050 (pathway-lite): consumers + Exchequer by mechanism", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    rp = "resources/public_ownership"
    parser.add_argument("--ledger-2030", default=f"{rp}/cp30_2030_no_acceleration_exchequer_ledger.json")
    parser.add_argument("--ledger-2050", default=f"{rp}/fes_ht_2050_exchequer_ledger.json")
    parser.add_argument("--register-2050", default=f"{rp}/fes_ht_2050_asset_finance_register.parquet")
    parser.add_argument("--case-settings", default=f"{rp}/fes_ht_2050_case_settings.csv")
    parser.add_argument("--config-2050", default=f"{rp}/FES25_HT_2050_finance_config.yaml")
    parser.add_argument("--legacy-profile-2030", default=f"{rp}/cp30_2030_no_acceleration_legacy_cash_flow_profile.csv")
    parser.add_argument("--snapshot", action="append", default=[], metavar="YEAR=LEDGER",
                        help="a solved year between 2030 and 2050 and its Exchequer ledger (repeatable; 2040 later)")
    parser.add_argument("--carried-acquisition", default=f"{rp}/fes_ht_2050_carried_acquisition.json")
    parser.add_argument("--fleet-flows", default=f"{rp}/fes_ht_2050_fleet_vintage_flows.csv")
    parser.add_argument("--fleet-vintages", default=f"{rp}/fes_ht_2050_fleet_vintages.csv")
    parser.add_argument("--fes-data", default="resources/FES/FES_2025_data.csv")
    parser.add_argument("--out-csv", default=f"{rp}/pathway_2030_2050.csv")
    parser.add_argument("--out-json", default=f"{rp}/pathway_2030_2050_summary.json")
    parser.add_argument("--out-figure", default=f"{rp}/pathway_2030_2050.png")
    args = parser.parse_args(argv)
    result = run(args)
    result["pathway"].to_csv(args.out_csv, index=False)
    Path(args.out_json).write_text(json.dumps(result["summary"], indent=1, default=float), encoding="utf-8")
    write_figure(result["lines"], Path(args.out_figure))
    s = result["summary"]
    print(f"wrote {args.out_csv}, {args.out_json}, {args.out_figure}; anchoring all pass: {s['anchoring']['all_pass']}")
    for case_id, entry in s["cases"].items():
        r = entry["parties"]["resource"]["total"]
        c = entry["parties"]["consumers"]["total"]
        b = entry["beyond_2050"]["parties"]["consumers"]["total"]
        print(f"  {entry['label']:8s} resource {r['undiscounted'] / 1e9:8.2f} bn (NPV 3.5% {r['npv_stpr_3p5'] / 1e9:7.2f}); "
              f"consumers {c['undiscounted'] / 1e9:8.2f} bn (NPV {c['npv_stpr_3p5'] / 1e9:7.2f}); "
              f"beyond 2050 consumers {b['undiscounted'] / 1e9:7.2f} bn (NPV {b['npv_stpr_3p5'] / 1e9:6.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
