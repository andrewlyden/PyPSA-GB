"""The policy timeline: which new build a public scenario can own (F28 (a), user decisions of 26 Oct 2026).

New build was state-financed from ``policy_start_year`` (2025) by commissioning date, which is
already past: plant financed or contracted before any plausible law would have been state-owned.
The legislation year ``L`` (2028 central; 2025 the "in force since 2025" bound) fixes the
boundary instead, by the date the plant was committed to:

* a CfD or DPA row by its contract award year (``cfd_award_year``); a modelled round (CP30's
  ``AR8_modelled`` contracts, the 2050 ``PAR7_<year>`` and ``DPA_<year>`` labels) is awarded its
  technology's award lead before commissioning (except modelled ``PAR7_`` nuclear, which is generic pathway
  capacity and always eligible: only Hinkley Point C and Sizewell C are committed nuclear);
* any other row by its investment date, commissioning less the technology's construction
  period (``data/public_ownership/financing/legislation_lead_times.csv``);
* an undated row (the 2030 register's capacity added since 2024, a REPD pipeline project) by a
  declared build profile: its capacity commissions evenly over the years from ``policy_start_year``
  (a pipeline project from the year after the REPD snapshot) to the model year, and the share
  whose date falls on or after ``L`` is eligible, pro rata. A project recorded as under
  construction in the snapshot was committed before any ``L``.

Plant that is not eligible was committed under today's rules: private under P1 and P2, bought at
its owners' value by P3 with the rest of the existing fleet on one transfer date (2030). In the
register it is ``legacy`` when it commissions by the transfer year and ``in_construction`` after it
(a 2050 register's later builds, which P3's 2030 purchase cannot price). Its CfD is ``pre_policy``.
A row only partly eligible is split: ``<asset>::pre_legislation`` and ``<asset>::new_build``, each
with ``cohort_share`` (its share of the network asset), which the settlement reads.

``legislation_status`` records the outcome on every row: ``pre_policy_fleet`` (commissioned
before ``policy_start_year``), ``pre_legislation`` or ``post_legislation``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from scripts.market.cfd_strike_table import MODELLED_ROUND

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEAD_TIMES = ROOT / "data" / "public_ownership" / "financing" / "legislation_lead_times.csv"

STATUS_FLEET = "pre_policy_fleet"
STATUS_PRE = "pre_legislation"
STATUS_POST = "post_legislation"
PRE_LEGISLATION_COHORT = "pre_legislation"
#: REPD statuses whose projects were already committed in the register's source snapshot.
COMMITTED_STATUSES = frozenset({"under construction"})
#: Register columns a split shares out.
SPLIT_COLUMNS = ("capacity_mw", "energy_capacity_mwh", "dispatch_mwh")
#: Support types whose contract award fixes the boundary.
CONTRACT_SUPPORT = frozenset({"cfd", "dpa"})
#: Rounds whose award year is modelled, not published (``cfd_round_award_years.csv`` ``AR8_modelled``: 2027).
MODELLED_ROUNDS = frozenset({MODELLED_ROUND})
MODELLED_ROUND_PREFIXES = ("PAR7_", "DPA_")
#: Modelled nuclear (the 2050 ``PAR7_<year>`` contracts) is generic pathway capacity, not committed projects: only
#: Hinkley Point C and Sizewell C are (user decision, 28 Oct 2026), so it is state new build whatever its date.
LEGISLATION_EXEMPT_CARRIER = "nuclear"
LEGISLATION_EXEMPT_ROUND_PREFIX = "PAR7_"


def load_lead_times(path: str | Path | None = None) -> pd.DataFrame:
    """The lead-time table, indexed by carrier, with a source on every row."""
    table = pd.read_csv(path or DEFAULT_LEAD_TIMES)
    missing = {"carrier", "investment_lead_years", "cfd_award_lead_years", "source"} - set(table.columns)
    if missing:
        raise ValueError(f"lead-time table lacks columns {sorted(missing)}")
    if table["carrier"].duplicated().any():
        raise ValueError(f"lead-time table repeats a carrier: {sorted(table.loc[table['carrier'].duplicated(), 'carrier'])}")
    if table["source"].isna().any() or table["source"].astype(str).str.strip().eq("").any():
        raise ValueError("every lead-time row needs a source")
    lead = pd.to_numeric(table["investment_lead_years"], errors="coerce")
    if lead.isna().any() or (lead < 0).any():
        raise ValueError("investment_lead_years must be a non-negative number on every row")
    return table.set_index("carrier")


def _text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _is_modelled_round(cfd_round: str) -> bool:
    return cfd_round in MODELLED_ROUNDS or cfd_round.startswith(MODELLED_ROUND_PREFIXES)


def commissioning_years(row: Mapping, *, policy_start_year: int, model_year: int, snapshot_year: int,
                        vintaged: bool) -> tuple[list[float], str]:
    """The year(s) a row commissions in: one dated year, or the declared even profile; and the basis."""
    if vintaged:
        built = pd.to_numeric(row.get("build_year"), errors="coerce")
        if pd.notna(built):
            return [float(built)], "build_year"
    built = pd.to_numeric(row.get("commissioning_year"), errors="coerce")
    if pd.notna(built):
        return [float(built)], "commissioning_year"
    status = _text(row.get("development_status")).lower()
    if status in COMMITTED_STATUSES:
        return [], "committed_in_snapshot"
    first = policy_start_year if status == "" else max(policy_start_year, snapshot_year + 1)
    years = [float(y) for y in range(int(first), int(model_year) + 1)]
    if not years:
        years = [float(model_year)]
    return years, "even_build_profile"


def eligible_share(row: Mapping, leads: pd.DataFrame, *, legislated_year: int, policy_start_year: int,
                   model_year: int, snapshot_year: int, vintaged: bool) -> dict:
    """One new-build row's eligible share under the legislation year, and how it was decided."""
    carrier = _text(row.get("carrier"))
    if carrier not in leads.index:
        raise ValueError(f"no lead time for carrier {carrier!r}: add it to the lead-time table")
    support = _text(row.get("support_type")).lower()
    cfd_round = _text(row.get("cfd_round"))
    award = pd.to_numeric(row.get("cfd_award_year"), errors="coerce")
    years, basis = commissioning_years(row, policy_start_year=policy_start_year, model_year=model_year,
                                       snapshot_year=snapshot_year, vintaged=vintaged)
    contract = support in CONTRACT_SUPPORT and cfd_round != ""
    built_on = _text(row.get("contract_round_at_build"))
    if not contract and built_on and _is_modelled_round(built_on):
        # S1/S2 audit (14 Nov 2026): a modelled contract that has expired by this model year was still awarded when
        # the row was built, so the row is committed at that award in every year (it was 'investment' once expired,
        # which moved 2031's gas CCS from in construction in 2040 to state new build in 2050).
        contract, cfd_round = True, built_on
    if contract and not _is_modelled_round(cfd_round) and pd.notna(award):
        return {"share": 1.0 if float(award) >= legislated_year else 0.0, "basis": "contract_award_year",
                "committed_year": float(award), "years": years}
    if contract and carrier == LEGISLATION_EXEMPT_CARRIER and cfd_round.startswith(LEGISLATION_EXEMPT_ROUND_PREFIX):
        return {"share": 1.0, "basis": "modelled_nuclear_exempt", "committed_year": np.nan, "years": years}
    if contract:
        lead = pd.to_numeric(leads.at[carrier, "cfd_award_lead_years"], errors="coerce")
        if pd.isna(lead):
            raise ValueError(f"carrier {carrier!r} holds a modelled contract ({cfd_round}) but has no award lead")
        kind = "modelled_award"
    else:
        lead = float(leads.at[carrier, "investment_lead_years"])
        kind = "investment"
    if not years:
        return {"share": 0.0, "basis": f"{kind}:{basis}", "committed_year": np.nan, "years": years}
    committed = [y - float(lead) for y in years]
    share = float(np.mean([c >= legislated_year for c in committed]))
    return {"share": share, "basis": f"{kind}:{basis}",
            "committed_year": committed[0] if len(committed) == 1 else np.nan, "years": years}


def eligible_share_at(register: pd.DataFrame, year: int, *, policy_start_year: int, model_year: int,
                      leads: pd.DataFrame | None = None, snapshot_year: int = 2025,
                      vintaged: bool = False) -> np.ndarray:
    """Each register row's share committed on or after ``year``, by the rules of :func:`eligible_share`.

    For a register already re-classed at its own legislation year ``L`` (:func:`apply_legislation_year`,
    ``legislated_year`` column) and an earlier ``year`` (GB Energy's, the GB Energy Act 2025; user
    decision, option E, 28 Oct 2026). Rows are not split again; each row carries the share of its own
    capacity committed on or after ``year``:

    * ``post_legislation``: 1 (committed on or after ``L``, so after ``year``);
    * ``pre_policy_fleet``: 0 (commissioned before ``policy_start_year``);
    * ``pre_legislation``, dated (a contract by its award year, other plant by its investment date,
      ``committed_year``): 1 if committed on or after ``year``, else 0;
    * ``pre_legislation`` on the declared even profile (the ``::pre_legislation`` part of a split row,
      or a whole undated row): the profile's years committed before ``L``, of which the share committed
      on or after ``year``. The part's ``commissioning_year`` was set to the mean of those years, so the
      profile is rebuilt from the row's own development status, as the split built it;
    * anything committed before the source snapshot, or without a status: 0.

    Summed over the parts of an asset, ``capacity x share`` is the capacity :func:`apply_legislation_year`
    at ``year`` would have made eligible (a test holds this).
    """
    leads = load_lead_times() if leads is None else leads
    out = np.zeros(len(register))
    status = register["legislation_status"].astype(str).to_numpy() if "legislation_status" in register.columns         else np.full(len(register), "", dtype=object)
    for i, (_, row) in enumerate(register.iterrows()):
        if status[i] == STATUS_POST:
            out[i] = 1.0
            continue
        if status[i] != STATUS_PRE:
            continue
        own_year = pd.to_numeric(row.get("legislated_year"), errors="coerce")
        if pd.isna(own_year) or int(year) > int(own_year):
            raise ValueError(f"{row.get('asset_id')}: a share at {year} needs a register re-classed at a later year")
        basis = _text(row.get("legislation_basis"))
        # The ::pre_legislation part of a split row is always on the profile (a dated row is never split), and
        # the split does not copy the basis onto it.
        split_part = str(row.get("asset_id", "")).endswith(f"::{PRE_LEGISLATION_COHORT}")
        if basis.endswith("even_build_profile") or (split_part and basis == ""):
            undated = dict(row)
            undated["commissioning_year"] = np.nan
            years, _ = commissioning_years(undated, policy_start_year=policy_start_year, model_year=model_year,
                                           snapshot_year=snapshot_year, vintaged=vintaged)
            lead = _lead(leads, row)
            before = [y - lead for y in years if y - lead < int(own_year)]
            out[i] = float(np.mean([c >= int(year) for c in before])) if before else 0.0
            continue
        committed = pd.to_numeric(row.get("committed_year"), errors="coerce")
        out[i] = 1.0 if pd.notna(committed) and float(committed) >= int(year) else 0.0
    return out


def _lead(leads: pd.DataFrame, row: Mapping) -> float:
    """The lead :func:`eligible_share` applied to a profiled row (award lead for a modelled contract)."""
    carrier = _text(row.get("carrier"))
    support = _text(row.get("support_type")).lower()
    if support in CONTRACT_SUPPORT and _text(row.get("cfd_round")) != "":
        return float(leads.at[carrier, "cfd_award_lead_years"])
    return float(leads.at[carrier, "investment_lead_years"])


def _split_ids(asset_id: str) -> tuple[str, str]:
    """``(ineligible id, eligible id)`` for a row split in two."""
    base = asset_id[: -len("::new_build")] if asset_id.endswith("::new_build") else asset_id
    return f"{base}::{PRE_LEGISLATION_COHORT}", f"{base}::new_build"


def apply_legislation_year(register: pd.DataFrame, *, legislated_year: int, policy_start_year: int,
                           model_year: int, transfer_year: int, leads: pd.DataFrame | None = None,
                           snapshot_year: int = 2025, vintaged: bool = False) -> pd.DataFrame:
    """Re-class the new build that was committed before the legislation year (see the module docstring).

    Args:
        register: The register after the cohort split (and, for a vintaged year, the build-year join).
        legislated_year: ``L``; on or after ``policy_start_year``.
        policy_start_year: The first commissioning year of the post-policy cohort (2025).
        model_year: The register's model year.
        transfer_year: P3's purchase date (2030); later builds committed before ``L`` are
            ``in_construction`` (only a vintaged register has them).
        leads: :func:`load_lead_times` (the default table when None).
        snapshot_year: The REPD snapshot's year (pipeline projects commission after it).
        vintaged: Commissioning years are the build-year join's.

    Returns:
        The register with ``legislated_year``, ``legislation_status``, ``legislation_basis``,
        ``committed_year`` and, on split rows, ``cohort_share``.
    """
    if int(legislated_year) < int(policy_start_year):
        raise ValueError(f"legislated year {legislated_year} precedes policy_start_year {policy_start_year}")
    leads = load_lead_times() if leads is None else leads
    out = register.copy()
    for column, default in (("legislation_status", None), ("legislation_basis", None),
                            ("committed_year", np.nan), ("cohort_share", np.nan)):
        if column not in out.columns:
            out[column] = default
    for column in ("legislation_status", "legislation_basis", "finance_vintage", "contract_vintage", "asset_id",
                   "vintage_method"):
        if column in out.columns:
            out[column] = out[column].astype(object)
    out["legislated_year"] = int(legislated_year)
    vintage = out["finance_vintage"].astype(str)
    out.loc[vintage.ne("new_build"), "legislation_status"] = STATUS_FLEET
    eligible_rows = out["finance_eligible"].eq(True) if "finance_eligible" in out.columns else pd.Series(True, index=out.index)
    candidates = out.index[vintage.eq("new_build") & eligible_rows]
    # Finance-ineligible new build (interconnector imports, CHP, ...) is no case's: left as it is.
    out.loc[vintage.eq("new_build") & ~eligible_rows, "legislation_status"] = STATUS_POST
    added, dropped = [], []
    for index in candidates:
        row = out.loc[index]
        decided = eligible_share(row, leads, legislated_year=int(legislated_year),
                                 policy_start_year=int(policy_start_year), model_year=int(model_year),
                                 snapshot_year=int(snapshot_year), vintaged=vintaged)
        share = decided["share"]
        out.at[index, "legislation_basis"] = decided["basis"]
        out.at[index, "committed_year"] = decided["committed_year"]
        if share >= 1.0 - 1e-12:
            out.at[index, "legislation_status"] = STATUS_POST
            continue
        # The part committed before L: commissioned by the transfer year it is existing plant.
        late = [y for y in decided["years"] if y > transfer_year]
        before_vintage = "in_construction" if decided["years"] and len(late) == len(decided["years"]) else "legacy"
        # A profiled row's committed-before part is dated at the mean of its years (the valuation
        # would otherwise date it at its carrier's median existing plant).
        early = [y for y in decided["years"]
                 if y - _lead(leads, row) < int(legislated_year)] if decided["basis"].endswith("even_build_profile") else []
        dated_at = float(np.mean(early)) if early else np.nan
        whole_row = share <= 1e-12
        if whole_row and not str(row["asset_id"]).endswith("::new_build"):
            out.at[index, "finance_vintage"] = before_vintage
            out.at[index, "legislation_status"] = STATUS_PRE
            if np.isfinite(dated_at):
                out.at[index, "commissioning_year"] = dated_at
            if _text(row.get("support_type")).lower() == "cfd":
                out.at[index, "contract_vintage"] = "pre_policy"
            continue
        # Split (a residual ::new_build cohort wholly committed before L is renamed the same way): the
        # eligible share stays new build, the rest is committed before L.
        asset_share = pd.to_numeric(row.get("cohort_share"), errors="coerce")
        if pd.isna(asset_share):
            legacy_share = pd.to_numeric(row.get("cohort_legacy_share"), errors="coerce")
            suffix_new = str(row["asset_id"]).endswith("::new_build")
            asset_share = (1.0 - float(legacy_share)) if suffix_new and pd.notna(legacy_share) else 1.0
        before_id, after_id = _split_ids(str(row["asset_id"]))
        before = row.copy()
        for column in SPLIT_COLUMNS:
            if column in out.columns:
                value = pd.to_numeric(row[column], errors="coerce")
                out.at[index, column] = float(value) * share
                before[column] = float(value) * (1.0 - share)
        out.at[index, "asset_id"] = after_id
        out.at[index, "cohort_share"] = float(asset_share) * share
        out.at[index, "legislation_status"] = STATUS_POST
        before["asset_id"] = before_id
        if np.isfinite(dated_at):
            before["commissioning_year"] = dated_at
        before["cohort_share"] = float(asset_share) * (1.0 - share)
        # The valuation reads a legacy cohort's own share of its asset (capacity / share = asset MW).
        before["cohort_legacy_share"] = float(asset_share) * (1.0 - share)
        before["finance_vintage"] = before_vintage
        before["legislation_status"] = STATUS_PRE
        if _text(row.get("support_type")).lower() == "cfd":
            before["contract_vintage"] = "pre_policy"
        added.append(before)
        if whole_row:
            dropped.append(index)
    if dropped:
        out = out.drop(index=dropped)
    if added:
        out = pd.concat([out, pd.DataFrame(added)], ignore_index=True)
        # The legacy sibling of a split residual row keeps its own share explicitly, so every part of the
        # asset settles on cohort_share alike.
        legacy_cohort = out["asset_id"].astype(str).str.endswith("::legacy") & out["cohort_share"].isna()
        bases = {str(a)[: -len("::pre_legislation")] for a in out["asset_id"].astype(str)
                 if str(a).endswith("::pre_legislation")}
        siblings = legacy_cohort & out["asset_id"].astype(str).str[: -len("::legacy")].isin(bases)
        out.loc[siblings, "cohort_share"] = pd.to_numeric(out.loc[siblings, "cohort_legacy_share"], errors="coerce")
    return out


#: F28 (d): the carriers whose plant is rebuilt like for like at the end of its life in an undated (2030)
#: register. Gas past its life is life-extended under the Capacity Market in CP30, not rebuilt; hydro and
#: pumped hydro are refurbished on the 2050 stock-flow's schedule, not in 2030 (declared).
REBUILT_CARRIERS = ("wind_onshore", "embedded_wind", "wind_offshore", "solar_pv", "embedded_solar", "marine",
                    "Battery", "battery")
REBUILT_METHOD = "rebuilt_at_end_of_life"


def rebuild_at_end_of_life(register: pd.DataFrame, lives: Mapping[str, float], *, policy_start_year: int,
                           model_year: int, carriers=REBUILT_CARRIERS) -> pd.DataFrame:
    """Plant reaching the end of its economic life is rebuilt, not life-extended (F28 (d), user 26 Oct 2026).

    A dated existing row of a rebuilt carrier whose commissioning year plus its technology's life
    falls from ``policy_start_year`` to the model year is repowered on its site as new build of
    that year (its commissioning year becomes it, at that year's new-build cost).  Whether the
    rebuild is the state's is the legislation year's test (:func:`apply_legislation_year`, run
    after this): a rebuild committed before it is private (P3 buys it with the fleet).  Plant past
    its life before ``policy_start_year`` was life-extended before any policy and stays existing.

    Args:
        register: The register after the crosswalk merge (``finance_technology``).
        lives: Economic life by finance technology (``technology_costs.csv``).
        policy_start_year: The first year a rebuild counts.
        model_year: The register's model year.
        carriers: The carriers rebuilt like for like.

    Returns:
        The register, rebuilt rows with ``finance_vintage`` ``new_build`` and ``vintage_method``
        :data:`REBUILT_METHOD` (``rebuilt_from_year`` keeps the old commissioning year).
    """
    out = register.copy()
    if "rebuilt_from_year" not in out.columns:
        out["rebuilt_from_year"] = np.nan
    life = out["finance_technology"].map({str(k): float(v) for k, v in lives.items()}) \
        if "finance_technology" in out.columns else pd.Series(np.nan, index=out.index)
    built = pd.to_numeric(out["commissioning_year"], errors="coerce")
    end = built + pd.to_numeric(life, errors="coerce")
    eligible = out["finance_eligible"].eq(True) if "finance_eligible" in out.columns else pd.Series(True, index=out.index)
    rebuilt = (out["finance_vintage"].astype(str).eq("legacy") & eligible & out["carrier"].astype(str).isin(carriers)
               & end.between(int(policy_start_year), int(model_year)))
    out.loc[rebuilt, "rebuilt_from_year"] = built[rebuilt]
    out.loc[rebuilt, "commissioning_year"] = end[rebuilt]
    out["finance_vintage"] = out["finance_vintage"].astype(object)
    out["vintage_method"] = out["vintage_method"].astype(object)
    out.loc[rebuilt, "finance_vintage"] = "new_build"
    out.loc[rebuilt, "vintage_method"] = REBUILT_METHOD
    return out


def summary(register: pd.DataFrame) -> dict:
    """Capacity by legislation status and carrier (MW), for the run's audit."""
    frame = register.assign(capacity_mw=pd.to_numeric(register["capacity_mw"], errors="coerce").fillna(0.0))
    by_status = frame.groupby("legislation_status")["capacity_mw"].sum().round(3).to_dict()
    pre = frame.loc[frame["legislation_status"].eq(STATUS_PRE)]
    return {"capacity_mw_by_status": by_status,
            "pre_legislation_mw_by_carrier": pre.groupby("carrier")["capacity_mw"].sum().round(3).to_dict()}
