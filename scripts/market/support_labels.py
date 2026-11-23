"""Support labels for a long-range model year's balancing stage (FES 2050 plan, task BM2050; decisions 4, 6,
22, 35; year-generic stage 5: the year is the scenario's ``modelled_year``, ``--model-year``).

A long-range network (HT50 and its successors) carries no support labels: its generators
are clustered (one wind farm row per bus and carrier) and each holds several build years,
so one row can be part AR7, part post-AR7 CfD (``PAR7_<year>``), part DPA and part
merchant. The contract each build year holds in the model year is decided by the vintaged
finance register (task Y10, ``fleet_vintages.label_support_contracts``) from the
FES stock-flow and the contract table of every CfD round (``cfd_strikes.known_contracts``).
This module runs **that same code** on
the unsolved parent network, so the balancing stage bids on exactly the contract
mix the register settles, without reading the register (which needs the solved
year, so would make the bids circular).

Path (b) of the BM2050 brief, declared: labels are attached at solve time from
this table, not at build. A build-time label would have to be a blend anyway (a
clustered generator holds many vintages), and the table keeps one source of
truth for "which build year holds which contract".

Output: one row per (network component, build-year cohort, contract), with the
cohort's capacity weight in its component, the contract (support type, round,
contract id, live share), and for CfDs the strike in 2024 prices (the bill's
``cfd_strikes.strike_table_for_bill`` keys for ``PAR7_<year>``; a known
contract's own strike for every earlier round, so Scottish AR7 offshore keeps its
89.49), its negative-pricing rule and its reference price; for DPAs the reference
CCGT's short-run cost and the variable-payment margin (``support_regime_2050``), so
the BM can price the DPA's net cost; for the Renewables Obligation (model years before
its end, 31 Mar 2037) the cohort's ROC banding (``support_schemes.csv``), which the BM
prices at banding x buyout as the 2030 chain does.

``--kind dpa_dispatch`` writes the rate-free DPA dispatch table instead (one row per
component: its live DPA weight, reference CCGT cost and margin), which Stage 1 and the
seasonal pass read (``marginal_cost_basis.dpa_dispatch.file``). It holds no strike, so
a financing-rate change re-runs only the balancing months.

Run by Snakemake (``rules/market.smk`` rules ``build_support_labels`` and ``build_dpa_dispatch``)::

    python scripts/market/support_labels.py NETWORK OUT.csv --summary OUT.json --model-year YEAR [options]
"""

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("support_labels")

POLICY_START_YEAR = 2025
COHORT_SEPARATOR = "::"
#: Support types a label can carry (the register's vocabulary, lower case; ``ro`` and
#: ``fit`` from :data:`SUPPORT_SCHEMES` in model years before each scheme's end).
SUPPORT_TYPES = ("cfd", "dpa", "rab", "ro", "fit", "merchant")
#: The round a contract's negative-pricing term is read as
#: (``revenue_tracking.negative_price_eligibility``): AR4 onward pay nothing in a
#: negative hour, AR2 and AR3 after six, HPC's contract has no such term (``cfd_strikes.cfd_settlement``).
NEGATIVE_PRICE_ROUND = {"ar4_plus": "AR4", "six_hour": "AR3", "none": "HPC"}
#: The support schemes with end dates (RO, FiT): which cohorts outside a CfD hold them.
SUPPORT_SCHEMES = ROOT / "data" / "market" / "support" / "support_schemes.csv"
#: Kinds of table the script writes.
KINDS = ("labels", "dpa_dispatch")
#: Carriers whose CfD settles on the season-ahead baseload reference (Y11).
BASELOAD_CARRIERS = ("nuclear", "CCS Biomass")
#: Carriers kept off contract-value pricing in the BM (config/defaults/base/support.yaml
#: support.bm_bids.exclude_carriers; must match).
UNPRICED_CARRIERS = ("nuclear", "CCS Biomass", "biomass")
#: DPA carrier -> ``support_regime_2050`` scheme carrier.
DPA_SCHEME_CARRIER = {"CCS Gas": "CCS Gas", "H2_turbine": "Hydrogen"}

LABEL_COLUMNS = (
    "component", "component_type", "carrier", "cohort_id", "stock_flow_technology", "build_year",
    "cohort_capacity_mw", "weight", "support_type", "cfd_round", "cfd_contract_id", "live_share",
    "strike_gbp_per_mwh_2024", "strike_key", "negative_price_rule", "reference_price",
    "dpa_reference_srmc_gbp_per_mwh", "dpa_margin_gbp_per_mwh", "support_basis", "ro_banding",
)
#: The rate-free DPA dispatch table (one row per component; ``support_bids.apply_dpa_dispatch``).
DPA_DISPATCH_COLUMNS = ("component", "component_type", "carrier", "dpa_cohorts", "dpa_weight",
                        "dpa_reference_srmc_gbp_per_mwh", "dpa_margin_gbp_per_mwh")


def component_name(asset_id) -> str:
    """``WIND_X::b2035`` -> ``WIND_X``: the network component a register cohort belongs to."""
    return str(asset_id).split(COHORT_SEPARATOR, 1)[0]


def fleet_vintage_mix(*, workbook, pathway, rule, life_source, repd, dukes, lives_path, closures,
                      target_year: int, base_year: int = 2024, life_overrides=None,
                      repowering=None, exclude_countries=(), stale_records=None, refurbishment=None):
    """The stock-flow target-year mix and lives, exactly as ``fleet_vintages.py`` (rule
    ``build_public_ownership_fleet_vintages``) computes them for the run group, with the
    group's per-technology life overrides (decision 48) applied to both, decision 62's
    repowering rule and evidence exclusions, and F28 (d)'s refurbishment rule when given."""
    from scripts.FES.fes2025_targets import read_es1
    from scripts.public_ownership import fleet_vintages as fv

    overrides = fv.parse_life_overrides(life_overrides)
    es1 = read_es1(workbook)
    trajectory = fv.fes_capacity_trajectory(es1, pathway=pathway, years=range(base_year, target_year + 1))
    evidence = fv.existing_fleet_evidence(pd.read_csv(repd, encoding="latin1", low_memory=False),
                                          pd.read_csv(dukes), base_year=base_year,
                                          exclude_countries=exclude_countries or (),
                                          exclude_ref_ids=fv.load_stale_records(stale_records))
    lives = fv.load_lives(lives_path, source=life_source, overrides=overrides)
    result = fv.build_fleet_vintages(trajectory, evidence, lives, base_year=base_year, target_year=target_year,
                                     rule=rule, closure_years=fv.load_closure_dates(closures),
                                     repowering=fv.parse_repowering(repowering),
                                     refurbishment=fv.parse_refurbishment(refurbishment))
    # The register rule reads the mix back from its CSV; round-trip it the same way.
    mix = pd.read_csv(io.StringIO(result["mix"].to_csv(index=False)))
    check_vintage_sources(mix)
    return mix, fv.load_lives(source=life_source, overrides=overrides)


#: Stock-flow ``source`` values the labelling knows how to label (``fleet_vintages.stock_flow``):
#: new build, and the 2024 stock dated from REPD, DUKES or spread evenly. The contract and
#: scheme rules (known contracts by build year, PAR7/DPA from 2031, RO by accreditation date)
#: are written for these; a cohort of any other kind (e.g. a refurbishment) is refused until
#: its contract treatment is decided, never labelled as if it were one of them.
KNOWN_VINTAGE_SOURCES = frozenset({"gross_addition", "base_stock_repd", "base_stock_dukes",
                                   "base_stock_uniform_fallback"})


def check_vintage_sources(mix: pd.DataFrame) -> None:
    """Refuse a stock-flow mix holding a cohort ``source`` the labelling has no rule for.

    Raises:
        ValueError: An unknown source (named with its technologies), or no ``source`` column.
    """
    if "source" not in mix.columns:
        raise ValueError("the fleet-vintage mix has no source column; the labels cannot tell new build from stock")
    unknown = sorted(set(mix["source"].astype(str)) - KNOWN_VINTAGE_SOURCES)
    if unknown:
        techs = sorted(set(mix.loc[mix["source"].astype(str).isin(unknown), "technology"].astype(str)))
        raise ValueError(f"fleet-vintage cohorts of source {unknown} ({techs}) have no support-label rule; "
                         "add their contract treatment to support_labels.KNOWN_VINTAGE_SOURCES first")


def register_cohorts(network_path, *, mix, lives, crosswalk, crosswalk_supplement, award_years=None,
                     eligible_award_years_from=None, legacy_plant_rounds=frozenset(),
                     support_type_overrides=None, join_mode: str = "pro_rata",
                     component_types=("Generator", "StorageUnit"), model_year: int,
                     policy_start_year: int = POLICY_START_YEAR, repowering=None,
                     no_cfd_on_repowering=(), register_options=None) -> pd.DataFrame:
    """The vintaged register's labelled cohorts for one network (``build_asset_finance_register``).

    The same call the register rule makes, on the parent network: the labels do not
    depend on dispatch, so the unsolved network gives the solved months' labels.
    ``register_options``: the register rule's F28 keywords (:func:`f28_register_options`).
    """
    from scripts.public_ownership.build_asset_finance_register import build_asset_finance_register

    return build_asset_finance_register(
        [network_path], crosswalk, model_year=model_year, policy_start_year=policy_start_year,
        component_types=tuple(component_types), support_type_overrides=support_type_overrides,
        award_years=award_years, eligible_award_years_from=eligible_award_years_from,
        legacy_plant_rounds=legacy_plant_rounds, fleet_vintage_mix=mix, fleet_vintage_lives=lives,
        fleet_vintage_mode=join_mode, crosswalk_supplement=crosswalk_supplement,
        fleet_vintage_repowering=repowering or None, no_cfd_on_repowering=tuple(no_cfd_on_repowering or ()),
        **(register_options or {}),
    )


def f28_register_options(args) -> dict:
    """The F28 keywords of ``build_asset_finance_register`` from the register rule's flags (the legislation
    year and its lead times, the end-of-life rebuild, the small-scale exclusion), read as its own CLI reads them."""
    from scripts.public_ownership.build_asset_finance_register import _load_lead_times, micro_capacity_by_carrier

    return dict(
        small_scale_data_sources=tuple(args.small_scale_data_source),
        legislated_year=args.legislated_year, transfer_year=args.transfer_year,
        lead_times=None if args.lead_times is None else _load_lead_times(args.lead_times),
        snapshot_year=args.snapshot_year,
        rebuild_lives=(None if args.rebuild_lives is None else
                       pd.read_csv(args.rebuild_lives).set_index("technology")["economic_life_years"].astype(float)
                       .to_dict()),
        micro_mw_by_carrier=(micro_capacity_by_carrier(pd.read_csv(args.micro_capacity_lines))
                             if args.micro_capacity_lines else None),
    )


#: The strike variant that bids the bill's own contract strikes (FES 2050 decision 49 (b)).
CALIBRATED_VARIANT = "calibrated"


def calibrated_strike_frame(study: str | None = None, *, model_year: int, snapshot=None) -> pd.DataFrame:
    """The bill's contract strike table: the study's decision-41 basis with its break-even calibration.

    The same call the bill's strike table makes (``contract_book_2050.build_strike_table_file``
    on the study's resolved config): ``basis_strike_frame`` with the calibration in
    ``public_ownership.vintage_capital_2050.strike_calibration``, insured on the study's D4(a)
    settings. The study's config is resolved from ``config/`` as the insurance block already is
    (``cfd_strikes.finance_insurance_config``). The calibrated variant is the HT50 2050
    setting; a year without a calibration file uses ``central`` (:func:`strike_table`).

    Args:
        study: The finance study (default ``cfd_strikes.DEFAULT_STUDY``).
        model_year: The scenario's model year.
        snapshot: The workbook snapshot (default ``cfd_strikes.load_snapshot()``).

    Raises:
        ValueError: The study has no term basis or no calibration file.
    """
    from scripts.market import cfd_strikes as cfd
    from scripts.public_ownership import contract_book_2050 as cb
    from scripts.public_ownership import vintage_capital_2050 as vk
    from scripts.public_ownership.study_config import load_studies

    study = study or cfd.DEFAULT_STUDY
    config = load_studies(str(ROOT / "config"), validate=False).configs[study]
    basis = vk.basis_from_config(config)
    if not basis["enabled"] or basis["strike_calibration"] is None:
        raise ValueError(f"{study}: the calibrated strike variant needs vintage_capital_2050 with a term amortisation "
                         "and a strike_calibration file")
    return cb.basis_strike_frame(basis, snapshot=cfd.load_snapshot() if snapshot is None else snapshot,
                                 insurance_config=cfd.finance_insurance_config(study=study),
                                 model_year=model_year, calibration=vk.load_calibration(basis["strike_calibration"]))


def strike_table(variant: str, known: pd.DataFrame, *, model_year: int, study: str | None = None, snapshot=None):
    """The ``StrikeTable`` the labels price CfD cohorts on.

    ``calibrated``: the bill's contract strikes (:func:`calibrated_strike_frame`), BECCS rows
    left out so its cohorts stay unpriced as before (the BM never prices them on contract
    value). Any other variant (``cfd_strikes.STRIKE_VARIANTS``): the levelised strikes at that
    variant (decision 6's economic-life central and its sensitivities), on the study's D4(a)
    insurance, or ``config/defaults/``'s for a scenario that is not a finance study (a
    private-only new year). ``central`` needs no calibration file, so it is the strike basis
    of a new year.
    """
    from scripts.market import cfd_strikes as cfd

    if variant == CALIBRATED_VARIANT:
        frame = calibrated_strike_frame(study, model_year=model_year, snapshot=snapshot)
        return cfd.strike_table_from_frame(frame.loc[frame["technology"].astype(str).ne("beccs")])
    if variant not in cfd.STRIKE_VARIANTS:
        raise ValueError(f"strike variant {variant!r}: one of {sorted(cfd.STRIKE_VARIANTS)} or {CALIBRATED_VARIANT!r}")
    insurance = cfd.finance_insurance_config(study=study or cfd.DEFAULT_STUDY, defaults_without_study=True)
    kwargs = dict(cfd.STRIKE_VARIANTS[variant], insurance_config=insurance)
    strikes = cfd.modelled_strikes(model_year=model_year, snapshot=snapshot, variants={variant: kwargs})
    return cfd.strike_table_for_bill(strikes, known, variant=variant)


def _support(value) -> str:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip().lower()
    return text if text in SUPPORT_TYPES else "merchant"


# ── support schemes with end dates (RO, FiT) ─────────────────────────────────

#: Support basis of a cohort labelled from :data:`SUPPORT_SCHEMES`.
SUPPORT_BASIS_SCHEME = "support_scheme"


def load_support_schemes(path: str | Path = SUPPORT_SCHEMES) -> pd.DataFrame:
    """The support schemes with end dates (``support_schemes.csv``), provenance and dates checked."""
    table = pd.read_csv(path)
    for column in ("source", "source_page", "status"):
        blank = table[column].isna() | table[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(f"{path}: rows without {column}")
    for column in ("opened", "closed_to_new"):
        table[column] = pd.to_datetime(table[column])
    table["scheme_end"] = pd.to_datetime(table["scheme_end"])
    if table.duplicated(["scheme", "technology"]).any():
        raise ValueError(f"{path} repeats a (scheme, technology) row")
    if table["scheme"].str.lower().isin(["ro"]).any() and table.loc[table["scheme"].str.lower().eq("ro"),
                                                                   "ro_banding"].isna().any():
        raise ValueError(f"{path}: every RO row needs its ro_banding")
    return table


def _year_fraction(date, *, end_of_day: bool = True) -> float:
    """A date as a decimal year: ``2037-03-31`` -> about 2037.25 (its end; its start with ``end_of_day=False``)."""
    stamp = pd.Timestamp(date) + pd.Timedelta(days=1 if end_of_day else 0)
    start = pd.Timestamp(year=stamp.year, month=1, day=1)
    days = (pd.Timestamp(year=stamp.year + 1, month=1, day=1) - start).days
    return stamp.year + (stamp - start).days / days


def scheme_live_share(build_year: int, model_year: int, *, opened: float, closed: float, term_years: float,
                      scheme_end: float | None, steps: int = 4000) -> float:
    """Expected share of calendar ``model_year`` a build year's cohort holds a scheme.

    Commissioning is spread evenly through the build year (as ``live_fraction``); a plant
    commissioned at ``t`` holds the scheme if ``opened <= t < closed`` and until
    ``min(t + term, scheme_end)``. The share is the mean over ``t`` of the part of the model
    year before that end, for accredited plant (midpoint rule on ``steps`` points; exact to
    well under 0.1%). Zero from the year after the scheme's end.
    """
    t = build_year + (np.arange(steps) + 0.5) / steps
    end = t + float(term_years)
    if scheme_end is not None and np.isfinite(scheme_end):
        end = np.minimum(end, scheme_end)
    held = (t >= opened) & (t < closed)
    live = np.clip(end - float(model_year), 0.0, 1.0)
    return float(np.mean(np.where(held, live, 0.0)))


def label_support_schemes(cohorts: pd.DataFrame, schemes: pd.DataFrame, *, model_year: int) -> pd.DataFrame:
    """Label the generation cohorts outside any contract that hold a support scheme in the model year.

    A cohort the register leaves merchant (no CfD, DPA or RAB label) of a technology a scheme
    row covers takes the scheme where :func:`scheme_live_share` is positive: support type
    ``RO`` (bid at banding x buyout in the BM) or ``FiT`` (embedded plant outside the BM,
    bid as merchant), ``cfd_live_share`` the share, ``ro_banding`` the scheme's banding.
    A FiT row labels only the ``cohort_share`` of a cohort it declares; a row without one
    labels nothing (the register does not separate small-scale plant). The RO ends on 31
    March 2037, so a 2040 or 2050 year has no scheme labels and is unchanged.

    Returns:
        A copy of ``cohorts`` with ``ro_banding`` added (NaN where no RO).
    """
    out = cohorts.copy()
    out["ro_banding"] = np.nan
    if schemes is None or schemes.empty:
        return out
    support = out["support_type"].map(_support)
    contract = out["cfd_contract_id"] if "cfd_contract_id" in out.columns else pd.Series(None, index=out.index)
    open_rows = (out["component_type"].astype(str).eq("Generator") & support.eq("merchant")
                 & (contract.isna() | contract.astype(str).str.strip().eq("")))
    year = pd.to_numeric(out["build_year"], errors="coerce")
    technology = out["stock_flow_technology"].astype(str) if "stock_flow_technology" in out.columns else None
    if technology is None:
        return out
    for s in schemes.itertuples(index=False):
        name = str(s.scheme)
        if name.lower() == "fit" and pd.isna(getattr(s, "cohort_share", np.nan)):
            continue                                     # not assigned until a FiT share is declared
        share_of_cohort = 1.0 if pd.isna(getattr(s, "cohort_share", np.nan)) else float(s.cohort_share)
        rows = open_rows & technology.eq(str(s.technology)) & year.notna()
        end = _year_fraction(s.scheme_end) if pd.notna(s.scheme_end) else None
        for i in out.index[rows]:
            share = scheme_live_share(int(year[i]), int(model_year), opened=_year_fraction(s.opened, end_of_day=False),
                                      closed=_year_fraction(s.closed_to_new), term_years=float(s.term_years),
                                      scheme_end=end) * share_of_cohort
            if share <= 0:
                continue
            out.at[i, "support_type"] = "RO" if name.lower() == "ro" else name
            out.at[i, "cfd_round"] = name.upper()
            out.at[i, "cfd_live_share"] = share
            out.at[i, "support_basis"] = SUPPORT_BASIS_SCHEME
            if name.lower() == "ro":
                out.at[i, "ro_banding"] = float(s.ro_banding)
            open_rows[i] = False
    return out


def _component_cohorts(cohorts: pd.DataFrame) -> pd.DataFrame:
    """The generation cohorts (generators and hydrogen-turbine links) with their component and capacity weight."""
    frame = cohorts.loc[cohorts["component_type"].astype(str).isin(("Generator", "Link"))].copy()
    frame["component"] = frame["asset_id"].map(component_name)
    capacity = pd.to_numeric(frame["capacity_mw"], errors="coerce").fillna(0.0)
    total = capacity.groupby(frame["component"]).transform("sum")
    frame["weight"] = np.where(total > 0, capacity / total.where(total > 0, 1.0), 0.0)
    frame["support"] = frame["support_type"].map(_support)
    return frame


def label_table(cohorts: pd.DataFrame, *, strike_table, known: pd.DataFrame, dpa_reference_srmc: float,
                dpa_margin: float, unpriced_carriers=UNPRICED_CARRIERS) -> pd.DataFrame:
    """One row per generation cohort with the contract it holds in the model year and what the BM needs to price it.

    Args:
        cohorts: :func:`register_cohorts` output (with :func:`label_support_schemes` applied).
        strike_table: ``cfd_strikes.strike_table_for_bill`` (a ``StrikeTable``, 2024 prices).
        known: ``cfd_strikes.known_contracts(model_year)``.
        dpa_reference_srmc: The reference unabated CCGT's short-run cost (GBP/MWh, 2024 prices).
        dpa_margin: The DPA variable payment's margin (GBP/MWh).
        unpriced_carriers: Carriers the BM never prices on contract value
            (``support.bm_bids.exclude_carriers``): a CfD cohort of one may
            lack a strike here (BECCS's electricity-leg strike is ``support_regime_2050``'s,
            not the CfD table's) and is kept with a blank strike.

    Returns:
        :data:`LABEL_COLUMNS`. ``weight`` sums to 1 over each component.

    Raises:
        ValueError: A CfD cohort has no strike, or a known contract id is not in ``known``.
    """
    frame = _component_cohorts(cohorts)
    phases = known.assign(key=known["contract_id"].astype(str) + "#" + known["phase"].astype(int).astype(str))
    phases = phases.set_index("key")
    rows, missing = [], []
    for r in frame.itertuples(index=False):
        row = {"component": r.component, "component_type": r.component_type, "carrier": r.carrier,
               "cohort_id": r.asset_id, "stock_flow_technology": getattr(r, "stock_flow_technology", None),
               "build_year": getattr(r, "build_year", np.nan), "cohort_capacity_mw": float(r.capacity_mw),
               "weight": float(r.weight), "support_type": r.support, "cfd_round": r.cfd_round,
               "cfd_contract_id": getattr(r, "cfd_contract_id", None), "live_share": np.nan,
               "strike_gbp_per_mwh_2024": np.nan, "strike_key": "", "negative_price_rule": "",
               "reference_price": "", "dpa_reference_srmc_gbp_per_mwh": np.nan, "dpa_margin_gbp_per_mwh": np.nan,
               "support_basis": getattr(r, "support_basis", None), "ro_banding": np.nan}
        live = pd.to_numeric(pd.Series([getattr(r, "cfd_live_share", np.nan)]), errors="coerce").iloc[0]
        if r.support == "cfd":
            contract = row["cfd_contract_id"]
            if isinstance(contract, str) and contract.strip():
                if contract not in phases.index:
                    raise ValueError(f"{r.asset_id}: known contract {contract!r} is not in the known contracts")
                c = phases.loc[contract]
                row.update(strike_gbp_per_mwh_2024=float(c["strike_gbp_per_mwh_2024"]),
                           strike_key=contract, negative_price_rule=str(c["negative_price_rule"]),
                           reference_price=str(c["reference_price"]),
                           live_share=float(c["live_share"]) if pd.isna(live) else float(live))
            else:
                strike, key, fallback = strike_table.lookup(r.carrier, r.cfd_round)
                if (strike is None or fallback) and r.carrier in set(unpriced_carriers):
                    row.update(strike_key="not_priced_excluded_carrier", live_share=1.0 if pd.isna(live) else float(live))
                    rows.append(row)
                    continue
                if strike is None or fallback:
                    missing.append((r.asset_id, r.carrier, r.cfd_round))
                    continue
                row.update(strike_gbp_per_mwh_2024=float(strike), strike_key=key, negative_price_rule="ar4_plus",
                           reference_price=("baseload_seasonal" if r.carrier in BASELOAD_CARRIERS
                                            else "intermittent_hourly"),
                           live_share=1.0 if pd.isna(live) else float(live))
        elif r.support in ("ro", "fit"):
            # A support scheme (label_support_schemes): the share of the year it is held; RO's banding.
            row.update(live_share=1.0 if pd.isna(live) else float(live),
                       ro_banding=float(getattr(r, "ro_banding", np.nan)))
        elif r.support == "dpa":
            row.update(live_share=1.0 if pd.isna(live) else float(live),
                       dpa_reference_srmc_gbp_per_mwh=float(dpa_reference_srmc),
                       dpa_margin_gbp_per_mwh=float(dpa_margin))
        rows.append(row)
    if missing:
        raise ValueError(f"{len(missing)} CfD cohorts have no strike in the model year (first: {missing[:5]}); "
                         "an unlabelled CfD is an error to find, not a strike to guess")
    out = pd.DataFrame(rows, columns=list(LABEL_COLUMNS))
    sums = out.groupby("component")["weight"].sum()
    bad = sums[(sums - 1.0).abs() > 1e-6]
    if len(bad):
        raise ValueError(f"cohort weights do not sum to 1 for {len(bad)} components (first {bad.index[:3].tolist()})")
    return out


def _as_label_csv_reads(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` as a reader of the label table's CSV sees it: written by pandas, parsed by its default reader.

    That parser is not exact to the last bit; reproducing it keeps the DPA table's numbers
    identical to the ones Stage 1 took from the label table before the table was split out.
    """
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer)


def dpa_dispatch_table(cohorts: pd.DataFrame, *, dpa_reference_srmc: float, dpa_margin: float) -> pd.DataFrame:
    """The rate-free DPA dispatch table: per component, its live DPA weight, reference CCGT cost and margin.

    Exactly what ``support_bids.apply_dpa_dispatch`` took from the label table: the DPA
    cohorts' ``weight x live_share`` summed per component in the label table's row order,
    from the values as the label table's CSV is parsed (:func:`_as_label_csv_reads`), so the
    sum, the reference cost and the margin are bit for bit the label route's. It holds no
    strike: it depends on the fleet, the contract terms and the fuel and carbon prices,
    never on a financing rate. ``dpa_cohorts`` counts a component's DPA cohorts (0: it holds
    none; listed so the reader can tell a DPA-carrier link without a label from one without
    a DPA). The reader parses it exactly (``float_precision="round_trip"``).
    """
    frame = _component_cohorts(cohorts)
    dpa = frame["support"].eq("dpa")
    counts = dpa.groupby(frame["component"], sort=False).sum()
    first = frame.groupby("component", sort=False)[["component_type", "carrier"]].first()
    live = pd.to_numeric(frame["cfd_live_share"], errors="coerce")
    sub = _as_label_csv_reads(pd.DataFrame({
        "component": frame.loc[dpa, "component"].to_numpy(), "weight": frame.loc[dpa, "weight"].to_numpy(),
        "live_share": live[dpa].fillna(1.0).to_numpy(),
        "dpa_reference_srmc_gbp_per_mwh": float(dpa_reference_srmc), "dpa_margin_gbp_per_mwh": float(dpa_margin)}))
    sub["component"] = sub["component"].astype(str)
    weight = (pd.to_numeric(sub["weight"], errors="coerce").fillna(0.0)
              * pd.to_numeric(sub["live_share"], errors="coerce").fillna(1.0).clip(0.0, 1.0)).groupby(sub["component"]).sum()
    reference = sub.groupby("component")["dpa_reference_srmc_gbp_per_mwh"].max()
    margin = sub.groupby("component")["dpa_margin_gbp_per_mwh"].max()
    out = first.assign(dpa_cohorts=counts.astype(int)).reset_index()
    out["dpa_weight"] = out["component"].map(weight).fillna(0.0)
    out["dpa_reference_srmc_gbp_per_mwh"] = out["component"].map(reference)
    out["dpa_margin_gbp_per_mwh"] = out["component"].map(margin)
    return out[list(DPA_DISPATCH_COLUMNS)]


def summarise(labels: pd.DataFrame) -> dict:
    """Capacity by carrier and support type (MW, live-share weighted for CfD, DPA and the schemes)."""
    live = labels["live_share"].fillna(1.0).where(labels["support_type"].isin(["cfd", "dpa", "ro", "fit"]), 1.0)
    frame = labels.assign(mw=labels["cohort_capacity_mw"] * live,
                          expired_mw=labels["cohort_capacity_mw"] * (1.0 - live))
    by = frame.groupby(["carrier", "support_type"])[["mw", "expired_mw"]].sum().round(1)
    return {"components": int(labels["component"].nunique()), "cohorts": int(len(labels)),
            "capacity_mw_by_carrier_and_support": {f"{c}|{s}": v for (c, s), v in by["mw"].items()},
            "part_live_contract_mw_outside_term": {f"{c}|{s}": v for (c, s), v in by["expired_mw"].items() if v > 0}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("network")
    parser.add_argument("output")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model-year", type=int, required=True,
                        help="the scenario's modelled_year: contracts live, stock-flow target and strikes")
    parser.add_argument("--kind", default="labels", choices=KINDS,
                        help="labels (the BM's table) or dpa_dispatch (Stage 1's rate-free DPA table)")
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--pathway", default="Holistic Transition")
    parser.add_argument("--rule", default="economic_life_with_extension")
    parser.add_argument("--life-source", default="central")
    parser.add_argument("--life-override", nargs="*", default=[],
                        help="TECHNOLOGY=YEARS over the life source (decision 48), as the register group's stock-flow")
    parser.add_argument("--repowering", nargs="*", default=[],
                        help="TECHNOLOGY=SHARE:MAX_LIFE (decision 62), as the register group's stock-flow")
    parser.add_argument("--refurbishment", default=None,
                        help="TECH;TECH=FIRST:LAST (F28 (d), strict rule only), as the register group's stock-flow")
    parser.add_argument("--exclude-country", nargs="*", default=[], help="REPD countries left out (decision 62)")
    parser.add_argument("--stale-records", default=None, help="REPD ref_ids left out (decision 62)")
    parser.add_argument("--no-cfd-on-repowering", nargs="*", default=[],
                        help="technologies whose repowered capacity takes no CfD (decision 62: 'Solar PV')")
    parser.add_argument("--join-mode", default="pro_rata")
    parser.add_argument("--repd", required=True)
    parser.add_argument("--dukes", required=True)
    parser.add_argument("--lives", required=True)
    parser.add_argument("--closures", required=True)
    parser.add_argument("--crosswalk", required=True)
    parser.add_argument("--crosswalk-2050", required=True)
    parser.add_argument("--award-years", default=None)
    parser.add_argument("--eligible-award-years-from", type=int, default=None)
    parser.add_argument("--support-type-override", nargs="*", default=[])
    # F28: the register rule's flags (_po_register_flags), passed on as build_asset_finance_register takes them.
    parser.add_argument("--legislated-year", type=int, default=None)
    parser.add_argument("--transfer-year", type=int, default=None)
    parser.add_argument("--snapshot-year", type=int, default=2025)
    parser.add_argument("--lead-times", default=None)
    parser.add_argument("--rebuild-lives", default=None)
    parser.add_argument("--small-scale-data-source", nargs="*", default=[])
    parser.add_argument("--micro-capacity-lines", default=None)
    parser.add_argument("--support-schemes", default=str(SUPPORT_SCHEMES),
                        help="the RO and FiT table with end dates (support_schemes.csv)")
    parser.add_argument("--future-costs", required=True, help="the scenario's fuel and carbon price file (2024 prices)")
    parser.add_argument("--strike-variant", default="central",
                        help=f"a cfd_strikes.STRIKE_VARIANTS variant, or {CALIBRATED_VARIANT!r}: the bill's calibrated "
                             "strikes (labels only)")
    parser.add_argument("--study", default=None,
                        help="the finance study the strikes' insurance (and the calibrated variant) resolve "
                             "(default FES25_HT_2050; a scenario without a finance block falls back to config/defaults/)")
    parser.add_argument("--snapshot", default=None,
                        help="the workbook rate snapshot (default cfd_strikes.DEFAULT_SNAPSHOT)")
    parser.add_argument("--no-cost-calibration", action="store_true",
                        help="price the strikes on DESNZ capex without the AR7 offshore calibration (a check only)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from scripts.market import cfd_strikes as cfd
    from scripts.public_ownership import support_regime_2050 as sr
    from scripts.public_ownership.build_asset_finance_register import (
        load_award_years, load_legacy_plant_rounds, parse_support_type_overrides)
    from scripts.public_ownership.fleet_vintages import parse_repowering

    model_year = int(args.model_year)
    if args.no_cost_calibration:
        cfd.COST_CALIBRATION_ENABLED = False
    mix, lives = fleet_vintage_mix(workbook=args.workbook, pathway=args.pathway, rule=args.rule,
                                   life_source=args.life_source, repd=args.repd, dukes=args.dukes,
                                   lives_path=args.lives, closures=args.closures, life_overrides=args.life_override,
                                   repowering=args.repowering, exclude_countries=args.exclude_country,
                                   stale_records=args.stale_records, refurbishment=args.refurbishment,
                                   target_year=model_year)
    cohorts = register_cohorts(
        args.network, mix=mix, lives=lives, crosswalk=pd.read_csv(args.crosswalk),
        crosswalk_supplement=pd.read_csv(args.crosswalk_2050),
        award_years=load_award_years(args.award_years) if args.award_years else None,
        eligible_award_years_from=args.eligible_award_years_from,
        legacy_plant_rounds=load_legacy_plant_rounds(args.award_years) if args.award_years else frozenset(),
        support_type_overrides=parse_support_type_overrides(args.support_type_override), join_mode=args.join_mode,
        repowering=parse_repowering(args.repowering), no_cfd_on_repowering=args.no_cfd_on_repowering,
        register_options=f28_register_options(args), model_year=model_year)
    cohorts = label_support_schemes(cohorts, load_support_schemes(args.support_schemes), model_year=model_year)
    prices = json.loads(Path(args.future_costs).read_text(encoding="utf-8"))["assumptions"]
    gas, carbon = float(prices["fuel_prices"]["gas"]), float(prices["carbon"]["uk_ets"])
    params = sr.load_parameters()
    reference = float(sr.reference_ccgt_srmc(gas, carbon, params))
    margin = sr._value(params["dpa"]["variable_payment_margin_gbp_per_mwh"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dpa_rule = ("min(plant SRMC, reference CCGT SRMC - margin): the plant's short-run "
                "cost less the variable payment (support_regime_2050.dpa_settlement)")
    if args.kind == "dpa_dispatch":
        table = dpa_dispatch_table(cohorts, dpa_reference_srmc=reference, dpa_margin=margin)
        table.to_csv(args.output, index=False)
        held = table.loc[table["dpa_cohorts"].gt(0)]
        summary = {"kind": "dpa_dispatch", "network": str(args.network), "model_year": model_year,
                   "components": int(len(table)), "dpa_components": int(len(held)),
                   "dpa_weight_sum": float(held["dpa_weight"].sum()),
                   "dpa_reference_ccgt_srmc_gbp_per_mwh": reference, "dpa_margin_gbp_per_mwh": margin,
                   "gas_gbp_per_mwh": gas, "carbon_gbp_per_t": carbon, "dpa_net_cost_rule": dpa_rule}
        Path(args.summary).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        LOGGER.info("DPA dispatch table: %d components, %d with a DPA -> %s", len(table), len(held), args.output)
        return
    snapshot = cfd.load_snapshot(args.snapshot) if args.snapshot else None
    known = cfd.known_contracts(model_year)
    table = strike_table(args.strike_variant, known, model_year=model_year, study=args.study, snapshot=snapshot)
    labels = label_table(cohorts, strike_table=table, known=known, dpa_reference_srmc=reference, dpa_margin=margin)
    labels.to_csv(args.output, index=False)
    summary = dict(summarise(labels), network=str(args.network), strike_variant=args.strike_variant,
                   dpa_reference_ccgt_srmc_gbp_per_mwh=reference, dpa_margin_gbp_per_mwh=margin,
                   gas_gbp_per_mwh=gas, carbon_gbp_per_t=carbon, dpa_net_cost_rule=dpa_rule)
    if args.life_override:
        summary["life_overrides"] = list(args.life_override)
    if args.repowering:
        summary["repowering"] = list(args.repowering)
    if args.refurbishment:
        summary["refurbishment"] = args.refurbishment
    summary["model_year"] = model_year
    Path(args.summary).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    LOGGER.info("Support labels: %d cohorts on %d components -> %s", len(labels), labels["component"].nunique(),
                args.output)


if __name__ == "__main__":
    main()
