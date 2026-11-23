"""
Revenue tracking for CfD and ROC subsidy schemes.

Post-solve module that calculates:
- CfD difference payments: (strike_price - wholesale_price) × dispatch × dt
- ROC income: ro_banding × roc_buyout_price × dispatch × dt

Input:
    - Solved network (with bus marginal prices from wholesale/balancing solve)
    - Generator attributes: support_type, ro_banding, cfd_round

Output:
    - Per-generator revenue breakdown CSV
    - System-level subsidy summary

Called after solve_wholesale or solve_balancing in the market pipeline.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network
from scripts.market.cfd_strike_table import MODELLED_ROUND, contract_technology, load_strike_table, strike_lookup

logger = setup_logging("revenue_tracking")


NEGATIVE_PRICE_RULES = ("none", "ar4_zero_payment_when_price_le_0", "gb_contract_terms")

#: ``gb_contract_terms``: the negative pricing rule each allocation round carries.
#: From AR4 no difference payment accrues in any hour whose reference price is
#: strictly negative. AR2 and AR3 lose it only inside a run of six or more consecutive
#: negative hours. AR1, Investment Contracts and Hinkley Point C carry no such term: the
#: rule came in for contracts offered from 1 January 2016 (the Commission's state-aid
#: condition; Explanatory Memorandum to SI 2015/1425, para 4.3), after AR1 was offered
#: in February 2015 (checked 6 Oct 2026).
#: AR7 and AR7a are assumed to keep the AR4 rule (no published change found,
#: 26 Sep 2026); the 2030 chain's modelled AR8 round (``AR8_modelled``, F18) keeps it too
#: (the AR8 framework, 6 Jul 2026, keeps the AR4 negative-pricing rule).
POST_AR4_ROUNDS = frozenset({"AR4", "AR5", "AR6", "AR7", "AR7a", MODELLED_ROUND})
#: FES 2050 task Y11/SETTLE2050: the modelled post-AR7 contracts are labelled
#: ``PAR7_<build year>`` (one round per vintage) and keep the AR4 rule.
POST_AR4_ROUND_PREFIX = "PAR7_"
SIX_HOUR_RULE_ROUNDS = frozenset({"AR2", "AR3"})
SIX_HOUR_RUN = 6
#: A price counts as negative below this. A barrier solve without crossover
#: returns hours the merchant fleet sets at zero as values like -3e-5; read
#: literally, those stripped AR4+ payments worth 6.4 m on one January week that
#: never truly cleared below zero. The contract-aware order puts genuinely
#: negative hours at -1 or below, far outside this band. Widened from 0.01 to 0.50
#: on 12 Oct 2026: the CP30 Stage 1 (s1_zf) bids AR4+ at -0.01 so they still curtail
#: first, and its surplus hours clear at -0.01 plus noise (to -0.01005), which a 0.01
#: band split at random. Matches project_future_bill.NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH.
NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH = 0.50


def _negative_run_lengths(negative: np.ndarray) -> np.ndarray:
    """Length of the run of consecutive ``True`` each element sits in, 0 where ``False`` (column-wise)."""
    out = np.zeros(negative.shape, dtype=int)
    for j in range(negative.shape[1]):
        column = negative[:, j]
        if not column.any():
            continue
        edges = np.diff(np.concatenate(([0], column.astype(int), [0])))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            out[start:stop, j] = stop - start
    return out


def negative_price_eligibility(reference, rounds, rule: str) -> np.ndarray:
    """Boolean (snapshots x units): does a difference payment accrue in that hour?

    ``reference`` is the settlement reference price, a (snapshots,) series or a
    (snapshots x units) array; ``rounds`` is each unit's CfD allocation round.
    Assumes consecutive hourly snapshots for the six-hour runs; a run crossing the
    edge of the settled window is counted only inside it.
    """
    if rule not in NEGATIVE_PRICE_RULES:
        raise ValueError(f"negative_price_rule must be one of {NEGATIVE_PRICE_RULES}, got {rule!r}")
    rounds = [str(r).strip() for r in rounds]
    ref = np.asarray(reference, dtype=float)
    if ref.ndim == 1:
        ref = np.repeat(ref[:, None], len(rounds), axis=1)
    if rule == "none":
        return np.ones(ref.shape, dtype=bool)
    if rule == "ar4_zero_payment_when_price_le_0":
        return ref > 0.0
    eligible = np.ones(ref.shape, dtype=bool)
    post_ar4 = np.array([r in POST_AR4_ROUNDS or r.startswith(POST_AR4_ROUND_PREFIX) for r in rounds])
    six_hour = np.array([r in SIX_HOUR_RULE_ROUNDS for r in rounds])
    negative = ref < -NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH
    if post_ar4.any():
        eligible[:, post_ar4] = ~negative[:, post_ar4]
    if six_hour.any():
        eligible[:, six_hour] = _negative_run_lengths(negative[:, six_hour]) < SIX_HOUR_RUN
    return eligible


def compute_cfd_payments(
    network,
    strike_prices: dict,
    logger: logging.Logger = logger,
    reference_price: pd.Series | None = None,
    negative_price_rule: str = "none",
    reference_price_overrides: dict | None = None,
    volume_caps: dict | None = None,
    price_bands: dict | None = None,
) -> pd.DataFrame:
    """
    Compute CfD difference payments for generators with support_type == 'CfD'.

    CfD payment per timestep = (strike_price - reference_price) × dispatch_MW × dt_hours
    If reference_price > strike_price, the generator pays back the difference.

    Parameters
    ----------
    network : pypsa.Network
        Solved network with generators_t.p (dispatch) and buses_t.marginal_price.
    strike_prices : dict
        Mapping of CfD round / carrier to strike price (£/MWh).
        Keys can be: 'AR1_offshore', 'AR2_offshore', ..., 'future_offshore', etc.
    logger : Logger
    reference_price : Series, optional
        System wholesale reference price by snapshot (intermittent CfDs settle
        against the hourly day-ahead reference).
    negative_price_rule : str
        One of ``NEGATIVE_PRICE_RULES``.
    reference_price_overrides : dict, optional
        ``{carrier: pd.Series indexed by snapshot}``. A generator whose carrier
        is present settles against that series instead of ``reference_price``.
        Baseload CfDs such as nuclear settle against a season-ahead baseload
        reference rather than the hourly price, and this is how a caller
        supplies it. Recorded per generator in ``reference_price_basis``.
    volume_caps : dict, optional
        ``{cfd_round: annual MWh}``. The supported volume of a collared contract
        (Drax's bridging CfD pays on about 6 TWh a year) is prorated to the hours
        this call settles, and a round's payments are scaled down when its
        dispatch exceeds that share. Recorded in ``volume_cap_factor``.
    price_bands : dict, optional
        ``{label: threshold GBP/MWh}`` (F18, 27 Oct 2026, a report): each generator's
        payment and dispatch in the hours whose hourly wholesale price (``reference_price``,
        else the unit's own reference) is at or below the threshold, after the
        negative-price rule and the volume cap, as ``cfd_payment_{label}`` and
        ``dispatch_MWh_{label}``. The payments themselves are unchanged.

    Returns
    -------
    DataFrame with columns: generator, carrier, bus, support_type, cfd_round,
        strike_price, total_dispatch_MWh, avg_reference_price, total_cfd_payment,
        net_cfd_cost (positive = cost to consumer, negative = generator pays back),
        reference_price_basis ("hourly_wholesale" or "override:<carrier>")
    """
    if 'support_type' not in network.generators.columns:
        logger.info("No support_type column — skipping CfD revenue tracking")
        return pd.DataFrame()

    if negative_price_rule not in NEGATIVE_PRICE_RULES:
        raise ValueError(
            f"negative_price_rule must be one of {NEGATIVE_PRICE_RULES}, got {negative_price_rule!r}"
        )

    cfd_gens = network.generators[network.generators['support_type'] == 'CfD']
    if cfd_gens.empty:
        logger.info("No CfD generators found")
        return pd.DataFrame()

    overrides = {}
    for override_carrier, series in (reference_price_overrides or {}).items():
        if not isinstance(series, pd.Series):
            raise ValueError(
                f"reference_price_overrides[{override_carrier!r}] must be a pandas Series indexed by snapshot"
            )
        aligned = series.reindex(network.snapshots)
        if aligned.isna().any() or not np.isfinite(aligned.to_numpy()).all():
            raise ValueError(
                f"reference_price_overrides[{override_carrier!r}] does not cover every snapshot with finite prices"
            )
        overrides[str(override_carrier)] = aligned.astype(float)

    # Get snapshot weighting (hours per snapshot)
    if hasattr(network, 'snapshot_weightings') and 'generators' in network.snapshot_weightings.columns:
        dt = network.snapshot_weightings['generators']
    elif hasattr(network, 'snapshot_weightings') and 'objective' in network.snapshot_weightings.columns:
        dt = network.snapshot_weightings['objective']
    else:
        dt = pd.Series(1.0, index=network.snapshots)

    # Vectorised over generators (the consumer-bill projector calls this once
    # per month per ownership case, on ~7,000 CfD units; the per-generator loop
    # it replaces took ~10 s a call).  Every step below is the loop's arithmetic
    # applied column-wise: strike and round lookup per generator, dispatch
    # masked before the contract start, the reference price per generator
    # (carrier override, else the supplied series, else the bus dual, else
    # zero), payment = (strike - ref) x dispatch x dt with the AR4 zero-price
    # rule masking hours where the reference is at or below zero.
    snapshots = pd.DatetimeIndex(network.snapshots)
    dt = pd.Series(dt).reindex(snapshots).astype(float)
    dispatch_table = network.generators_t.p
    # The strike lookup is per (carrier, round) pair, not per generator, and the
    # support-start parse is one call on the column: on ~5,500 CfD units a month
    # the per-generator version of this block took ~3 s a month.
    carriers = cfd_gens['carrier'].fillna('') if 'carrier' in cfd_gens.columns else pd.Series('', index=cfd_gens.index)
    rounds = cfd_gens['cfd_round'].fillna('') if 'cfd_round' in cfd_gens.columns else pd.Series('', index=cfd_gens.index)
    buses = cfd_gens['bus'].fillna('') if 'bus' in cfd_gens.columns else pd.Series('', index=cfd_gens.index)
    starts_raw = (cfd_gens['cfd_support_start'] if 'cfd_support_start' in cfd_gens.columns
                  else pd.Series(pd.NaT, index=cfd_gens.index))
    starts_all = pd.to_datetime(starts_raw, errors='coerce')
    # The contract's own strike-table technology where known (floating, permitted
    # reduction, Scottish pot, remote island wind): cfd_strike_table.contract_technology.
    technologies = contract_technology(cfd_gens)
    lookups = {key: _lookup_strike_price(key[0], key[1], strike_prices, key[2])
               for key in set(zip(carriers.tolist(), rounds.tolist(), technologies.tolist()))}
    has_dispatch = set(dispatch_table.columns)
    rows = []
    for gen_name, carrier, cfd_round, technology, bus, start in zip(
            cfd_gens.index, carriers.tolist(), rounds.tolist(), technologies.tolist(), buses.tolist(),
            starts_all.tolist()):
        strike, strike_key, is_fallback = lookups[(carrier, cfd_round, technology)]
        if strike is None:
            logger.debug(f"No strike price for {gen_name} (carrier={carrier}, round={cfd_round})")
            continue
        if gen_name not in has_dispatch:
            logger.debug(f"No dispatch data for {gen_name}")
            continue
        rows.append((gen_name, carrier, bus, cfd_round, float(strike), strike_key, bool(is_fallback), start))
    if not rows:
        df = pd.DataFrame()
    else:
        meta = pd.DataFrame(rows, columns=['generator', 'carrier', 'bus', 'cfd_round', 'strike_price', 'strike_key',
                                           'is_fallback_strike', 'support_start'])
        meta['support_start'] = pd.to_datetime(meta['support_start'], errors='coerce')
        names = meta['generator'].tolist()
        dispatch = dispatch_table[names].reindex(snapshots).astype(float).to_numpy(copy=True)
        # Explicit historical tranches may be physically commissioned before
        # their individual CfD begins.  Such output remains physical/merchant
        # generation but must not accrue a difference payment.
        naive = snapshots.tz_localize(None) if snapshots.tz is not None else snapshots
        starts = meta['support_start']
        for j, start in enumerate(starts):
            if pd.notna(start):
                dispatch[np.asarray(naive < start), j] = 0.0
        # Reference price per generator: one column per basis, chosen per unit.
        hourly = None
        if reference_price is not None:
            hourly = pd.Series(reference_price).reindex(snapshots).ffill().bfill().fillna(0.0).astype(float).to_numpy()
        marginal = getattr(getattr(network, 'buses_t', None), 'marginal_price', None)
        ref = np.zeros_like(dispatch)
        basis = []
        for j, (carrier, bus) in enumerate(zip(meta['carrier'], meta['bus'])):
            if str(carrier) in overrides:
                ref[:, j] = overrides[str(carrier)].to_numpy()
                basis.append(f"override:{carrier}")
            elif hourly is not None:
                ref[:, j] = hourly
                basis.append("hourly_wholesale")
            elif marginal is not None and bus in marginal.columns:
                # Backward-compatible fallback for standalone use only.
                ref[:, j] = marginal[bus].reindex(snapshots).to_numpy()
                basis.append("hourly_wholesale")
            else:
                basis.append("hourly_wholesale")
        weighted = dispatch * dt.to_numpy()[:, None]
        payment = (meta['strike_price'].to_numpy()[None, :] - ref) * weighted
        # The negative pricing rule (negative_price_eligibility): gb_contract_terms
        # applies each round's own term, which bites once the contract-aware
        # wholesale order (curtailment_order.py) lets hours clear below zero.
        if negative_price_rule != "none":
            payment = np.where(negative_price_eligibility(ref, meta['cfd_round'], negative_price_rule), payment, 0.0)
        total_dispatch = weighted.sum(axis=0)
        avg_ref_price = (ref * weighted).sum(axis=0) / np.maximum(total_dispatch, 1e-6)
        total_payment = payment.sum(axis=0)
        cap_factor = np.ones(len(names))
        period_hours = float(dt.sum())
        for capped_round, annual_mwh in (volume_caps or {}).items():
            members = meta['cfd_round'].to_numpy() == capped_round
            round_mwh = float(total_dispatch[members].sum())
            allowed = float(annual_mwh) * period_hours / 8760.0
            if members.any() and round_mwh > allowed > 0:
                cap_factor[members] = allowed / round_mwh
        total_payment = total_payment * cap_factor
        banded = {}
        for label, threshold in (price_bands or {}).items():
            price = hourly[:, None] if hourly is not None else ref
            low = price <= float(threshold)
            banded[f'cfd_payment_{label}'] = np.where(low, payment, 0.0).sum(axis=0) * cap_factor
            banded[f'dispatch_MWh_{label}'] = np.where(low, weighted, 0.0).sum(axis=0)
        df = pd.DataFrame({
            'generator': names,
            'carrier': meta['carrier'].to_numpy(),
            'bus': meta['bus'].to_numpy(),
            'support_type': 'CfD',
            'cfd_round': meta['cfd_round'].to_numpy(),
            'strike_price': meta['strike_price'].to_numpy(),
            'strike_key': meta['strike_key'].to_numpy(),
            'is_fallback_strike': meta['is_fallback_strike'].to_numpy(),
            'total_dispatch_MWh': total_dispatch,
            'avg_reference_price': avg_ref_price,
            'total_cfd_payment': total_payment,
            'net_cfd_cost': total_payment,
            'reference_price_basis': basis,
            'volume_cap_factor': cap_factor,
            **banded,
        })
    if not df.empty:
        total_cost = df['net_cfd_cost'].sum()
        total_mwh = df['total_dispatch_MWh'].sum()
        logger.info(f"CfD payments: {len(df)} generators, "
                     f"{total_mwh:,.0f} MWh, £{total_cost:,.0f} net cost")
    return df


def compute_roc_income(
    network,
    roc_buyout_price: float | pd.Series,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Compute ROC income for generators with support_type == 'ROC'.

    ROC income per timestep = ro_banding × roc_buyout_price × dispatch_MW × dt_hours

    Parameters
    ----------
    network : pypsa.Network
        Solved network.
    roc_buyout_price : float or pd.Series
        ROC buyout price (£/ROC), optionally aligned to each snapshot so
        calendar-year accounting can respect the April obligation-year change.
    logger : Logger

    Returns
    -------
    DataFrame with columns: generator, carrier, bus, support_type, ro_banding,
        total_dispatch_MWh, roc_income_total
    """
    if 'support_type' not in network.generators.columns:
        logger.info("No support_type column — skipping ROC revenue tracking")
        return pd.DataFrame()

    roc_gens = network.generators[network.generators['support_type'] == 'ROC']
    if roc_gens.empty:
        logger.info("No ROC generators found")
        return pd.DataFrame()

    if hasattr(network, 'snapshot_weightings') and 'generators' in network.snapshot_weightings.columns:
        dt = network.snapshot_weightings['generators']
    elif hasattr(network, 'snapshot_weightings') and 'objective' in network.snapshot_weightings.columns:
        dt = network.snapshot_weightings['objective']
    else:
        dt = pd.Series(1.0, index=network.snapshots)

    results = []
    for gen_name, gen in roc_gens.iterrows():
        carrier = gen.get('carrier', '')
        bus = gen.get('bus', '')
        ro_banding = gen.get('ro_banding', 0.0)

        if not pd.notna(ro_banding) or ro_banding <= 0:
            continue

        if gen_name in network.generators_t.p.columns:
            dispatch = network.generators_t.p[gen_name]
        else:
            continue

        total_dispatch = (dispatch * dt).sum()
        eligible_share = gen.get('roc_eligible_share', 1.0)
        eligible_share = float(eligible_share) if pd.notna(eligible_share) else 1.0
        eligible_dispatch = total_dispatch * eligible_share
        if isinstance(roc_buyout_price, pd.Series):
            price = roc_buyout_price.reindex(dispatch.index)
            if not np.isfinite(price).all() or (price < 0).any():
                raise ValueError("ROC price series must cover every dispatch snapshot with finite non-negative prices")
            roc_income = float(ro_banding) * eligible_share * (dispatch * dt * price).sum()
        else:
            roc_income = float(ro_banding) * roc_buyout_price * eligible_dispatch

        results.append({
            'generator': gen_name,
            'carrier': carrier,
            'bus': bus,
            'support_type': 'ROC',
            'ro_banding': ro_banding,
            'roc_eligible_share': eligible_share,
            'total_dispatch_MWh': total_dispatch,
            'roc_eligible_dispatch_MWh': eligible_dispatch,
            'roc_income_total': roc_income,
        })

    df = pd.DataFrame(results)
    if not df.empty:
        total_income = df['roc_income_total'].sum()
        total_mwh = df['total_dispatch_MWh'].sum()
        logger.info(f"ROC income: {len(df)} generators, "
                     f"{total_mwh:,.0f} MWh, £{total_income:,.0f} total")
    return df


def _lookup_strike_price(carrier: str, cfd_round, strike_prices, technology=None):
    """Return ``(strike, key, is_fallback)`` for a contract.

    ``is_fallback`` is true when the generator's own allocation round could not
    be matched and a carrier-wide fallback strike was used instead (the modelled
    AR8 round in a strike table; ``future_*`` in the legacy flat mapping). The
    caller records it, because a fleet of separately auctioned farms settling
    at one price is the largest assumption behind the CfD line and it is
    invisible in the payment total alone.

    ``strike_prices`` is either a ``StrikeTable`` (forecast cases, published
    2012-price clearing prices indexed to the model basis) or the legacy flat
    mapping from ``support.labels.cfd_strike_prices``.
    """
    if not strike_prices:
        return None, None, True
    if hasattr(strike_prices, "lookup"):
        return strike_lookup(strike_prices, carrier, cfd_round, technology)

    # Try exact round+carrier match (e.g., "AR2_offshore")
    tech_suffix = _carrier_to_tech_suffix(carrier)
    if pd.notna(cfd_round) and cfd_round:
        round_str = str(cfd_round).strip()
        key = f"{round_str}_{tech_suffix}"
        if key in strike_prices:
            return strike_prices[key], key, False
        # Try just the round
        if round_str in strike_prices:
            return strike_prices[round_str], round_str, False

    # Fallback in the legacy flat mapping (subsidy_tracking.cfd_strike_prices, historical
    # and FES cases, not the 2030 CP30 chain): its "future_{tech}" keys
    future_key = f"future_{tech_suffix}"
    if future_key in strike_prices:
        return strike_prices[future_key], future_key, True

    return None, None, True


def _carrier_to_tech_suffix(carrier: str) -> str:
    """Map PyPSA carrier to CfD technology suffix."""
    mapping = {
        'wind_offshore': 'offshore',
        'wind_onshore': 'onshore',
        'solar_pv': 'solar',
        'nuclear': 'nuclear',
        'tidal_stream': 'tidal',
        'shoreline_wave': 'wave',
    }
    return mapping.get(carrier, carrier)


STRIKE_TABLE_KEYS = ("file", "indexation", "model_price_year")


def _resolve_strike_prices(sub_config: dict):
    """Return ``(strike_prices, source_label)`` for the archived CfD cost.

    ``support.labels.cfd_strike_table`` (``file``, ``indexation`` in the
    ``cfd_strike_indexation`` schema, ``model_price_year``) selects the
    published-round ``StrikeTable``; otherwise the legacy flat
    ``support.labels.cfd_strike_prices`` mapping is used. A present but
    malformed block is an error rather than a silent fall-back, because the
    two sources differ materially (the legacy map lacks HPC, Investment
    Contract and AR7 keys).
    """
    table_config = sub_config.get('cfd_strike_table')
    if table_config is None:
        return sub_config.get('cfd_strike_prices', {}), 'legacy_subsidy_tracking.cfd_strike_prices'
    if not isinstance(table_config, dict):
        raise ValueError("support.labels.cfd_strike_table must be a mapping with keys "
                         f"{STRIKE_TABLE_KEYS}, got {type(table_config).__name__}")
    if missing := [key for key in STRIKE_TABLE_KEYS if key not in table_config]:
        raise ValueError(f"support.labels.cfd_strike_table is missing {missing}")
    if not isinstance(table_config['indexation'], dict):
        raise ValueError("support.labels.cfd_strike_table.indexation must be a mapping "
                         "(from_price_year, to_price_year, from_index, to_index, factor, index)")
    path = Path(str(table_config['file']))
    if not path.exists():
        raise ValueError(f"support.labels.cfd_strike_table.file does not exist: {path}")
    table = load_strike_table(path, table_config['indexation'], int(table_config['model_price_year']))
    return table, f"strike_table:{table_config['file']}"


def compute_revenue_tracking(
    network,
    scenario_config: dict,
    logger: logging.Logger = logger,
    wholesale_reference_price: pd.Series | None = None,
) -> dict:
    """
    Run full revenue tracking computation.

    Parameters
    ----------
    network : pypsa.Network
        Solved network.
    scenario_config : dict
        Full scenario configuration.
    logger : Logger

    Returns
    -------
    dict with keys: 'cfd_df', 'roc_df', 'summary'
    """
    support = scenario_config.get('support', {})
    rev_config = support.get('revenue', {})
    sub_config = support.get('labels', {})
    mc_sub = support.get('dispatch_costs', {})

    results = {'cfd_df': pd.DataFrame(), 'roc_df': pd.DataFrame(), 'summary': {}}

    cfd_strike_source = None
    if rev_config.get('include_cfd', True):
        strike_prices, cfd_strike_source = _resolve_strike_prices(sub_config)
        # The same negative-price rule the BM bids and the bill apply (review
        # 26 Sep 2026); without it the archived CfD cost paid AR4+ units through
        # the hours the contract-aware order clears below zero.
        bids_cfg = ((support.get('bm_bids') or {}).get('contract_value_bids') or {})
        results['cfd_df'] = compute_cfd_payments(
            network, strike_prices, logger, reference_price=wholesale_reference_price,
            negative_price_rule=str(bids_cfg.get('negative_price_rule', 'none')),
        )

    if rev_config.get('include_roc', True):
        modelled_year = scenario_config.get('modelled_year', 2024)
        buyout_prices = mc_sub.get('roc_buyout_prices', {})
        # Reuse the interpolation helper from apply_marginal_costs
        from scripts.generators.apply_marginal_costs import _get_roc_buyout_price
        roc_price = _get_roc_buyout_price(buyout_prices, modelled_year)
        results['roc_df'] = compute_roc_income(network, roc_price, logger)

    # Build summary
    summary = {}
    if not results['cfd_df'].empty:
        cfd = results['cfd_df']
        summary['cfd_total_cost'] = float(cfd['net_cfd_cost'].sum())
        summary['cfd_total_dispatch_MWh'] = float(cfd['total_dispatch_MWh'].sum())
        summary['cfd_generator_count'] = len(cfd)
    if not results['roc_df'].empty:
        roc = results['roc_df']
        summary['roc_total_income'] = float(roc['roc_income_total'].sum())
        summary['roc_total_dispatch_MWh'] = float(roc['total_dispatch_MWh'].sum())
        summary['roc_generator_count'] = len(roc)
    if wholesale_reference_price is not None:
        summary['cfd_reference_price_source'] = 'wholesale_system_price'
    if cfd_strike_source is not None:
        summary['cfd_strike_source'] = cfd_strike_source
    results['summary'] = summary

    return results
