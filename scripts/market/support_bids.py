"""Balancing bids and offers for a long-range fleet from its support labels (task BM2050; decisions 35, 22).

The 2030 contract-value bid stack (``cfd_bm_bids.build_contract_value_bid_frames``)
prices one generator on one contract. A clustered 2050 generator holds several
build years, each with its own contract in the model year (``support_labels.py``), so
here each generator's bid is the capacity-weighted blend of its cohorts' bids,
the same blend the BM1 ownership stack already uses for state shares:

* **CfD** (every round's live phases, HPC and post-AR7 ``PAR7_<year>``): what the unit
  loses by being turned down, ``max(strike - reference, 0)`` per snapshot on the Stage 1
  clearing price (``bid_basis: contract_value``), or the whole strike
  (``strike``), and zero in an hour its contract pays nothing (its round's negative
  rule: AR4 onward and every post-AR7 contract any negative hour, AR2 and AR3 after six).
  A phase live for part of the year (``live_share``) is CfD for that share and merchant
  for the rest, spread evenly.
* **RO** (model years before 31 March 2037): the 2030 chain's rule
  (``market_utils._apply_renewable_support_pricing``): the ROC value it forgoes,
  ``ro_banding x buyout x support.bm_bids.roc.bid_multiplier``, the buyout of
  the model year (``support.dispatch_costs.roc_buyout_prices``), on the live share; offers
  keep their static price. None in 2040 or 2050: the RO has ended.
* **FiT**: embedded plant outside the BM, bid as merchant.
* **Merchant renewables** (expired contracts): the configured merchant bid
  (``contract_value_bids.merchant.bid_price``, 20 GBP/MWh as in 2030).
* **DPA** (gas CCS here; hydrogen turbines are links, :func:`dpa_link_prices`): the
  plant is paid the variable payment ``max(SRMC - SRMC_ref + margin, 0)`` on its
  output (``support_regime_2050.dpa_settlement``), so its net short-run cost, and so
  its offer and its bid, is ``min(SRMC, SRMC_ref - margin)``, declared symmetric:
  offer at that price, bid at minus it (it hands back the cost it saves). No FTI
  markup on the DPA share. Merchant CCS (a DPA expired) keeps its derived prices.
* Excluded carriers (nuclear, BECCS, biomass: ``exclude_carriers``) and every carrier
  without a label keep the price the rest of the pricing chain gave them.

Item BM1 (P2's per-share bid stacks): a generator's state share bids
``public_bid_price``. The share table is per generator; the state's part is taken
from the generator's post-2025 cohorts pro rata (L4 owns a share of each post-2025
technology and build-year cell; declared approximation of the register's whole-row
pick). Only renewables' state shares bid at cost (``support_labels.state_carriers``,
default the merchant carriers and marine); DPA and thermal plant keep their own
blend under P2 (declared). The state-first step
lands on each generator's private merchant weight, objective only, as in 2030.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.market.cfd_bm_bids import (
    AUDIT_COLUMNS,
    BID_BASES,
    DEFAULT_MERCHANT_BID,
    DEFAULT_MERCHANT_CARRIERS,
    ContractValueBids,
    demand_weighted_reference,
    resolve_general_turn_up_floor,
    resolve_public_ownership_bids,
    resolve_reference_price,
)
from scripts.market.revenue_tracking import negative_price_eligibility

#: The columns the BM reads from the label table (``support_labels.LABEL_COLUMNS``; ``ro_banding`` optional).
REQUIRED_LABEL_COLUMNS = ("component", "component_type", "carrier", "build_year", "weight", "support_type",
                          "live_share", "strike_gbp_per_mwh_2024", "negative_price_rule", "reference_price",
                          "dpa_reference_srmc_gbp_per_mwh", "dpa_margin_gbp_per_mwh")
#: ``negative_price_rule`` -> the round ``negative_price_eligibility`` reads it as.
NEGATIVE_PRICE_ROUND = {"ar4_plus": "AR4", "six_hour": "AR3", "none": "HPC"}
#: Must match config/defaults/base/support.yaml support.bm_bids.exclude_carriers.
DEFAULT_EXCLUDE = ("nuclear", "CCS Biomass", "biomass")
#: Generator carriers priced on a DPA in the BM (the hydrogen turbines are links).
DEFAULT_DPA_CARRIERS = ("CCS Gas",)
#: Link carriers priced on a DPA in the BM (decision 35).
DEFAULT_DPA_LINK_CARRIERS = ("H2_turbine",)
#: Build years from which a cohort is post-2025 new build (the state's L4 boundary).
POLICY_START_YEAR = 2025
#: Extra audit columns the label path writes after :data:`AUDIT_COLUMNS`.
EXTRA_AUDIT_COLUMNS = ("cfd_weight", "merchant_weight", "dpa_weight", "static_weight", "dpa_offer_gbp_per_mwh",
                       "label_rounds", "ro_weight")


def support_labels_block(bm_bids: dict) -> dict:
    """``support.bm_bids.contract_value_bids.support_labels`` ({} when absent: the 2030 path).

    ``bm_bids`` is the scenario's ``support.bm_bids`` block.
    """
    return dict((((bm_bids or {}).get("contract_value_bids") or {}))
                .get("support_labels") or {})


def load_support_labels(path) -> pd.DataFrame:
    """Read the label table written by ``support_labels.py``."""
    if not path:
        raise ValueError("contract_value_bids.support_labels needs a file (support_labels.py's output)")
    if not Path(path).exists():
        raise FileNotFoundError(f"support label table not found: {path}")
    labels = pd.read_csv(path, low_memory=False)
    missing = set(REQUIRED_LABEL_COLUMNS) - set(labels.columns)
    if missing:
        raise ValueError(f"support label table {path} lacks columns {sorted(missing)}")
    labels["support_type"] = labels["support_type"].astype(str).str.strip().str.lower()
    labels["component"] = labels["component"].astype(str)
    return labels


def dpa_net_cost(srmc, reference_srmc, margin):
    """A DPA plant's short-run cost net of its variable payment: ``min(SRMC, SRMC_ref - margin)``.

    ``support_regime_2050.dpa_settlement`` pays ``max(SRMC - SRMC_ref + margin, 0)`` per
    MWh of output, so the plant's cost of one more MWh is SRMC less that.
    """
    return np.minimum(np.asarray(srmc, dtype=float),
                      np.asarray(reference_srmc, dtype=float) - np.asarray(margin, dtype=float))


def _live(labels: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(labels["live_share"], errors="coerce").fillna(1.0).clip(0.0, 1.0)


def cfd_bid_profiles(groups: pd.DataFrame, hourly_reference: pd.Series, baseload_reference: pd.Series | None,
                     bid_basis: str, apply_negative_rule: bool) -> pd.DataFrame:
    """Per-snapshot bid of each distinct (strike, negative rule, reference) contract group (columns)."""
    out = {}
    for key, g in groups.iterrows():
        reference = (baseload_reference if g["reference_price"] == "baseload_seasonal" and baseload_reference is not None
                     else hourly_reference)
        strike = float(g["strike_gbp_per_mwh_2024"])
        bid = (pd.Series(strike, index=reference.index) if bid_basis == "strike"
               else (strike - reference).clip(lower=0.0))
        if apply_negative_rule:
            rule = NEGATIVE_PRICE_ROUND.get(str(g["negative_price_rule"]).strip().lower(), "AR4")
            eligible = negative_price_eligibility(reference.to_numpy(), [rule], "gb_contract_terms")[:, 0]
            bid = bid.where(eligible, 0.0)
        out[key] = bid.to_numpy(dtype=float)
    return pd.DataFrame(out, index=hourly_reference.index)


def build_label_bid_frames(network, scenario_config: dict, balancing: dict, gen_offer: pd.Series,
                           gen_bid: pd.Series, reference_price, logger: logging.Logger,
                           labels: pd.DataFrame | None = None) -> ContractValueBids:
    """The 2050 contract-value frames: every labelled generator bids the blend of its cohorts' contracts.

    Same return type as ``cfd_bm_bids.build_contract_value_bid_frames``; generators
    with no label, or of an excluded carrier, keep their static prices.
    """
    support_cfg = ((scenario_config or {}).get("support", {}) or {}).get("bm_bids", {}) or {}
    cfg = support_cfg.get("contract_value_bids", {}) or {}
    block = support_labels_block(support_cfg)
    labels = load_support_labels(block.get("file")) if labels is None else labels
    bid_basis = str(cfg.get("bid_basis", "contract_value")).strip().lower()
    if bid_basis not in BID_BASES:
        raise ValueError(f"contract_value_bids.bid_basis must be one of {BID_BASES}, got {bid_basis!r}")
    if str(cfg.get("contract_value_floor") or "zero").strip().lower() != "zero":
        # The unfloored FTI form (cfd_bm_bids.contract_value_floor, 16 Oct 2026) is a 2030
        # CP30 choice; the 2050 label stack and its settlement floor at zero.
        raise ValueError("contract_value_bids.contract_value_floor 'none' is not implemented for support-label "
                         "(2050) bids; the label stack floors the contract value at zero")
    negative_rule = str(cfg.get("negative_price_rule", "gb_contract_terms"))
    if negative_rule not in ("none", "gb_contract_terms"):
        raise ValueError("support-label bids apply each contract's own negative-pricing term: "
                         "negative_price_rule must be gb_contract_terms or none")
    ownership_block = cfg.get("public_ownership_bids") or {}
    if (ownership_block.get("tie_break") or {}).get("enabled", False):
        raise ValueError("support-label bids do not implement the P3 tie-break (decision 20)")
    state_shares, public_bid_price = resolve_public_ownership_bids(ownership_block, logger, network=network)
    merchant_cfg = cfg.get("merchant", {}) or {}
    merchant_price = float(merchant_cfg.get("bid_price", DEFAULT_MERCHANT_BID))
    merchant_carriers = {str(c) for c in (merchant_cfg.get("carriers", DEFAULT_MERCHANT_CARRIERS) or [])}
    exclude = {str(c) for c in (support_cfg.get("exclude_carriers", DEFAULT_EXCLUDE) or [])}
    dpa_carriers = {str(c) for c in (block.get("dpa_carriers", DEFAULT_DPA_CARRIERS) or [])}
    # Item BM1 under P2: the carriers whose state share bids at cost (0): the renewables.
    # DPA and thermal plant keep their own blend (declared).
    state_carriers = {str(c) for c in (block.get("state_carriers", sorted(merchant_carriers | {"marine"})) or [])}

    snapshots = pd.DatetimeIndex(network.snapshots)
    hourly = resolve_reference_price(network, reference_price)
    names = network.generators.index
    offer_tv = pd.DataFrame(np.tile(gen_offer.reindex(names).to_numpy(dtype=float), (len(snapshots), 1)),
                            index=snapshots, columns=names)
    bid_tv = pd.DataFrame(np.tile(gen_bid.reindex(names).to_numpy(dtype=float), (len(snapshots), 1)),
                          index=snapshots, columns=names)
    carrier = network.generators["carrier"].astype(str)

    lab = labels.loc[labels["component_type"].astype(str).eq("Generator") & labels["component"].isin(names)].copy()
    lab = lab.loc[~lab["carrier"].astype(str).isin(exclude)]
    unlabelled = names[~names.isin(labels["component"]) & carrier.isin(merchant_carriers | dpa_carriers).to_numpy()]
    if len(unlabelled):
        raise ValueError(f"{len(unlabelled)} generators of support-priced carriers have no support label "
                         f"(first: {list(unlabelled[:5])}); rebuild the label table on this network")
    if lab.empty:
        logger.warning("Support-label bids enabled, but no labelled generator is support-priced")
        return ContractValueBids(offer_tv, bid_tv, pd.DataFrame(columns=list(AUDIT_COLUMNS)), hourly)

    live = _live(lab)
    weight = pd.to_numeric(lab["weight"], errors="coerce").fillna(0.0)
    year = pd.to_numeric(lab["build_year"], errors="coerce")
    # Item BM1: the state's share comes out of each generator's post-2025 cohorts, pro rata.
    state = pd.Series(0.0, index=names)
    if state_shares is not None:
        is_dpa_row = ~lab["carrier"].isin(state_carriers)        # rows the state share never takes
        post = weight.where(year.ge(POLICY_START_YEAR) & ~is_dpa_row, 0.0).groupby(lab["component"]).sum()
        eligible = lab.loc[~is_dpa_row, "component"].unique()
        wanted = state_shares.reindex(post.index).fillna(0.0).clip(0.0, 1.0)
        wanted = wanted.where(wanted.index.isin(eligible), 0.0)
        over = wanted - post
        if (over > 1e-6).any():
            logger.warning("Item BM1: %d generators' state share exceeds their post-2025 capacity (max %.4f); "
                           "clipped", int((over > 1e-6).sum()), float(over.max()))
        taken = np.minimum(wanted, post)
        scale = (1.0 - taken / post.where(post > 0, 1.0)).where(post > 0, 1.0)
        factor = lab["component"].map(scale).where(year.ge(POLICY_START_YEAR) & ~is_dpa_row, 1.0)
        weight = weight * factor
        state = taken.reindex(names).fillna(0.0)
    is_cfd = lab["support_type"].eq("cfd")
    is_dpa = lab["support_type"].eq("dpa") & lab["carrier"].isin(dpa_carriers)
    is_ro = lab["support_type"].eq("ro")
    cfd_w = weight.where(is_cfd, 0.0) * live
    dpa_w = weight.where(is_dpa, 0.0) * live
    ro_w = weight.where(is_ro, 0.0) * live
    # Merchant, RAB, FiT and the expired part of a part-live contract or scheme.
    rest_w = weight - cfd_w - dpa_w - ro_w
    renewable = lab["carrier"].isin(merchant_carriers)
    merch_w = rest_w.where(renewable, 0.0)
    static_w = rest_w.where(~renewable, 0.0)
    comp = lab["component"]
    columns = pd.Index(comp.unique())

    # CfD: one bid profile per distinct contract group, blended by weight.
    groups = (lab.loc[is_cfd, ["strike_gbp_per_mwh_2024", "negative_price_rule", "reference_price"]]
              .fillna({"negative_price_rule": "ar4_plus", "reference_price": "intermittent_hourly"})
              .drop_duplicates().reset_index(drop=True))
    if groups["strike_gbp_per_mwh_2024"].isna().any():
        raise ValueError("a support-priced CfD cohort has no strike in the label table")
    baseload = (demand_weighted_reference(hourly, network)
                if groups["reference_price"].eq("baseload_seasonal").any() else None)
    profiles = cfd_bid_profiles(groups, hourly, baseload, bid_basis, negative_rule != "none")
    key = pd.Series(-1, index=lab.index)
    if len(groups):
        cfd_rows = lab.loc[is_cfd, ["strike_gbp_per_mwh_2024", "negative_price_rule", "reference_price"]].fillna(
            {"negative_price_rule": "ar4_plus", "reference_price": "intermittent_hourly"})
        merged = cfd_rows.reset_index().merge(groups.reset_index().rename(columns={"index": "group"}),
                                              on=["strike_gbp_per_mwh_2024", "negative_price_rule",
                                                  "reference_price"], how="left")
        key.loc[merged["index"].to_numpy()] = merged["group"].to_numpy()
    w_matrix = (pd.DataFrame({"component": comp, "group": key, "w": cfd_w}).loc[lambda f: f["group"] >= 0]
                .groupby(["group", "component"])["w"].sum().unstack("component", fill_value=0.0)
                .reindex(index=profiles.columns, columns=columns, fill_value=0.0))
    cfd_bid = profiles.to_numpy() @ w_matrix.to_numpy()                       # snapshots x columns
    cfd_weight = cfd_w.groupby(comp).sum().reindex(columns).fillna(0.0)
    merchant_weight = merch_w.groupby(comp).sum().reindex(columns).fillna(0.0)
    dpa_weight = dpa_w.groupby(comp).sum().reindex(columns).fillna(0.0)
    static_weight = static_w.groupby(comp).sum().reindex(columns).fillna(0.0)
    state_w = state.reindex(columns).fillna(0.0)
    ro_weight = ro_w.groupby(comp).sum().reindex(columns).fillna(0.0)
    # The RO bid per unit of capacity: banding x buyout x multiplier, blended by live weight.
    ro_bid = (pd.Series(0.0, index=columns) if not (ro_weight > 0).any()
              else (ro_w * _ro_banding(lab)).groupby(comp).sum().reindex(columns).fillna(0.0)
              * roc_bid_value(scenario_config, support_cfg))
    # Only generators holding a priced contract share are re-priced; the rest keep
    # their static prices exactly (no blend of a price with itself).
    priced = (cfd_weight + merchant_weight + dpa_weight + state_w + ro_weight) > 0.0
    keep = priced.to_numpy()
    columns = columns[keep]
    cfd_bid = cfd_bid[:, keep]
    cfd_weight, merchant_weight, dpa_weight = cfd_weight[columns], merchant_weight[columns], dpa_weight[columns]
    static_weight, state_w = static_weight[columns], state_w[columns]
    ro_weight, ro_bid = ro_weight[columns], ro_bid[columns]

    srmc = pd.to_numeric(network.generators["marginal_cost"], errors="coerce").reindex(columns).fillna(0.0)
    reference_srmc = (pd.to_numeric(lab["dpa_reference_srmc_gbp_per_mwh"], errors="coerce").groupby(comp).max()
                      .reindex(columns))
    margin = pd.to_numeric(lab["dpa_margin_gbp_per_mwh"], errors="coerce").groupby(comp).max().reindex(columns)
    if (dpa_weight > 0).any() and (reference_srmc[dpa_weight > 0].isna().any() or margin[dpa_weight > 0].isna().any()):
        raise ValueError("a DPA cohort has no reference CCGT cost or margin in the label table")
    dpa_price = pd.Series(np.where(dpa_weight > 0, dpa_net_cost(srmc, reference_srmc.fillna(0.0),
                                                                margin.fillna(0.0)), 0.0), index=columns)
    static_bid = bid_tv[columns].to_numpy()
    static_offer = offer_tv[columns].to_numpy()
    bid = (cfd_bid + merchant_weight.to_numpy()[None, :] * merchant_price
           + static_weight.to_numpy()[None, :] * static_bid
           - (dpa_weight * dpa_price).to_numpy()[None, :]
           + state_w.to_numpy()[None, :] * public_bid_price)
    if (ro_weight > 0).any():
        bid = bid + ro_bid.to_numpy()[None, :]
    offer_static_share = (1.0 - dpa_weight).to_numpy()[None, :]
    offer = offer_static_share * static_offer + (dpa_weight * dpa_price).to_numpy()[None, :]
    if cfg.get("cfd_offer_price") is not None:
        offer = offer + (cfd_weight.to_numpy()[None, :] * (float(cfg["cfd_offer_price"]) - static_offer))
    bid_tv.loc[:, columns] = bid
    offer_tv.loc[:, columns] = offer

    audit = _audit(lab, columns, carrier, network, hourly, pd.DataFrame(bid, index=snapshots, columns=columns),
                   gen_bid, cfd_weight, merchant_weight, dpa_weight, static_weight, state_w, dpa_price, bid_basis,
                   negative_rule, is_cfd, ro_weight)
    offer_regulariser = resolve_general_turn_up_floor(cfg, carrier, logger)
    bid_regulariser = None
    step = float(ownership_block.get("state_first_step_gbp_per_mwh", 0.0) or 0.0)
    if state_shares is not None and step > 0.0:
        private_merchant = merchant_weight[merchant_weight > 0]
        bid_regulariser = step * private_merchant
        logger.info("Item BM1: state-first step of %.2f GBP/MWh on %d generators' private merchant weight "
                    "(objective only)", step, len(private_merchant))
    _log(audit, merchant_price, bid_basis, len(snapshots), logger)
    return ContractValueBids(offer_tv, bid_tv, audit, hourly, offer_regulariser, bid_regulariser)


def _ro_banding(labels: pd.DataFrame) -> pd.Series:
    """Each label row's ROC banding (0 where none; a label table written before the RO column has none)."""
    if "ro_banding" not in labels.columns:
        return pd.Series(0.0, index=labels.index)
    return pd.to_numeric(labels["ro_banding"], errors="coerce").fillna(0.0)


def roc_bid_value(scenario_config: dict, support_cfg: dict) -> float:
    """GBP per MWh an RO unit forgoes per ROC of banding: the model year's buyout x ``roc.bid_multiplier``.

    The 2030 chain's rule (``market_utils._apply_renewable_support_pricing``), reused: the
    buyout of ``modelled_year`` from ``support.dispatch_costs.roc_buyout_prices``.

    Raises:
        ValueError: RO labels but no buyout price for the year.
    """
    from scripts.generators.apply_marginal_costs import _get_roc_buyout_price

    scenario_config = scenario_config or {}
    subsidies = ((scenario_config.get("support") or {}).get("dispatch_costs") or {})
    year = int(scenario_config.get("modelled_year", 0))
    buyout = _get_roc_buyout_price(subsidies.get("roc_buyout_prices", {}), year)
    if not buyout > 0:
        raise ValueError(f"RO support labels need support.dispatch_costs.roc_buyout_prices for {year}")
    multiplier = float(((support_cfg or {}).get("roc") or {}).get("bid_multiplier", 1.0))
    return float(buyout) * multiplier


def _audit(lab, columns, carrier, network, hourly, bid, gen_bid, cfd_weight, merchant_weight, dpa_weight,
           static_weight, state_w, dpa_price, bid_basis, negative_rule, is_cfd, ro_weight=None) -> pd.DataFrame:
    p_nom = pd.to_numeric(network.generators["p_nom"], errors="coerce").reindex(columns).fillna(0.0)
    cfd = lab.loc[is_cfd]
    w = (pd.to_numeric(cfd["weight"], errors="coerce") * _live(cfd))
    strike = (cfd["strike_gbp_per_mwh_2024"] * w).groupby(cfd["component"]).sum() / w.groupby(cfd["component"]).sum()
    rounds = lab.groupby("component")["cfd_round"].agg(
        lambda s: "|".join(sorted({str(v) for v in s.dropna() if str(v).strip()})))
    rows = []
    for name in columns:
        weights = {"CfD": cfd_weight[name], "merchant": merchant_weight[name], "DPA": dpa_weight[name],
                   "static": static_weight[name]}
        if ro_weight is not None and ro_weight[name] > 0:
            weights["ROC"] = ro_weight[name]
        series = bid[name]
        rows.append({
            "generator": name, "carrier": str(carrier.get(name, "")),
            "support_type": max(weights, key=weights.get) if max(weights.values()) >= 0.999 else "mixed",
            # The pricing tag keeps its first name: solved audits carry it and the bill reads it
            # (extract_future_balancing_costs.LABEL_PRICING).
            "cfd_round": "", "p_nom_mw": float(p_nom[name]), "pricing": "support_labels_2050",
            "strike_key": "capacity_weighted_live_cfd_cohorts",
            "strike_price_gbp_per_mwh": float(strike.get(name, np.nan)), "is_fallback_strike": False,
            "reference_price_basis": "hourly_wholesale", "mean_reference_price_gbp_per_mwh": float(hourly.mean()),
            "mean_bid_gbp_per_mwh": float(series.mean()), "min_bid_gbp_per_mwh": float(series.min()),
            "max_bid_gbp_per_mwh": float(series.max()), "hours_bid_at_zero": int((series <= 0.0).sum()),
            "previous_static_bid_gbp_per_mwh": float(gen_bid.get(name, np.nan)),
            "state_share": float(state_w[name]), "contract_value_mean_bid_gbp_per_mwh": float(series.mean()),
            "tie_break_step_gbp_per_mwh": 0.0, "negative_price_rule": negative_rule, "bid_basis": bid_basis,
            "cfd_weight": float(cfd_weight[name]), "merchant_weight": float(merchant_weight[name]),
            "dpa_weight": float(dpa_weight[name]), "static_weight": float(static_weight[name]),
            "dpa_offer_gbp_per_mwh": float(dpa_price[name]) if dpa_weight[name] > 0 else np.nan,
            "label_rounds": rounds.get(name, ""),
            "ro_weight": float(ro_weight[name]) if ro_weight is not None else 0.0,
        })
    return pd.DataFrame(rows, columns=list(AUDIT_COLUMNS) + list(EXTRA_AUDIT_COLUMNS))


def _log(audit: pd.DataFrame, merchant_price: float, bid_basis: str, n_snapshots: int, logger) -> None:
    mw = audit["p_nom_mw"]
    logger.info(
        "Support-label BM bids (BM2050) over %d snapshots on %d generators, %s basis: CfD %.0f MW, merchant "
        "%.0f MW at %.2f, DPA %.0f MW, other %.0f MW (capacity-weighted); state-owned %.0f MW",
        n_snapshots, len(audit), bid_basis, float((audit["cfd_weight"] * mw).sum()),
        float((audit["merchant_weight"] * mw).sum()), merchant_price, float((audit["dpa_weight"] * mw).sum()),
        float((audit["static_weight"] * mw).sum()), float((audit["state_share"] * mw).sum()))
    for carrier, g in audit.groupby("carrier"):
        w = g["p_nom_mw"]
        logger.info("  %-14s %5d gens %9.0f MW: mean bid %7.2f GBP/MWh (CfD %.2f, merchant %.2f, DPA %.2f)",
                    carrier, len(g), float(w.sum()), float((g["mean_bid_gbp_per_mwh"] * w).sum() / max(w.sum(), 1e-9)),
                    float((g["cfd_weight"] * w).sum() / max(w.sum(), 1e-9)),
                    float((g["merchant_weight"] * w).sum() / max(w.sum(), 1e-9)),
                    float((g["dpa_weight"] * w).sum() / max(w.sum(), 1e-9)))


# ── hydrogen turbines on their DPA (decision 35) ─────────────────────────────

def dpa_link_block(balancing: dict) -> dict:
    """``market.balancing.hydrogen_links.dpa_pricing`` ({} when absent: the 2030 path)."""
    return dict(((balancing or {}).get("hydrogen_links") or {}).get("dpa_pricing") or {})


def dpa_link_prices(network, names, increase: pd.DataFrame, decrease: pd.DataFrame, hydrogen_value: pd.Series,
                    labels: pd.DataFrame, logger: logging.Logger, carriers=DEFAULT_DPA_LINK_CARRIERS):
    """Decision 35: the hydrogen turbines' BM prices on their DPA, blended with the merchant share.

    Prices are per MWh of ``p0`` (hydrogen in), as ``market_utils.hydrogen_link_prices``.
    A turbine's resource cost per MWh electric is ``(lambda + mc) / efficiency``
    (lambda the Stage 1 hydrogen value, mc the link's cost on p0); on its DPA it is
    ``min(that, SRMC_ref - margin)`` (:func:`dpa_net_cost`), offered at that price
    and bid at minus it, times the efficiency to put it on p0. The share of a
    turbine whose DPA has expired keeps the resource-cost prices it had.

    Returns:
        ``(increase, decrease, audit)``; the frames are copies.
    """
    increase, decrease = increase.copy(), decrease.copy()
    links = network.links
    turbines = [lk for lk in names if str(links.at[lk, "carrier"]) in set(carriers)]
    lab = labels.loc[labels["component_type"].astype(str).eq("Link") & labels["component"].isin(turbines)]
    missing = sorted(set(turbines) - set(lab["component"]))
    if missing:
        raise ValueError(f"{len(missing)} hydrogen turbines have no support label (first {missing[:5]})")
    w = (pd.to_numeric(lab["weight"], errors="coerce").fillna(0.0) * _live(lab)).where(lab["support_type"].eq("dpa"), 0.0)
    dpa_weight = w.groupby(lab["component"]).sum().reindex(turbines).fillna(0.0)
    reference = pd.to_numeric(lab["dpa_reference_srmc_gbp_per_mwh"], errors="coerce").groupby(lab["component"]).max()
    margin = pd.to_numeric(lab["dpa_margin_gbp_per_mwh"], errors="coerce").groupby(lab["component"]).max()
    eff = pd.to_numeric(links.loc[turbines, "efficiency"], errors="coerce").fillna(1.0)
    mc = pd.to_numeric(links.loc[turbines, "marginal_cost"], errors="coerce").fillna(0.0)
    lam = pd.to_numeric(hydrogen_value, errors="coerce").reindex(increase.index).fillna(0.0).to_numpy()[:, None]
    srmc_e = (lam + mc.to_numpy()[None, :]) / eff.to_numpy()[None, :]
    held = dpa_weight[dpa_weight > 0].index
    if len(held) and (reference.reindex(held).isna().any() or margin.reindex(held).isna().any()):
        raise ValueError("a DPA hydrogen turbine has no reference CCGT cost or margin in the label table")
    ref = reference.reindex(turbines).fillna(np.inf).to_numpy()[None, :]
    net_e = np.minimum(srmc_e, ref - margin.reindex(turbines).fillna(0.0).to_numpy()[None, :])
    w_arr = dpa_weight.to_numpy()[None, :]
    on_p0 = net_e * eff.to_numpy()[None, :]
    increase.loc[:, turbines] = (1.0 - w_arr) * increase[turbines].to_numpy() + w_arr * on_p0
    decrease.loc[:, turbines] = (1.0 - w_arr) * decrease[turbines].to_numpy() - w_arr * on_p0
    audit = pd.DataFrame({"link": turbines, "dpa_weight": dpa_weight.to_numpy(),
                          "mean_resource_cost_gbp_per_mwh_e": srmc_e.mean(axis=0),
                          "mean_dpa_net_cost_gbp_per_mwh_e": net_e.mean(axis=0),
                          "p_nom_electric_mw": (pd.to_numeric(links.loc[turbines, "p_nom"], errors="coerce") * eff).to_numpy()})
    cap = audit["p_nom_electric_mw"]
    logger.info(
        "Decision 35: %d hydrogen turbines (%.0f MW electric) in the BM on their DPA; DPA share %.1f%% of capacity; "
        "offer/bid at min(resource cost, reference CCGT - margin): mean %.2f GBP/MWh_e against resource cost %.2f",
        len(turbines), float(cap.sum()), 100.0 * float((audit["dpa_weight"] * cap).sum() / max(cap.sum(), 1e-9)),
        float((audit["mean_dpa_net_cost_gbp_per_mwh_e"] * cap).sum() / max(cap.sum(), 1e-9)),
        float((audit["mean_resource_cost_gbp_per_mwh_e"] * cap).sum() / max(cap.sum(), 1e-9)))
    return increase, decrease, audit


# ── DPA plant in Stage 1 and the annual pass (coordinator follow-up, decision 22) ──

#: ``market.wholesale.marginal_cost_basis.dpa_dispatch`` keys (long-range years only).
DPA_DISPATCH_KEYS = ("enabled", "file", "generator_carriers", "link_carriers", "hydrogen_value_gbp_per_mwh_h2")


def apply_dpa_dispatch(network, block: dict, logger: logging.Logger) -> dict:
    """The 2050 equivalent of ``apply_marginal_costs._apply_dpa_dispatch`` (2030), at solve time.

    2030 sets a DPA unit's dispatch cost to ``min(own MC, cheapest unabated CCGT - margin)``
    at build from its support label. A 2050 network has no unabated CCGT and no labels at
    build, so this applies the same rule to the DPA share of each generator and
    hydrogen-turbine link from the support label table, against the declared reference
    CCGT of ``support_regime_2050`` (the table's ``dpa_reference_srmc_gbp_per_mwh``):

    * a generator's cost becomes ``(1 - w) x MC + w x min(MC, ref - margin)``, ``w`` its
      live DPA share (gas CCS: MC 66.33 < 85.45, so unchanged in HT50);
    * a turbine link's cost on ``p0`` (hydrogen in) is lowered by ``w x efficiency x
      top-up``, the top-up per MWh electric ``max((lambda + mc) / efficiency - (ref - margin), 0)``
      at the declared hydrogen value ``lambda`` (``hydrogen_value_gbp_per_mwh_h2``: the
      reformer-set value, flat at 36.75 all year in the annual pass). With hydrogen at that
      value the DPA share dispatches at ``ref - margin``; the top-up does not follow the
      hydrogen price within the window (declared linearisation; the pass dual is flat).

    Blending by share is the BM bids' declared approximation: a link or generator has one
    cost, so its expired (merchant) share is dispatched at the blend too.

    ``file`` is the rate-free DPA dispatch table (``support_labels.py --kind dpa_dispatch``,
    one row per component), so Stage 1 does not re-run when a financing rate moves; a full
    label table is still read (the same numbers, summed here).

    Returns:
        A summary (units, capacity, the cost change) for the log.
    """
    block = dict(block or {})
    unknown = sorted(set(block) - set(DPA_DISPATCH_KEYS))
    if unknown:
        raise ValueError(f"marginal_cost_basis.dpa_dispatch: unknown keys {unknown}")
    w, reference, margin, components = load_dpa_dispatch(block.get("file"))
    gen_carriers = set(block.get("generator_carriers", DEFAULT_DPA_CARRIERS) or [])
    link_carriers = set(block.get("link_carriers", DEFAULT_DPA_LINK_CARRIERS) or [])
    if "hydrogen_value_gbp_per_mwh_h2" not in block and link_carriers:
        raise ValueError("marginal_cost_basis.dpa_dispatch needs hydrogen_value_gbp_per_mwh_h2 for the turbines")
    lam = float(block.get("hydrogen_value_gbp_per_mwh_h2", 0.0))
    summary = {}
    gens = network.generators
    names = gens.index[gens["carrier"].astype(str).isin(gen_carriers) & gens.index.isin(w.index)]
    if len(names):
        if (getattr(network.generators_t, "marginal_cost", pd.DataFrame()).columns.isin(names)).any():
            raise ValueError("DPA generators with a time-varying marginal cost are not supported here")
        mc = pd.to_numeric(gens.loc[names, "marginal_cost"], errors="coerce").fillna(0.0)
        new = (1.0 - w[names]) * mc + w[names] * dpa_net_cost(mc, reference[names], margin[names])
        network.generators.loc[names, "marginal_cost"] = new.to_numpy()
        summary["generators"] = {"units": int(len(names)), "mean_before": float(mc.mean()),
                                 "mean_after": float(new.mean()), "mean_dpa_share": float(w[names].mean())}
    links = network.links
    lnames = links.index[links["carrier"].astype(str).isin(link_carriers) & links.index.isin(w.index)]
    missing = links.index[links["carrier"].astype(str).isin(link_carriers) & ~links.index.isin(components)]
    if len(missing):
        raise ValueError(f"{len(missing)} DPA-carrier links have no support label (first {list(missing[:5])})")
    if len(lnames):
        if (getattr(network.links_t, "marginal_cost", pd.DataFrame()).columns.isin(lnames)).any():
            raise ValueError("DPA links with a time-varying marginal cost are not supported here")
        eff = pd.to_numeric(links.loc[lnames, "efficiency"], errors="coerce").fillna(1.0)
        mc = pd.to_numeric(links.loc[lnames, "marginal_cost"], errors="coerce").fillna(0.0)
        top_up_e = np.maximum((lam + mc) / eff - (reference[lnames] - margin[lnames]), 0.0)
        new = mc - w[lnames] * eff * top_up_e
        network.links.loc[lnames, "marginal_cost"] = new.to_numpy()
        summary["links"] = {"units": int(len(lnames)), "mc_p0_before": float(mc.mean()), "mc_p0_after": float(new.mean()),
                            "mean_dpa_share": float(w[lnames].mean()), "hydrogen_value": lam,
                            "resource_cost_e": float(((lam + mc) / eff).mean()),
                            "dispatch_cost_e_at_hydrogen_value": float(((lam + new) / eff).mean())}
    logger.info("DPA dispatch (long-range year, solve time): %s", summary)
    return summary


def load_dpa_dispatch(path):
    """``(dpa weight, reference CCGT cost, margin, components)`` from a DPA dispatch table or a label table.

    The weight, cost and margin are indexed by the components holding at least one DPA
    cohort; ``components`` lists every labelled component (the missing-label check).
    """
    if not path:
        raise ValueError("marginal_cost_basis.dpa_dispatch needs a file (support_labels.py --kind dpa_dispatch)")
    if not Path(path).exists():
        raise FileNotFoundError(f"DPA dispatch table not found: {path}")
    head = pd.read_csv(path, nrows=0).columns
    if "dpa_weight" in head:
        # Parsed exactly: the writer already holds the values as the label table's reader saw them.
        table = pd.read_csv(path, low_memory=False, float_precision="round_trip")
        table["component"] = table["component"].astype(str)
        held = table.loc[pd.to_numeric(table["dpa_cohorts"], errors="coerce").fillna(0).gt(0)].set_index("component")
        return (pd.to_numeric(held["dpa_weight"], errors="coerce"),
                pd.to_numeric(held["dpa_reference_srmc_gbp_per_mwh"], errors="coerce"),
                pd.to_numeric(held["dpa_margin_gbp_per_mwh"], errors="coerce"), pd.Index(table["component"]))
    labels = load_support_labels(path)
    dpa = labels.loc[labels["support_type"].eq("dpa")]
    w = (pd.to_numeric(dpa["weight"], errors="coerce").fillna(0.0) * _live(dpa)).groupby(dpa["component"]).sum()
    reference = pd.to_numeric(dpa["dpa_reference_srmc_gbp_per_mwh"], errors="coerce").groupby(dpa["component"]).max()
    margin = pd.to_numeric(dpa["dpa_margin_gbp_per_mwh"], errors="coerce").groupby(dpa["component"]).max()
    return w, reference, margin, pd.Index(labels["component"])
