"""Extract per-month constraint costs, including CfD curtailment compensation.

The bill's CfD line is recomputed on balancing (post-curtailment) dispatch, so
every MWh of constrained-off CfD wind *lowers* the reported CfD cost. That is
only half of the real settlement: the generator is compensated for the foregone
difference payment through its balancing-mechanism bid. In the CP30 runs
``support.bm_bids.cfd.bid_price`` is 0.0, so tens of TWh of CfD
turn-down cost the model nothing and the compensation appears nowhere. Booking
the CfD saving without the compensation understates the consumer cost of
constraints by roughly the saving itself.

This script computes that compensation explicitly, per generator and hour:

    compensation = max(p_wholesale - p_balancing, 0)
                 * max(strike - reference_price, 0)
                 * snapshot hours

Both factors are floored at zero on purpose. Turn-*up* in balancing is not
curtailment, and an hour whose reference price already exceeds the strike costs
the generator nothing to lose, so neither creates a compensation claim.

There is no double counting between the CfD line and this compensation. The
CfD line is settled on balancing dispatch, so it pays difference payments only
on the MWh that were actually delivered. This term pays the floored turn-down
(wholesale minus balancing, clipped at zero) times the floored strike premium.
Because CP30 balancing prices CfD bids at 0.0, ``accepted_bid_cost`` contains
nothing for CfD generators, so the same MWh is never charged twice. The sum of
the two is not identical to CfD-on-wholesale-dispatch, however: turn-up hours
add difference payments on the CfD line that wholesale dispatch would not have
carried, and payback hours (balancing above wholesale after an earlier
turn-down) are credited on the CfD line at balancing volume while the
compensation here stays floored at zero.

The strike lookup is the same ``StrikeTable`` the CfD line uses, so the two
move together by construction rather than by coincidence.

Interconnector re-trading (the cost of unwinding wholesale interconnector
positions in balancing) is read from the market summary when present and is
added to the constraint cost only when the bill assumptions opt in through
``model_linked_rules.bsuos_constraints.include_interconnector_retrading``.

Item B8 (26 Sep 2026): the accepted offer and bid costs include the VOLL
feasibility slack (load shedding) and the turn-down of balancing
non-participants priced at the LP's penalty bid (``market_accounting``); neither
is a balancing purchase, so both are subtracted from the constraint cost.

Item C7 (27 Sep 2026): under P3 the decision-20 tie-break steps sit inside the
state-owned CfD bids, so accepted bid cost carries ``step x turn-down`` that
only orders a tie the state is indifferent to. It is subtracted from the
constraint cost exactly as ``decompose_public_bm_bids._net_bid_costs`` nets it
(the audit's share-weighted step x the redispatch summary's decrease_MWh).
The CfD compensation recompute keeps the step so it still reconciles with the
LP's own bid cost.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts.market.cfd_strike_table import (
    CFD_TECHNOLOGY_COLUMN, bm_strike_table_path, contract_technology, load_strike_table, strike_lookup,
    strike_overrides,
)
from scripts.market.cfd_bm_bids import contract_value_rate as cfd_contract_value_rate
from scripts.market.cfd_bm_bids import foregone_rate as cfd_foregone_rate
from scripts.market.market_accounting import _non_participant_turn_down, _slack_mask, _sum
from scripts.market.revenue_tracking import negative_price_eligibility
# Pro rata zonal turn-down (20 Oct 2026): every per-unit BM read goes through the month's basis.
from scripts.market.turn_down_allocation import (
    balancing_dispatch_in_force, bid_cost_uplift_in_force, month_basis, read_redispatch_in_force,
)


# How the CfD curtailment compensation enters the constraint cost.
#
#   add_post_hoc            the balancing LP priced CfD bids at a scalar (0.0 in
#                           the CP30 configuration), so the compensation is
#                           genuinely missing from accepted_bid_cost and is
#                           added here.
#   reconcile_against_lp_bid
#                           the LP priced CfD bids on contract value, so the
#                           compensation is already inside accepted_bid_cost.
#                           The term is still recomputed, but only as a check
#                           against the LP's own CfD bid cost; adding it would
#                           double count. See scripts/market/cfd_bm_bids.py.
COMPENSATION_TREATMENTS = ("add_post_hoc", "reconcile_against_lp_bid")
DEFAULT_COMPENSATION_TREATMENT = "add_post_hoc"

# How far the recomputed compensation may sit from the LP's own CfD bid cost
# before the reconciliation counts as failed, as a share of the recompute. The
# two are the same quantity by construction - floored turn-down times the
# floored strike premium - so the honest residual is only the hours where a
# generator is turned both up and down and the LP's decrease variable is not
# exactly max(p_wholesale - p_balancing, 0). Measured at 0.010% on the one-week
# reinforced run (GBP -430 on GBP 4,424,648), so 1% absorbs that while still
# catching a wrong strike table, a wrong reference series or a double count.
DEFAULT_RECONCILIATION_TOLERANCE = 0.01
#: 13 Nov 2026: a breach must also exceed this many GBP. A month whose reconciled set is near empty (2040 Jan,
#: P2 100%: one geothermal unit, 0.09 MWh, GBP 3 against GBP 5) cannot meet a share alone; real months reconcile
#: GBP millions, where 1,000 is far inside the 1% share.
RECONCILIATION_TOLERANCE_ABS_GBP = 1000.0

BASE_CONSTRAINT_COMPONENTS = (
    "accepted_offer_cost_gbp",
    "accepted_bid_cost_gbp",
)
CONSTRAINT_COMPONENTS = (
    *BASE_CONSTRAINT_COMPONENTS,
    "cfd_curtailment_compensation_gbp",
)


def constraint_components(treatment: str) -> tuple:
    """The accounting rows summed into the constraint cost under ``treatment``."""
    if treatment not in COMPENSATION_TREATMENTS:
        raise ValueError(
            f"cfd_curtailment_compensation_treatment must be one of "
            f"{list(COMPENSATION_TREATMENTS)}, got {treatment!r}"
        )
    if treatment == "add_post_hoc":
        return CONSTRAINT_COMPONENTS
    return BASE_CONSTRAINT_COMPONENTS


def lp_cfd_bid_cost(redispatch_summary_path, cfd_names) -> float:
    """The balancing LP's own accepted bid cost for the CfD fleet.

    Read from the per-generator redispatch summary rather than the market
    summary, which reports only a fleet total.
    """
    frame = read_redispatch_in_force(redispatch_summary_path, index_col=0)
    if "bid_cost" not in frame.columns:
        raise ValueError(
            f"{redispatch_summary_path} has no bid_cost column; the CfD bid-cost "
            "reconciliation cannot be formed"
        )
    present = pd.Index(cfd_names).intersection(frame.index)
    return float(pd.to_numeric(frame.loc[present, "bid_cost"], errors="coerce").fillna(0.0).sum())


def tie_break_step_cost(redispatch_summary_path, contract_value_bids_path) -> float:
    """The decision-20 tie steps inside accepted bid cost (GBP; item C7).

    Share-weighted step per unit (``tie_break_step_gbp_per_mwh`` in the bid
    audit) times its turn-down (``decrease_MWh`` in the redispatch summary),
    the same netting ``decompose_public_bm_bids`` applies. Zero when either
    file is missing or no unit carries a step (every case but P3).
    """
    if redispatch_summary_path is None or contract_value_bids_path is None:
        return 0.0
    if not Path(contract_value_bids_path).exists() or not Path(redispatch_summary_path).exists():
        return 0.0
    audit = pd.read_csv(contract_value_bids_path)
    if audit.empty or "tie_break_step_gbp_per_mwh" not in audit.columns:
        return 0.0
    steps = pd.to_numeric(audit.set_index("generator")["tie_break_step_gbp_per_mwh"], errors="coerce").fillna(0.0)
    steps = steps[steps.ne(0.0)]
    if steps.empty:
        return 0.0
    # Indexed on its first column (``component``), as lp_cfd_bid_cost reads it.
    frame = read_redispatch_in_force(redispatch_summary_path, index_col=0)
    if "decrease_MWh" not in frame.columns:
        return 0.0
    frame = frame.loc[~frame.index.astype(str).isin(["TOTAL"])]
    if "type" in frame.columns:
        frame = frame.loc[frame["type"].astype(str).eq("generator")]
    decrease = pd.to_numeric(frame["decrease_MWh"], errors="coerce").fillna(0.0)
    decrease = decrease.groupby(level=0).sum()
    return float((steps.groupby(level=0).first().reindex(decrease.index).fillna(0.0) * decrease).sum())


def reconciliation_net_of_penalty(redispatch_summary_path, reconciled: pd.DataFrame,
                                  reconciled_compensation: float, lp_bid_cost: float) -> dict | None:
    """The CfD reconciliation on the units the LP priced at their bid, not at the non-participant penalty.

    A unit the participation policy leaves out of the balancing mechanism is
    turned down at the penalty bid (item B8), which the constraint line already
    nets out, while the recompute prices its turn-down at contract value. On a
    contract-value stack that difference is noise inside the tolerance; on P3's
    stack, where every state-owned unit bids about zero plus its tie step, it is
    most of the LP's CfD bid cost (Jan 2030: 7.4k of 17.8k GBP). Returns the gap
    with those units taken out of both sides, or ``None`` when none is present.
    """
    if redispatch_summary_path is None or reconciled.empty:
        return None
    frame = read_redispatch_in_force(redispatch_summary_path, index_col=0)
    frame = frame.loc[frame.index.astype(str).isin(set(reconciled.generator.astype(str)))]
    forced = _non_participant_turn_down(frame)
    if forced.empty:
        return None
    names = set(forced.index.astype(str))
    lp_penalty = float(pd.to_numeric(forced["bid_cost"], errors="coerce").fillna(0.0).sum())
    recompute_penalty = float(reconciled.loc[reconciled.generator.astype(str).isin(names), "compensation_gbp"].sum())
    compensation = reconciled_compensation - recompute_penalty
    gap = compensation - (lp_bid_cost - lp_penalty)
    return {"penalty_priced_units": len(names), "lp_penalty_bid_cost_gbp": lp_penalty,
            "gap_gbp": gap, "gap_share": gap / compensation if compensation > 0 else None}


#: ``pricing`` values of a bid audit that price CfD turn-down on contract value: the 2030
#: per-unit bids (``cfd_bm_bids``) and the 2050 support-label blend (``support_bids_2050``,
#: task BM2050), where a generator bids its cohorts' contracts blended by capacity.
LABEL_PRICING = "support_labels_2050"
CONTRACT_VALUE_PRICINGS = ("cfd_contract_value", LABEL_PRICING)


def _contract_value_rows(frame: pd.DataFrame) -> pd.Series:
    """Rows of a bid audit that price CfD turn-down on contract value (a 2050 blend only where it holds CfD)."""
    pricing = frame["pricing"].astype(str)
    rows = pricing.eq("cfd_contract_value")
    if pricing.eq(LABEL_PRICING).any():
        weight = pd.to_numeric(frame.get("cfd_weight"), errors="coerce").fillna(0.0)
        rows |= pricing.eq(LABEL_PRICING) & weight.gt(0.0)
    return rows


def audit_declares_contract_value_bids(path) -> bool:
    """True when the balancing solve recorded contract-value CfD bids.

    ``solve_balancing`` always writes the audit; it carries rows only when
    ``support.bm_bids.contract_value_bids`` was enabled. Reading it
    here turns the assumptions-file declaration into an enforced claim about
    the run rather than a comment about it.
    """
    if path is None or not Path(path).exists():
        return False
    frame = pd.read_csv(path)
    if frame.empty or "pricing" not in frame.columns:
        return False
    if bool(_contract_value_rows(frame).any()):
        return True
    # Item BM1 (FES 2050, decision 16): a public-ownership bid stack prices each unit as
    # (1 - s) x contract value + s x (public price + step). When the state holds every CfD
    # unit (P2 at 100% in 2050) no private CfD weight is left, so no row carries it, but the
    # solve still priced CfD on that rule: nothing private to compensate, nothing to add.
    if "state_share" in frame.columns and frame["pricing"].astype(str).eq(LABEL_PRICING).any():
        return bool(pd.to_numeric(frame["state_share"], errors="coerce").fillna(0.0).gt(0.0).any())
    return False


def ownership_from_audit(path) -> dict | None:
    """Item BM1: the state's share of each CfD unit, its tie step and the public bid price.

    The balancing solve bid ``(1 - s) x contract value + s x (public price + step)``
    and recorded ``state_share``, the share-weighted ``tie_break_step`` and both mean
    bids in its audit. The public price is recovered from those means, row by row,
    and must be one price. ``None`` when nothing is state-owned, which is every run
    before BM1, so the recompute is then exactly as it was.
    """
    if path is None or not Path(path).exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "state_share" not in frame.columns:
        return None
    frame = frame.loc[frame["pricing"].astype(str).eq("cfd_contract_value")].set_index("generator")
    share = pd.to_numeric(frame["state_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if not share.gt(0.0).any():
        return None
    step = (pd.to_numeric(frame["tie_break_step_gbp_per_mwh"], errors="coerce").fillna(0.0)
            if "tie_break_step_gbp_per_mwh" in frame.columns else pd.Series(0.0, index=frame.index))
    owned = share.gt(0.0)
    mean_bid = pd.to_numeric(frame["mean_bid_gbp_per_mwh"], errors="coerce")
    contract_value = pd.to_numeric(frame["contract_value_mean_bid_gbp_per_mwh"], errors="coerce")
    price = ((mean_bid - (1.0 - share) * contract_value - step) / share)[owned]
    if price.isna().any() or (price.max() - price.min()) > 1e-6:
        raise ValueError(
            f"{path}: the state-owned rows imply more than one public bid price "
            f"({price.min():.6f} to {price.max():.6f}); the audit does not describe one blend")
    return {"state_share": share, "tie_break_step": step, "public_bid_price": float(price.median())}


def _dispatch(dataset: xr.Dataset, names: pd.Index, snapshots: pd.DatetimeIndex) -> pd.DataFrame:
    """Return hourly dispatch for ``names``, zero-filled where a series is absent."""
    available = pd.Index(dataset["generators_t_p_i"].values)
    present = names.intersection(available)
    # Read the whole table then subset in pandas: a label ``.sel`` on the
    # NetCDF-backed array costs ~0.14 s per generator, the full read <1 s.
    frame = dataset["generators_t_p"].to_pandas()[present]
    frame.index = snapshots
    return frame.reindex(columns=names, fill_value=0.0)


def unlabelled_support_audit(dataset: xr.Dataset, snapshots, weights) -> pd.DataFrame:
    """Capacity and physical energy of generators carrying no support label.

    Unlabelled generators accrue no CfD difference payment and no ROC in this
    bill. In the CP30 rebuild nuclear is placed by the station schedule, so
    Sizewell B is merchant (correctly unlabelled) and Hinkley Point C carries a
    CfD label; what remains unlabelled is the EU market balance generators and
    merchant plant. Reporting the unlabelled energy lets that be checked rather
    than assumed.
    """
    names = pd.Index(dataset["generators_i"].values)
    support = pd.Series(dataset["generators_support_type"].values, index=names)
    carrier = pd.Series(dataset["generators_carrier"].values, index=names)
    p_nom = pd.Series(dataset["generators_p_nom"].values, index=names, dtype=float)
    unlabelled = support.index[support.fillna("").astype(str).str.strip().eq("")]
    if not len(unlabelled):
        return pd.DataFrame(columns=["carrier", "capacity_mw", "physical_mwh", "generators"])
    dispatch = _dispatch(dataset, unlabelled, snapshots)
    energy = dispatch.mul(weights, axis=0).sum()
    frame = pd.DataFrame({
        "carrier": carrier.reindex(unlabelled),
        "capacity_mw": p_nom.reindex(unlabelled),
        "physical_mwh": energy.reindex(unlabelled).fillna(0.0),
    })
    return frame.groupby("carrier", as_index=False).agg(
        capacity_mw=("capacity_mw", "sum"),
        physical_mwh=("physical_mwh", "sum"),
        generators=("capacity_mw", "size"),
    )


def _cfd_generators(dataset: xr.Dataset) -> pd.DataFrame:
    names = pd.Index(dataset["generators_i"].values)
    if "generators_support_type" not in dataset:
        raise ValueError("Physical model lacks support labels for CfD accounting")
    support = pd.Series(dataset["generators_support_type"].values, index=names)
    cfd_names = support.index[support.eq("CfD")]
    generators = pd.DataFrame(index=cfd_names)
    for column in ("carrier", "cfd_round", CFD_TECHNOLOGY_COLUMN):
        key = f"generators_{column}"
        generators[column] = (
            pd.Series(dataset[key].values, index=names).reindex(cfd_names)
            if key in dataset else ""
        )
    return generators


def negative_price_rule_from_audit(path) -> str:
    """The negative pricing rule the balancing solve's contract-value bids applied ('none' if unrecorded)."""
    if path is None or not Path(path).exists():
        return "none"
    frame = pd.read_csv(path)
    if frame.empty or "negative_price_rule" not in frame.columns:
        return "none"
    rules = frame.loc[_contract_value_rows(frame), "negative_price_rule"]
    rules = set(rules.dropna().astype(str))
    if len(rules) > 1:
        raise ValueError(f"{path}: contract-value bids record more than one negative pricing rule: {sorted(rules)}")
    return rules.pop() if rules else "none"


def bid_basis_from_audit(path) -> str:
    """The basis the balancing solve's CfD bids used: 'contract_value' (also when unrecorded) or 'strike'.

    ``contract_value_bids.bid_basis`` (scripts/market/cfd_bm_bids.py). Under 'strike'
    a turned-down CfD unit was paid its strike, so the recompute prices it the same
    way and still reconciles with the LP's own bid cost.
    """
    if path is None or not Path(path).exists():
        return "contract_value"
    frame = pd.read_csv(path)
    if frame.empty or "bid_basis" not in frame.columns:
        return "contract_value"
    bases = set(frame.loc[_contract_value_rows(frame), "bid_basis"]
                .dropna().astype(str).str.strip()) - {""}
    if len(bases) > 1:
        raise ValueError(f"{path}: CfD bids record more than one bid basis: {sorted(bases)}")
    return bases.pop() if bases else "contract_value"


def contract_value_floor_from_audit(path) -> str:
    """The contract-value floor the CfD bids applied: 'zero' (also when unrecorded) or 'none'.

    ``contract_value_bids.contract_value_floor`` (scripts/market/cfd_bm_bids.py, 16 Oct
    2026). Under 'none' a turned-down unit was paid ``strike - reference`` even when
    that is negative (it then pays NESO the difference payment it saves), so the
    recompute prices it the same way.
    """
    if path is None or not Path(path).exists():
        return "zero"
    frame = pd.read_csv(path)
    if frame.empty or "contract_value_floor" not in frame.columns:
        return "zero"
    floors = set(frame.loc[_contract_value_rows(frame), "contract_value_floor"]
                 .dropna().astype(str).str.strip()) - {""}
    if len(floors) > 1:
        raise ValueError(f"{path}: CfD bids record more than one contract-value floor: {sorted(floors)}")
    return floors.pop() if floors else "zero"


def label_priced_cfd_assets(path) -> set | None:
    """Generators a 2050 support-label stack bid wholly on CfD contract value (task SETTLE2050), or None.

    A blended generator's accepted bid cost also carries its merchant and DPA cohorts'
    bids, so the compensation recompute (CfD units only) reconciles with the LP's bid
    cost only where the generator is all CfD (``cfd_weight`` at least 0.999).
    """
    if path is None or not Path(path).exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "pricing" not in frame.columns or not frame["pricing"].astype(str).eq(LABEL_PRICING).any():
        return None
    weight = pd.to_numeric(frame.get("cfd_weight"), errors="coerce").fillna(0.0)
    return set(frame.loc[frame["pricing"].astype(str).eq(LABEL_PRICING) & weight.ge(0.999), "generator"].astype(str))


def contract_value_units(path):
    """Generators the balancing bids priced on contract value, or ``None`` with no audit.

    A CfD-labelled unit of an excluded carrier (Drax's bridging CfD, Power BECCS,
    nuclear) keeps its carrier bid in the LP, so its turn-down is not priced at the
    strike and cannot be reconciled against a strike-priced recompute (review 28
    Sep 2026). The reconciliation therefore runs over the audited units only.
    """
    if path is None or not Path(path).exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "pricing" not in frame.columns:
        return None
    return set(frame.loc[frame["pricing"].astype(str).eq("cfd_contract_value"), "generator"].astype(str))


def recorded_negative_price_rule(path):
    """The rule the bid audit recorded, or ``None`` when it recorded none.

    Unlike :func:`negative_price_rule_from_audit`, which defaults to 'none' for
    settlement, this tells an audit that recorded 'none' apart from one that
    predates the column, so a consistency check can skip only the latter.
    """
    if path is None or not Path(path).exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "negative_price_rule" not in frame.columns or "pricing" not in frame.columns:
        return None
    rules = set(frame.loc[_contract_value_rows(frame), "negative_price_rule"].dropna().astype(str))
    if len(rules) > 1:
        raise ValueError(f"{path}: contract-value bids record more than one negative pricing rule: {sorted(rules)}")
    return rules.pop() if rules else None


def check_negative_price_rule(bill_rule, audit_rule, where=""):
    """Refuse a bill whose CfD negative pricing rule differs from the one the balancing bids applied.

    The CfD line settles every round on ``model_linked_rules.cfd.negative_price_rule``;
    the balancing LP bid each CfD unit its foregone payment under the rule in
    ``contract_value_bids.negative_price_rule`` (recorded in the bid audit), and the
    compensation reconciliation follows the audit. If the two differ, turned-down
    energy is priced on one contract and settled output on another. ``audit_rule``
    ``None`` (no audit recorded a rule) is not checked.
    """
    if audit_rule is None:
        return
    if str(bill_rule) != str(audit_rule):
        raise ValueError(
            f"{where}: the bill settles CfDs under negative_price_rule {bill_rule!r}, but the balancing "
            f"bids applied {audit_rule!r} (contract_value_bids audit). Align "
            "model_linked_rules.cfd.negative_price_rule with "
            "support.bm_bids.contract_value_bids.negative_price_rule and re-solve "
            "or re-project."
        )


def _book_generators(dataset: xr.Dataset, contract_book):
    """The 2050 contract book's CfD units on a network, and their dispatch (task SETTLE2050)."""
    from scripts.public_ownership.contract_book_2050 import cfd_units, read_contract_book

    names = pd.Index(dataset["generators_i"].values)
    static = pd.DataFrame({"carrier": pd.Series(dataset["generators_carrier"].values, index=names).astype(str),
                           "bus": pd.Series(dataset["generators_bus"].values, index=names).astype(str)}, index=names)
    return cfd_units(read_contract_book(contract_book), static)


def compute_cfd_curtailment_compensation(wholesale_path, balancing_path,
                                         reference_price: pd.Series, strike_table, ownership=None,
                                         negative_price_rule: str = "none", bid_basis: str = "contract_value",
                                         contract_book=None, contract_value_floor: str = "zero"):
    """Return per-carrier CfD turn-down volume and foregone difference payment.

    With ``ownership`` (item BM1, :func:`ownership_from_audit`) each unit is paid the
    blend it bid: the private part its foregone difference payment, the state part
    the public price plus its tie step; the two are reported apart. The foregone
    payment follows the round's ``negative_price_rule``, as the bid did, and the
    bid basis (:func:`bid_basis_from_audit`): under 'strike' the private part is paid
    the full strike. ``contract_value_floor`` (:func:`contract_value_floor_from_audit`):
    under 'none' the contract-value rate is not floored at zero.

    ``contract_book`` (a path; task SETTLE2050): the CfD units are the 2050 book's, each
    its network asset's dispatch x ``energy_factor``, and the rows carry ``network_asset``.
    """
    if contract_book is not None:
        return _book_compensation(wholesale_path, balancing_path, reference_price, strike_table,
                                  negative_price_rule, bid_basis, contract_book, contract_value_floor)
    with xr.open_dataset(wholesale_path, engine="netcdf4") as wholesale:
        snapshots = pd.DatetimeIndex(wholesale["snapshots_snapshot"].values)
        if not snapshots.equals(pd.DatetimeIndex(reference_price.index)):
            raise ValueError("Wholesale dispatch and reference price snapshots do not match")
        weights = pd.Series(wholesale["snapshots_generators"].values, index=snapshots, dtype=float)
        generators = _cfd_generators(wholesale)
        wholesale_dispatch = _dispatch(wholesale, generators.index, snapshots)

    with xr.open_dataset(balancing_path, engine="netcdf4") as balancing:
        balancing_snapshots = pd.DatetimeIndex(balancing["snapshots_snapshot"].values)
        if not balancing_snapshots.equals(snapshots):
            raise ValueError("Wholesale and balancing snapshots do not match")
        balancing_generators = _cfd_generators(balancing)
        balancing_dispatch = balancing_dispatch_in_force(_dispatch(balancing, generators.index, snapshots),
                                                         balancing_path)
        unlabelled = unlabelled_support_audit(balancing, snapshots, weights)

    missing_labels = generators.index.difference(balancing_generators.index)
    if len(missing_labels):
        raise ValueError(
            f"{len(missing_labels)} CfD generators are labelled in the wholesale model but not "
            "the balancing model; compensation cannot be attributed"
        )

    turn_down = (wholesale_dispatch - balancing_dispatch).clip(lower=0.0)
    if not np.isfinite(turn_down.to_numpy()).all():
        raise ValueError("CfD turn-down contains missing or non-finite values")

    rows = []
    technology = contract_technology(generators)
    for name, generator in generators.iterrows():
        strike, strike_key, is_fallback = strike_lookup(
            strike_table, generator.carrier, generator.cfd_round, technology[name])
        if strike is None:
            raise ValueError(f"No strike price for CfD generator {name} (carrier={generator.carrier})")
        # Floored (contract_value_floor zero): an hour whose reference price already
        # exceeds the strike costs the generator nothing to forgo, so it is not
        # compensated; unfloored ('none') it pays the saved difference back. Under the
        # strike bid basis it was paid the whole strike.
        contract_value_rate = cfd_contract_value_rate(float(strike), reference_price, contract_value_floor)
        foregone_rate = cfd_foregone_rate(strike, reference_price, bid_basis, contract_value_floor)
        if negative_price_rule != "none":
            eligible = negative_price_eligibility(
                reference_price.to_numpy(), [str(generator.cfd_round or "")], negative_price_rule)[:, 0]
            foregone_rate = foregone_rate.where(eligible, 0.0)
            contract_value_rate = contract_value_rate.where(eligible, 0.0)
        volume = turn_down[name] * weights
        share = float(ownership["state_share"].get(name, 0.0)) if ownership else 0.0
        step = float(ownership["tie_break_step"].get(name, 0.0)) if ownership else 0.0
        public_price = float(ownership["public_bid_price"]) if ownership else 0.0
        private = float((volume * foregone_rate).sum()) * (1.0 - share)
        # Review 28 Sep 2026: what the strike basis pays a private unit above what it
        # forgoes (strike less the difference payment it loses = min(reference, strike)
        # in a paid hour). A transfer to CfD generators inside the constraint line,
        # reported so the bill can label it; zero under the contract-value basis.
        strike_premium = float((volume * (foregone_rate - contract_value_rate)).sum()) * (1.0 - share)
        state = float(volume.sum()) * (share * public_price + step)
        rows.append({
            "generator": name,
            "carrier": generator.carrier,
            "cfd_round": str(generator.cfd_round or ""),
            "strike_key": strike_key,
            "is_fallback_strike": bool(is_fallback),
            "strike_price_gbp_per_mwh": float(strike),
            "curtailed_mwh": float(volume.sum()),
            "state_share": share,
            "compensation_private_gbp": private,
            "compensation_state_gbp": state,
            "compensation_gbp": private + state,
            "strike_basis_premium_over_contract_value_gbp": strike_premium,
        })
    return pd.DataFrame(rows), unlabelled


def _book_compensation(wholesale_path, balancing_path, reference_price, strike_table, negative_price_rule,
                       bid_basis, contract_book, contract_value_floor="zero"):
    """:func:`compute_cfd_curtailment_compensation` on the 2050 contract book's CfD units (task SETTLE2050).

    The same rate, unit by unit: ``max(p_wholesale - p_balancing, 0)`` of the unit (its
    asset's turn-down x ``energy_factor``) at the foregone difference payment, or the
    strike under the strike basis, zero in hours the round's negative pricing rule leaves
    unpaid. Ownership (item BM1) is not read: a 2050 P2 stack's state share is blended
    per generator, not per contract unit (declared; compensation is private).
    """
    with xr.open_dataset(wholesale_path, engine="netcdf4") as wholesale:
        snapshots = pd.DatetimeIndex(wholesale["snapshots_snapshot"].values)
        if not snapshots.equals(pd.DatetimeIndex(reference_price.index)):
            raise ValueError("Wholesale dispatch and reference price snapshots do not match")
        weights = pd.Series(wholesale["snapshots_generators"].values, index=snapshots, dtype=float)
        units, _ = _book_generators(wholesale, contract_book)
        assets = pd.Index(units["network_asset"].unique())
        p_w = _dispatch(wholesale, assets, snapshots)
    with xr.open_dataset(balancing_path, engine="netcdf4") as balancing:
        if not pd.DatetimeIndex(balancing["snapshots_snapshot"].values).equals(snapshots):
            raise ValueError("Wholesale and balancing snapshots do not match")
        p_b = balancing_dispatch_in_force(_dispatch(balancing, assets, snapshots), balancing_path)
        names = pd.Index(balancing["generators_i"].values)
        support = (pd.Series(balancing["generators_support_type"].values, index=names).fillna("").astype(str).str.strip()
                   if "generators_support_type" in balancing else pd.Series("", index=names))
        if support.isin(["CfD", "ROC", "DPA"]).any():
            raise ValueError("the network carries support labels and the bill names a contract book; settle one")
        booked = set(assets)
        carrier = pd.Series(balancing["generators_carrier"].values, index=names).astype(str)
        p_nom = pd.Series(balancing["generators_p_nom"].values, index=names, dtype=float)
        outside = names[~names.isin(booked)]
        energy = _dispatch(balancing, outside, snapshots).mul(weights, axis=0).sum()
        unlabelled = pd.DataFrame({"carrier": carrier.reindex(outside), "capacity_mw": p_nom.reindex(outside),
                                   "physical_mwh": energy.reindex(outside).fillna(0.0)}).groupby(
            "carrier", as_index=False).agg(capacity_mw=("capacity_mw", "sum"), physical_mwh=("physical_mwh", "sum"),
                                           generators=("capacity_mw", "size"))
    turn_down = (p_w - p_b).clip(lower=0.0)
    if not np.isfinite(turn_down.to_numpy()).all():
        raise ValueError("CfD turn-down contains missing or non-finite values")
    rows = []
    for name, unit in units.iterrows():
        # 13 Nov 2026: the contract's own strike technology, as the book priced it (contract_book_2050:
        # 2040's AR3 advanced-conversion and AR4 energy-from-waste CfDs sit on biogas rows).
        strike, strike_key, is_fallback = strike_table.lookup(unit.carrier, unit.cfd_round,
                                                              unit.get(CFD_TECHNOLOGY_COLUMN))
        if strike is None or is_fallback:
            raise ValueError(f"No 2050 strike for contract unit {name} ({unit.carrier}, {unit.cfd_round})")
        contract_value_rate = cfd_contract_value_rate(float(strike), reference_price, contract_value_floor)
        foregone_rate = cfd_foregone_rate(strike, reference_price, bid_basis, contract_value_floor)
        if negative_price_rule != "none":
            eligible = negative_price_eligibility(reference_price.to_numpy(), [str(unit.cfd_round)],
                                                  negative_price_rule)[:, 0]
            foregone_rate = foregone_rate.where(eligible, 0.0)
            contract_value_rate = contract_value_rate.where(eligible, 0.0)
        volume = turn_down[unit.network_asset] * float(unit.energy_factor) * weights
        private = float((volume * foregone_rate).sum())
        rows.append({
            "generator": name, "network_asset": unit.network_asset, "carrier": unit.carrier,
            "cfd_round": str(unit.cfd_round), "strike_key": strike_key, "is_fallback_strike": False,
            "strike_price_gbp_per_mwh": float(strike), "curtailed_mwh": float(volume.sum()), "state_share": 0.0,
            "compensation_private_gbp": private, "compensation_state_gbp": 0.0, "compensation_gbp": private,
            "strike_basis_premium_over_contract_value_gbp": float((volume * (foregone_rate - contract_value_rate)).sum()),
        })
    return pd.DataFrame(rows), unlabelled


#: Item F24.2: the per-unit table :func:`cfd_bid_change_at_price_level` returns.
PRICE_LEVEL_BID_COLUMNS = ["generator", "carrier", "state_share", "curtailed_mwh", "compensation_model_gbp",
                           "compensation_at_price_level_gbp", "gross_change_gbp", "change_gbp"]


def cfd_bid_change_at_price_level(wholesale_path, balancing_path, model_price: pd.Series, level_price: pd.Series,
                                  strike_table, contract_value_bids_path, redispatch_summary_path=None) -> pd.DataFrame:
    """Item F24.2 (user decision, 24 Oct 2026): each CfD unit's BM bid moved to a price level's reference.

    A contract-value bid is the foregone difference payment, strike less the Stage 1
    reference (:func:`compute_cfd_curtailment_compensation`, with the audit's bid basis,
    floor and negative-price rule), so at a price level (the ``cp32`` twins) the bid on
    the same turn-down moves with the reference. Volumes are as solved, on the month's
    turn-down basis. Only what the LP priced on contract value moves: the audited
    contract-value units (:func:`contract_value_units`), less any turned down at the
    non-participant penalty; the state-owned share bids its public price, which has no
    reference in it, so a unit's bid moves by ``(1 - state share)`` of its gross change.
    The bill's constraint line (:func:`extract_future_balancing_costs`) and the twin's
    asset settlement (``price_level_settlement``) both take their change from here, unit
    for unit, so the bill-to-register identities keep closing.

    Returns :data:`PRICE_LEVEL_BID_COLUMNS`: ``gross_change_gbp`` on the whole turn-down
    (the settlement's compensation column), ``change_gbp`` the money the bid moves.
    """
    if not audit_declares_contract_value_bids(contract_value_bids_path):
        raise ValueError(
            f"{contract_value_bids_path}: a price level re-prices the CfD bids only when the balancing solve "
            "priced them on contract value; this month's bids carry no reference to move (item F24.2)")
    model_price = model_price.astype(float)
    level_price = level_price.astype(float).reindex(model_price.index)
    if level_price.isna().any():
        raise ValueError(f"the price level's reference misses {int(level_price.isna().sum())} of the month's hours")
    rules = {"negative_price_rule": negative_price_rule_from_audit(contract_value_bids_path),
             "bid_basis": bid_basis_from_audit(contract_value_bids_path),
             "contract_value_floor": contract_value_floor_from_audit(contract_value_bids_path)}
    model, _ = compute_cfd_curtailment_compensation(wholesale_path, balancing_path, model_price, strike_table,
                                                    **rules)
    level, _ = compute_cfd_curtailment_compensation(wholesale_path, balancing_path, level_price, strike_table,
                                                    **rules)
    if model.empty:
        return pd.DataFrame(columns=PRICE_LEVEL_BID_COLUMNS)
    model = model.set_index(model.generator.astype(str))
    level = level.set_index(level.generator.astype(str)).reindex(model.index)
    keep = model.index.isin(contract_value_units(contract_value_bids_path) or set())
    if redispatch_summary_path is not None:
        frame = read_redispatch_in_force(redispatch_summary_path, index_col=0)
        keep &= ~model.index.isin(set(_non_participant_turn_down(frame).index.astype(str)))
    ownership = ownership_from_audit(contract_value_bids_path)
    share = (ownership["state_share"].reindex(model.index).fillna(0.0) if ownership
             else pd.Series(0.0, index=model.index))
    out = pd.DataFrame({
        "generator": model.index, "carrier": model.carrier.to_numpy(), "state_share": share.to_numpy(),
        "curtailed_mwh": model.curtailed_mwh.to_numpy(),
        "compensation_model_gbp": model.compensation_gbp.to_numpy(),
        "compensation_at_price_level_gbp": level.compensation_gbp.to_numpy(),
    })[keep].reset_index(drop=True)
    out["gross_change_gbp"] = out.compensation_at_price_level_gbp - out.compensation_model_gbp
    out["change_gbp"] = out.gross_change_gbp * (1.0 - out.state_share)
    return out[PRICE_LEVEL_BID_COLUMNS]


def price_level_references(case, wholesale_networks, prices):
    """``(mapped, record)``: the year on the case's price level, mapped exactly as its bill maps it (item F24.2).

    The bill (``project_future_bill.aggregate_inputs``) and the twin's asset settlement
    (``price_level_settlement``) map the same way, so all three move one hourly series.
    ``(None, None)`` at the model's own price level.
    """
    from scripts.consumer_bill import price_level_mapping as plm
    from scripts.consumer_bill.project_future_bill import DEFAULT_PRICE_LEVEL, map_year_to_price_level, price_level_of

    level = price_level_of(case)
    if level == DEFAULT_PRICE_LEVEL:
        return None, None
    hourly = plm.read_market_year(wholesale_networks, prices)
    mapped, record = map_year_to_price_level(hourly, (case or {}).get("price_level_mapping"))
    return mapped, {"price_level": level, **record}


def extract_month(scenario, wholesale_path, balancing_path, price_path, summary_path,
                  strike_table, include_interconnector_retrading: bool = False,
                  compensation_treatment: str = DEFAULT_COMPENSATION_TREATMENT,
                  redispatch_summary_path=None, contract_value_bids_path=None,
                  reconciliation_tolerance: float = DEFAULT_RECONCILIATION_TOLERANCE,
                  contract_book=None, level_price: pd.Series | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Return per-carrier compensation rows and the month's constraint accounting.

    ``level_price`` (item F24.2): the month's reference on a price level (a ``cp32``
    twin). Everything is extracted and reconciled at the model's price as before, then
    the CfD contract-value bids move to that reference (:func:`cfd_bid_change_at_price_level`)
    and the constraint cost takes the change, recorded as ``cfd_bid_price_level_change_gbp``.
    """
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if summary.get("scenario") != scenario:
        raise ValueError(f"Market summary does not identify {scenario}")
    accounting = summary["accounting"]
    # Older market summaries predate the re-trading term; degrade to zero and
    # say so, rather than failing the whole extraction on one missing key.
    retrading_raw = accounting.get("interconnector_retrading_cost_GBP")
    retrading_available = retrading_raw is not None
    retrading = float(retrading_raw) if retrading_available else 0.0
    if not np.isfinite(retrading):
        raise ValueError(f"{scenario}: interconnector re-trading cost is not finite")

    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)["wholesale_price"]
    if prices.index.has_duplicates:
        raise ValueError(f"{scenario}: wholesale prices have duplicate timestamps")

    ownership = ownership_from_audit(contract_value_bids_path)
    details, unlabelled = compute_cfd_curtailment_compensation(
        wholesale_path, balancing_path, prices, strike_table, ownership=ownership,
        negative_price_rule=negative_price_rule_from_audit(contract_value_bids_path),
        bid_basis=bid_basis_from_audit(contract_value_bids_path),
        contract_value_floor=contract_value_floor_from_audit(contract_value_bids_path),
        **({"contract_book": contract_book} if contract_book is not None else {}),
    )
    unlabelled.insert(0, "scenario", scenario)
    by_carrier = (
        details.groupby("carrier", as_index=False)[["curtailed_mwh", "compensation_gbp"]].sum()
        if not details.empty else
        pd.DataFrame(columns=["carrier", "curtailed_mwh", "compensation_gbp"])
    )
    by_carrier.insert(0, "scenario", scenario)

    compensation = float(details.compensation_gbp.sum()) if not details.empty else 0.0
    curtailed = float(details.curtailed_mwh.sum()) if not details.empty else 0.0
    demand = float(accounting["demand_MWh"])

    # The declared treatment must match what the balancing solve actually did.
    # Getting this wrong either double counts the compensation or drops it.
    components = constraint_components(compensation_treatment)
    priced_on_contract_value = audit_declares_contract_value_bids(contract_value_bids_path)
    expected = "reconcile_against_lp_bid" if priced_on_contract_value else "add_post_hoc"
    if compensation_treatment != expected:
        raise ValueError(
            f"{scenario}: the bill declares cfd_curtailment_compensation_treatment="
            f"{compensation_treatment!r}, but the balancing solve "
            f"{'did' if priced_on_contract_value else 'did not'} price CfD bids on contract "
            f"value, which requires {expected!r}. Adding a post-hoc compensation on top of a "
            "contract-value LP bid double counts it; omitting it after a scalar-zero LP bid "
            "drops it."
        )
    # Reconcile only the units the LP priced on contract value (see contract_value_units).
    priced_units = contract_value_units(contract_value_bids_path)
    whole_cfd_assets = label_priced_cfd_assets(contract_value_bids_path) if contract_book is not None else None
    if whole_cfd_assets is not None:
        # SETTLE2050: a blended 2050 stack reconciles on its all-CfD generators only.
        reconciled = details.loc[details.network_asset.astype(str).isin(whole_cfd_assets)] if not details.empty else details
        bid_names = reconciled.network_asset.drop_duplicates() if not reconciled.empty else pd.Index([])
    else:
        reconciled = (details if priced_units is None or details.empty
                      else details.loc[details.generator.astype(str).isin(priced_units)])
        bid_names = reconciled.generator if not reconciled.empty else pd.Index([])
    reconciled_compensation = float(reconciled.compensation_gbp.sum()) if not reconciled.empty else 0.0
    lp_bid_cost = (
        lp_cfd_bid_cost(redispatch_summary_path, bid_names)
        if redispatch_summary_path is not None and not reconciled.empty else None
    )
    gap = reconciled_compensation - lp_bid_cost if lp_bid_cost is not None else None
    gap_share = gap / reconciled_compensation if gap is not None and reconciled_compensation > 0 else None
    # Only consulted when the gross check fails, so a month that passes it is extracted
    # exactly as before; one that passes only net of penalty-priced units records it.
    net_of_penalty = None
    if (compensation_treatment == "reconcile_against_lp_bid" and gap_share is not None
            and abs(gap_share) > reconciliation_tolerance and abs(gap) > RECONCILIATION_TOLERANCE_ABS_GBP):
        net_of_penalty = reconciliation_net_of_penalty(redispatch_summary_path, reconciled,
                                                       reconciled_compensation, lp_bid_cost)
    if (compensation_treatment == "reconcile_against_lp_bid" and gap_share is not None
            and abs(gap_share) > reconciliation_tolerance and abs(gap) > RECONCILIATION_TOLERANCE_ABS_GBP
            and not (net_of_penalty is not None and net_of_penalty["gap_share"] is not None
                     and abs(net_of_penalty["gap_share"]) <= reconciliation_tolerance)):
        raise ValueError(
            f"{scenario}: the recomputed CfD curtailment compensation "
            f"(GBP {reconciled_compensation:,.0f}) and the balancing LP's own accepted CfD bid cost "
            f"(GBP {lp_bid_cost:,.0f}) differ by {100 * gap_share:+.3f}%, beyond the "
            f"{100 * reconciliation_tolerance:.3f}% tolerance. They are the same quantity by "
            "construction, so a breach means the LP and the bill are not pricing the same "
            "contracts: check that the bid stack and the bill read the same strike table, that "
            "the reference price given to the solve is the wholesale series the bill settles "
            "against, and that no CfD unit was priced out of the balancing mechanism by the "
            "participation policy while still being turned down."
        )
    month = {
        "scenario": scenario,
        "reported_period_hours": float(accounting["reported_period_hours"]),
        "demand_mwh": demand,
        "accepted_offer_cost_gbp": float(accounting["accepted_offer_cost_GBP"]),
        # On the month's turn-down basis: as solved, or + the pro rata zonal uplift on the wind bids.
        "accepted_bid_cost_gbp": float(accounting["accepted_bid_cost_GBP"]) + bid_cost_uplift_in_force(balancing_path),
        "accepted_bid_cost_as_solved_gbp": float(accounting["accepted_bid_cost_GBP"]),
        "turn_down_pro_rata_uplift_gbp": bid_cost_uplift_in_force(balancing_path),
        "turn_down_basis": month_basis(balancing_path),
        # Slack is unserved load priced at VOLL, not a balancing purchase, and is
        # reported separately so it never enters the bill.
        "feasibility_slack_cost_gbp": float(accounting["feasibility_slack_cost_GBP"]),
        "feasibility_slack_energy_mwh": float(accounting["feasibility_slack_energy_MWh"]),
        "cfd_curtailment_compensation_gbp": compensation,
        # Item BM1: what private owners were paid, and what the state paid itself.
        "cfd_curtailment_compensation_private_gbp": (
            float(details.compensation_private_gbp.sum()) if not details.empty else 0.0),
        "cfd_curtailment_compensation_state_gbp": (
            float(details.compensation_state_gbp.sum()) if not details.empty else 0.0),
        "state_owned_cfd_units": int((details.state_share > 0).sum()) if not details.empty else 0,
        # Inside the accepted bid cost; a labelled transfer, not an added cost.
        "cfd_strike_basis_premium_over_contract_value_gbp": (
            float(details.strike_basis_premium_over_contract_value_gbp.sum()) if not details.empty else 0.0),
        "cfd_bid_basis": bid_basis_from_audit(contract_value_bids_path),
        "cfd_contract_value_floor": contract_value_floor_from_audit(contract_value_bids_path),
        "cfd_curtailed_mwh": curtailed,
        "cfd_curtailment_compensation_treatment": compensation_treatment,
        "cfd_bids_priced_on_contract_value": priced_on_contract_value,
        # What the bids applied, for the bill's consistency check (None: not recorded).
        "balancing_negative_price_rule": recorded_negative_price_rule(contract_value_bids_path),
        "lp_cfd_accepted_bid_cost_gbp": lp_bid_cost,
        # The recompute over the contract-value-priced units the check compares.
        "cfd_compensation_reconciled_units_gbp": reconciled_compensation,
        "cfd_compensation_outside_contract_value_pricing_gbp": compensation - reconciled_compensation,
        "cfd_compensation_reconciliation_gap_gbp": gap,
        "cfd_compensation_reconciliation_gap_share": gap_share,
        "cfd_compensation_reconciliation_tolerance": reconciliation_tolerance,
        "cfd_compensation_reconciliation_tolerance_abs_gbp": RECONCILIATION_TOLERANCE_ABS_GBP,
        "cfd_fallback_strike_share_of_compensation": (
            float(details.loc[details.is_fallback_strike, "compensation_gbp"].sum() / compensation)
            if compensation > 0 else None
        ),
        # ROC generators need no post-hoc term: the balancing bid already prices
        # banding x buyout, so their compensation is inside the accepted bid cost.
        "roc_compensation_treatment": "priced_in_balancing_bid",
        "interconnector_retrading_cost_gbp": retrading,
        "interconnector_retrading_available": retrading_available,
        "include_interconnector_retrading": bool(include_interconnector_retrading),
    }
    # Item B8: the VOLL slack and the penalty-priced turn-down of balancing
    # non-participants both sit inside the accepted offer and bid costs. What is
    # subtracted is exactly what is inside them: the slack and penalty rows of the
    # per-generator redispatch summary (market_accounting's own definitions),
    # falling back to the market summary's accounting when no summary is given.
    slack_in, penalty, penalty_mwh, basis = _slack_and_penalty_in_accepted_costs(redispatch_summary_path, accounting)
    if net_of_penalty is not None:
        # Passed only once the penalty-priced units were taken out of both sides (P3).
        month["cfd_compensation_reconciliation_gap_net_of_penalty_share"] = net_of_penalty["gap_share"]
        month["cfd_compensation_reconciliation_penalty_priced_units"] = net_of_penalty["penalty_priced_units"]
    month["feasibility_slack_cost_in_accepted_costs_gbp"] = slack_in
    month["non_participant_turn_down_cost_at_penalty_gbp"] = penalty
    month["non_participant_turn_down_mwh"] = penalty_mwh
    month["slack_and_penalty_basis"] = basis
    # Item C7: the P3 tie steps order a tie; they are not money.
    tie_steps = tie_break_step_cost(redispatch_summary_path, contract_value_bids_path)
    month["tie_break_step_cost_gbp"] = tie_steps
    month["constraint_cost_gbp"] = (sum(month[component] for component in components)
                                    - slack_in - penalty - tie_steps)
    if include_interconnector_retrading:
        month["constraint_cost_gbp"] += retrading
    if level_price is not None:
        # Item F24.2: the CfD contract-value bids on the price level's reference, volumes as solved.
        moved = cfd_bid_change_at_price_level(wholesale_path, balancing_path, prices, level_price, strike_table,
                                              contract_value_bids_path, redispatch_summary_path)
        month["cfd_bid_price_level_change_gbp"] = float(moved.change_gbp.sum())
        month["cfd_bid_price_level_units"] = int(len(moved))
        month["constraint_cost_gbp"] += month["cfd_bid_price_level_change_gbp"]
    month["constraint_cost_gbp_per_mwh_demand"] = month["constraint_cost_gbp"] / demand
    month["cfd_curtailed_share_of_demand"] = curtailed / demand
    return by_carrier, month, unlabelled


def _slack_and_penalty_in_accepted_costs(redispatch_summary_path, accounting) -> tuple:
    """The VOLL slack and non-participant penalty cost inside the accepted offer and bid costs (item B8).

    Returns ``(slack_gbp, penalty_gbp, penalty_mwh, basis)``.
    """
    if redispatch_summary_path is not None:
        frame = read_redispatch_in_force(redispatch_summary_path, index_col=0)
        slack = frame.loc[_slack_mask(frame)]
        slack_cost = _sum(slack, "offer_cost") + _sum(slack, "bid_cost")
        forced = _non_participant_turn_down(frame)
        penalty = _sum(forced, "bid_cost")
        volume = _sum(forced, "decrease_MWh")
        return slack_cost, penalty, volume, "redispatch_summary_rows"
    return (float(accounting.get("feasibility_slack_cost_GBP") or 0.0),
            float(accounting.get("non_participant_turn_down_cost_at_penalty_GBP") or 0.0),
            float(accounting.get("non_participant_turn_down_MWh") or 0.0), "market_summary_accounting")


def extract_future_balancing_costs(scenarios, wholesale_networks, balancing_networks, prices,
                                   summaries, strike_table_path, indexation, model_price_year,
                                   include_interconnector_retrading: bool = False,
                                   cp30_pathway_curtailment_twh: dict | None = None,
                                   compensation_treatment: str = DEFAULT_COMPENSATION_TREATMENT,
                                   redispatch_summaries=None, contract_value_bids=None,
                                   reconciliation_tolerance: float = DEFAULT_RECONCILIATION_TOLERANCE,
                                   bill_negative_price_rule: str | None = None, strike_table_overrides=None,
                                   contract_book=None, case: dict | None = None):
    """Union the twelve monthly constraint extractions into one annual view.

    ``contract_book`` (a path; FES 2050 task SETTLE2050) recomputes the compensation on
    the 2050 book's CfD units instead of the network's labels.

    ``case`` (item F24.2): the forecast case. At a price level (``price_level: cp32``)
    the year is mapped as the case's bill maps it (:func:`price_level_references`) and
    each month's CfD contract-value bids move to the mapped reference; the annual record
    then says ``price_level``. Absent or at ``model`` nothing changes.

    ``strike_table_overrides`` are the bill's declared strike overrides
    (``model_linked_rules.cfd.strike_overrides``), so compensation is valued at the
    strike the CfD line settles.

    ``bill_negative_price_rule`` (the bill's ``model_linked_rules.cfd`` rule), when
    given, is checked against the rule the bid audits recorded
    (:func:`check_negative_price_rule`).

    ``cp30_pathway_curtailment_twh`` is the optional CP30 pathway curtailment by
    carrier (TWh) from the bill assumptions; when given, the annual CfD
    curtailed volume is expressed as a ratio to it, carrier by carrier.
    """
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Balancing-cost extraction needs non-empty, unique scenarios")
    lengths = {len(scenarios), len(wholesale_networks), len(balancing_networks),
               len(prices), len(summaries)}
    if len(lengths) != 1:
        raise ValueError("Balancing-cost inputs must align with the source scenarios")

    components = constraint_components(compensation_treatment)
    redispatch_summaries = list(redispatch_summaries or [None] * len(scenarios))
    contract_value_bids = list(contract_value_bids or [None] * len(scenarios))
    if len(redispatch_summaries) != len(scenarios) or len(contract_value_bids) != len(scenarios):
        raise ValueError("Redispatch summaries and bid audits must align with the source scenarios")

    strike_table = load_strike_table(strike_table_path, indexation, model_price_year,
                                     overrides=strike_table_overrides)
    # Item F24.2: the year on the case's price level (None at the model's).
    mapped, level_record = price_level_references(case, wholesale_networks, prices)
    if mapped is not None and contract_book is not None:
        raise ValueError("a price level is not implemented for a 2050 contract book (item F24.2)")
    carrier_frames, months, unlabelled_frames = [], [], []
    for scenario, wholesale, balancing, price, summary, redispatch, bid_audit in zip(
        scenarios, wholesale_networks, balancing_networks, prices, summaries,
        redispatch_summaries, contract_value_bids,
    ):
        by_carrier, month, unlabelled = extract_month(
            scenario, wholesale, balancing, price, summary, strike_table,
            include_interconnector_retrading=include_interconnector_retrading,
            compensation_treatment=compensation_treatment,
            redispatch_summary_path=redispatch, contract_value_bids_path=bid_audit,
            reconciliation_tolerance=reconciliation_tolerance,
            **({"contract_book": contract_book} if contract_book is not None else {}),
            **({"level_price": mapped} if mapped is not None else {}),
        )
        carrier_frames.append(by_carrier)
        months.append(month)
        unlabelled_frames.append(unlabelled)

    monthly = pd.DataFrame(months)
    hours = float(monthly.reported_period_hours.sum())
    if hours not in (8760.0, 8784.0):
        raise ValueError(
            f"Monthly constraint extractions cover {hours:g} hours, not one calendar year"
        )
    by_carrier = (
        pd.concat(carrier_frames)
        .groupby("carrier", as_index=False)[["curtailed_mwh", "compensation_gbp"]].sum()
        .sort_values("compensation_gbp", ascending=False)
    )
    # Capacity is a stock, so take the maximum month rather than the sum;
    # energy is a flow and sums over the year.
    unlabelled = pd.concat(unlabelled_frames)
    if not unlabelled.empty:
        unlabelled = unlabelled.groupby("carrier", as_index=False).agg(
            capacity_mw=("capacity_mw", "max"),
            physical_mwh=("physical_mwh", "sum"),
            generators=("generators", "max"),
        ).sort_values("physical_mwh", ascending=False)
    demand = float(monthly.demand_mwh.sum())
    compensation = float(monthly.cfd_curtailment_compensation_gbp.sum())
    # The reconciled subset (contract-value-priced units); the whole recompute for
    # months extracted before the subset was recorded.
    reconciled = float(monthly.cfd_compensation_reconciled_units_gbp.sum()
                       if "cfd_compensation_reconciled_units_gbp" in monthly else compensation)
    retrading = float(monthly.interconnector_retrading_cost_gbp.sum())
    retrading_available = bool(monthly.interconnector_retrading_available.all())
    # Item F24.2: the CfD bids' move to the price level (only a price-level extraction has the column).
    level_change = (float(monthly.cfd_bid_price_level_change_gbp.sum())
                    if "cfd_bid_price_level_change_gbp" in monthly else 0.0)
    constraint_cost_excluding_retrading = float(
        sum(monthly[component].sum() for component in components)
        - monthly.feasibility_slack_cost_in_accepted_costs_gbp.sum()
        - monthly.non_participant_turn_down_cost_at_penalty_gbp.sum()
        - monthly.tie_break_step_cost_gbp.sum()
        + level_change
    )
    lp_bid_cost = (
        float(monthly.lp_cfd_accepted_bid_cost_gbp.sum())
        if monthly.lp_cfd_accepted_bid_cost_gbp.notna().all() else None
    )
    constraint_cost = float(monthly.constraint_cost_gbp.sum())
    recorded = sorted({str(rule) for rule in monthly.balancing_negative_price_rule if rule is not None
                       and not (isinstance(rule, float) and np.isnan(rule))})
    if len(recorded) > 1:
        raise ValueError(f"The months' balancing bids applied different negative pricing rules: {recorded}")
    balancing_negative_price_rule = recorded[0] if recorded else None
    if bill_negative_price_rule is not None:
        check_negative_price_rule(bill_negative_price_rule, balancing_negative_price_rule,
                                  where="balancing-cost extraction")

    # CP30 pathway comparison: model network curtailment by carrier against the
    # pathway's own curtailment figures, where the assumptions supply them.
    network_curtailment_twh = {
        str(row.carrier): float(row.curtailed_mwh) / 1e6 for row in by_carrier.itertuples()
    }
    # The assumptions block carries a ``source`` string beside the numeric
    # carriers; keep only entries that are pathway volumes.
    pathway = {
        carrier: float(value)
        for carrier, value in dict(cp30_pathway_curtailment_twh or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    pathway_source = (cp30_pathway_curtailment_twh or {}).get("source")
    pathway_ratio = {}
    for carrier, pathway_twh in pathway.items():
        pathway_ratio[carrier] = (
            network_curtailment_twh.get(carrier, 0.0) / pathway_twh if pathway_twh > 0 else None
        )

    retrading_method = (
        " Interconnector re-trading cost is included in the constraint cost "
        "(include_interconnector_retrading = true)."
        if include_interconnector_retrading else
        " Interconnector re-trading cost is reported alongside but not included "
        "(include_interconnector_retrading = false)."
    )
    if not retrading_available:
        retrading_method += (
            " One or more market summaries predate the re-trading term, so it is "
            "carried as 0.0 for those months (interconnector_retrading_available = false)."
        )
    annual = {
        "reported_period_hours": hours,
        "demand_mwh": demand,
        "model_price_year": int(model_price_year),
        **{
            component: float(monthly[component].sum())
            for component in (*CONSTRAINT_COMPONENTS, "feasibility_slack_cost_gbp",
                              "feasibility_slack_energy_mwh", "cfd_curtailed_mwh",
                              "feasibility_slack_cost_in_accepted_costs_gbp",
                              "non_participant_turn_down_cost_at_penalty_gbp", "non_participant_turn_down_mwh",
                              "tie_break_step_cost_gbp")
        },
        # Item B8: the bill refuses an extraction without this flag.
        "constraint_cost_excludes_slack_and_penalty": True,
        # Pro rata zonal turn-down (20 Oct 2026): the basis each month was read on, and what it added.
        "turn_down_basis": sorted(set(monthly.turn_down_basis)),
        "turn_down_pro_rata_uplift_gbp": float(monthly.turn_down_pro_rata_uplift_gbp.sum()),
        "accepted_bid_cost_as_solved_gbp": float(monthly.accepted_bid_cost_as_solved_gbp.sum()),
        "slack_and_penalty_basis": sorted(set(monthly.slack_and_penalty_basis)),
        "constraint_cost_gbp": constraint_cost,
        "constraint_cost_excluding_retrading_gbp": constraint_cost_excluding_retrading,
        "interconnector_retrading_cost_gbp": retrading,
        "interconnector_retrading_available": retrading_available,
        "include_interconnector_retrading": bool(include_interconnector_retrading),
        "cfd_curtailment_compensation_share_of_constraint_cost": (
            compensation / constraint_cost if constraint_cost else None
        ),
        "cfd_curtailment_compensation_treatment": compensation_treatment,
        "cfd_bids_priced_on_contract_value": bool(
            monthly.cfd_bids_priced_on_contract_value.all()
        ),
        "balancing_negative_price_rule": balancing_negative_price_rule,
        "lp_cfd_accepted_bid_cost_gbp": lp_bid_cost,
        "cfd_compensation_reconciled_units_gbp": reconciled,
        "cfd_compensation_outside_contract_value_pricing_gbp": compensation - reconciled,
        # Review 28 Sep 2026: the strike bid basis pays turned-down private CfD units
        # min(reference, strike) above what they forgo. Inside the accepted bid cost,
        # so not added; a transfer to generators, not a resource cost.
        "cfd_strike_basis_premium_over_contract_value_gbp": float(
            monthly.cfd_strike_basis_premium_over_contract_value_gbp.sum()
            if "cfd_strike_basis_premium_over_contract_value_gbp" in monthly else 0.0),
        "cfd_bid_basis": sorted({str(b) for b in monthly.get("cfd_bid_basis", pd.Series(dtype=str)).dropna()}),
        "cfd_compensation_reconciliation_gap_gbp": (
            reconciled - lp_bid_cost if lp_bid_cost is not None else None
        ),
        "cfd_compensation_reconciliation_gap_share": (
            (reconciled - lp_bid_cost) / reconciled
            if lp_bid_cost is not None and reconciled > 0 else None
        ),
        "cfd_compensation_reconciliation_tolerance": reconciliation_tolerance,
        "network_curtailment_twh_by_carrier": network_curtailment_twh,
        "cp30_pathway_curtailment_twh": pathway or None,
        "cp30_pathway_curtailment_source": pathway_source,
        "cfd_curtailment_vs_cp30_pathway_ratio": pathway_ratio or None,
        "strike_table": strike_table_path if isinstance(strike_table_path, str)
                        else str(strike_table_path),
        "strike_indexation_factor": float(indexation["factor"]),
        "method": (
            "Constraint cost = " + " + ".join(
                component.replace("_gbp", "").replace("_", " ") for component in components
            ) + " - feasibility slack - non-participant penalty turn-down - P3 tie-break steps, summed over "
            "twelve complete months. Feasibility slack (unserved load priced at VOLL) and the forced turn-down "
            "of units outside the balancing mechanism, priced at the LP's penalty bid, both sit inside the "
            "accepted offer and bid costs and are not balancing purchases, so both are subtracted (item B8); "
            "so are the decision-20 tie steps inside state-owned CfD bids, which order a tie and cost "
            "nothing (item C7, as decompose_public_bm_bids nets them)."
            + (
                " CfD curtailment compensation is recomputed but not added: the balancing LP "
                "priced CfD bids on contract value, so it is already inside the accepted bid "
                "cost, and the recomputation serves as a reconciliation check against the LP's "
                "own CfD bid cost."
                if compensation_treatment == "reconcile_against_lp_bid" else
                " CfD curtailment compensation is added because the balancing LP priced CfD "
                "bids at a scalar, so it is absent from the accepted bid cost."
            ) + retrading_method
        ),
        "limitations": [
            "Accepted offer and bid costs are modelled redispatch against fixed wholesale "
            "positions. They are not NESO settlement expenditure.",
            (
                "CfD compensation is a reconciliation check: the balancing LP prices CfD bids "
                "at max(strike - reference, 0) per snapshot, so the settlement is inside the "
                "accepted bid cost. A non-zero reconciliation gap is a price-basis or volume "
                "difference between the LP and the recomputation, not a missing payment."
                if compensation_treatment == "reconcile_against_lp_bid" else
                "CfD compensation is a post-hoc term because the balancing LP prices CfD bids "
                "at a scalar 0.0, so curtailment is indifferent to strike. Contract-value bids "
                "(support.bm_bids.contract_value_bids) fix that and "
                "turn this term into a check."
            ),
            "Only the constraint part of BSUoS comes from the model; response, reserve, reactive, "
            "restoration and NESO internal costs remain an external allowance.",
        ],
        "unlabelled_support": unlabelled.to_dict("records") if not unlabelled.empty else [],
        "unlabelled_support_note": (
            "Generators carrying no support label. Their output accrues no CfD difference "
            "payment and no ROC in this bill. In the CP30 rebuild nuclear is placed by the "
            "station schedule: Sizewell B is merchant (correctly unlabelled) and Hinkley Point C "
            "is CfD-labelled, so the remaining unlabelled capacity is the EU market balance "
            "generators and merchant plant."
        ),
    }
    if level_record is not None:
        # Item F24.2: the bill checks this mapping is its own before it reads the line.
        annual.update({
            "price_level": level_record["price_level"],
            "price_level_mapping_check": {
                side: {key: float(level_record[side][key])
                       for key in ("time_weighted_gbp_per_mwh", "demand_weighted_gbp_per_mwh")}
                for side in ("model", "mapped")},
            "cfd_bid_price_level_change_gbp": level_change,
        })
        annual["method"] += (
            f" At price level {level_record['price_level']} (item F24.2) the CfD contract-value bids are re-priced "
            "on the mapped reference, volumes as solved (cfd_bid_price_level_change_gbp, added to the constraint "
            "cost); everything else in the line is at the model's price.")
    annual["constraint_cost_gbp_per_mwh_demand"] = annual["constraint_cost_gbp"] / demand
    return monthly, by_carrier, annual


def bsuos_rule_options(assumptions: dict) -> tuple[bool, dict | None, str, float]:
    """Read the BSUoS constraint rules from the bill assumptions.

    Returns the opt-in re-trading flag, the CP30 pathway curtailment volumes, the
    declared CfD compensation treatment and the reconciliation tolerance.
    """
    rule = assumptions.get("model_linked_rules", {}).get("bsuos_constraints", {})
    include = bool(rule.get("include_interconnector_retrading", False))
    pathway = rule.get("cp30_pathway_curtailment_twh")
    if pathway is not None and not isinstance(pathway, dict):
        raise ValueError("cp30_pathway_curtailment_twh must map carrier to TWh")
    treatment = str(rule.get("cfd_curtailment_compensation_treatment",
                             DEFAULT_COMPENSATION_TREATMENT))
    if treatment not in COMPENSATION_TREATMENTS:
        raise ValueError(
            f"cfd_curtailment_compensation_treatment must be one of "
            f"{list(COMPENSATION_TREATMENTS)}, got {treatment!r}"
        )
    tolerance = float(rule.get("cfd_compensation_reconciliation_tolerance",
                               DEFAULT_RECONCILIATION_TOLERANCE))
    if not 0.0 < tolerance < 1.0:
        raise ValueError(
            "cfd_compensation_reconciliation_tolerance is a share of the recomputed "
            f"compensation and must lie in (0, 1), got {tolerance}"
        )
    return include, pathway, treatment, tolerance


def contract_book_from_assumptions(assumptions: dict):
    """The 2050 contract book the bill names (``model_linked_rules.cfd.contract_book``), or None (every 2030 bill)."""
    from scripts.consumer_bill.project_future_bill import contract_book_path

    return contract_book_path(((assumptions or {}).get("model_linked_rules") or {}).get("cfd"))


def write_balancing_costs(scenarios, wholesale_networks, balancing_networks, prices, summaries,
                          strike_table_path, indexation, model_price_year,
                          output_monthly, output_carriers, output_summary,
                          include_interconnector_retrading: bool = False,
                          cp30_pathway_curtailment_twh: dict | None = None,
                          compensation_treatment: str = DEFAULT_COMPENSATION_TREATMENT,
                          redispatch_summaries=None, contract_value_bids=None,
                          reconciliation_tolerance: float = DEFAULT_RECONCILIATION_TOLERANCE,
                          bill_negative_price_rule: str | None = None, strike_table_overrides=None,
                          contract_book=None, case: dict | None = None):
    monthly, by_carrier, annual = extract_future_balancing_costs(
        scenarios, wholesale_networks, balancing_networks, prices, summaries,
        strike_table_path, indexation, model_price_year,
        include_interconnector_retrading=include_interconnector_retrading,
        cp30_pathway_curtailment_twh=cp30_pathway_curtailment_twh,
        compensation_treatment=compensation_treatment,
        redispatch_summaries=redispatch_summaries,
        contract_value_bids=contract_value_bids,
        reconciliation_tolerance=reconciliation_tolerance,
        bill_negative_price_rule=bill_negative_price_rule,
        strike_table_overrides=strike_table_overrides,
        **({"contract_book": contract_book} if contract_book is not None else {}),
        **({"case": case} if case is not None else {}),
    )
    for path in (output_monthly, output_carriers, output_summary):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_monthly, index=False)
    by_carrier.to_csv(output_carriers, index=False)
    Path(output_summary).write_text(
        json.dumps({"annual": annual, "by_carrier": by_carrier.to_dict("records")},
                   indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return annual


def compensation_strike_source(assumptions: dict, strike_table_path):
    """The table the CfD curtailment compensation is valued on, and its overrides.

    FES 2050 decision 41: when the bill names the table the solved BM bid on
    (``model_linked_rules.cfd.bm_strike_table``), the compensation reconciles against those
    accepted bids, so it is valued there, without the bill's contract overrides. Every 2030
    bill names none and keeps ``strike_table`` with its overrides.
    """
    bm = bm_strike_table_path(assumptions)
    if bm:
        return (bm if Path(bm).is_absolute() else str(Path(__file__).resolve().parents[2] / bm)), None
    return strike_table_path, strike_overrides(assumptions)


if "snakemake" in globals():
    case = snakemake.params.case
    _assumptions = json.loads(Path(snakemake.input.assumptions).read_text(encoding="utf-8"))
    _include_retrading, _pathway, _treatment, _tolerance = bsuos_rule_options(_assumptions)
    _strike_path, _strike_overrides = compensation_strike_source(_assumptions, snakemake.input.strike_table)
    write_balancing_costs(
        case["source_scenarios"],
        list(snakemake.input.wholesale_networks),
        list(snakemake.input.balancing_networks),
        list(snakemake.input.prices),
        list(snakemake.input.summaries),
        _strike_path,
        snakemake.params.indexation,
        case["model_price_year"],
        snakemake.output.monthly,
        snakemake.output.carriers,
        snakemake.output.summary,
        include_interconnector_retrading=_include_retrading,
        cp30_pathway_curtailment_twh=_pathway,
        compensation_treatment=_treatment,
        redispatch_summaries=list(snakemake.input.redispatch_summaries),
        contract_value_bids=list(snakemake.input.contract_value_bids),
        reconciliation_tolerance=_tolerance,
        bill_negative_price_rule=_assumptions["model_linked_rules"]["cfd"].get("negative_price_rule"),
        strike_table_overrides=_strike_overrides,
        contract_book=contract_book_from_assumptions(_assumptions),
        # Item F24.2: a price-level case (a cp32 twin) moves its CfD bids to the mapped reference.
        case=case,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", required=True)
    parser.add_argument("--wholesale", action="append", required=True)
    parser.add_argument("--balancing", action="append", required=True)
    parser.add_argument("--price", action="append", required=True)
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--redispatch-summary", action="append")
    parser.add_argument("--contract-value-bids", action="append")
    parser.add_argument("--strike-table", required=True)
    parser.add_argument("--assumptions", required=True)
    parser.add_argument("--model-price-year", type=int, required=True)
    parser.add_argument("--output-monthly", required=True)
    parser.add_argument("--output-carriers", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    assumptions = json.loads(Path(args.assumptions).read_text(encoding="utf-8"))
    include_retrading, pathway, treatment, tolerance = bsuos_rule_options(assumptions)
    strike_path, strike_table_overrides = compensation_strike_source(assumptions, args.strike_table)
    result = write_balancing_costs(
        args.scenario, args.wholesale, args.balancing, args.price, args.summary,
        strike_path, assumptions["cfd_strike_indexation"], args.model_price_year,
        args.output_monthly, args.output_carriers, args.output_summary,
        include_interconnector_retrading=include_retrading,
        cp30_pathway_curtailment_twh=pathway,
        compensation_treatment=treatment,
        redispatch_summaries=args.redispatch_summary,
        contract_value_bids=args.contract_value_bids,
        reconciliation_tolerance=tolerance,
        bill_negative_price_rule=assumptions["model_linked_rules"]["cfd"].get("negative_price_rule"),
        strike_table_overrides=strike_table_overrides,
        contract_book=contract_book_from_assumptions(assumptions),
    )
    print(f"constraint cost: GBP {result['constraint_cost_gbp'] / 1e6:,.1f}M "
          f"({result['constraint_cost_gbp_per_mwh_demand']:.3f}/MWh demand); "
          f"interconnector re-trading GBP {result['interconnector_retrading_cost_gbp'] / 1e6:,.1f}M "
          f"({'included' if result['include_interconnector_retrading'] else 'excluded'})")
