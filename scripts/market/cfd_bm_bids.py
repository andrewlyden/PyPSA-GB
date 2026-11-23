"""Contract-value balancing bids for supported generation (backlog item B1).

**The defect this replaces.** ``_apply_renewable_support_pricing`` in
``market_utils`` gives every CfD generator one scalar bid, and in the CP30
configuration that scalar is the evidence-gap fallback
``support.bm_bids.cfd.bid_price`` = 0.0. With 73 GW of CfD capacity
all bidding the same zero, the balancing LP has no reason to prefer turning down
a cheap contract over an expensive one: in the January 2030 run the Investment
Contract offshore fleet, at a 200.84 GBP/MWh indexed strike, was curtailed at
two to three times the per-GW rate of AR4 at 52.04. Curtailment volume was set
by the network, but its *composition* - and therefore the constraint cost, the
CfD curtailment compensation and the BSUoS bill line - was arbitrary.

**What a CfD unit actually loses by being turned down.** It keeps no wholesale
revenue either way, because the balancing stage buys back its wholesale
position. What it loses is the difference payment, ``strike - reference``, on
every MWh it does not deliver, and nothing when the reference price already
exceeds the strike. So the bid, in the ESO-cost convention this model uses
(positive = it costs the ESO to decrease output), is

    bid(i, t) = max(strike_i - reference_t, 0)

which is exactly the rate ``extract_future_balancing_costs.py`` already applies
*after* the solve to reconstruct the compensation the zero bid left out. Moving
it into the LP is what makes the reconstruction a reconciliation check rather
than a correction.

**Bid basis** (``contract_value_bids.bid_basis``). ``contract_value`` is the
reservation price above. ``strike`` bids the full strike in every hour the
contract pays, which is what CfD wind actually bids: 2024 BM acceptances on CfD
wind BMUs averaged 64.6 GBP/MWh (AR1 59, AR2 65) while strike minus reference was
well below it, and NESO prices CP30 constraint bids from allocation round data
(user decision, 27 Sep 2026: strike for CP30). The negative pricing rule applies
under both, and the audit records the basis so the post-solve compensation and
the settlement price turned-down energy the same way.

**Contract-value floor** (``contract_value_bids.contract_value_floor``, 16 Oct 2026).
``zero`` (the default) is the reservation price above, ``max(strike - reference, 0)``.
``none`` is FTI's NESO baseline ("CfD strike price - Wholesale price", FTI for NESO,
Mar 2025, slide 53) without the floor: in an hour whose reference exceeds the strike
the unit would pay the difference back on every MWh it generates, so being turned
down saves it ``reference - strike`` and it bids ``strike - reference < 0`` in this
ESO-cost convention (NESO receives that saving; in ELEXON's convention, where a bid
is what the unit pays, it is ``reference - strike > 0``). The floor applies to the
contract-value basis only, and the audit records it beside the basis, so the
compensation recompute and the settlement use the same rate
(:func:`foregone_rate`).

Merchant renewables are the mirror-image error: they hold no contract, so their
cost of turning down is their short-run marginal cost, near zero. Today they
inherit ``carrier_overrides`` written before support labels existed - onshore 25,
offshore 90, solar 55 GBP/MWh - a blended subsidy they do not have, which makes
them the *last* renewables the LP turns down rather than the first. This module
prices them at the configured merchant bid, 20 GBP/MWh by default since 27 Sep
2026: what merchant wind was paid to turn down in the 2024 BM (lost REGO and PPA
volume). Until then it was -0.5, the Stage 1 merchant variable cost
(``renewable_curtailment_order.merchant_variable_cost_gbp_per_mwh``, +0.5,
decision D1) handed back to NESO as a negative bid in this ESO-cost convention,
which turned merchant units down first even in a negative-reference hour.

The ROC branch is already right and is left alone: banding x buyout is what a
ROC generator forgoes, it does not vary with the reference price, and the scalar
carries it correctly.

**Carriers excluded from support pricing keep their carrier override.** The
exclusion list is shared with ``_apply_renewable_support_pricing`` and exists
for nuclear: Hinkley Point C is a CfD unit, and pricing it on contract value
would let the balancing LP turn it up and down against a 92.50 strike instead of
the ``nuclear`` override that represents its restart cost.

**Reference price.** Intermittent contracts settle against the hourly day-ahead
reference, proxied here by the wholesale clearing price the Stage 1 solve
produced for the same snapshots. Baseload contracts settle against a season-ahead
baseload reference; where a carrier is listed in ``reference_price_basis`` this
module uses the demand-weighted mean price of the solve window, which for the
monthly CP30 runs is a single-month proxy for a seasonal quantity and is
recorded as such in the audit. Under the CP30 configuration nuclear is the only
baseload CfD carrier and it is excluded, so that branch is inert.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.market.cfd_strike_table import contract_technology, load_strike_table, strike_lookup
from scripts.market.revenue_tracking import negative_price_eligibility


SUMMER_MONTHS = (4, 5, 6, 7, 8, 9)

# Carriers whose ``carrier_overrides`` bid price is a pre-support-label blended
# subsidy rather than an opportunity cost. Defaults here must match
# config/defaults/base/support.yaml support.bm_bids
# .contract_value_bids.merchant.carriers.
#: Must match config/defaults/base/support.yaml ...contract_value_bids.merchant.bid_price:
#: 20 GBP/MWh from 27 Sep 2026 (2024 merchant wind BM bids; was -0.5, the Stage 1
#: merchant variable cost, see the module docstring).
DEFAULT_MERCHANT_BID = 20.0

DEFAULT_MERCHANT_CARRIERS = (
    "wind_onshore",
    "wind_offshore",
    "solar_pv",
    "embedded_wind",
    "embedded_solar",
)

AUDIT_COLUMNS = (
    "generator", "carrier", "support_type", "cfd_round", "p_nom_mw",
    "pricing", "strike_key", "strike_price_gbp_per_mwh", "is_fallback_strike",
    "reference_price_basis", "mean_reference_price_gbp_per_mwh",
    "mean_bid_gbp_per_mwh", "min_bid_gbp_per_mwh", "max_bid_gbp_per_mwh",
    "hours_bid_at_zero", "previous_static_bid_gbp_per_mwh",
    "state_share", "contract_value_mean_bid_gbp_per_mwh", "tie_break_step_gbp_per_mwh",
    "negative_price_rule", "bid_basis", "contract_value_floor",
)

#: ``contract_value_bids.bid_basis``: what a privately held CfD unit bids to be turned down.
#: Must match config/defaults/base/support.yaml ...contract_value_bids.bid_basis.
BID_BASES = ("contract_value", "strike")
DEFAULT_BID_BASIS = "strike"
#: ``contract_value_bids.contract_value_floor``: ``zero`` floors the contract-value bid at
#: 0; ``none`` lets it go negative when the reference exceeds the strike (FTI's NESO
#: baseline). Must match config/defaults/base/support.yaml ...contract_value_bids.contract_value_floor.
CONTRACT_VALUE_FLOORS = ("zero", "none")
DEFAULT_CONTRACT_VALUE_FLOOR = "zero"


def resolve_contract_value_floor(value) -> str:
    """``contract_value_floor`` validated (``None`` or absent: the default)."""
    floor = str(DEFAULT_CONTRACT_VALUE_FLOOR if value is None else value).strip().lower()
    if floor not in CONTRACT_VALUE_FLOORS:
        raise ValueError(f"contract_value_bids.contract_value_floor must be one of {CONTRACT_VALUE_FLOORS}, "
                         f"got {value!r}")
    return floor


def contract_value_rate(strike, reference, floor: str = DEFAULT_CONTRACT_VALUE_FLOOR):
    """``strike - reference``, floored at zero unless ``floor`` is ``none`` (Series or array in, same out)."""
    rate = strike - reference
    return np.maximum(rate, 0.0) if resolve_contract_value_floor(floor) == "zero" else rate


def foregone_rate(strike, reference, bid_basis: str = DEFAULT_BID_BASIS,
                  floor: str = DEFAULT_CONTRACT_VALUE_FLOOR):
    """What a privately held CfD unit is paid per MWh turned down: the bid it made.

    ``strike`` basis: the whole strike in every hour; ``contract_value``: the contract
    value, floored per ``floor``. ``reference`` is a Series (hourly) and the result a
    Series on its index; ``strike`` a scalar.
    """
    if str(bid_basis) == "strike":
        return pd.Series(float(strike), index=reference.index)
    return contract_value_rate(strike, reference, floor)

#: Decision 20: how P3's all-state-owned tie is broken. ``fifo`` exists only as the
#: reverse ordering the two-orderings gate compares against.
TIE_BREAK_RULES = ("lifo_commissioning", "fifo_commissioning")
#: Small enough not to reorder anything but the tie, large enough to sit above the
#: barrier's convergence tolerance (decision 20: 0.01 to 1 GBP/MWh per year).
TIE_BREAK_STEP_RANGE = (0.01, 1.0)


class ContractValueBids:
    """The two per-snapshot price frames and the per-generator audit.

    ``offer_regulariser`` (generator -> GBP/MWh, or ``None``) is added to the
    offers in the balancing objective only, never to the prices costs are
    reported at; see :func:`resolve_turn_up_floor`.
    """

    def __init__(self, offer_tv: pd.DataFrame, bid_tv: pd.DataFrame,
                 audit: pd.DataFrame, reference_price: pd.Series,
                 offer_regulariser: pd.Series | None = None,
                 bid_regulariser: pd.Series | None = None):
        self.offer_tv = offer_tv
        self.bid_tv = bid_tv
        self.audit = audit
        self.reference_price = reference_price
        self.offer_regulariser = offer_regulariser
        self.bid_regulariser = bid_regulariser


def season_labels(index) -> pd.Series:
    """Label each snapshot summer (Apr-Sep) or winter (Oct-Mar)."""
    index = pd.DatetimeIndex(index)
    return pd.Series(np.where(index.month.isin(SUMMER_MONTHS), "summer", "winter"),
                     index=index)


def demand_weighted_reference(reference_price: pd.Series, network) -> pd.Series:
    """Flat demand-weighted mean of ``reference_price`` over the solve window.

    The baseload proxy. A season-ahead reference cannot be formed inside one
    monthly solve, so the window's own demand-weighted mean stands in for it and
    the audit records which window it came from.
    """
    demand = _network_demand_mw(network, reference_price.index)
    weight = demand * _snapshot_hours(network, reference_price.index)
    total = float(weight.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "A baseload reference price needs positive demand-weighted hours in the "
            "solve window"
        )
    mean = float((reference_price * weight).sum() / total)
    return pd.Series(mean, index=reference_price.index, dtype=float)


def _snapshot_hours(network, snapshots) -> pd.Series:
    weightings = getattr(network, "snapshot_weightings", None)
    if weightings is not None and "generators" in getattr(weightings, "columns", []):
        return pd.to_numeric(weightings["generators"], errors="coerce").reindex(
            snapshots).fillna(1.0).astype(float)
    return pd.Series(1.0, index=snapshots, dtype=float)


def _network_demand_mw(network, snapshots) -> pd.Series:
    """Total load per snapshot, from the time-varying set-points where present."""
    loads_t = getattr(network, "loads_t", None)
    p_set = getattr(loads_t, "p_set", None) if loads_t is not None else None
    if p_set is not None and len(getattr(p_set, "columns", [])) > 0:
        series = p_set.sum(axis=1).reindex(snapshots)
        static = pd.to_numeric(network.loads.get("p_set", 0.0), errors="coerce")
        missing = network.loads.index.difference(p_set.columns)
        if len(missing):
            series = series + float(static.reindex(missing).fillna(0.0).sum())
        return series.astype(float)
    static_total = float(
        pd.to_numeric(network.loads.get("p_set", 0.0), errors="coerce").fillna(0.0).sum()
    ) if len(network.loads) else 0.0
    return pd.Series(static_total, index=snapshots, dtype=float)


def resolve_reference_price(network, reference_price) -> pd.Series:
    """Align the Stage 1 clearing price to the balancing snapshots."""
    if reference_price is None:
        raise ValueError(
            "Contract-value balancing bids need the Stage 1 wholesale clearing price as "
            "the CfD reference; attach it as network._bm_reference_price before "
            "calculate_bid_offer_prices, or disable "
            "support.bm_bids.contract_value_bids"
        )
    if isinstance(reference_price, pd.DataFrame):
        if "wholesale_price" not in reference_price.columns:
            raise ValueError("Reference price frame has no 'wholesale_price' column")
        reference_price = reference_price["wholesale_price"]
    series = pd.to_numeric(pd.Series(reference_price), errors="coerce")
    series.index = pd.DatetimeIndex(series.index)
    aligned = series.reindex(pd.DatetimeIndex(network.snapshots))
    if aligned.isna().any():
        missing = int(aligned.isna().sum())
        raise ValueError(
            f"The wholesale reference price is missing for {missing} of "
            f"{len(aligned)} balancing snapshots; CfD bids cannot be priced"
        )
    return aligned.astype(float)


def load_scenario_strike_table(scenario_config: dict):
    """Build the bill's :class:`StrikeTable` from ``support.labels``.

    Deliberately the same table, file and indexation the CfD bill line uses, so
    the LP bid and the post-solve compensation cannot drift apart.
    """
    block = ((((scenario_config or {}).get("support", {}) or {}).get("labels", {}) or {}).get(
        "cfd_strike_table", {}) or {})
    path = block.get("file")
    if not path:
        raise ValueError(
            "Contract-value balancing bids need support.labels.cfd_strike_table.file, "
            "the same published strike table the CfD bill line settles against"
        )
    if not Path(path).exists():
        raise FileNotFoundError(f"CfD strike table not found: {path}")
    return load_strike_table(path, block.get("indexation", {}) or {},
                             int(block.get("model_price_year", 2024)))


def _support_labels(network) -> pd.Series:
    if "support_type" not in network.generators.columns:
        raise ValueError(
            "Contract-value balancing bids need generator support_type labels; the "
            "network carries none"
        )
    return (network.generators["support_type"].astype("string").fillna("merchant")
            .str.strip().str.lower())


def resolve_public_ownership_bids(block, logger, network=None):
    """Item BM1: the state ownership shares the CfD bid blends towards, or nothing.

    Returns ``(shares, public_bid_price)``.  ``shares`` is ``None`` when the block
    is absent or disabled, which is the default and reproduces the pre-BM1 stack
    exactly -- the blend then multiplies by a zero share and cannot move a number.

    The share table is a small per-generator artefact written by
    ``scripts/public_ownership/ownership_shares.py``; the balancing stage must not
    read the ~600 MB register itself, and going through the register means the bid
    stack and the bill cannot disagree about who owns what.
    """
    block = dict(block or {})
    if not block or not block.get("enabled", False):
        return None, 0.0
    from scripts.public_ownership.ownership_shares import load_shares

    case_id = block.get("case_id")
    shares_file = block.get("shares_file")
    if not case_id or not shares_file:
        raise ValueError(
            "contract_value_bids.public_ownership_bids needs case_id and shares_file "
            "(the per-generator table written by scripts/public_ownership/ownership_shares.py)"
        )
    if not Path(shares_file).exists():
        raise FileNotFoundError(f"ownership share table not found: {shares_file}")
    price = float(block.get("public_bid_price", 0.0))
    if price < 0.0:
        raise ValueError("contract_value_bids.public_ownership_bids.public_bid_price must not be negative")
    if network is not None:
        # A stale table (units renamed by a network rebuild) would hand state plant back to
        # private owners without an error; refuse it before bidding (share_coverage.py).
        from scripts.public_ownership.share_coverage import check_share_coverage

        check_share_coverage(shares_file, network, case_id, logger=logger)
    shares = load_shares(shares_file, case_id)
    owned = float((shares > 0).sum())
    logger.info(
        "Item BM1: state-owned CfD units bid towards %.2f GBP/MWh on case %r; the share table "
        "carries %d generators, %d of them at least partly state-owned",
        price, case_id, len(shares), int(owned),
    )
    return shares, price


def lifo_tie_steps(years: pd.Series, step: float, rule: str) -> pd.Series:
    """Per-generator bid step that orders an all-at-cost tie by commissioning year.

    ``lifo_commissioning``: step x (newest year - year), so the newest plant bids
    lowest and is turned down first. ``fifo_commissioning`` is its reverse.
    """
    if rule not in TIE_BREAK_RULES:
        raise ValueError(f"tie-break rule must be one of {TIE_BREAK_RULES}, not {rule!r}")
    low, high = TIE_BREAK_STEP_RANGE
    if not (low <= float(step) <= high):
        raise ValueError(f"tie-break step must be within {low} to {high} GBP/MWh per year, not {step}")
    years = pd.to_numeric(pd.Series(years), errors="coerce").astype(float)
    if years.isna().any():
        missing = list(years.index[years.isna()][:5])
        raise ValueError(
            f"{int(years.isna().sum())} state-owned units have no commissioning year to rank on "
            f"(first: {missing}); the tie-break cannot order them")
    if rule == "lifo_commissioning":
        return float(step) * (years.max() - years)
    return float(step) * (years - years.min())


def resolve_tie_break(tie_block, ownership_block, state_shares, logger, eligible=None):
    """Decision 20: the per-generator tie step for state-owned units, or ``None``.

    Only generators with a positive state share are ranked; the step is added to
    the state-owned part of the bid alone, so private plant keeps its contract value.
    ``eligible`` limits the ranking to the generators whose bid carries the step
    (CfD units priced on contract value): under P3 every unit is state-owned, and
    merchant plant, storage and the excluded carriers never see a step, so they
    need no year. Years come from ``tie_break.years_file`` when declared (the
    table ``commissioning_years.py`` writes), else from the share table.
    """
    tie_block = dict(tie_block or {})
    if not tie_block or not tie_block.get("enabled", False) or state_shares is None:
        return None
    from scripts.public_ownership.ownership_shares import load_commissioning_years

    missing = [key for key in ("rule", "step_gbp_per_mwh") if key not in tie_block]
    if missing:
        # Declared per scenario, like the rest of public_ownership_bids: no silent default.
        raise ValueError(f"public_ownership_bids.tie_break needs {missing} when enabled")
    rule = str(tie_block["rule"])
    step = float(tie_block["step_gbp_per_mwh"])
    years_file = tie_block.get("years_file") or ownership_block["shares_file"]
    years = load_commissioning_years(years_file, ownership_block["case_id"])
    owned = state_shares.index[state_shares.gt(0.0)]
    if eligible is not None:
        owned = owned.intersection(pd.Index(eligible))
    steps = lifo_tie_steps(years.reindex(owned), step, rule)
    logger.info(
        "Decision 20: %s tie-break on %d state-owned units, %.2f GBP/MWh per year, steps 0 to %.2f",
        rule, len(steps), step, float(steps.max()) if len(steps) else 0.0,
    )
    return steps


#: Carriers that bid and offer close to zero once state-owned, between which a
#: balancing swap is otherwise free.
TURN_UP_FLOOR_CARRIERS = frozenset({"wind_offshore", "wind_onshore", "solar_pv", "marine"})
#: A regulariser, not a price: large enough for the solver to see, small beside
#: every real offer (the cheapest thermal offers are tens of GBP/MWh).
TURN_UP_FLOOR_MAX = 5.0


def resolve_turn_up_floor(tie_block, state_shares, carriers: pd.Series, logger):
    """Decision 20: an objective-only price on turning state-owned renewables up, or ``None``.

    Under P3 state-owned wind and solar bid their tie step (0 for the zero-step
    vintage) and offer about 0.02 GBP/MWh, so turning one down and another up in the
    same hour costs nothing the solver can resolve: on the P3 week the size of those
    swaps was arbitrary, 117.5 GWh under LIFO against 102.7 under FIFO, the whole
    1.3% gap between the orderings. Every swap has a turn-up leg, so a floor on
    turning up any state-owned unit of these carriers prices every swap route
    whichever unit is turned down. It is added to the objective only and its cost is
    stripped back out of the reported balancing cost, like the storage dispatch floor.
    """
    tie_block = dict(tie_block or {})
    floor = float(tie_block.get("turn_up_floor_gbp_per_mwh", 0.0) or 0.0)
    if not tie_block.get("enabled", False) or state_shares is None or floor <= 0.0:
        return None
    if floor > TURN_UP_FLOOR_MAX:
        raise ValueError(f"tie_break.turn_up_floor_gbp_per_mwh must be at most {TURN_UP_FLOOR_MAX}, not {floor}")
    owned = state_shares[state_shares.gt(0.0)]
    renewable = pd.Index(carriers.index[carriers.astype(str).str.lower().isin(TURN_UP_FLOOR_CARRIERS)])
    shares = owned.reindex(owned.index.intersection(renewable))
    logger.info("Decision 20: objective-only turn-up floor of %.2f GBP/MWh on %d state-owned renewable units",
                floor, len(shares))
    return floor * shares


def resolve_general_turn_up_floor(cfg: dict, carriers: pd.Series, logger) -> pd.Series | None:
    """An objective-only floor on turning ANY wind, solar or marine unit up, or ``None``.

    The P3 floor (:func:`resolve_turn_up_floor`) covers state-owned plant; the
    contract-aware wholesale order moved the same free swap onto merchant plant,
    which bids and offers about 0 in the balancing stage: on the 3-9 Jan test
    week renewable swaps rose from 69 to 161 GWh while the balancing cost fell,
    i.e. volume the solver could not tell from no swap. Two units of one
    technology genuinely are indifferent, so this is a regulariser for the
    reported volumes, not a price, and its cost is stripped like the P3 floor
    (user decision, 26 Sep 2026).
    """
    floor = float((cfg or {}).get("turn_up_floor_gbp_per_mwh", 0.0) or 0.0)
    if floor <= 0.0:
        return None
    if floor > TURN_UP_FLOOR_MAX:
        raise ValueError(f"contract_value_bids.turn_up_floor_gbp_per_mwh must be at most {TURN_UP_FLOOR_MAX}, "
                         f"not {floor}")
    renewable = carriers.index[carriers.astype(str).str.lower().isin(TURN_UP_FLOOR_CARRIERS)]
    logger.info("Objective-only turn-up floor of %.2f GBP/MWh on %d renewable units", floor, len(renewable))
    return pd.Series(floor, index=pd.Index(renewable), dtype=float)


def resolve_state_first_step(ownership_block, state_shares, merchant_units: pd.Index, logger):
    """Item BM1: an objective-only step on turning private merchant plant down, or ``None``.

    Under P2 a state-owned CfD share bids the public price (0) and a merchant
    renewable bids 0 too, so which of them the balancing stage turns down is an
    exact tie the solver splits arbitrarily; on the 2030 months 6.3 TWh of near-free
    turn-down landed on CfD units in P2 against about none in base, and that split
    feeds the settlement and the BM1 transfer. The user's rule (26 Sep 2026): the
    state turns down first, because an at-cost contract recovers its revenue whatever
    it generates while a merchant unit loses its market margin. The step lands on
    each merchant unit's private share, is added to the objective only and is
    stripped from the reported balancing cost.
    """
    block = dict(ownership_block or {})
    step = float(block.get("state_first_step_gbp_per_mwh", 0.0) or 0.0)
    if not block.get("enabled", False) or state_shares is None or step <= 0.0 or not len(merchant_units):
        return None
    if step > TURN_UP_FLOOR_MAX:
        raise ValueError(f"public_ownership_bids.state_first_step_gbp_per_mwh must be at most "
                         f"{TURN_UP_FLOOR_MAX}, not {step}")
    private = 1.0 - state_shares.reindex(merchant_units).fillna(0.0).clip(0.0, 1.0)
    private = private[private.gt(0.0)]
    logger.info("Item BM1: state-first step of %.2f GBP/MWh on turning down %d private merchant units "
                "(objective only)", step, len(private))
    return step * private


#: Item F8: carriers whose state-owned part bids and offers at short-run cost when
#: ``public_ownership_bids.thermal_at_cost`` is on.
STATE_THERMAL_CARRIERS = frozenset({"ccgt", "ocgt", "gas_engine", "chp", "ccs gas", "oil"})


def apply_state_thermal_at_cost(network, ownership_block, state_shares, offer_tv: pd.DataFrame,
                                bid_tv: pd.DataFrame, gen_offer: pd.Series, gen_bid: pd.Series,
                                carrier_lower: pd.Series, logger, turn_down_saving: bool = True) -> int:
    """Item F8: the state-owned share of thermal plant bids and offers its short-run cost.

    Private gas offers its marginal cost with the balancing markup (CP30: 1.29 x SRMC,
    FTI) and bids it less a discount; a state-owned unit run at cost forgoes only its
    short-run cost either way. Like the CfD blend, ownership reaches the LP as a
    capacity-weighted blend: price = (1 - share) x private price + share x state price.
    The state offers its marginal cost; it bids its avoided marginal cost handed back
    to NESO, which in the bids' ESO-cost convention is ``-mc`` under ``net_saving``
    (the private bid is ``-0.9 x mc``) and ``+mc`` only under ``legacy_cost``. Before
    13 Oct 2026 the state bid ``+mc`` under ``net_saving`` too, so NESO paid the state
    to turn its gas down and P3 turned almost none down (0.01 TWh against 0.51 private).
    Off unless ``public_ownership_bids.thermal_at_cost`` is true. Returns the units moved.
    """
    block = dict(ownership_block or {})
    if not block.get("thermal_at_cost", False) or state_shares is None:
        return 0
    thermal = carrier_lower.index[carrier_lower.isin(STATE_THERMAL_CARRIERS)]
    shares = state_shares.reindex(thermal).fillna(0.0).clip(0.0, 1.0)
    owned = shares.index[shares.gt(0.0)]
    if not len(owned):
        logger.info("Item F8: thermal_at_cost on, but the case owns no thermal plant")
        return 0
    s = shares.loc[owned].to_numpy()
    cost = pd.to_numeric(network.generators.loc[owned, "marginal_cost"], errors="coerce").fillna(0.0).to_numpy()
    state_bid = -cost if turn_down_saving else cost
    offer_tv.loc[:, owned] = offer_tv.loc[:, owned].to_numpy() * (1.0 - s) + cost * s
    bid_tv.loc[:, owned] = bid_tv.loc[:, owned].to_numpy() * (1.0 - s) + state_bid * s
    moved = (gen_offer.reindex(owned) - pd.Series(cost, index=owned)).mul(s).abs()
    logger.info("Item F8: %d state-owned thermal units bid and offer at short-run cost (state share mean "
                "%.2f); offers lowered by up to %.2f GBP/MWh", len(owned), float(s.mean()), float(moved.max()))
    return len(owned)


def apply_state_renewables_at_cost(ownership_block, state_shares, bid_tv: pd.DataFrame, support: pd.Series,
                                   carrier_lower: pd.Series, excluded_mask: pd.Series, public_bid_price: float,
                                   logger) -> pd.Series:
    """The state-owned share of RO and merchant renewables bids the public price, like its CfD units.

    User decision, 13 Oct 2026: under P3 the state owns RO and merchant plant, and the
    tariff removes the RO line from the bill, so turning that plant down costs the state
    nothing beyond its avoided short-run cost (about zero). Without this its share kept
    the private bid: the forgone ROCs (about 72 GBP/MWh onshore) or the merchant bid.
    Same capacity-weighted blend as the CfD loop: (1 - share) x private + share x public
    price. No tie step: decision 20 ranks only the CfD units. Off unless
    ``public_ownership_bids.renewables_at_cost`` is true. Returns the shares moved.
    """
    block = dict(ownership_block or {})
    if not block.get("renewables_at_cost", False) or state_shares is None:
        return pd.Series(dtype=float)
    carriers = {str(c).strip().lower() for c in
                (block.get("renewables_at_cost_carriers", DEFAULT_MERCHANT_CARRIERS) or [])}
    mask = support.isin(("roc", "merchant")) & carrier_lower.isin(carriers) & ~excluded_mask
    shares = state_shares.reindex(support.index[mask]).fillna(0.0).clip(0.0, 1.0)
    shares = shares[shares.gt(0.0)]
    if not len(shares):
        logger.info("renewables_at_cost on, but the case owns no RO or merchant renewables")
        return shares
    s = shares.to_numpy()
    owned = shares.index
    bid_tv.loc[:, owned] = bid_tv.loc[:, owned].to_numpy() * (1.0 - s) + float(public_bid_price) * s
    logger.info("State-owned RO and merchant renewables bid at cost: %d units (RO %d, merchant %d), "
                "state share mean %.2f", len(owned), int(support.reindex(owned).eq("roc").sum()),
                int(support.reindex(owned).eq("merchant").sum()), float(s.mean()))
    return shares


def build_contract_value_bid_frames(
    network,
    scenario_config: dict,
    balancing: dict,
    gen_offer: pd.Series,
    gen_bid: pd.Series,
    reference_price,
    logger: logging.Logger,
) -> ContractValueBids:
    """Return per-snapshot bid and offer frames priced on contract value.

    The frames start as the static prices broadcast across snapshots, so every
    generator this module does not touch keeps exactly the price the existing
    pricing chain gave it.
    """
    support_cfg = ((scenario_config or {}).get("support", {}) or {}).get("bm_bids", {}) or {}
    cfg = support_cfg.get("contract_value_bids", {}) or {}
    if cfg.get("support_labels"):
        # Task BM2050: a 2050 fleet's generators each hold several build-year
        # contracts, priced from its support label table (scripts/market/
        # support_bids.py). No 2030 scenario sets the key.
        from scripts.market.support_bids import build_label_bid_frames

        return build_label_bid_frames(network, scenario_config, balancing, gen_offer, gen_bid,
                                      reference_price, logger)
    merchant_cfg = cfg.get("merchant", {}) or {}
    state_shares, public_bid_price = resolve_public_ownership_bids(
        cfg.get("public_ownership_bids"), logger, network=network)
    ownership_block = cfg.get("public_ownership_bids") or {}
    negative_price_rule = str(cfg.get("negative_price_rule", "none"))
    bid_basis = str(cfg.get("bid_basis", DEFAULT_BID_BASIS)).strip().lower()
    if bid_basis not in BID_BASES:
        raise ValueError(f"contract_value_bids.bid_basis must be one of {BID_BASES}, got {bid_basis!r}")
    floor = resolve_contract_value_floor(cfg.get("contract_value_floor"))
    missing_policy = str(cfg.get("missing_strike_policy", "error")).lower()
    if missing_policy not in {"error", "retain_static"}:
        raise ValueError(
            "contract_value_bids.missing_strike_policy must be error or retain_static"
        )

    snapshots = pd.DatetimeIndex(network.snapshots)
    if len(snapshots) == 0:
        raise ValueError("Contract-value balancing bids need a non-empty snapshot index")
    hourly_reference = resolve_reference_price(network, reference_price)

    names = network.generators.index
    offer_tv = pd.DataFrame(np.tile(gen_offer.reindex(names).to_numpy(dtype=float),
                                    (len(snapshots), 1)),
                            index=snapshots, columns=names)
    bid_tv = pd.DataFrame(np.tile(gen_bid.reindex(names).to_numpy(dtype=float),
                                  (len(snapshots), 1)),
                          index=snapshots, columns=names)

    support = _support_labels(network)
    carrier = network.generators["carrier"].astype("string").fillna("").str.strip()
    carrier_lower = carrier.str.lower()
    p_nom = pd.to_numeric(network.generators.get("p_nom", 0.0), errors="coerce").fillna(0.0)
    rounds = (network.generators["cfd_round"].astype("string").fillna("").str.strip()
              if "cfd_round" in network.generators.columns
              else pd.Series("", index=names, dtype="string"))

    # Shared with _apply_renewable_support_pricing: an excluded carrier keeps the
    # explicit carrier bid/offer price rather than a contract value.
    exclude = {str(c).strip().lower()
               for c in (support_cfg.get("exclude_carriers", ["nuclear", "CCS Biomass", "biomass"]) or [])}
    excluded_mask = carrier_lower.isin(exclude)

    baseload = {str(c).strip().lower()
                for c in ((cfg.get("reference_price_basis", {}) or {}).get(
                    "baseload_carriers", []) or [])}
    baseload_reference = (demand_weighted_reference(hourly_reference, network)
                          if baseload else None)

    strike_table = load_scenario_strike_table(scenario_config)
    audit_rows = []
    unresolved = []

    cfd_mask = support.eq("cfd") & ~excluded_mask
    # Decision 20: only the units this loop prices carry a tie step, so only they are ranked.
    tie_steps = resolve_tie_break(ownership_block.get("tie_break"), ownership_block, state_shares, logger,
                                  eligible=names[cfd_mask.to_numpy()])
    offer_regulariser = resolve_turn_up_floor(ownership_block.get("tie_break"), state_shares, carrier, logger)
    general_floor = resolve_general_turn_up_floor(cfg, carrier, logger)
    if general_floor is not None:
        # Each unit takes the larger of the general and the P3 ownership floor.
        offer_regulariser = (general_floor if offer_regulariser is None
                             else pd.concat([general_floor, offer_regulariser], axis=1).max(axis=1))
    # One strike lookup per (carrier, round) pair rather than per generator:
    # 6,782 CfD units in the CP30 fleet collapse to about thirty pairs.
    # The contract's own strike technology (floating, permitted reduction, Scottish pot,
    # remote island wind) where the build recorded it: the same key the bill settles on.
    technology = contract_technology(network.generators)
    for (gen_carrier, cfd_round, gen_technology), group in network.generators[cfd_mask].groupby(
            [carrier[cfd_mask], rounds[cfd_mask], technology[cfd_mask]], dropna=False):
        strike, strike_key, is_fallback = strike_lookup(strike_table, gen_carrier, cfd_round, gen_technology)
        columns = group.index
        if strike is None:
            unresolved.append((str(gen_carrier), str(cfd_round), len(columns),
                               float(p_nom.reindex(columns).sum())))
            continue
        is_baseload = str(gen_carrier).strip().lower() in baseload
        reference = baseload_reference if is_baseload else hourly_reference
        basis = ("window_demand_weighted_mean" if is_baseload else "hourly_wholesale")
        # contract_value_floor zero: an hour whose reference already clears above the
        # strike costs the generator nothing to forgo, so it bids zero; none: it bids
        # strike - reference < 0, the difference payment it saves (FTI). Under the
        # strike basis it bids the whole strike instead (see the docstring).
        bid = foregone_rate(strike, reference, bid_basis, floor)
        # Nor does an hour in which its contract pays nothing (the round's negative
        # pricing rule; same function as the CfD bill line and the settlement).
        if negative_price_rule != "none":
            bid = bid.where(negative_price_eligibility(reference.to_numpy(), [cfd_round], negative_price_rule)[:, 0],
                            0.0)
        # Item BM1: a state-owned unit on an at-cost contract forgoes nothing by
        # being turned down -- its capital is sunk and recovered by contract either
        # way -- so what it loses is its avoided short-run cost. Ownership reaches
        # the LP as a capacity-weighted blend because the register owns COHORTS and
        # the LP bids per generator (see ownership_shares.py).
        shares = state_shares.reindex(columns).fillna(0.0).to_numpy() if state_shares is not None             else np.zeros(len(columns))
        # Decision 20: the state-owned part bids its at-cost price plus the tie step.
        steps = (tie_steps.reindex(columns).fillna(0.0).to_numpy() if tie_steps is not None
                 else np.zeros(len(columns)))
        blended = np.outer(bid.to_numpy(), 1.0 - shares) + (public_bid_price + steps) * shares
        bid_tv.loc[:, columns] = blended
        if cfg.get("cfd_offer_price") is not None:
            offer_tv.loc[:, columns] = float(cfg["cfd_offer_price"])
        audit_rows.extend(
            _audit_row(name, gen_carrier, "CfD", cfd_round, p_nom.get(name, 0.0),
                       "cfd_contract_value", strike_key, strike, is_fallback,
                       basis, reference,
                       pd.Series(blended[:, j], index=snapshots),
                       gen_bid.get(name, np.nan),
                       state_share=float(shares[j]), contract_value_bid=bid,
                       tie_break_step=float(steps[j] * shares[j]),
                       negative_price_rule=negative_price_rule, bid_basis=bid_basis,
                       contract_value_floor=floor)
            for j, name in enumerate(columns)
        )

    if unresolved:
        detail = ", ".join(
            f"{c or '(no carrier)'}/{r or '(no round)'}: {n} units / {mw:,.0f} MW"
            for c, r, n, mw in unresolved
        )
        message = (
            "No strike-table entry, not even the modelled AR8 fallback strike, for CfD "
            f"generators: {detail}"
        )
        if missing_policy == "error":
            raise ValueError(message)
        logger.warning("%s; these units retain their static bid", message)

    merchant_price = float(merchant_cfg.get("bid_price", DEFAULT_MERCHANT_BID))
    merchant_carriers = {str(c).strip().lower() for c in
                         (merchant_cfg.get("carriers", DEFAULT_MERCHANT_CARRIERS) or [])}
    merchant_mask = (support.eq("merchant") & carrier_lower.isin(merchant_carriers)
                     & ~excluded_mask)
    merchant_columns = names[merchant_mask]
    if len(merchant_columns):
        bid_tv.loc[:, merchant_columns] = merchant_price
        merchant_bid = pd.Series(merchant_price, index=snapshots, dtype=float)
        audit_rows.extend(
            _audit_row(name, carrier.get(name, ""), "merchant", "", p_nom.get(name, 0.0),
                       "merchant_short_run_cost", "", np.nan, False,
                       "not_applicable", hourly_reference, merchant_bid,
                       gen_bid.get(name, np.nan))
            for name in merchant_columns
        )

    from scripts.market.market_utils import _derived_bid_turn_down_saving
    apply_state_thermal_at_cost(network, ownership_block, state_shares, offer_tv, bid_tv,
                                gen_offer, gen_bid, carrier_lower, logger,
                                turn_down_saving=_derived_bid_turn_down_saving(balancing))
    private_bids = bid_tv.copy() if ownership_block.get("renewables_at_cost", False) else None
    renewable_shares = apply_state_renewables_at_cost(ownership_block, state_shares, bid_tv, support,
                                                      carrier_lower, excluded_mask, public_bid_price, logger)
    if len(renewable_shares):
        # The merchant rows above carry the private bid; restate the owned ones, and add
        # a row for each owned RO unit (no other row prices RO plant), so the audit holds
        # the bid the LP saw and the private bid beside it.
        audit_rows = [row for row in audit_rows if row["generator"] not in renewable_shares.index]
        audit_rows.extend(
            _audit_row(name, carrier.get(name, ""), str(support.get(name)), rounds.get(name, ""),
                       p_nom.get(name, 0.0), f"{support.get(name)}_state_at_cost", "", np.nan, False,
                       "not_applicable", hourly_reference, bid_tv[name], gen_bid.get(name, np.nan),
                       state_share=float(share), contract_value_bid=private_bids[name])
            for name, share in renewable_shares.items()
        )

    audit = (pd.DataFrame(audit_rows, columns=list(AUDIT_COLUMNS))
             if audit_rows else pd.DataFrame(columns=list(AUDIT_COLUMNS)))
    _log_summary(audit, support, carrier_lower, p_nom, excluded_mask, merchant_price,
                 len(snapshots), logger)
    bid_regulariser = resolve_state_first_step(ownership_block, state_shares, merchant_columns, logger)
    return ContractValueBids(offer_tv, bid_tv, audit, hourly_reference, offer_regulariser, bid_regulariser)


def _audit_row(name, gen_carrier, support_type, cfd_round, capacity, pricing,
               strike_key, strike, is_fallback, basis, reference, bid, previous_bid,
               *, state_share=0.0, contract_value_bid=None, tie_break_step=0.0, negative_price_rule="none",
               bid_basis="", contract_value_floor=""):
    return {
        "generator": name,
        "carrier": str(gen_carrier),
        "support_type": support_type,
        "cfd_round": str(cfd_round or ""),
        "p_nom_mw": float(capacity),
        "pricing": pricing,
        "strike_key": str(strike_key or ""),
        "strike_price_gbp_per_mwh": float(strike) if strike is not None and np.isfinite(
            float(strike)) else np.nan,
        "is_fallback_strike": bool(is_fallback),
        "reference_price_basis": basis,
        "mean_reference_price_gbp_per_mwh": float(reference.mean()),
        "mean_bid_gbp_per_mwh": float(bid.mean()),
        "min_bid_gbp_per_mwh": float(bid.min()),
        "max_bid_gbp_per_mwh": float(bid.max()),
        "hours_bid_at_zero": int((bid <= 0.0).sum()),
        "previous_static_bid_gbp_per_mwh": float(previous_bid) if pd.notna(previous_bid)
        else np.nan,
        # Item BM1: what the state owns, and what the bid would have been without
        # it, so the transfer can be separated from the saving after the solve.
        "state_share": float(state_share),
        "contract_value_mean_bid_gbp_per_mwh": (
            float(pd.Series(contract_value_bid).mean()) if contract_value_bid is not None
            else float(pd.Series(bid).mean())),
        # Decision 20: the share-weighted tie step inside this unit's bid, so a money
        # total can be stated net of it (the steps order the tie, they cost nothing).
        "tie_break_step_gbp_per_mwh": float(tie_break_step),
        # The rule the bid applied; post-solve compensation reads it from here.
        "negative_price_rule": str(negative_price_rule),
        # And the basis (contract_value or strike), likewise; empty on merchant rows.
        "bid_basis": str(bid_basis),
        # And the contract-value floor (zero or none) the basis applied; empty on merchant rows.
        "contract_value_floor": str(contract_value_floor),
    }


def _log_summary(audit, support, carrier_lower, p_nom, excluded_mask, merchant_price,
                 n_snapshots, logger) -> None:
    if audit.empty:
        logger.warning(
            "Contract-value balancing bids are enabled but no CfD or merchant renewable "
            "generator was re-priced; the bid stack is unchanged"
        )
        return
    by_pricing = audit.groupby("pricing").agg(
        units=("generator", "size"), capacity_mw=("p_nom_mw", "sum"),
        mean_bid=("mean_bid_gbp_per_mwh", "mean"))
    detail = ", ".join(
        f"{label}: {int(row.units)} units / {row.capacity_mw:,.0f} MW, "
        f"mean bid {row.mean_bid:.2f} GBP/MWh"
        for label, row in by_pricing.iterrows()
    )
    logger.info(
        "Contract-value BM bids over %d snapshots (%s); merchant renewables at "
        "%.2f GBP/MWh", n_snapshots, detail, merchant_price
    )
    cfd = audit[audit.pricing.eq("cfd_contract_value")]
    if not cfd.empty:
        by_strike = cfd.groupby(["strike_key"]).agg(
            units=("generator", "size"), capacity_mw=("p_nom_mw", "sum"),
            strike=("strike_price_gbp_per_mwh", "first"),
            mean_bid=("mean_bid_gbp_per_mwh", "first"),
        ).sort_values("strike", ascending=False)
        for key, row in by_strike.iterrows():
            logger.info(
                "  %-34s strike %7.2f -> mean bid %7.2f GBP/MWh (%4d units, %8.0f MW)",
                key, row.strike, row.mean_bid, int(row.units), row.capacity_mw,
            )
        fallback_mw = float(cfd.loc[cfd.is_fallback_strike, "p_nom_mw"].sum())
        total_mw = float(cfd.p_nom_mw.sum())
        logger.info(
            "  fallback (modelled AR8) strike carried %.0f MW of %.0f MW CfD capacity (%.1f%%)",
            fallback_mw, total_mw, 100.0 * fallback_mw / total_mw if total_mw else 0.0,
        )
    excluded_capacity = float(p_nom[excluded_mask & support.isin(["cfd", "roc"])].sum())
    if excluded_capacity > 0:
        logger.info(
            "  %s excluded from contract-value pricing (%.0f MW); the carrier "
            "bid/offer override is retained",
            ", ".join(sorted(set(carrier_lower[excluded_mask &
                                               support.isin(["cfd", "roc"])]))),
            excluded_capacity,
        )
