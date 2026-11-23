"""The balancing stage's pricing basis: bids and offers, or every unit at its marginal cost.

``market.balancing.pricing_basis``:

``bid_offer`` (default)
    Units are priced from ``default_offer_markup``, ``default_bid_discount`` and
    ``carrier_overrides`` as configured.
``at_cost``
    The operator instructs its own schedule at cost (central dispatch,
    config/defaults/dispatch/central.yaml): the default markup and discount are zero and every
    ``carrier_overrides`` entry is read as markup mode at zero, its ``offer_price`` and
    ``bid_price`` dropped. The entries' other keys are kept, and markup mode reads none of them.

Every reader of the markups goes through :func:`priced_balancing`: the bid/offer prices
(``market_utils.calculate_bid_offer_prices``) and the counter-trade's foreign plant multipliers
(``interconnector_counter_trade``), which borrow the CCGT markups.
"""

from __future__ import annotations

from typing import Any, Dict

BID_OFFER, AT_COST = "bid_offer", "at_cost"
PRICING_BASES = (BID_OFFER, AT_COST)
#: Must match config/defaults/base/market/balancing.yaml.
DEFAULT_PRICING_BASIS = BID_OFFER


def pricing_basis(balancing: Dict[str, Any] | None) -> str:
    """The configured basis, checked."""
    basis = str((balancing or {}).get("pricing_basis") or DEFAULT_PRICING_BASIS).lower()
    if basis not in PRICING_BASES:
        raise ValueError(f"market.balancing.pricing_basis must be one of {PRICING_BASES}, got {basis!r}")
    return basis


def priced_balancing(balancing: Dict[str, Any] | None) -> Dict[str, Any]:
    """``market.balancing`` as the bid/offer pricing reads it: unchanged under ``bid_offer``,
    a copy with every markup and discount at zero under ``at_cost``."""
    balancing = balancing or {}
    if pricing_basis(balancing) == BID_OFFER:
        return balancing
    out = dict(balancing)
    out["default_offer_markup"] = 0.0
    out["default_bid_discount"] = 0.0
    overrides = {}
    for carrier, spec in (balancing.get("carrier_overrides") or {}).items():
        if isinstance(spec, dict):
            spec = {k: v for k, v in spec.items() if k not in ("offer_price", "bid_price")}
            spec.update(mode="markup", offer_markup=0.0, bid_discount=0.0)
        overrides[carrier] = spec
    out["carrier_overrides"] = overrides
    return out
