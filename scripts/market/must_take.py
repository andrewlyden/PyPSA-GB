"""Must-take generation: units that are never curtailed or redispatched (16 Oct 2026).

``market.must_take_generation`` names generators, by the ``data_source`` the build
stamped on them, that run at their available output in both market stages:

* **Stage 1** (``apply_must_take_stage1``): ``p_min_pu = p_max_pu`` hour by hour, so the
  wholesale cannot curtail them whatever their marginal cost. The contract-aware
  curtailment order (``curtailment_order.py``) still re-prices them, but a fixed unit
  sets no price.
* **Balancing stage** (``apply_must_take_balancing``): they leave the balancing
  mechanism and are held at their Stage 1 position (the solve's ``fixed`` set), so they
  submit no bids or offers and are never turned down.

The CP30 use (user decision, 16 Oct 2026, NESO review item S1-08): the 12.96 GW of FES
small-scale solar the build keeps at its GSP nodes (``data_source``
``CP30_small_scale_solar_gsp``, domestic and small commercial G98/G83 PV, labelled
merchant) is behind-the-meter in practice: no market curtails it. As merchant plant at
+0.5 it was the first renewable the wholesale curtailed (3.84 TWh, 29% of its energy).
Capacity, profiles and support labels are untouched, so the ES5 totals and the energy
accounting (available, dispatched, curtailed) are unchanged; only the curtailment
order moves.

Code default: no unit (``data_sources: []``), the behaviour before this module. The
key sits at ``market`` level because both stages read it, and the ``wholesale_from``
check (Snakefile) makes a balancing family carry the same value as its Stage 1.
Defaults must match config/defaults/ ``market.must_take_generation``.
"""

import logging

import pandas as pd

#: Code default of ``market.must_take_generation.data_sources`` (must match config/defaults/).
DEFAULT_DATA_SOURCES: tuple = ()


def resolve_must_take_config(market_config: dict | None) -> dict:
    """``market.must_take_generation`` validated: ``{"data_sources": [...]}``."""
    block = dict((market_config or {}).get("must_take_generation") or {})
    unknown = sorted(set(block) - {"data_sources"})
    if unknown:
        raise ValueError(f"market.must_take_generation has unknown keys {unknown}; supported: ['data_sources']")
    sources = block.get("data_sources", list(DEFAULT_DATA_SOURCES))
    if sources is None:
        sources = []
    if isinstance(sources, str) or not all(isinstance(s, str) and s.strip() for s in sources):
        raise ValueError("market.must_take_generation.data_sources must be a list of data_source labels")
    return {"data_sources": [s.strip() for s in sources]}


def must_take_generators(network, market_config: dict | None) -> pd.Index:
    """The generators ``market.must_take_generation`` holds (an empty Index when it names none).

    A configured label that matches no generator is refused: a typo would silently
    restore the old behaviour.
    """
    sources = resolve_must_take_config(market_config)["data_sources"]
    if not sources:
        return pd.Index([])
    gens = network.generators
    if "data_source" not in gens.columns:
        raise ValueError("market.must_take_generation names data_source labels, but the network's generators "
                         "carry no data_source column")
    labels = gens["data_source"].fillna("").astype(str).str.strip()
    missing = [s for s in sources if not labels.eq(s).any()]
    if missing:
        raise ValueError(f"market.must_take_generation: no generator has data_source {missing}")
    return gens.index[labels.isin(sources)]


def apply_must_take_stage1(network, market_config: dict | None, logger: logging.Logger) -> pd.Index:
    """Stage 1: hold the must-take units at their available output (``p_min_pu = p_max_pu``).

    Call after the solve period and any forecast error are applied, so the floor is the
    availability Stage 1 actually schedules on. Returns the units held.
    """
    names = must_take_generators(network, market_config)
    if not len(names):
        return names
    snapshots = network.snapshots
    p_max = network.get_switchable_as_dense("Generator", "p_max_pu", snapshots)[names]
    floor = p_max.clip(lower=0.0)
    p_min_t = network.generators_t.p_min_pu
    network.generators_t.p_min_pu = pd.concat([p_min_t.drop(columns=list(names), errors="ignore"), floor], axis=1)
    p_nom = pd.to_numeric(network.generators.loc[names, "p_nom"], errors="coerce").fillna(0.0)
    weights = network.snapshot_weightings["generators"].reindex(snapshots).fillna(1.0)
    energy = float(floor.mul(p_nom, axis=1).mul(weights, axis=0).to_numpy().sum())
    logger.info("Must-take generation (market.must_take_generation): %d units, %.0f MW, held at availability in "
                "Stage 1 (%.1f GWh over the period, never curtailed)", len(names), float(p_nom.sum()), energy / 1e3)
    return names


def apply_must_take_balancing(network, market_config: dict | None, logger: logging.Logger) -> pd.Index:
    """Balancing stage: take the must-take units out of the BM and hold them at their Stage 1 position.

    Call after ``_apply_participation_policy`` (``calculate_bid_offer_prices``) has
    attached the participation sets. Returns the units moved.
    """
    names = must_take_generators(network, market_config)
    if not len(names):
        return names
    keep = set(names)
    for attr in ("_bm_eligible_generators", "_bm_participating_generators"):
        current = getattr(network, attr, None)
        if current is None:
            current = network.generators.index
        setattr(network, attr, pd.Index([g for g in current if g not in keep]))
    fixed = getattr(network, "_bm_fixed_generators", None)
    fixed = pd.Index([]) if fixed is None else pd.Index(fixed)
    network._bm_fixed_generators = fixed.append(pd.Index([g for g in names if g not in set(fixed)]))
    network._bm_must_take_generators = names
    p_nom = float(pd.to_numeric(network.generators.loc[names, "p_nom"], errors="coerce").fillna(0.0).sum())
    logger.info("Must-take generation: %d units (%.0f MW) out of the balancing mechanism, held at their Stage 1 "
                "position (no bids or offers)", len(names), p_nom)
    return names


def hold_at_stage1(network, wholesale_gen: pd.DataFrame, names, logger: logging.Logger) -> pd.DataFrame:
    """``wholesale_gen`` with the must-take units' Stage 1 positions clipped to the BM's availability.

    The balancing stage fixes them at these positions; Stage 1 wrote them at the same
    availability, so the clip only removes solver noise above the bound (and, under a
    Stage 1 forecast error, the forecast's excess over outturn). Returns a copy.
    """
    cols = [g for g in names if g in wholesale_gen.columns]
    if not cols:
        return wholesale_gen
    out = wholesale_gen.copy()
    snapshots = out.index
    p_nom = pd.to_numeric(network.generators.loc[cols, "p_nom"], errors="coerce").fillna(0.0)
    avail = network.get_switchable_as_dense("Generator", "p_max_pu", network.snapshots)[cols]
    cap = avail.reindex(snapshots).ffill().bfill().mul(p_nom, axis=1)
    before = out[cols]
    out[cols] = before.clip(lower=0.0).where(before.le(cap), cap)
    moved = float((before - out[cols]).abs().to_numpy().sum())
    if moved > 1.0:
        logger.info("Must-take generation: Stage 1 positions clipped to balancing availability by %.1f MW-h in all",
                    moved)
    return out
