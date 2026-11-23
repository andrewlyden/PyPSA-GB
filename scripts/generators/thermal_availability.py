"""Reproducible non-UC thermal planned and forced availability profiles."""

import logging

import numpy as np
import pandas as pd


def _carrier_config(config, carrier):
    carriers = config.get("carriers", {})
    if carrier in carriers:
        return carriers[carrier]
    carrier_lower = str(carrier).lower()
    for name, value in carriers.items():
        if str(name).lower() == carrier_lower:
            return value
    return None


def _outage_profile(snapshots, cfg, seed):
    """Build a 0/1 availability profile with contiguous planned/forced events."""
    n_hours = len(snapshots)
    profile = np.ones(n_hours, dtype=float)
    if not n_hours:
        return profile

    rng = np.random.default_rng(seed)
    planned_duration = max(1, int(cfg.get("planned_outage_duration_hours", 168)))
    months = tuple(cfg.get("planned_outage_months", [4, 5, 6, 7, 8, 9]))
    candidates = np.flatnonzero(snapshots.month.isin(months))
    candidates = candidates[candidates <= n_hours - planned_duration]
    for _ in range(max(0, int(cfg.get("planned_outages_per_year", 0)))):
        if not len(candidates):
            break
        start = int(rng.choice(candidates))
        profile[start:start + planned_duration] = 0.0

    forced_rate = max(0.0, float(cfg.get("forced_outage_rate", 0.0)))
    mean_duration = max(1.0, float(cfg.get("forced_outage_mean_duration_hours", 24)))
    expected_events = n_hours * forced_rate / mean_duration
    for _ in range(int(rng.poisson(expected_events))):
        start = int(rng.integers(0, n_hours))
        duration = max(1, int(rng.exponential(mean_duration)))
        profile[start:min(start + duration, n_hours)] = 0.0
    return profile


def apply_thermal_outage_availability(network, config, logger: logging.Logger):
    """Apply planned/forced thermal outages to ``generators_t.p_max_pu``.

    This is an availability derating only: it adds no commitment variables,
    start-up logic, or inter-temporal dispatch constraints.
    """
    if not config.get("enabled", False):
        return pd.DataFrame(columns=["carrier", "generators", "outage_hours"])

    snapshots = network.snapshots
    seed = int(config.get("seed", 42))
    records = []
    eligible = network.generators.index[
        network.generators["carrier"].map(
            lambda carrier: _carrier_config(config, carrier) is not None
        )
    ]
    # A unit whose availability is observed (station-month rows, e.g. from ELEXON notifications, or a historical
    # dispatch anchor) already carries its outages: random ones on top double-count them and can push the cap below a
    # held minimum, which made a 2020 window infeasible (18 Nov 2026). Its seed offset is kept so every other unit's
    # outage draw is unchanged.
    observed = set()
    for column in ("observed_availability", "historical_dispatch_anchor"):
        if column in network.generators.columns:
            flags = network.generators[column]
            observed |= set(network.generators.index[flags.notna() & flags.astype(str).ne("") & flags.ne(False)])
    for offset, gen_name in enumerate(sorted(eligible)):
        if gen_name in observed:
            continue
        carrier = network.generators.at[gen_name, "carrier"]
        carrier_cfg = _carrier_config(config, carrier)
        outage = _outage_profile(snapshots, carrier_cfg, seed + offset)
        if gen_name in network.generators_t.p_max_pu.columns:
            base = network.generators_t.p_max_pu[gen_name].reindex(snapshots).fillna(1.0)
        else:
            base = pd.Series(
                network.generators.at[gen_name, "p_max_pu"], index=snapshots
            ).fillna(1.0)
        network.generators_t.p_max_pu[gen_name] = base * outage
        records.append(
            {
                "carrier": carrier,
                "generator": gen_name,
                "outage_hours": int((outage == 0.0).sum()),
            }
        )

    summary = pd.DataFrame(records)
    if summary.empty:
        logger.info("Thermal outage availability enabled but no configured carriers were found")
    else:
        grouped = summary.groupby("carrier").agg(
            generators=("generator", "count"), outage_hours=("outage_hours", "sum")
        )
        logger.info("Applied planned/forced availability to %d thermal generators", len(summary))
        for carrier, row in grouped.iterrows():
            logger.info(
                "  %s: %d generators, %d aggregate outage-hours",
                carrier, row["generators"], row["outage_hours"],
            )
    return summary
