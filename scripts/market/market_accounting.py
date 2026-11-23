"""Definitions and reconciliation for two-stage market output accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEASIBILITY_SLACK_CARRIERS = frozenset(
    {"load_shedding", "load shedding", "voll", "unserved_energy"}
)
#: A unit left out of the balancing mechanism (``participation.min_p_nom_mw``,
#: excluded carriers) carries the non-participant penalty bid (6,000 GBP/MWh). No
#: real bid comes near this (ELEXON bids are capped at 500), so an implied bid at or
#: above it marks turn-down the unit could not avoid, e.g. forecast error leaving
#: sub-1 MW plant short of its schedule under central dispatch: ~92 GWh a year priced
#: at ~549 m, 82% of that case's reported redispatch cost (audit of 26 Sep 2026).
NON_PARTICIPANT_BID_THRESHOLD_GBP_PER_MWH = 1000.0

METRIC_DEFINITIONS = {
    "wholesale_price": (
        "Copperplate marginal energy price from the wholesale LP. It is not a "
        "retail tariff and excludes balancing, network and policy charges."
    ),
    "modelled_redispatch_cost": (
        "Accepted offer cost plus accepted bid cost relative to fixed wholesale "
        "positions in the balancing LP. It is not total NESO balancing expenditure."
    ),
    "ordinary_redispatch_cost": (
        "Modelled redispatch cost excluding feasibility-slack resources such as "
        "load shedding."
    ),
    "feasibility_slack": (
        "Energy from a VOLL-backed feasibility resource. It represents unserved "
        "load and is reported separately, not as ordinary balancing cost."
    ),
    "non_participant_turn_down": (
        "Turn-down of units left out of the balancing mechanism, which the LP can "
        "only price at the non-participant penalty bid. It is forced (the unit "
        "cannot meet its schedule, e.g. under forecast error), not an accepted "
        "bid, so it is reported separately and left out of ordinary redispatch cost."
    ),
    "storage_burn": (
        "Energy storage destroyed by charging and discharging in the same hour in "
        "the balancing solution: sum over unit-hours of min(p_dispatch, p_store) x "
        "(1/efficiency_dispatch - efficiency_store), counting overlaps above "
        "storage_burn_tolerance_pu x p_nom. The pair is net zero at the bus, so the "
        "system disposed of surplus energy through storage losses instead of "
        "curtailing it: it is reported as curtailed energy (decision D6), and no "
        "dispatch is changed."
    ),
    "balancing_nodal_dual": (
        "A balancing-network power-balance optimisation dual at demand buses. "
        "It is a diagnostic marginal value, not an imbalance or cash-out price."
    ),
    "gross_redispatch_volume": (
        "The sum of upward and downward redispatch energy. It is reported as a "
        "share of served-model demand, without netting opposing actions. "
        "gross_redispatch_MWh (alias gross_redispatch_including_foreign_side_MWh) "
        "counts every row of the redispatch summary, including the foreign-market "
        "balance generators behind the interconnectors, and divides by demand_MWh."
    ),
    "gb_gross_redispatch_volume": (
        "Gross redispatch by GB assets only: the foreign-market balance generators "
        "(foreign_market_role, External buses, carriers EU_market/EU_import) move "
        "because the cables' far ends re-balance, not because NESO accepted a bid "
        "or offer. Divided by gb_electricity_demand_MWh."
    ),
    "demand_MWh": (
        "Every modelled Load's p_set (GB electrical demand plus the hydrogen offtake on "
        "the hydrogen bus, which is MWh of hydrogen, not of electricity) plus what consumer "
        "flexibility draws (consumer_flexibility_MWh, F5 27 Oct 2026: its energy is carved "
        "out of the Loads). The bill projector checks its own Load total against "
        "demand_MWh - consumer_flexibility_MWh."
    ),
    "consumer_flexibility_MWh": (
        "Electricity the consumer-flexibility storage units draw in the balancing solution "
        "(demand_carriers.DEMAND_FLEX_STORAGE_CARRIERS): p_store - p_dispatch. Inside demand_MWh "
        "and gb_electricity_demand_MWh."
    ),
    "gb_electricity_demand": (
        "Electricity drawn in GB: Loads on GB electrical buses plus the electricity "
        "the conversion links (electrolysers) draw from them in the balancing "
        "solution. The hydrogen offtake is excluded; it is met from that "
        "electricity (at the electrolysis efficiency) and from hydrogen storage."
    ),
    "neso_comparable_redispatch_cost": (
        "The measure comparable with NESO's thermal constraint cost (CP30 main report "
        "Table 1, GBP 3.57 bn for FFR): ordinary redispatch cost (accepted offers and "
        "bids of GB units, excluding feasibility slack at VoLL and non-participant "
        "turn-down at the penalty bid) plus interconnector re-trading. The two "
        "excluded parts are reported on their own lines (feasibility_slack_cost_GBP, "
        "non_participant_turn_down_cost_at_penalty_GBP). None when re-trading could "
        "not be valued."
    ),
    "churn_sink_held": (
        "Energy the Stage 1 churn sink absorbed, held at its Stage 1 output in the "
        "balancing stage (market.wholesale.churn_sink). It is not a balancing action "
        "and is in no redispatch volume or cost (item F22)."
    ),
    "interconnector_retrading_cost": (
        "Bid/ask valuation of the change in interconnector schedules between "
        "the wholesale (day-ahead) and balancing stages: for each cable and "
        "hour the balancing LP pays the foreign ask on imports and earns the "
        "efficiency-adjusted foreign bid on exports, so the cost is "
        "sum(hours * max(ask*p, eff*bid*p)) on the balancing schedule minus "
        "the same valuation of the day-ahead schedule, plus the counter-trade "
        "premium paid per MWh of |balancing - day-ahead| where the balancing "
        "network carries one (market_counter_trade_premium; FTI 2023 GBP 100/MWh "
        "from 2030, like NESO counter-trades recovered through BSUoS). It sits in "
        "the balancing objective but is not part of accepted offer or bid costs, "
        "which only cover GB generators and storage. A positive value is a net "
        "payment to foreign markets and traders for re-trading; a negative value "
        "is a net receipt."
    ),
}


def _snapshot_hours(network) -> pd.Series:
    """Return energy weights for a network, falling back to timestamp spacing."""
    snapshots = pd.Index(network.snapshots)
    weightings = getattr(network, "snapshot_weightings", None)
    if weightings is not None:
        for column in ("generators", "objective"):
            if column in weightings.columns:
                hours = pd.to_numeric(
                    weightings[column].reindex(snapshots), errors="coerce"
                )
                if hours.notna().all() and (hours >= 0).all():
                    return hours.astype(float)

    hours = pd.Series(1.0, index=snapshots, dtype=float)
    if len(snapshots) > 1:
        spacing = snapshots.to_series().diff().dt.total_seconds().median()
        if pd.notna(spacing) and spacing > 0:
            hours[:] = float(spacing) / 3600.0
    return hours


def consumer_flex_draw(carriers: pd.Series, p_store: pd.DataFrame | None = None,
                       p_dispatch: pd.DataFrame | None = None, p: pd.DataFrame | None = None,
                       inflow: pd.DataFrame | None = None,
                       efficiency_store: pd.Series | None = None) -> pd.DataFrame:
    """Electricity (MW, per unit) that consumer flexibility draws: demand, not storage (F5, 27 Oct 2026).

    Units are the storage units whose carrier is in ``demand_carriers.DEMAND_FLEX_STORAGE_CARRIERS``
    (EV smart charging, V2G, heat, storage heaters, DSR, DACCS). A solved network: what they buy,
    net of what V2G and demand response give back, ``p_store - p_dispatch`` (else ``-p``). A built
    network (no dispatch yet): the drain's electricity, ``-inflow / efficiency_store``. Columns are
    the units found; an empty frame when there are none.
    """
    from scripts.demand.demand_carriers import DEMAND_FLEX_STORAGE_CARRIERS

    carriers = pd.Series(carriers).astype(str)
    units = list(carriers.index[carriers.isin(DEMAND_FLEX_STORAGE_CARRIERS)])

    def _has(frame):
        return frame is not None and not frame.empty

    if not units:
        return pd.DataFrame()
    # A solved network: a unit missing from a saved series is zero there (netCDF keeps the
    # columns that moved), never its drain.
    if _has(p_store) or _has(p_dispatch):
        index = (p_store if _has(p_store) else p_dispatch).index
        store = p_store.reindex(columns=units) if _has(p_store) else pd.DataFrame(0.0, index=index, columns=units)
        dispatch = (p_dispatch.reindex(index=index, columns=units) if _has(p_dispatch)
                    else pd.DataFrame(0.0, index=index, columns=units))
        return store.fillna(0.0) - dispatch.fillna(0.0)
    if _has(p):
        return -p.reindex(columns=units).fillna(0.0)
    if inflow is not None and not inflow.empty:
        cols = [u for u in units if u in inflow.columns]
        eta = (pd.Series(1.0, index=cols) if efficiency_store is None
               else pd.to_numeric(pd.Series(efficiency_store).reindex(cols), errors="coerce").fillna(1.0))
        return (-inflow[cols].fillna(0.0).clip(upper=0.0)).div(eta.clip(lower=1e-9), axis=1)
    return pd.DataFrame()


def _network_flex_draw(network, snapshots) -> pd.DataFrame:
    """:func:`consumer_flex_draw` on an in-memory network, on ``snapshots``."""
    su = getattr(network, "storage_units", pd.DataFrame())
    if len(su) == 0 or "carrier" not in su.columns:
        return pd.DataFrame(index=snapshots)
    su_t = network.storage_units_t
    get = lambda name: getattr(su_t, name, None)  # noqa: E731
    draw = consumer_flex_draw(su["carrier"], get("p_store"), get("p_dispatch"), get("p"), get("inflow"),
                              su.get("efficiency_store"))
    return draw.reindex(index=snapshots).fillna(0.0)


def _demand_energy_mwh(network, hours: pd.Series) -> float:
    """Modelled gross demand with the same energy weights as dispatch: every Load plus consumer flexibility.

    Consumer flexibility (storage units on the demand carriers) is demand (F5, 27 Oct 2026): the
    2050 chain's units and CP30 2030's carve their energy out of the Loads, so leaving them out
    understated the demand by what they draw.
    """
    snapshots = pd.Index(network.snapshots)
    try:
        demand = network.get_switchable_as_dense("Load", "p_set", snapshots)
    except (AttributeError, KeyError):
        demand = getattr(network.loads_t, "p_set", pd.DataFrame(index=snapshots))
        demand = demand.reindex(index=snapshots).fillna(0.0)
    flex = _network_flex_draw(network, snapshots)
    return float(demand.sum(axis=1).mul(hours).sum()) + float(flex.sum(axis=1).mul(hours).sum())


#: Carriers of the generators that stand for the foreign markets at each
#: cable's far end; a fallback when the network carries no foreign_market_role.
FOREIGN_MARKET_CARRIERS = frozenset({"EU_market", "EU_import"})
FOREIGN_MARKET_ROLES = frozenset({"import_supply", "export_demand", "market_balance"})
ELECTRICAL_BUS_CARRIERS = frozenset({"", "ac", "dc"})


def _is_gb_electrical_bus(buses: pd.DataFrame) -> pd.Series:
    """GB buses that carry electricity (not hydrogen, not a foreign market node)."""
    carrier = buses.get("carrier", pd.Series("", index=buses.index)).fillna("").astype(str).str.strip().str.lower()
    electrical = carrier.isin(ELECTRICAL_BUS_CARRIERS)
    external = pd.Series(buses.index.astype(str).str.contains("External"), index=buses.index)
    if "country" in buses.columns:
        country = buses["country"].fillna("").astype(str).str.strip()
        # An empty country is GB unless the bus is marked External (the CP30
        # buses all carry "GB"; only the hydrogen bus is blank, and it is not electrical).
        in_gb = country.isin({"GB", ""}) & ~external
    else:
        in_gb = ~external
    return electrical & in_gb


def _gb_electricity_demand_mwh(network, hours: pd.Series) -> float:
    """Loads on GB electrical buses plus electricity drawn by conversion links from them."""
    snapshots = pd.Index(network.snapshots)
    buses = network.buses
    gb_electrical = set(buses.index[_is_gb_electrical_bus(buses)])
    loads = network.loads
    total = 0.0
    if len(loads):
        try:
            demand = network.get_switchable_as_dense("Load", "p_set", snapshots)
        except (AttributeError, KeyError):
            demand = getattr(network.loads_t, "p_set", pd.DataFrame(index=snapshots))
            demand = demand.reindex(index=snapshots).fillna(0.0)
        columns = [c for c in demand.columns if c in loads.index and loads.at[c, "bus"] in gb_electrical]
        total += float(demand[columns].sum(axis=1).mul(hours).sum())
    flex = _network_flex_draw(network, snapshots)
    if not flex.empty and len(flex.columns):
        on_gb = [c for c in flex.columns if network.storage_units.at[c, "bus"] in gb_electrical]
        total += float(flex[on_gb].sum(axis=1).mul(hours).sum())
    links = network.links
    p0 = getattr(network.links_t, "p0", pd.DataFrame())
    if len(links) and not p0.empty:
        carrier = buses.get("carrier", pd.Series("", index=buses.index)).fillna("").astype(str).str.strip().str.lower()
        electrical = set(buses.index[carrier.isin(ELECTRICAL_BUS_CARRIERS)])
        # Electricity in at a GB bus, something else (hydrogen) out: electrolysers.
        conversion = links.index[links["bus0"].isin(gb_electrical) & links["bus1"].isin(set(buses.index) - electrical)]
        columns = [c for c in conversion if c in p0.columns]
        if columns:
            drawn = p0[columns].reindex(index=snapshots).fillna(0.0).clip(lower=0.0)
            total += float(drawn.sum(axis=1).mul(hours).sum())
    return total


def _foreign_side_mask(frame: pd.DataFrame, network) -> pd.Series:
    """Redispatch rows that are foreign-market balance generators, not GB assets."""
    if frame.empty:
        return pd.Series(False, index=frame.index)
    carrier = frame.get("carrier", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    mask = carrier.isin(FOREIGN_MARKET_CARRIERS)
    if network is not None and "component" in frame.columns and len(network.generators):
        generators = network.generators
        foreign = pd.Series(False, index=generators.index)
        if "foreign_market_role" in generators.columns:
            role = generators["foreign_market_role"].fillna("").astype(str).str.strip().str.lower()
            foreign |= role.isin(FOREIGN_MARKET_ROLES)
        bus_ok = _is_gb_electrical_bus(network.buses)
        on_foreign_bus = ~generators["bus"].map(bus_ok).fillna(False).astype(bool)
        foreign |= on_foreign_bus & generators["bus"].astype(str).str.contains("External")
        names = set(generators.index[foreign])
        is_generator = frame.get("type", pd.Series("generator", index=frame.index)).astype(str).eq("generator")
        mask |= frame["component"].isin(names) & is_generator
    return mask


def _sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _slack_mask(frame: pd.DataFrame) -> pd.Series:
    if "carrier" not in frame.columns:
        return pd.Series(False, index=frame.index)
    carrier = (
        frame["carrier"].fillna("").astype(str).str.strip().str.lower()
        .str.replace("-", "_", regex=False)
    )
    return carrier.isin(FEASIBILITY_SLACK_CARRIERS)


def _non_participant_turn_down(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows whose turn-down was priced at the non-participant penalty (slack carriers excluded)."""
    if frame.empty or not {"decrease_MWh", "bid_cost"}.issubset(frame.columns):
        return frame.iloc[0:0]
    decrease = pd.to_numeric(frame["decrease_MWh"], errors="coerce").fillna(0.0)
    bid_cost = pd.to_numeric(frame["bid_cost"], errors="coerce").fillna(0.0)
    implied = bid_cost.div(decrease.where(decrease > 1e-6))
    mask = implied.ge(NON_PARTICIPANT_BID_THRESHOLD_GBP_PER_MWH).fillna(False) & ~_slack_mask(frame)
    return frame.loc[mask]


#: A unit-hour counts as simultaneous charge and discharge when BOTH gross legs
#: exceed this share of the unit's p_nom. Below it the overlap is barrier noise
#: (a no-crossover interior solution leaves small flows on idle legs). On the
#: Dec / Jun 2030 test weeks 1% of p_nom keeps 98% / 94% of the energy burnt at
#: 1e-4 pu (459 of 481 MWh, 803 of 855 MWh) in 1,246 / 345 unit-hours instead
#: of 40,317 / 23,841; the old fixed 1e-3 MW flagged 45% of all unit-hours.
STORAGE_BURN_TOLERANCE_PU = 1.0e-2


def storage_simultaneous_burn(network, hours: pd.Series | None = None,
                              tol_pu: float = STORAGE_BURN_TOLERANCE_PU) -> dict:
    """Energy storage destroyed by charging and discharging in the same hour (decision D6).

    Per unit-hour the overlapping pair ``min(p_dispatch, p_store)`` is net zero at
    the bus but loses ``min(d, s) x (1/eta_dispatch - eta_store)`` of stored
    energy: surplus the system disposed of through storage losses instead of
    curtailing it, so it is reported as curtailed energy. Read from the gross
    legs; net ``p`` cannot show it. Only overlaps above ``tol_pu x p_nom`` count.
    Returns MWh (total and by carrier), the flagged unit-hours and the
    sub-tolerance noise, all zero when the network has no gross split.
    """
    empty = {"burn_MWh": 0.0, "burn_by_carrier_MWh": {}, "unit_hours": 0, "total_unit_hours": 0,
             "noise_MWh": 0.0, "tolerance_pu": float(tol_pu)}
    su = getattr(network, "storage_units", pd.DataFrame())
    su_t = getattr(network, "storage_units_t", None)
    if su_t is None or len(su) == 0:
        return empty
    dispatch = getattr(su_t, "p_dispatch", pd.DataFrame())
    store = getattr(su_t, "p_store", pd.DataFrame())
    if dispatch is None or store is None or dispatch.empty or store.empty:
        return empty
    # Consumer flexibility is demand, not storage (F5): its overlaps are load shifting, not burnt surplus.
    from scripts.demand.demand_carriers import DEMAND_FLEX_STORAGE_CARRIERS

    flex = set(su.index[su["carrier"].astype(str).isin(DEMAND_FLEX_STORAGE_CARRIERS)]) if "carrier" in su.columns \
        else set()
    names = [c for c in dispatch.columns if c in store.columns and c in su.index and c not in flex]
    if not names:
        return empty
    if hours is None:
        hours = _snapshot_hours(network)
    hours = pd.to_numeric(pd.Series(hours), errors="coerce").reindex(dispatch.index).fillna(1.0)

    def _attr(name, default):
        if name in su.columns:
            return pd.to_numeric(su.loc[names, name], errors="coerce").fillna(default)
        return pd.Series(default, index=names, dtype=float)

    eta_s = _attr("efficiency_store", 1.0)
    eta_d = _attr("efficiency_dispatch", 1.0).clip(lower=1e-6)
    loss = (1.0 / eta_d - eta_s).clip(lower=0.0)
    p_nom = _attr("p_nom", 0.0).abs()
    overlap = np.minimum(dispatch[names].fillna(0.0).clip(lower=0.0), store[names].fillna(0.0).clip(lower=0.0))
    flagged = overlap.gt(p_nom * float(tol_pu), axis=1) & overlap.gt(0.0)
    burnt = overlap.mul(loss, axis=1).mul(hours, axis=0)
    counted = burnt.where(flagged, 0.0)
    carrier = su.loc[names, "carrier"] if "carrier" in su.columns else pd.Series("", index=names)
    by_carrier = counted.sum().groupby(carrier).sum()
    return {
        "burn_MWh": float(counted.to_numpy().sum()),
        "burn_by_carrier_MWh": {str(k): float(v) for k, v in by_carrier.items() if v > 0.0},
        "unit_hours": int(flagged.to_numpy().sum()),
        "total_unit_hours": int(flagged.size),
        "noise_MWh": float(burnt.where(~flagged, 0.0).to_numpy().sum()),
        "tolerance_pu": float(tol_pu),
    }


def _physical_feasibility_slack(balancing_network, hours: pd.Series):
    """Return solved load-shedding energy/cost; fall back only when unavailable."""
    generators = balancing_network.generators
    carriers = generators["carrier"].fillna("").astype(str).str.strip().str.lower().str.replace("-", "_", regex=False)
    assets = generators.index[carriers.isin(FEASIBILITY_SLACK_CARRIERS)]
    dispatch = getattr(balancing_network.generators_t, "p", pd.DataFrame())
    if assets.empty or dispatch.empty:
        return None
    dispatch = dispatch.reindex(index=hours.index, columns=assets, fill_value=0.0).fillna(0.0)
    energy = float(dispatch.mul(hours, axis=0).sum().sum())
    cost = float(dispatch.mul(generators.loc[assets, "marginal_cost"].astype(float), axis=1).mul(hours, axis=0).sum().sum())
    return energy, cost

def churn_sink_held_mwh(balancing_network, hours: pd.Series) -> float:
    """Energy (MWh, >= 0) the churn sink units absorbed in the balancing solution (0 without a sink)."""
    generators = balancing_network.generators
    if "carrier" not in generators.columns:
        return 0.0
    sink = generators.index[generators["carrier"].astype(str).eq("churn_sink")]
    dispatch = getattr(balancing_network.generators_t, "p", pd.DataFrame())
    if sink.empty or dispatch.empty:
        return 0.0
    p = dispatch.reindex(index=hours.index, columns=sink, fill_value=0.0).fillna(0.0)
    return float(-(p.clip(upper=0.0).mul(hours, axis=0)).to_numpy().sum())


def create_market_accounting_report(
    balancing_network,
    redispatch_summary: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Return a JSON-safe accounting block and an auditable metric table."""
    hours = _snapshot_hours(balancing_network)
    period_hours = float(hours.sum())
    demand_mwh = _demand_energy_mwh(balancing_network, hours)
    gb_demand_mwh = _gb_electricity_demand_mwh(balancing_network, hours)
    flex_demand_mwh = float(_network_flex_draw(balancing_network, pd.Index(balancing_network.snapshots))
                            .sum(axis=1).mul(hours).sum())

    redispatch = redispatch_summary.copy()
    if redispatch.empty:
        redispatch = pd.DataFrame(
            columns=["carrier", "increase_MWh", "decrease_MWh", "offer_cost", "bid_cost"]
        )
    slack = redispatch.loc[_slack_mask(redispatch)]

    increase_mwh = _sum(redispatch, "increase_MWh")
    decrease_mwh = _sum(redispatch, "decrease_MWh")
    offer_cost = _sum(redispatch, "offer_cost")
    bid_cost = _sum(redispatch, "bid_cost")
    redispatch_cost = offer_cost + bid_cost

    physical_slack = _physical_feasibility_slack(balancing_network, hours)
    if physical_slack is None:
        slack_energy_mwh = _sum(slack, "increase_MWh")
        slack_cost = _sum(slack, "offer_cost") + _sum(slack, "bid_cost")
    else:
        slack_energy_mwh, slack_cost = physical_slack
    burn = storage_simultaneous_burn(balancing_network, hours)
    non_participant = _non_participant_turn_down(redispatch)
    non_participant_mwh = _sum(non_participant, "decrease_MWh")
    non_participant_cost = _sum(non_participant, "bid_cost")
    ordinary_cost = redispatch_cost - slack_cost - non_participant_cost
    gross_redispatch_mwh = increase_mwh + decrease_mwh
    # The foreign-market balance generators re-balance the cables' far ends;
    # on CP30 December 2030 they were 4.6 of 14.7 TWh gross, taking the share of
    # demand from 35% (GB assets) to 52% (audit of 26 Sep 2026).
    gb_rows = redispatch.loc[~_foreign_side_mask(redispatch, balancing_network)]
    gb_gross_redispatch_mwh = _sum(gb_rows, "increase_MWh") + _sum(gb_rows, "decrease_MWh")
    is_short_run = period_hours < 8760.0

    def per_demand(value: float) -> float | None:
        return float(value / demand_mwh) if demand_mwh > 0 else None

    def per_gb_demand(value: float) -> float | None:
        return float(value / gb_demand_mwh) if gb_demand_mwh > 0 else None

    warnings = []
    if is_short_run:
        warnings.append(
            "Short-run diagnostic: do not annualise costs, volumes or price "
            "statistics from this report."
        )
    if slack_energy_mwh > 0.1:
        warnings.append(
            "Feasibility slack/load shedding occurred; its cost is separated from "
            "ordinary redispatch and the run should not be treated as adequate."
        )
    if burn["burn_MWh"] > 1.0:
        warnings.append(
            f"Storage charged and discharged at once in {burn['unit_hours']:,} unit-hours, "
            f"destroying {burn['burn_MWh']:,.1f} MWh; reported as curtailed energy (storage_burn_MWh)."
        )
    if non_participant_mwh > 1.0:
        warnings.append(
            f"{non_participant_mwh:,.0f} MWh of forced turn-down by balancing non-participants is "
            f"priced at the penalty bid (GBP {non_participant_cost:,.0f}); it is excluded from "
            "ordinary redispatch cost, and modelled_redispatch_cost_GBP includes it."
        )

    accounting = {
        "definitions_version": "WP0",
        "reported_period_hours": period_hours,
        "is_short_run": is_short_run,
        "annualisation_allowed": not is_short_run,
        "warnings": warnings,
        "demand_MWh": demand_mwh,
        "consumer_flexibility_MWh": flex_demand_mwh,
        "accepted_offer_cost_GBP": offer_cost,
        "accepted_bid_cost_GBP": bid_cost,
        "modelled_redispatch_cost_GBP": redispatch_cost,
        "modelled_redispatch_cost_GBP_per_MWh_demand": per_demand(redispatch_cost),
        "ordinary_redispatch_cost_excluding_feasibility_slack_GBP": ordinary_cost,
        "ordinary_redispatch_cost_GBP_per_MWh_demand": per_demand(ordinary_cost),
        "feasibility_slack_energy_MWh": slack_energy_mwh,
        "feasibility_slack_cost_GBP": slack_cost,
        "feasibility_slack_cost_GBP_per_MWh_demand": per_demand(slack_cost),
        "non_participant_turn_down_MWh": non_participant_mwh,
        "non_participant_turn_down_cost_at_penalty_GBP": non_participant_cost,
        # Decision D6: storage losses from simultaneous charge/discharge, curtailed energy.
        # Item F22: the churn sink's held absorption, outside every redispatch volume.
        "churn_sink_held_MWh": churn_sink_held_mwh(balancing_network, hours),
        "storage_burn_MWh": burn["burn_MWh"],
        "storage_burn_by_carrier_MWh": burn["burn_by_carrier_MWh"],
        "storage_burn_unit_hours": burn["unit_hours"],
        "storage_burn_tolerance_pu": burn["tolerance_pu"],
        "curtailed_energy_storage_burn_MWh": burn["burn_MWh"],
        "gross_redispatch_MWh": gross_redispatch_mwh,
        "gross_redispatch_pct_demand": per_demand(gross_redispatch_mwh * 100.0),
        "gross_redispatch_including_foreign_side_MWh": gross_redispatch_mwh,
        "gb_electricity_demand_MWh": gb_demand_mwh,
        "gb_gross_redispatch_MWh": gb_gross_redispatch_mwh,
        "gb_gross_redispatch_pct_gb_electricity_demand": per_gb_demand(gb_gross_redispatch_mwh * 100.0),
        "definitions": METRIC_DEFINITIONS,
    }

    rows = [
        ("reported_period_hours", period_hours, "h", "reported_period"),
        ("demand_MWh", demand_mwh, "MWh", "modelled_demand"),
        ("consumer_flexibility_MWh", flex_demand_mwh, "MWh", "modelled_demand"),
        ("accepted_offer_cost_GBP", offer_cost, "GBP", "redispatch"),
        ("accepted_bid_cost_GBP", bid_cost, "GBP", "redispatch"),
        ("modelled_redispatch_cost_GBP", redispatch_cost, "GBP", "redispatch"),
        (
            "modelled_redispatch_cost_GBP_per_MWh_demand",
            per_demand(redispatch_cost),
            "GBP/MWh demand",
            "redispatch",
        ),
        (
            "ordinary_redispatch_cost_excluding_feasibility_slack_GBP",
            ordinary_cost,
            "GBP",
            "redispatch",
        ),
        ("feasibility_slack_energy_MWh", slack_energy_mwh, "MWh", "feasibility"),
        ("feasibility_slack_cost_GBP", slack_cost, "GBP", "feasibility"),
        ("non_participant_turn_down_MWh", non_participant_mwh, "MWh", "non_participant"),
        ("non_participant_turn_down_cost_at_penalty_GBP", non_participant_cost, "GBP", "non_participant"),
        ("churn_sink_held_MWh", accounting["churn_sink_held_MWh"], "MWh", "churn_sink"),
        ("storage_burn_MWh", burn["burn_MWh"], "MWh", "curtailment"),
        ("storage_burn_unit_hours", burn["unit_hours"], "unit-hours", "curtailment"),
        ("gross_redispatch_MWh", gross_redispatch_mwh, "MWh", "redispatch"),
        ("gross_redispatch_pct_demand", per_demand(gross_redispatch_mwh * 100.0), "%", "redispatch"),
        ("gross_redispatch_including_foreign_side_MWh", gross_redispatch_mwh, "MWh", "redispatch"),
        ("gb_electricity_demand_MWh", gb_demand_mwh, "MWh", "modelled_demand"),
        ("gb_gross_redispatch_MWh", gb_gross_redispatch_mwh, "MWh", "redispatch"),
        (
            "gb_gross_redispatch_pct_gb_electricity_demand",
            per_gb_demand(gb_gross_redispatch_mwh * 100.0),
            "%",
            "redispatch",
        ),
    ]
    report = pd.DataFrame(rows, columns=["metric", "value", "unit", "scope"])
    return accounting, report


def attach_interconnector_retrading(
    accounting: dict,
    report: pd.DataFrame,
    retrading_cost_gbp: float | None,
) -> pd.DataFrame:
    """Add the interconnector re-trading cost to the accounting block and table.

    The balancing LP re-trades the cross-border cables with a convex bid/ask
    cost that appears in the solver objective but not in accepted offers or
    bids. This records that cost alongside the redispatch metrics so the gap
    between ``modelled_redispatch_cost_GBP`` and the balancing objective is
    explained. ``accounting`` is updated in place; the returned DataFrame is
    ``report`` with the matching rows appended (same four-column shape).

    Parameters
    ----------
    accounting : dict
        Output of :func:`create_market_accounting_report`.
    report : pd.DataFrame
        Metric table from :func:`create_market_accounting_report`.
    retrading_cost_gbp : float or None
        Sum of per-cable ``retrading_cost_GBP``; ``None`` when the balancing
        network carries no piecewise-priced interconnectors or the prices
        could not be evaluated.
    """
    scope = "interconnector_retrading"
    demand_mwh = float(accounting.get("demand_MWh") or 0.0)

    def per_demand(value):
        if value is None or demand_mwh <= 0.0:
            return None
        return float(value / demand_mwh)

    if retrading_cost_gbp is None or pd.isna(retrading_cost_gbp):
        retrading = None
        including = None
        neso_comparable = None
    else:
        retrading = float(retrading_cost_gbp)
        redispatch_cost = accounting.get("modelled_redispatch_cost_GBP")
        including = (
            float(redispatch_cost) + retrading if redispatch_cost is not None else None
        )
        ordinary = accounting.get("ordinary_redispatch_cost_excluding_feasibility_slack_GBP")
        neso_comparable = float(ordinary) + retrading if ordinary is not None else None

    accounting["interconnector_retrading_cost_GBP"] = retrading
    accounting["interconnector_retrading_cost_GBP_per_MWh_demand"] = per_demand(retrading)
    accounting["modelled_redispatch_cost_including_retrading_GBP"] = including
    accounting["modelled_redispatch_cost_including_retrading_GBP_per_MWh_demand"] = (
        per_demand(including)
    )
    # NESO-comparable (16 Oct 2026): ordinary redispatch + re-trading, no slack or penalties.
    accounting["neso_comparable_redispatch_cost_GBP"] = neso_comparable
    definitions = accounting.get("definitions")
    if isinstance(definitions, dict):
        if definitions is METRIC_DEFINITIONS:
            definitions = dict(definitions)
            accounting["definitions"] = definitions
        definitions["interconnector_retrading_cost"] = METRIC_DEFINITIONS[
            "interconnector_retrading_cost"
        ]
        definitions["neso_comparable_redispatch_cost"] = METRIC_DEFINITIONS[
            "neso_comparable_redispatch_cost"
        ]

    rows = [
        ("interconnector_retrading_cost_GBP", retrading, "GBP", scope),
        (
            "interconnector_retrading_cost_GBP_per_MWh_demand",
            per_demand(retrading),
            "GBP/MWh demand",
            scope,
        ),
        (
            "modelled_redispatch_cost_including_retrading_GBP",
            including,
            "GBP",
            scope,
        ),
        (
            "modelled_redispatch_cost_including_retrading_GBP_per_MWh_demand",
            per_demand(including),
            "GBP/MWh demand",
            scope,
        ),
        ("neso_comparable_redispatch_cost_GBP", neso_comparable, "GBP", scope),
    ]
    extra = pd.DataFrame(rows, columns=["metric", "value", "unit", "scope"])
    return pd.concat([report, extra], ignore_index=True)

