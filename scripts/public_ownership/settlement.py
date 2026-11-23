"""WP-D: link each register asset to its market and support settlement.

For every generator and storage unit in the twelve monthly runs:

* ``wholesale_revenue_gbp``      = sum over hours of wholesale position x clearing price;
* ``balancing_settlement_gbp``   = accepted offer and bid payments from the balancing
  stage (``net_cost`` in the per-asset redispatch summary the market solve writes,
  in the ESO-cost convention: a positive number is paid to the asset);
* ``cfd_support_gbp``            = difference payments on balancing dispatch, settled
  through the same calculator and strike table as the bill's CfD line, with the
  baseload reference for the configured carriers;
* ``cfd_curtailment_compensation_gbp`` = the foregone difference payment restored
  through the balancing bid (the same formula as the bill's constraint line);
* ``roc_support_gbp``            = balancing dispatch x ROC banding x notional ROC value;
* ``dpa_support_gbp``            = the bill's Net Zero Teesside DPA line (``remaining_ccs``),
  paid to the plant it funds (item B11), spread over DPA plant by balancing dispatch;
* ``fuel_carbon_gbp``            = balancing dispatch x the plant's resource fuel and carbon
  cost from the build-time breakdown (``{build_source}_marginal_costs_breakdown.csv``):
  fuel plus carbon, net of a CHP heat credit, with no variable O&M (the register adds
  its own) and none of the balancing solve's cost floors (item B1).  The network's
  marginal cost is the fallback for a generator missing from the breakdown.  Storage
  burns no fuel: its purchased energy is already negative market revenue.

Hydrogen links (item B10).  Electrolysers buy and hydrogen turbines sell at the Stage 1
price, and the balancing stage moves them without an offer or a bid.  They are not
register assets, so the money had no destination; each link is settled here as a row
with ``component_type = Link`` and ``payee_class = hydrogen_off_register``: its Stage 1
injection into the AC network x the price (negative for an electrolyser), and any
balancing payment the redispatch summary records for it.  Every other row is
``register_asset``.

Hydrogen fuel (FES 2050 task Y14).  From the vintaged model years the hydrogen
turbines are register assets (task Y14-prep), so a turbine with no fuel cost would
keep its whole revenue as margin.  With ``hydrogen_fuel_price="bus0_marginal_price"``
(passed for vintaged groups only; the default keeps every 2030 row as it was) each
``H2_turbine`` row's ``fuel_carbon_gbp`` is the hydrogen it burns priced at its
hydrogen bus: hour by hour, the AC-side balancing delivery divided by the link
efficiency, times the Stage 1 ``buses_t.marginal_price`` of the link's ``bus0``
(the wholesale network's hydrogen price; the Stage 1 network keeps prices, and the
AC-side flows are the ones the settlement already derives), with
``fuel_carbon_basis = hydrogen_bus_marginal_price``.  Electrolysers keep no fuel
cost: their purchased electricity is already negative market revenue.

Each row also carries its bus and the bus coordinates (``bus_lat``, ``bus_lon``), which
the register uses to put the asset in its TNUoS charging zone (item D4(a)).

``cfd_curtailment_compensation_gbp`` describes the same money as the CfD share of
``balancing_settlement_gbp`` whenever the balancing solve priced CfD bids on contract
value, so ``support_revenue_gbp`` counts it only when the solve did *not* do that. This is
the treatment the bill already applies to its constraint line
(``scripts/consumer_bill/extract_future_balancing_costs.py``, ``add_post_hoc`` versus
``reconcile_against_lp_bid``), taken from the same audit the balancing solve writes rather
than declared by hand. Adding it on top of a contract-value bid double counts it: on the
CP30 2030 year it overstated CfD contract revenue by GBP 0.994 bn (5.1%), which inflated
the public account's rent and drove the at-cost strike basis to negative strikes.

Market revenue is a transfer, not a resource cost: it never enters the required-revenue
account and is used only for the producer and public account (plan section 10).
Everything is in the model's price year (GBP real 2024).

The 2050 contract book (FES 2050 task SETTLE2050).  A vintaged register settles its own
contract labels: ``settle_assets(..., contract_book=, register_assets=)`` settles every
network asset as above, then writes one row per *register row* (``settlement_level =
register_row``): each row takes its vintaged share of its network asset's amounts, and
its own CfD and DPA support from the book (:mod:`contract_book_2050`): CfD difference
payments and curtailment compensation per contract unit through the same calculator,
DPA availability and variable payments less gainshare per unit.  ``attach_settlement``
joins such a file row for row.  Without a book nothing changes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from scripts.consumer_bill.extract_future_balancing_costs import (
    audit_declares_contract_value_bids, bid_basis_from_audit, contract_value_floor_from_audit, contract_value_units,
    negative_price_rule_from_audit,
    ownership_from_audit,
)
from scripts.market.market_accounting import _non_participant_turn_down
from scripts.consumer_bill.project_future_bill import (
    read_hourly_inputs, seasonal_demand_weighted_reference,
)
from scripts.market.cfd_bm_bids import contract_value_rate
from scripts.market.cfd_strike_table import contract_technology
from scripts.market.revenue_tracking import compute_cfd_payments, negative_price_eligibility


# How far the recomputed compensation may sit from the CfD fleet's accepted
# balancing bids before the reconciliation counts as failed, as a share of the
# recompute. The two are the same quantity by construction when the solve
# priced CfD bids on contract value. Measured at 0.02% on the CP30 2030 year
# (GBP 0.2 m on GBP 993.5 m); the same 1% the bill allows. The comparison is
# with the bid side only: offers pay a CfD unit for turning *up* (Drax and MGT
# Teesside earned GBP 15.5 m of them on the 25 Sep 2026 run), which the
# compensation never covers, and comparing against the net settlement failed
# the check at 2.39% on that alone.
COMPENSATION_RECONCILIATION_TOLERANCE = 0.01
# Below this absolute gap the check passes whatever the ratio. P3 (29 Sep 2026)
# state-owns the whole CfD fleet, so the private compensation left to reconcile
# was GBP 879 against GBP 712 of bids: an 18.96% "gap" worth 0.0000 GBP a
# household, where 1% of it is GBP 9 of solver noise.
COMPENSATION_RECONCILIATION_FLOOR_GBP = 100_000.0
# Thermal CfDs keep their fuel-cost bids in the balancing stage (support.dispatch_costs.
# cfd_dispatch_cost_exempt_carriers: Drax, MGT, BECCS), so their accepted bids are not contract value and
# are left out of the comparison. On the UPPER network's P3 bid stack (25 Oct 2026) Drax's recomputed
# compensation (GBP 1.21 m) against its fuel-cost bids (0.90 m) failed the check above the floor.
COMPENSATION_CHECK_EXEMPT_CARRIERS = frozenset({"biomass", "CCS Biomass"})

SETTLEMENT_COLUMNS = [
    "component_type", "asset_id", "carrier", "support_type", "payee_class", "bus", "bus_lat", "bus_lon",
    "wholesale_mwh", "balancing_mwh",
    "wholesale_revenue_gbp", "balancing_settlement_gbp", "cfd_support_gbp",
    "cfd_curtailment_compensation_gbp", "roc_support_gbp", "dpa_support_gbp", "fuel_carbon_gbp",
    "fuel_carbon_basis", "market_revenue_gbp", "support_revenue_gbp", "months",
]
#: Settlement columns that describe the asset rather than an amount: copied onto
#: cohort rows as they are, never scaled by the cohort share.
DESCRIPTIVE_COLUMNS = ("payee_class", "bus", "bus_lat", "bus_lon", "fuel_carbon_basis")
# Components the register also drops.
NON_FINANCE_CARRIERS = {"load_shedding", "load shedding", "voll", "unserved_energy"}
#: Consumer flexibility (EVs, V2G, heat, storage heaters, DSR, DACCS) is demand, not an asset: no settlement
#: row (F5, 27 Oct 2026; scripts/demand/demand_carriers.py).
from scripts.demand.demand_carriers import DEMAND_FLEX_STORAGE_CARRIERS  # noqa: E402
PAYEE_REGISTER = "register_asset"
PAYEE_HYDROGEN = "hydrogen_off_register"
#: Link carriers settled as the hydrogen payee (item B10).
DEFAULT_HYDROGEN_LINK_CARRIERS = ("electrolysis", "H2_turbine")
FUEL_BASIS_RESOURCE = "resource_breakdown"
FUEL_BASIS_NETWORK = "network_marginal_cost_fallback"
FUEL_BASIS_NONE = "no_fuel"
#: Task Y14: the hydrogen-turbine fuel basis, and the option that switches it on.
FUEL_BASIS_HYDROGEN_BUS = "hydrogen_bus_marginal_price"
HYDROGEN_FUEL_BUS0_PRICE = "bus0_marginal_price"
HYDROGEN_FUEL_PRICE_BASES = (HYDROGEN_FUEL_BUS0_PRICE,)
#: Link carriers that burn hydrogen bought at their bus0 price (task Y14).
HYDROGEN_FUEL_LINK_CARRIERS = ("H2_turbine",)
#: Breakdown sources whose marginal cost is not a fuel cost at all.
_NON_FUEL_SOURCES = ("subsidy_", "renewable_default", "protected")


def resource_fuel_carbon_rates(breakdown: pd.DataFrame) -> pd.Series:
    """GBP/MWh of fuel and carbon per generator, from the build-time marginal cost breakdown (item B1).

    The balancing network's ``marginal_cost`` is the wrong measure of what a
    generator burns: the balancing solve floors some carriers (waste to energy
    at 30 GBP/MWh, whose gate fee covers its fuel and carbon, booked 494.9 m of
    fuel on the CP30 year), and the ``formula+vom`` sources fold variable O&M in,
    which the register adds again from DESNZ.  So:

    * a subsidy or renewable source (``subsidy_*``, ``renewable_default``,
      ``protected``) burns ``fuel + carbon`` (zero for every such row);
    * every other source costs its build-time marginal cost less its own
      variable O&M, floored at zero: fuel plus carbon for the formula sources,
      net of the heat credit for CHP (the electricity's share of the fuel), the
      DPA ordering for Net Zero Teesside, and the nuclear override (the
      model's only nuclear fuel cost; DESNZ-side nuclear VOM is zero).
    """
    frame = breakdown.drop_duplicates("generator").set_index("generator")
    source = frame["source"].fillna("").astype(str)
    fuel = pd.to_numeric(frame.get("fuel_cost_gbp_per_mwh"), errors="coerce").fillna(0.0)
    carbon = pd.to_numeric(frame.get("carbon_cost_gbp_per_mwh"), errors="coerce").fillna(0.0)
    vom = pd.to_numeric(frame.get("variable_om_gbp_per_mwh"), errors="coerce").fillna(0.0)
    mc = pd.to_numeric(frame["marginal_cost"], errors="coerce").fillna(0.0)
    non_fuel = source.apply(lambda s: any(s.startswith(prefix) for prefix in _NON_FUEL_SOURCES))
    return pd.Series(np.where(non_fuel, fuel + carbon, np.maximum(mc - vom, 0.0)), index=frame.index, dtype=float)


def read_marginal_cost_breakdowns(paths) -> pd.DataFrame | None:
    """Concatenate the build-time breakdowns (one per build source) and refuse conflicting generators."""
    frames = [pd.read_csv(path) for path in (paths or [])]
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    rates = combined.assign(_rate=resource_fuel_carbon_rates(combined).reindex(combined["generator"]).to_numpy())
    spread = rates.groupby("generator")["marginal_cost"].nunique()
    if (spread > 1).any():
        raise ValueError(f"marginal cost breakdowns disagree for {int((spread > 1).sum())} generators")
    return combined


def _bus_coordinates(ds: xr.Dataset) -> pd.DataFrame:
    """Bus latitude and longitude (WGS84), from ``buses_lat``/``buses_lon`` or converted from OSGB36 x/y."""
    index = pd.Index(ds["buses_i"].values) if "buses_i" in ds else pd.Index([])
    if "buses_lat" in ds and "buses_lon" in ds:
        return pd.DataFrame({"lat": ds["buses_lat"].values, "lon": ds["buses_lon"].values}, index=index)
    if "buses_x" not in ds or "buses_y" not in ds:
        return pd.DataFrame({"lat": np.nan, "lon": np.nan}, index=index)
    x, y = np.asarray(ds["buses_x"].values, dtype=float), np.asarray(ds["buses_y"].values, dtype=float)
    if np.nanmax(x) - np.nanmin(x) > 1_000:
        from pyproj import Transformer

        lon, lat = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True).transform(x, y)
        return pd.DataFrame({"lat": lat, "lon": lon}, index=index)
    return pd.DataFrame({"lat": y, "lon": x}, index=index)


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _static(ds: xr.Dataset, prefix: str, column: str, index: pd.Index, default=""):
    """A static attribute for ``index`` (a subset of the component index), defaulted where absent."""
    key = f"{prefix}_{column}"
    if key not in ds:
        return pd.Series(default, index=index)
    full = pd.Series(ds[key].values, index=pd.Index(ds[f"{prefix}_i"].values))
    return full.reindex(index).fillna(default)


def _table(ds: xr.Dataset, key: str, columns: pd.Index, snapshots: pd.DatetimeIndex) -> pd.DataFrame:
    """Read a whole time-series table and subset in pandas (label .sel is slow on NetCDF)."""
    if key not in ds:
        return pd.DataFrame(0.0, index=snapshots, columns=columns)
    frame = ds[key].to_pandas()
    frame.index = snapshots
    return frame.reindex(columns=columns, fill_value=0.0)


def non_money_balancing_gbp(redispatch: pd.DataFrame, audit_path=None) -> pd.Series:
    """Per component, the part of the accepted bid cost that is not a payment (review 28 Sep 2026).

    Two terms sit inside ``bid_cost`` (and so ``net_cost``) without being money a
    generator receives, and the bill's constraint line already nets both
    (``extract_future_balancing_costs``): the turn-down of a balancing
    non-participant priced at the LP's penalty bid (item B8), and the P3
    tie-break steps (decision 20), share-weighted step x turn-down. Without this
    the register booked them as balancing revenue. ``redispatch`` is indexed by
    component.
    """
    out = pd.Series(0.0, index=redispatch.index.astype(str), dtype=float)
    frame = redispatch.copy()
    frame.index = frame.index.astype(str)
    forced = _non_participant_turn_down(frame)
    if len(forced):
        out = out.add(pd.to_numeric(forced["bid_cost"], errors="coerce").fillna(0.0).groupby(level=0).sum(),
                      fill_value=0.0)
    if audit_path is not None and Path(audit_path).exists() and "decrease_MWh" in frame.columns:
        audit = pd.read_csv(audit_path)
        if not audit.empty and "tie_break_step_gbp_per_mwh" in audit.columns:
            steps = pd.to_numeric(audit.set_index("generator")["tie_break_step_gbp_per_mwh"],
                                  errors="coerce").fillna(0.0)
            steps = steps[steps.ne(0.0)].groupby(level=0).first()
            if not steps.empty:
                generators = frame.loc[frame["type"].astype(str).eq("generator")] if "type" in frame.columns else frame
                decrease = pd.to_numeric(generators["decrease_MWh"], errors="coerce").fillna(0.0).groupby(level=0).sum()
                out = out.add(steps.reindex(decrease.index).fillna(0.0) * decrease, fill_value=0.0)
    return out.groupby(level=0).sum()


def _month_settlement(wholesale_path, balancing_path, price: pd.Series, redispatch_path,
                      strike_table, reference_overrides, roc_value_gbp_per_roc: float,
                      volume_caps: dict | None = None, negative_price_rule: str = "none",
                      bid_negative_price_rule: str = "none", fuel_rates: pd.Series | None = None,
                      hydrogen_link_carriers=DEFAULT_HYDROGEN_LINK_CARRIERS,
                      bid_basis: str = "contract_value", audit_path=None,
                      hydrogen_fuel_price: str | None = None,
                      contract_value_floor: str = "zero") -> pd.DataFrame:
    rows = []
    with xr.open_dataset(wholesale_path, engine="netcdf4") as dw, xr.open_dataset(balancing_path, engine="netcdf4") as db:
        snapshots = pd.DatetimeIndex(db["snapshots_snapshot"].values)
        if not pd.DatetimeIndex(dw["snapshots_snapshot"].values).equals(snapshots):
            raise ValueError("Wholesale and balancing snapshots do not match")
        if not snapshots.equals(pd.DatetimeIndex(price.index)):
            raise ValueError("Price series and network snapshots do not match")
        hours = pd.Series(db["snapshots_generators"].values, index=snapshots, dtype=float)
        # Pro rata zonal turn-down (20 Oct 2026): per-unit bids and volumes on the month's basis.
        from scripts.market.turn_down_allocation import balancing_dispatch_in_force, read_redispatch_in_force

        redispatch = read_redispatch_in_force(redispatch_path).set_index("component")
        if not {"offer_cost", "bid_cost", "net_cost"}.issubset(redispatch.columns):
            raise ValueError(f"{redispatch_path} lacks offer_cost, bid_cost and net_cost")
        # Penalty-priced turn-down and P3 tie steps are not payments: take them out
        # of the bid and net settlement, as the bill's constraint line does.
        redispatch = redispatch.loc[~redispatch.index.astype(str).isin(["TOTAL"])].copy()
        non_money = non_money_balancing_gbp(redispatch, audit_path)
        if non_money.abs().sum() > 0.0:
            adjust = non_money.reindex(redispatch.index.astype(str)).fillna(0.0).to_numpy()
            for column in ("bid_cost", "net_cost"):
                redispatch[column] = pd.to_numeric(redispatch[column], errors="coerce").fillna(0.0) - adjust
        coordinates = _bus_coordinates(db)

        for component_type, prefix in (("Generator", "generators"), ("StorageUnit", "storage_units")):
            names = pd.Index(db[f"{prefix}_i"].values)
            carrier = _text(_static(db, prefix, "carrier", names))
            keep = names[~carrier.str.lower().isin(NON_FINANCE_CARRIERS) & ~carrier.isin(DEMAND_FLEX_STORAGE_CARRIERS)]
            if not len(keep):
                continue
            carrier = carrier.reindex(keep)
            support = _text(_static(db, prefix, "support_type", keep))
            bus = _text(_static(db, prefix, "bus", keep))
            marginal_cost = pd.to_numeric(_static(db, prefix, "marginal_cost", keep, 0.0), errors="coerce").fillna(0.0)
            # Item B1: resource fuel and carbon, never the balancing cost floors.
            if component_type == "StorageUnit":
                fuel_rate = pd.Series(0.0, index=keep)
                fuel_basis = pd.Series(FUEL_BASIS_NONE, index=keep)
            elif fuel_rates is not None:
                found = keep.isin(fuel_rates.index)
                fuel_rate = pd.Series(np.where(found, fuel_rates.reindex(keep).fillna(0.0), marginal_cost.clip(lower=0.0)),
                                      index=keep)
                fuel_basis = pd.Series(np.where(found, FUEL_BASIS_RESOURCE, FUEL_BASIS_NETWORK), index=keep)
            else:
                fuel_rate = marginal_cost.clip(lower=0.0)
                fuel_basis = pd.Series(FUEL_BASIS_NETWORK, index=keep)
            p_b = _table(db, f"{prefix}_t_p", keep, snapshots)
            if component_type == "Generator":
                p_b = balancing_dispatch_in_force(p_b, balancing_path)
            p_w = _table(dw, f"{prefix}_t_p", keep, snapshots)
            if not np.isfinite(p_b.to_numpy()).all() or not np.isfinite(p_w.to_numpy()).all():
                raise ValueError("Dispatch contains missing or non-finite values")
            energy_b = p_b.mul(hours, axis=0)
            energy_w = p_w.mul(hours, axis=0)
            frame = pd.DataFrame({
                "component_type": component_type,
                "asset_id": keep.astype(str),
                "carrier": carrier.to_numpy(),
                "support_type": support.to_numpy(),
                "payee_class": PAYEE_REGISTER,
                "bus": bus.to_numpy(),
                "bus_lat": coordinates["lat"].reindex(bus.to_numpy()).to_numpy(),
                "bus_lon": coordinates["lon"].reindex(bus.to_numpy()).to_numpy(),
                # Generators cannot run negative; storage is signed (charging is a purchase).
                "wholesale_mwh": (energy_w.clip(lower=0.0) if component_type == "Generator" else energy_w).sum().to_numpy(),
                "balancing_mwh": (energy_b.clip(lower=0.0) if component_type == "Generator" else energy_b).sum().to_numpy(),
                "wholesale_revenue_gbp": energy_w.mul(price, axis=0).sum().to_numpy(),
                "balancing_settlement_gbp": pd.to_numeric(redispatch["net_cost"], errors="coerce").reindex(keep).fillna(0.0).to_numpy(),
                # Kept only for the compensation reconciliation; not in the output.
                "balancing_bid_gbp": pd.to_numeric(redispatch["bid_cost"], errors="coerce").reindex(keep).fillna(0.0).to_numpy(),
                "fuel_carbon_gbp": energy_b.clip(lower=0.0).mul(fuel_rate, axis=1).sum().to_numpy(),
                "fuel_carbon_basis": fuel_basis.to_numpy(),
                "cfd_support_gbp": 0.0,
                "cfd_curtailment_compensation_gbp": 0.0,
                "roc_support_gbp": 0.0,
                "dpa_support_gbp": 0.0,
            }, index=keep)

            if component_type == "Generator":
                cfd_names = keep[support.eq("CfD").to_numpy()]
                if len(cfd_names) and strike_table is not None:
                    generators = pd.DataFrame(index=cfd_names)
                    generators["support_type"] = "CfD"
                    # cfd_technology: the contract's own strike technology, as the bill keys it.
                    for column in ("carrier", "bus", "cfd_round", "cfd_support_start", "cfd_technology"):
                        generators[column] = _text(_static(db, prefix, column, cfd_names))
                    network = SimpleNamespace(
                        generators=generators, snapshots=snapshots,
                        generators_t=SimpleNamespace(p=p_b[cfd_names]),
                        snapshot_weightings=pd.DataFrame({"generators": hours}),
                    )
                    overrides = (
                        {c: s.loc[snapshots] for c, s in reference_overrides.items()} if reference_overrides else None
                    )
                    details = compute_cfd_payments(network, strike_table, reference_price=price,
                                                   reference_price_overrides=overrides,
                                                   volume_caps=volume_caps,
                                                   negative_price_rule=negative_price_rule)
                    if not details.empty:
                        payments = details.set_index("generator")["net_cfd_cost"]
                        frame.loc[payments.index, "cfd_support_gbp"] = payments.to_numpy()
                        strikes = details.set_index("generator")["strike_price"]
                        # Compensation for constrained-off CfD output: the same floored
                        # formula the bill's constraint line uses.
                        turn_down = (p_w[strikes.index] - p_b[strikes.index]).clip(lower=0.0).mul(hours, axis=0)
                        # The basis the bids used (contract_value_bids.bid_basis): under
                        # 'strike' a turned-down unit was paid its whole strike.
                        # contract_value_floor 'none' (FTI): not floored at zero, as the bid.
                        premium = (np.broadcast_to(strikes.to_numpy()[None, :], (len(price), len(strikes))).copy()
                                   if bid_basis == "strike"
                                   else contract_value_rate(strikes.to_numpy()[None, :], price.to_numpy()[:, None],
                                                            contract_value_floor))
                        # The rule the balancing bids applied, so this stays the same money as the bid.
                        if bid_negative_price_rule != "none":
                            premium = premium * negative_price_eligibility(
                                price.to_numpy(), details.set_index("generator")["cfd_round"].reindex(strikes.index),
                                bid_negative_price_rule)
                        compensation = (turn_down.to_numpy() * premium).sum(axis=0)
                        frame.loc[strikes.index, "cfd_curtailment_compensation_gbp"] = compensation
                roc_names = keep[support.eq("ROC").to_numpy()]
                if len(roc_names):
                    banding = pd.to_numeric(_static(db, prefix, "ro_banding", roc_names, 1.0), errors="coerce").fillna(1.0)
                    frame.loc[roc_names, "roc_support_gbp"] = (
                        energy_b[roc_names].clip(lower=0.0).sum() * banding * float(roc_value_gbp_per_roc)
                    ).to_numpy()
            rows.append(frame)
        hydrogen = _hydrogen_link_rows(dw, db, wholesale_path, snapshots, hours, price, redispatch,
                                       coordinates, hydrogen_link_carriers, hydrogen_fuel_price=hydrogen_fuel_price)
        if hydrogen is not None:
            rows.append(hydrogen)
    return pd.concat(rows, ignore_index=True)


def _hydrogen_link_rows(dw, db, wholesale_path, snapshots, hours, price, redispatch, coordinates,
                        carriers, hydrogen_fuel_price: str | None = None) -> pd.DataFrame | None:
    """Item B10: each hydrogen link settled as an off-register payee.

    Stage 1 link flows are read from ``<stage1>_wholesale_links.csv`` (Stage 1
    networks solved before item F14, 13 Oct 2026, store no link time series);
    Stage 2 flows are the balancing
    network's ``links_t_p0``.  A link's injection into the AC network is
    ``-p0`` where bus0 is AC and ``efficiency x p0`` where bus1 is AC.

    ``hydrogen_fuel_price`` (task Y14, vintaged groups only) books each hydrogen
    turbine's fuel: :func:`hydrogen_turbine_fuel_gbp`.  None (the default) books none.
    """
    carriers = set(carriers or ())
    if not carriers or "links_i" not in db:
        return None
    names = pd.Index(db["links_i"].values)
    carrier = _text(_static(db, "links", "carrier", names))
    keep = names[carrier.isin(carriers).to_numpy()]
    if not len(keep):
        return None
    links_csv = Path(str(wholesale_path).replace("_wholesale.nc", "_wholesale_links.csv"))
    if not links_csv.exists():
        raise ValueError(f"hydrogen links need the Stage 1 link flows {links_csv}")
    bus_carrier = (pd.Series(db["buses_carrier"].values, index=pd.Index(db["buses_i"].values)).astype(str)
                   if "buses_carrier" in db else pd.Series(dtype=str))
    bus0 = _text(_static(db, "links", "bus0", keep))
    bus1 = _text(_static(db, "links", "bus1", keep))
    efficiency = pd.to_numeric(_static(db, "links", "efficiency", keep, 1.0), errors="coerce").fillna(1.0)
    ac0 = bus0.map(bus_carrier).eq("AC").to_numpy()
    ac1 = bus1.map(bus_carrier).eq("AC").to_numpy()
    stage1 = pd.read_csv(links_csv, index_col=0, parse_dates=True)
    stage1.index = pd.DatetimeIndex(stage1.index)
    p0_w = stage1.reindex(index=snapshots, columns=keep, fill_value=0.0).fillna(0.0)
    p0_b = _table(db, "links_t_p0", keep, snapshots)

    def injection(p0):
        return p0.mul(np.where(ac1, efficiency.to_numpy(), 0.0) - np.where(ac0, 1.0, 0.0), axis=1)

    inj_w, inj_b = injection(p0_w), injection(p0_b)
    ac_bus = np.where(ac1, bus1, bus0)
    fuel = pd.Series(0.0, index=keep)
    fuel_basis = pd.Series(FUEL_BASIS_NONE, index=keep)
    if hydrogen_fuel_price is not None:
        is_turbine = carrier.reindex(keep).isin(HYDROGEN_FUEL_LINK_CARRIERS).to_numpy()
        turbines = keep[is_turbine]
        if len(turbines):
            h2_buses = bus0.reindex(turbines)
            prices = _table_strict(dw, "buses_t_marginal_price", pd.Index(h2_buses.unique()), snapshots,
                                   "the Stage 1 hydrogen bus price")
            fuel.loc[turbines] = hydrogen_turbine_fuel_gbp(
                inj_b[turbines], efficiency.reindex(turbines), h2_buses, ac1[is_turbine], prices, hours,
                hydrogen_fuel_price).to_numpy()
            fuel_basis.loc[turbines] = FUEL_BASIS_HYDROGEN_BUS
    net = pd.to_numeric(redispatch["net_cost"], errors="coerce") if "net_cost" in redispatch else pd.Series(dtype=float)
    return pd.DataFrame({
        "component_type": "Link", "asset_id": keep.astype(str), "carrier": carrier.reindex(keep).to_numpy(),
        "support_type": "", "payee_class": PAYEE_HYDROGEN, "bus": ac_bus,
        "bus_lat": coordinates["lat"].reindex(ac_bus).to_numpy(), "bus_lon": coordinates["lon"].reindex(ac_bus).to_numpy(),
        "wholesale_mwh": inj_w.mul(hours, axis=0).sum().to_numpy(),
        "balancing_mwh": inj_b.mul(hours, axis=0).sum().to_numpy(),
        "wholesale_revenue_gbp": inj_w.mul(hours * price, axis=0).sum().to_numpy(),
        "balancing_settlement_gbp": net.reindex(keep).fillna(0.0).to_numpy(),
        "balancing_bid_gbp": 0.0, "fuel_carbon_gbp": fuel.to_numpy(), "fuel_carbon_basis": fuel_basis.to_numpy(),
        "cfd_support_gbp": 0.0, "cfd_curtailment_compensation_gbp": 0.0, "roc_support_gbp": 0.0,
        "dpa_support_gbp": 0.0,
    }, index=keep)


def _table_strict(ds, key: str, columns: pd.Index, snapshots: pd.DatetimeIndex, what: str) -> pd.DataFrame:
    """A time-series table that must carry every requested column (no silent zero fill)."""
    if key not in ds:
        raise ValueError(f"{what} needs {key} in the network file")
    frame = ds[key].to_pandas()
    frame.index = snapshots
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what}: {key} has no column for {missing[:5]}")
    return frame[list(columns)]


def hydrogen_turbine_fuel_gbp(ac_delivery_mw: pd.DataFrame, efficiency: pd.Series, bus0: pd.Series, ac_is_bus1,
                              h2_price: pd.DataFrame, hours: pd.Series,
                              basis: str = HYDROGEN_FUEL_BUS0_PRICE) -> pd.Series:
    """The hydrogen a turbine burns, priced at its hydrogen bus (task Y14).

    ``fuel = sum over hours of max(AC delivery, 0) / efficiency x price(bus0) x hours``:
    the AC-side dispatch the settlement and the register already derive, turned into
    hydrogen use by the link efficiency (electricity out per MWh of hydrogen in), at the
    hydrogen bus marginal price, the value of the hydrogen the turbine takes off the
    hydrogen balance.

    Args:
        ac_delivery_mw: Hourly AC-side delivery per turbine (MW), columns the links.
        efficiency: Link efficiency per turbine.
        bus0: Hydrogen bus (``bus0``) per turbine.
        ac_is_bus1: True where the AC end is ``bus1`` (every hydrogen turbine).
        h2_price: Hourly marginal price (GBP/MWh of hydrogen) per hydrogen bus.
        hours: Snapshot weights.
        basis: One of :data:`HYDROGEN_FUEL_PRICE_BASES`.

    Returns:
        GBP per turbine.

    Raises:
        ValueError: An unknown basis, a turbine whose AC end is not ``bus1``, or a
            non-positive efficiency.
    """
    if basis not in HYDROGEN_FUEL_PRICE_BASES:
        raise ValueError(f"hydrogen fuel price basis {basis!r}; supported: {HYDROGEN_FUEL_PRICE_BASES}")
    names = ac_delivery_mw.columns
    if not np.asarray(ac_is_bus1, dtype=bool).all():
        raise ValueError("a hydrogen turbine must take hydrogen at bus0 and deliver to an AC bus1")
    eff = pd.to_numeric(efficiency.reindex(names), errors="coerce")
    if not (eff > 0).all():
        raise ValueError(f"hydrogen turbines need a positive efficiency; got {eff[~(eff > 0)].to_dict()}")
    use = ac_delivery_mw.clip(lower=0.0).div(eff, axis=1)
    price = pd.DataFrame(h2_price[bus0.reindex(names).to_numpy()].to_numpy(), index=use.index, columns=names)
    return use.mul(price).mul(np.asarray(hours, dtype=float), axis=0).sum()


def settle_assets(scenarios, wholesale_paths, balancing_paths, price_paths, redispatch_paths, *,
                  strike_table, reference_price_basis: dict | None, roc_value_gbp_per_roc: float,
                  contract_value_bids_paths=None, volume_caps: dict | None = None,
                  negative_price_rule: str = "none", marginal_cost_breakdown: pd.DataFrame | None = None,
                  dpa_support_gbp: float | None = None,
                  hydrogen_link_carriers=DEFAULT_HYDROGEN_LINK_CARRIERS,
                  hydrogen_fuel_price: str | None = None, contract_book: pd.DataFrame | None = None,
                  register_assets: pd.DataFrame | None = None, dpa_prices: dict | None = None,
                  bm_strike_table=None) -> pd.DataFrame:
    """Annual settlement per asset over the twelve monthly runs.

    ``contract_book`` with ``register_assets`` (task SETTLE2050, vintaged groups only):
    the result is per register row (:func:`register_row_settlement`), with each row's own
    CfD and DPA support settled from the book; ``dpa_prices`` (``gas``, ``carbon``,
    GBP 2024) set the DPAs' reference CCGT. Its ``attrs["dpa_units"]`` hold the DPA
    payments by unit and ``attrs["compensation_treatment"]`` the compensation rule.

    ``contract_value_bids_paths`` are the per-month audits the balancing solve
    writes. They decide whether ``cfd_curtailment_compensation_gbp`` is already
    inside ``balancing_settlement_gbp`` (contract-value CfD bids) or has to be
    added to support revenue, the same rule the bill's constraint line applies.
    Passing ``None`` keeps the historical behaviour of adding it, which is only
    correct for a solve that priced CfD bids at a scalar.

    ``marginal_cost_breakdown`` is the build-time breakdown fuel and carbon are
    read from (item B1); without it the network's marginal cost is used and
    every row says so in ``fuel_carbon_basis``.  ``dpa_support_gbp`` is the
    bill's DPA line in the model's price year (item B11).

    ``hydrogen_fuel_price`` (task Y14) prices hydrogen turbines' fuel at their
    hydrogen bus (:func:`hydrogen_turbine_fuel_gbp`); pass it for vintaged (2050)
    groups only.  None, the default, books no hydrogen fuel, as before.
    """
    if hydrogen_fuel_price is not None and hydrogen_fuel_price not in HYDROGEN_FUEL_PRICE_BASES:
        raise ValueError(f"hydrogen_fuel_price={hydrogen_fuel_price!r}; supported: {HYDROGEN_FUEL_PRICE_BASES}")
    lengths = {len(scenarios), len(wholesale_paths), len(balancing_paths), len(price_paths), len(redispatch_paths)}
    if len(lengths) != 1 or not scenarios:
        raise ValueError("settlement inputs must be non-empty and aligned")
    if contract_value_bids_paths is not None and len(contract_value_bids_paths) != len(scenarios):
        raise ValueError("contract_value_bids_paths must align with scenarios")
    # Pass 1: the full-year hourly series, because the baseload reference is a
    # season-level quantity computed over the calendar year.
    # The Stage 1 link schedule beside each wholesale network: the bill's demand
    # basis counts the electrolysers' draw (item D1), so the baseload reference
    # is weighted on the same demand here.
    def _links(path):
        candidate = Path(str(path).replace("_wholesale.nc", "_wholesale_links.csv"))
        return candidate if candidate.exists() else None

    frames = [read_hourly_inputs(w, p, _links(w)) for w, p in zip(wholesale_paths, price_paths)]
    hourly = pd.concat(frames).sort_index()
    overrides = None
    carriers = [str(c) for c in ((reference_price_basis or {}).get("baseload_carriers") or [])]
    if carriers:
        series, _ = seasonal_demand_weighted_reference(hourly)
        overrides = {c: series for c in carriers}
    # Pass 2: settle each month.
    months = []
    audits = list(contract_value_bids_paths) if contract_value_bids_paths else [None] * len(scenarios)
    fuel_rates = resource_fuel_carbon_rates(marginal_cost_breakdown) if marginal_cost_breakdown is not None else None
    for scenario, w, b, p, r, frame, audit in zip(scenarios, wholesale_paths, balancing_paths, price_paths,
                                                  redispatch_paths, frames, audits):
        month = _month_settlement(w, b, frame["price_gbp_per_mwh"], r, strike_table, overrides, roc_value_gbp_per_roc,
                                  volume_caps, negative_price_rule=negative_price_rule,
                                  bid_negative_price_rule=negative_price_rule_from_audit(audit),
                                  fuel_rates=fuel_rates, hydrogen_link_carriers=hydrogen_link_carriers,
                                  bid_basis=bid_basis_from_audit(audit), audit_path=audit,
                                  hydrogen_fuel_price=hydrogen_fuel_price,
                                  contract_value_floor=contract_value_floor_from_audit(audit))
        month["scenario"] = scenario
        months.append(month)
    annual = pd.concat(months, ignore_index=True)
    numeric = ["wholesale_mwh", "balancing_mwh", "wholesale_revenue_gbp", "balancing_settlement_gbp",
               "balancing_bid_gbp", "cfd_support_gbp", "cfd_curtailment_compensation_gbp", "roc_support_gbp",
               "dpa_support_gbp", "fuel_carbon_gbp"]
    out = annual.groupby(["component_type", "asset_id"], as_index=False).agg(
        carrier=("carrier", "first"), support_type=("support_type", "first"),
        **{column: (column, "first") for column in DESCRIPTIVE_COLUMNS},
        **{column: (column, "sum") for column in numeric}, months=("scenario", "nunique"),
    )
    out = allocate_dpa_support(out, dpa_support_gbp)
    priced_on_contract_value = bool(contract_value_bids_paths) and any(
        audit_declares_contract_value_bids(path) for path in contract_value_bids_paths
    )
    shares = _state_shares_from_audits(contract_value_bids_paths)
    if shares is not None:
        out["state_share"] = out["asset_id"].map(shares).fillna(0.0)
    if contract_book is not None:
        if register_assets is None:
            raise ValueError("a contract book settles register rows: pass register_assets")
        units = settle_contract_book(
            list(wholesale_paths), list(balancing_paths), frames, contract_book, strike_table=strike_table,
            overrides=overrides, volume_caps=volume_caps, negative_price_rule=negative_price_rule, audits=audits,
            marginal_cost_breakdown=marginal_cost_breakdown, dpa_prices=dpa_prices,
            bm_strike_table=bm_strike_table)
        rows = register_row_settlement(out, register_assets, units["cfd"], units["dpa"])
        rows = combine_register_rows(rows, contract_book, priced_on_contract_value=priced_on_contract_value)
        rows = rows[SETTLEMENT_COLUMNS + REGISTER_ROW_COLUMNS]
        rows.attrs["dpa_units"] = units["dpa"]
        rows.attrs["compensation_treatment"] = ("reconcile_against_lp_bid" if priced_on_contract_value
                                                else "add_post_hoc")
        rows.attrs["period_share"] = units["period_share"]
        return rows
    elif register_assets is not None:
        raise ValueError("register_assets is only read with a contract book")
    out = combine_settlement_revenue(out, priced_on_contract_value=priced_on_contract_value)
    return out[SETTLEMENT_COLUMNS]


def _state_shares_from_audits(paths) -> pd.Series | None:
    """Item BM1: each CfD unit's state share, as the balancing solve recorded it.

    ``None`` when nothing is state-owned, which is every run except the
    public-bids cases. The share is a property of the unit, so the months agree;
    the largest is taken so a unit absent from one month's audit is not diluted.
    """
    found = [o["state_share"] for o in (ownership_from_audit(p) for p in paths or []) if o]
    if not found:
        return None
    return pd.concat(found).groupby(level=0).max()


def combine_settlement_revenue(out: pd.DataFrame, *, priced_on_contract_value: bool) -> pd.DataFrame:
    """Add ``market_revenue_gbp`` and ``support_revenue_gbp`` to the annual frame.

    The compensation is the CfD share of the balancing settlement whenever the
    solve priced CfD bids on contract value, so counting both would pay the same
    curtailment twice (see the module docstring).
    """
    out = out.copy()
    out["market_revenue_gbp"] = out["wholesale_revenue_gbp"] + out["balancing_settlement_gbp"]
    dpa = out["dpa_support_gbp"] if "dpa_support_gbp" in out.columns else 0.0
    if priced_on_contract_value:
        _check_compensation_matches_balancing(out)
        out["support_revenue_gbp"] = out["cfd_support_gbp"] + out["roc_support_gbp"] + dpa
    else:
        out["support_revenue_gbp"] = (
            out["cfd_support_gbp"] + out["cfd_curtailment_compensation_gbp"] + out["roc_support_gbp"] + dpa
        )
    return out


def allocate_dpa_support(out: pd.DataFrame, dpa_support_gbp: float | None) -> pd.DataFrame:
    """Item B11: the bill's DPA line, paid to the DPA plant in proportion to its balancing dispatch.

    Net Zero Teesside Power is a register row (support label ``DPA``) whose DPA
    payments are on the bill through ``remaining_ccs``.  That line is the ECC
    full business case's operating budget, an upper bound that also covers the
    Exchequer-funded T&S support, so the rent this books is an upper bound too.
    """
    out = out.copy()
    if "dpa_support_gbp" not in out.columns:
        out["dpa_support_gbp"] = 0.0
    if not dpa_support_gbp:
        return out
    is_dpa = out["support_type"].fillna("").astype(str).str.upper().eq("DPA") & out["component_type"].eq("Generator")
    if not is_dpa.any():
        raise ValueError("the bill carries a DPA line but no generator is labelled DPA")
    energy = out.loc[is_dpa, "balancing_mwh"].clip(lower=0.0)
    weights = energy / energy.sum() if energy.sum() > 0 else pd.Series(1.0 / is_dpa.sum(), index=energy.index)
    out.loc[is_dpa, "dpa_support_gbp"] = float(dpa_support_gbp) * weights
    return out


def _check_compensation_matches_balancing(out: pd.DataFrame) -> None:
    """Fail if the recomputed compensation is not the CfD fleet's accepted bids.

    Under contract-value bids the two are the same quantity, so a material gap
    means a different strike table, a different reference series, or a change in
    how the solve prices CfD bids - any of which would make dropping the
    compensation from support revenue wrong. Offers are left out: they pay for
    turning up, which the compensation does not cover. A frame without the bid
    column falls back to the net settlement.

    Under item BM1 a unit's state share bids the public price instead of its
    contract value, so only the private share of the compensation is in the
    bids (P2, 25 Sep 2026: 181.7 of 1,542.6 m, matching the bids to 0.01%).
    The state share's public-price-plus-tie-step payment is left out of the
    expectation; it is a few thousand pounds against the 1% tolerance. A gap
    below ``COMPENSATION_RECONCILIATION_FLOOR_GBP`` passes on any ratio.
    """
    cfd = out.loc[out["support_type"].astype(str).eq("CfD")
                  & ~out["carrier"].astype(str).isin(COMPENSATION_CHECK_EXEMPT_CARRIERS)]
    private = 1.0 - cfd["state_share"].clip(0.0, 1.0) if "state_share" in cfd else 1.0
    compensation = float((cfd["cfd_curtailment_compensation_gbp"] * private).sum())
    column = "balancing_bid_gbp" if "balancing_bid_gbp" in cfd else "balancing_settlement_gbp"
    balancing = float(cfd[column].sum())
    if compensation <= 0.0:
        return
    gap = abs(compensation - balancing) / compensation
    if abs(compensation - balancing) <= COMPENSATION_RECONCILIATION_FLOOR_GBP:
        return
    if gap > COMPENSATION_RECONCILIATION_TOLERANCE:
        raise ValueError(
            "The balancing solve priced CfD bids on contract value, so the recomputed "
            f"privately owned curtailment compensation (GBP {compensation:,.0f}) should equal the CfD fleet's "
            f"accepted balancing bids (GBP {balancing:,.0f}), but they differ by {gap:.2%}, above "
            f"the {COMPENSATION_RECONCILIATION_TOLERANCE:.0%} tolerance. Check the strike table "
            "and the reference price basis before trusting the settlement."
        )


def vintaged_settlement_share(assets: pd.DataFrame) -> pd.Series:
    """Each vintaged-register row's share of its network asset's settlement (NaN without ``cohort_share``).

    ``cohort_share`` (``fleet_vintages.assign_build_years``) is a build-year cohort's
    share of the row it was split from; ``site_cohort_share`` (written by
    ``build_asset_finance_register.vintage_register``) is that row's share of the
    network asset after an existing-site split, 1.0 (or absent) when there was none.

    Args:
        assets: The register.

    Returns:
        ``cohort_share x site_cohort_share``, indexed as ``assets``.

    Raises:
        ValueError: The shares of one network asset add to more than one (a double count).
    """
    cohort = pd.to_numeric(assets["cohort_share"], errors="coerce")
    site = (pd.to_numeric(assets["site_cohort_share"], errors="coerce").fillna(1.0)
            if "site_cohort_share" in assets.columns else pd.Series(1.0, index=assets.index))
    share = cohort * site
    keyed = share.notna()
    if keyed.any():
        base = assets.loc[keyed, "asset_id"].astype(str).str.split("::", n=1).str[0]
        totals = share[keyed].groupby([assets.loc[keyed, "component_type"].astype(str), base]).sum()
        over = totals[totals > 1.0 + 1e-6]
        if len(over):
            raise ValueError(f"vintaged cohort shares of {len(over)} assets add to more than one "
                             f"(e.g. {over.head(3).round(6).to_dict()}): the settlement would be double counted")
    return share


def attach_settlement(assets: pd.DataFrame, settlement: pd.DataFrame) -> pd.DataFrame:
    """Join the annual settlement onto the (possibly cohort-split) asset register.

    A row ``asset::legacy`` or ``asset::new_build`` takes its cohort share of
    the asset's settlement (``cohort_legacy_share`` or its complement), so the
    two cohorts sum back to the asset exactly.

    A vintaged register (FES 2050 task Y10; the only registers with a
    ``cohort_share`` column) splits rows into build-year cohorts (``asset::b<year>``,
    or ``asset::legacy::b<year>`` below an existing-site split). Such a row takes
    :func:`vintaged_settlement_share` of the asset's settlement (task Y14-prep: the
    share had been 1.0 per cohort, counting the settlement once per build year). A
    register without ``cohort_share`` (every 2030 register) is split exactly as before.
    """
    required = {"component_type", "asset_id", "market_revenue_gbp", "support_revenue_gbp", "fuel_carbon_gbp"}
    if missing := required - set(settlement.columns):
        raise ValueError(f"settlement lacks columns: {sorted(missing)}")
    if SETTLEMENT_LEVEL_COLUMN in settlement.columns:
        return attach_register_row_settlement(assets, settlement)
    assets = assets.copy()
    base = assets["asset_id"].astype(str).str.split("::", n=1)
    assets["_base_id"] = base.str[0]
    assets["_cohort"] = base.str[1].fillna("")
    legacy_share = pd.to_numeric(
        assets["cohort_legacy_share"] if "cohort_legacy_share" in assets.columns
        else pd.Series(np.nan, index=assets.index), errors="coerce",
    )
    share = pd.Series(1.0, index=assets.index)
    share[assets["_cohort"].eq("legacy")] = legacy_share[assets["_cohort"].eq("legacy")]
    share[assets["_cohort"].eq("new_build")] = 1.0 - legacy_share[assets["_cohort"].eq("new_build")]
    if "cohort_share" in assets.columns:
        vintaged = vintaged_settlement_share(assets)
        share = share.where(vintaged.isna(), vintaged)
    if share.isna().any():
        raise ValueError("cohort rows need cohort_legacy_share to split the settlement")
    descriptive = [c for c in DESCRIPTIVE_COLUMNS if c in settlement.columns]
    columns = [c for c in settlement.columns
               if c not in ("component_type", "asset_id", "carrier", "support_type", "months", *descriptive)]
    keyed = settlement.set_index(["component_type", "asset_id"])
    joined = keyed.reindex(pd.MultiIndex.from_arrays([assets["component_type"], assets["_base_id"]]))
    # F27 item 11 (23 Oct 2026): a row the settlement carries under its own id (a second network
    # unit of one site, e.g. 'Drax Power Station (biomass units)::DRAX42') takes its own money,
    # not the site's first unit's: both Drax rows had carried the first unit's settlement.
    exact = keyed.reindex(pd.MultiIndex.from_arrays([assets["component_type"], assets["asset_id"].astype(str)]))
    own = (exact[columns].notna().all(axis=1) & assets["_cohort"].ne("").to_numpy()).to_numpy()
    if own.any():
        joined = joined.copy()
        joined.iloc[own] = exact.iloc[own]
        share[own] = 1.0
    matched = joined[columns].notna().all(axis=1).to_numpy()
    for column in columns:
        assets[column] = pd.to_numeric(joined[column], errors="coerce").to_numpy() * share.to_numpy()
    # The asset's bus and payee describe the asset, so a cohort row takes them as they are.
    for column in descriptive:
        assets[column] = joined[column].to_numpy()
    assets["settlement_status"] = np.where(matched, "settled", "no_settlement_record")
    assets.loc[~matched, columns] = 0.0
    return assets.drop(columns=["_base_id", "_cohort"])


# ── central dispatch (institution I3; P1/P2 under central dispatch, 20 Oct 2026) ──

#: Under central dispatch the real-time correction of a generator's output (its physical dispatch less
#: the central schedule) is paid at its own fuel and carbon rate out of the forecast-error charge that
#: replaces it, so owners neither gain nor lose by it (``realtime_at_cost``); the charge less what it
#: pays is the resource the borrowed charge stands for (``investor_ledger``).
DEFAULT_REALTIME_AT_COST = True
#: The I3 settlement's per-asset breakdown of ``balancing_settlement_gbp`` (GBP), written beside it.
CENTRAL_TRANSFER_COLUMNS = ("cost_recovery_gbp", "firm_access_margin_gbp", "firm_access_cfd_support_gbp",
                            "firm_access_roc_support_gbp", "storage_make_whole_gbp", "realtime_at_cost_gbp")


def central_month_units(units: pd.DataFrame, central_scenario: str) -> pd.DataFrame:
    """One month's per-unit FTI transfers (``fti_settle_year``'s units file), indexed by component and asset."""
    from scripts.market.fti_settlement import UNIT_TRANSFER_ROWS

    rows = units.loc[units["central_scenario"].astype(str).eq(str(central_scenario))]
    if rows.empty:
        raise ValueError(f"the FTI settlement's per-unit transfers carry no rows for {central_scenario}")
    out = rows.groupby(["component_type", "asset_id"])[list(UNIT_TRANSFER_ROWS)].sum()
    return out


def _central_month_settlement(central_wholesale_path, central_balancing_path, price: pd.Series,
                              month_units: pd.DataFrame, strike_table, reference_overrides,
                              roc_value_gbp_per_roc: float, volume_caps: dict | None = None,
                              negative_price_rule: str = "none", fuel_rates: pd.Series | None = None,
                              hydrogen_link_carriers=DEFAULT_HYDROGEN_LINK_CARRIERS,
                              realtime_at_cost: bool = DEFAULT_REALTIME_AT_COST) -> pd.DataFrame:
    """One month of the I3 asset settlement (FTI's method, the user's decisions of 14 and 20 Oct 2026).

    * wholesale: the central schedule (the central year's Stage 1) at the national price, the
      market twin's copperplate price (decision 2);
    * balancing settlement: the unit's own FTI transfers: cost recovery, firm access with the lost
      CfD top-up and ROC value (decision 1), the storage weekly make-whole (decision 3), and its
      real-time correction at its own fuel and carbon rate (:data:`DEFAULT_REALTIME_AT_COST`);
    * CfD support: difference payments on metered central output (decision 1), through the bill's
      calculator and strike table; ROCs on metered central output; the lost top-up is inside the
      balancing settlement, so ``cfd_curtailment_compensation_gbp`` reports it and is never added;
    * fuel and carbon on the central physical dispatch, as for the market year.
    """
    rows = []
    with xr.open_dataset(central_wholesale_path, engine="netcdf4") as dw, \
            xr.open_dataset(central_balancing_path, engine="netcdf4") as db:
        snapshots = pd.DatetimeIndex(db["snapshots_snapshot"].values)
        if not pd.DatetimeIndex(dw["snapshots_snapshot"].values).equals(snapshots):
            raise ValueError("Central wholesale and balancing snapshots do not match")
        if not snapshots.equals(pd.DatetimeIndex(price.index)):
            raise ValueError("The national price and the central-dispatch snapshots do not match")
        hours = pd.Series(db["snapshots_generators"].values, index=snapshots, dtype=float)
        coordinates = _bus_coordinates(db)
        for component_type, prefix in (("Generator", "generators"), ("StorageUnit", "storage_units")):
            names = pd.Index(db[f"{prefix}_i"].values)
            carrier = _text(_static(db, prefix, "carrier", names))
            keep = names[~carrier.str.lower().isin(NON_FINANCE_CARRIERS) & ~carrier.isin(DEMAND_FLEX_STORAGE_CARRIERS)]
            if not len(keep):
                continue
            carrier = carrier.reindex(keep)
            support = _text(_static(db, prefix, "support_type", keep))
            bus = _text(_static(db, prefix, "bus", keep))
            marginal_cost = pd.to_numeric(_static(db, prefix, "marginal_cost", keep, 0.0), errors="coerce").fillna(0.0)
            if component_type == "StorageUnit":
                fuel_rate = pd.Series(0.0, index=keep)
                fuel_basis = pd.Series(FUEL_BASIS_NONE, index=keep)
            elif fuel_rates is not None:
                found = keep.isin(fuel_rates.index)
                fuel_rate = pd.Series(np.where(found, fuel_rates.reindex(keep).fillna(0.0), marginal_cost.clip(lower=0.0)),
                                      index=keep)
                fuel_basis = pd.Series(np.where(found, FUEL_BASIS_RESOURCE, FUEL_BASIS_NETWORK), index=keep)
            else:
                fuel_rate = marginal_cost.clip(lower=0.0)
                fuel_basis = pd.Series(FUEL_BASIS_NETWORK, index=keep)
            # A central-dispatch year has no turn-down re-pricing: its physical dispatch is read as solved.
            p_b = _table(db, f"{prefix}_t_p", keep, snapshots)
            p_w = _table(dw, f"{prefix}_t_p", keep, snapshots)
            if not np.isfinite(p_b.to_numpy()).all() or not np.isfinite(p_w.to_numpy()).all():
                raise ValueError("Dispatch contains missing or non-finite values")
            energy_b = p_b.mul(hours, axis=0)
            energy_w = p_w.mul(hours, axis=0)
            transfers = (month_units.xs(component_type, level="component_type")
                         if component_type in month_units.index.get_level_values("component_type")
                         else pd.DataFrame(columns=month_units.columns, dtype=float))
            transfers = transfers.reindex(keep).fillna(0.0)
            if component_type == "Generator" and realtime_at_cost:
                realtime = (energy_b.clip(lower=0.0) - energy_w.clip(lower=0.0)).mul(fuel_rate, axis=1).sum()
            else:
                realtime = pd.Series(0.0, index=keep)
            frame = pd.DataFrame({
                "component_type": component_type,
                "asset_id": keep.astype(str),
                "carrier": carrier.to_numpy(),
                "support_type": support.to_numpy(),
                "payee_class": PAYEE_REGISTER,
                "bus": bus.to_numpy(),
                "bus_lat": coordinates["lat"].reindex(bus.to_numpy()).to_numpy(),
                "bus_lon": coordinates["lon"].reindex(bus.to_numpy()).to_numpy(),
                "wholesale_mwh": (energy_w.clip(lower=0.0) if component_type == "Generator" else energy_w).sum().to_numpy(),
                "balancing_mwh": (energy_b.clip(lower=0.0) if component_type == "Generator" else energy_b).sum().to_numpy(),
                "wholesale_revenue_gbp": energy_w.mul(price, axis=0).sum().to_numpy(),
                "cost_recovery_gbp": transfers["cost_recovery_gbp"].to_numpy(),
                "firm_access_margin_gbp": transfers["firm_access_margin_gbp"].to_numpy(),
                "firm_access_cfd_support_gbp": transfers["firm_access_cfd_support_gbp"].to_numpy(),
                "firm_access_roc_support_gbp": transfers["firm_access_roc_support_gbp"].to_numpy(),
                "storage_make_whole_gbp": transfers["storage_make_whole_gbp"].to_numpy(),
                "realtime_at_cost_gbp": realtime.reindex(keep).fillna(0.0).to_numpy(),
                "balancing_bid_gbp": 0.0,
                "fuel_carbon_gbp": energy_b.clip(lower=0.0).mul(fuel_rate, axis=1).sum().to_numpy(),
                "fuel_carbon_basis": fuel_basis.to_numpy(),
                "cfd_support_gbp": 0.0,
                # The lost CfD top-up firm access pays, inside the balancing settlement (never added).
                "cfd_curtailment_compensation_gbp": transfers["firm_access_cfd_support_gbp"].to_numpy(),
                "roc_support_gbp": 0.0,
                "dpa_support_gbp": 0.0,
            }, index=keep)
            frame["balancing_settlement_gbp"] = frame[list(CENTRAL_TRANSFER_COLUMNS)].sum(axis=1)
            if component_type == "Generator":
                cfd_names = keep[support.eq("CfD").to_numpy()]
                if len(cfd_names) and strike_table is not None:
                    generators = pd.DataFrame(index=cfd_names)
                    generators["support_type"] = "CfD"
                    for column in ("carrier", "bus", "cfd_round", "cfd_support_start", "cfd_technology"):
                        generators[column] = _text(_static(db, prefix, column, cfd_names))
                    network = SimpleNamespace(generators=generators, snapshots=snapshots,
                                              generators_t=SimpleNamespace(p=p_b[cfd_names]),
                                              snapshot_weightings=pd.DataFrame({"generators": hours}))
                    overrides = ({c: s.loc[snapshots] for c, s in reference_overrides.items()}
                                 if reference_overrides else None)
                    details = compute_cfd_payments(network, strike_table, reference_price=price,
                                                   reference_price_overrides=overrides, volume_caps=volume_caps,
                                                   negative_price_rule=negative_price_rule)
                    if not details.empty:
                        payments = details.set_index("generator")["net_cfd_cost"]
                        frame.loc[payments.index, "cfd_support_gbp"] = payments.to_numpy()
                roc_names = keep[support.eq("ROC").to_numpy()]
                if len(roc_names):
                    banding = pd.to_numeric(_static(db, prefix, "ro_banding", roc_names, 1.0), errors="coerce").fillna(1.0)
                    frame.loc[roc_names, "roc_support_gbp"] = (
                        energy_b[roc_names].clip(lower=0.0).sum() * banding * float(roc_value_gbp_per_roc)).to_numpy()
            rows.append(frame)
        # Hydrogen links: their central Stage 1 injection at the national price; no balancing payments
        # (their real-time moves are part of the correction the forecast-error charge replaces).
        hydrogen = _hydrogen_link_rows(dw, db, central_wholesale_path, snapshots, hours, price,
                                       pd.DataFrame(columns=["net_cost"]), coordinates, hydrogen_link_carriers)
        if hydrogen is not None:
            for column in CENTRAL_TRANSFER_COLUMNS:
                hydrogen[column] = 0.0
            rows.append(hydrogen)
        # Every paid unit must be a unit of this network (a non-finance unit, load shedding, is left out
        # of the register here as its balancing payments are in the market year).
        on_network = pd.MultiIndex.from_tuples(
            [("Generator", str(n)) for n in db["generators_i"].values]
            + [("StorageUnit", str(n)) for n in (db["storage_units_i"].values if "storage_units_i" in db else [])])
    month = pd.concat(rows, ignore_index=True)
    stray = month_units.index.difference(on_network)
    if len(stray):
        raise ValueError(f"FTI transfers paid to {len(stray)} units this network does not carry (e.g. {list(stray[:3])}): "
                         "the settlement and the central-dispatch months are not the same build")
    return month


def settle_assets_central_dispatch(scenarios, central_scenarios, market_wholesale_paths, central_wholesale_paths,
                                   central_balancing_paths, price_paths, units: pd.DataFrame, *, strike_table,
                                   reference_price_basis: dict | None, roc_value_gbp_per_roc: float,
                                   volume_caps: dict | None = None, negative_price_rule: str = "none",
                                   marginal_cost_breakdown: pd.DataFrame | None = None,
                                   dpa_support_gbp: float | None = None,
                                   hydrogen_link_carriers=DEFAULT_HYDROGEN_LINK_CARRIERS,
                                   realtime_at_cost: bool = DEFAULT_REALTIME_AT_COST) -> pd.DataFrame:
    """The I3 annual settlement per asset: the twin run group's (P1/P2 under central dispatch, 20 Oct 2026).

    ``scenarios`` are the market months (whose Stage 1 networks and copperplate prices,
    ``market_wholesale_paths`` and ``price_paths``, give the national price and the demand the
    baseload reference is weighted on, exactly as the bill's); ``central_scenarios`` the
    central-dispatch months, paired month by month, with their Stage 1 and Stage 2 networks;
    ``units`` the year FTI settlement's per-unit transfers (``fti_settle_year``'s units file).
    Returns :data:`SETTLEMENT_COLUMNS`, with the per-asset breakdown of the balancing
    settlement in ``attrs["transfers"]``.
    """
    lengths = {len(scenarios), len(central_scenarios), len(market_wholesale_paths), len(central_wholesale_paths),
               len(central_balancing_paths), len(price_paths)}
    if len(lengths) != 1 or not scenarios:
        raise ValueError("central-dispatch settlement inputs must be non-empty and aligned month by month")

    def _links(path):
        candidate = Path(str(path).replace("_wholesale.nc", "_wholesale_links.csv"))
        return candidate if candidate.exists() else None

    frames = [read_hourly_inputs(w, p, _links(w)) for w, p in zip(market_wholesale_paths, price_paths)]
    hourly = pd.concat(frames).sort_index()
    overrides = None
    carriers = [str(c) for c in ((reference_price_basis or {}).get("baseload_carriers") or [])]
    if carriers:
        series, _ = seasonal_demand_weighted_reference(hourly)
        overrides = {c: series for c in carriers}
    fuel_rates = resource_fuel_carbon_rates(marginal_cost_breakdown) if marginal_cost_breakdown is not None else None
    months = []
    for scenario, central, w, b, frame in zip(scenarios, central_scenarios, central_wholesale_paths,
                                             central_balancing_paths, frames):
        month = _central_month_settlement(w, b, frame["price_gbp_per_mwh"], central_month_units(units, central),
                                          strike_table, overrides, roc_value_gbp_per_roc, volume_caps,
                                          negative_price_rule=negative_price_rule, fuel_rates=fuel_rates,
                                          hydrogen_link_carriers=hydrogen_link_carriers,
                                          realtime_at_cost=realtime_at_cost)
        month["scenario"] = scenario
        months.append(month)
    annual = pd.concat(months, ignore_index=True)
    numeric = ["wholesale_mwh", "balancing_mwh", "wholesale_revenue_gbp", "balancing_settlement_gbp",
               "balancing_bid_gbp", "cfd_support_gbp", "cfd_curtailment_compensation_gbp", "roc_support_gbp",
               "dpa_support_gbp", "fuel_carbon_gbp", *CENTRAL_TRANSFER_COLUMNS]
    out = annual.groupby(["component_type", "asset_id"], as_index=False).agg(
        carrier=("carrier", "first"), support_type=("support_type", "first"),
        **{column: (column, "first") for column in DESCRIPTIVE_COLUMNS},
        **{column: (column, "sum") for column in numeric}, months=("scenario", "nunique"),
    )
    out = allocate_dpa_support(out, dpa_support_gbp)
    # The lost top-up is inside the balancing settlement (firm access), so it is reported, never added.
    out["market_revenue_gbp"] = out["wholesale_revenue_gbp"] + out["balancing_settlement_gbp"]
    out["support_revenue_gbp"] = out["cfd_support_gbp"] + out["roc_support_gbp"] + out["dpa_support_gbp"]
    result = out[SETTLEMENT_COLUMNS].copy()
    result.attrs["transfers"] = out[["component_type", "asset_id", "carrier", *CENTRAL_TRANSFER_COLUMNS]].copy()
    result.attrs["realtime_at_cost"] = bool(realtime_at_cost)
    return result


# ── the 2050 contract book (task SETTLE2050) ─────────────────────────────────

#: Marks a settlement written per register row (a vintaged group with a contract book).
SETTLEMENT_LEVEL_COLUMN = "settlement_level"
SETTLEMENT_LEVEL_REGISTER_ROW = "register_row"
#: Columns a register-row settlement carries after :data:`SETTLEMENT_COLUMNS`.
REGISTER_ROW_COLUMNS = [SETTLEMENT_LEVEL_COLUMN, "network_asset_id", "settlement_share", "cfd_paid_mwh"]
#: Settlement amounts a register row takes from the book, not from its network asset's share.
BOOK_SUPPORT_COLUMNS = ("cfd_support_gbp", "cfd_curtailment_compensation_gbp", "dpa_support_gbp")


def _period_share(frames) -> float:
    """The settled hours over the calendar year's (1 for a full year)."""
    hourly = pd.concat(frames)
    year = int(pd.DatetimeIndex(hourly.index).year.min())
    days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    return float(hourly["hours"].sum()) / (24.0 * days)


def _dpa_reference(dpa_prices: dict | None, params: dict):
    from scripts.public_ownership.support_regime_2050 import _value, reference_ccgt_srmc

    prices = params["commodity_prices"]
    gas = float((dpa_prices or {}).get("gas", _value(prices["gas_gbp_per_mwh_thermal"])))
    carbon = float((dpa_prices or {}).get("carbon", _value(prices["carbon_gbp_per_tco2"])))
    return float(reference_ccgt_srmc(gas, carbon, params)), {"gas_gbp_per_mwh_thermal": gas, "carbon_gbp_per_tco2": carbon}


def settle_contract_book(wholesale_paths, balancing_paths, frames, book: pd.DataFrame, *, strike_table, overrides,
                         volume_caps, negative_price_rule: str, audits, marginal_cost_breakdown=None,
                         dpa_prices: dict | None = None, bm_strike_table=None) -> dict:
    """Settle the book's CfD and DPA units over the months (task SETTLE2050).

    CfD units: their network asset's balancing dispatch x ``energy_factor`` through
    ``compute_cfd_payments`` with the bill's strike table, reference series and negative
    pricing rule (the same call as the bill's CfD line, so the two agree to the pound);
    curtailment compensation on the same units with the formula
    :func:`_month_settlement` applies to a labelled network. DPA units: per network asset,
    the year's output, variable top-up, market revenue and short-run cost
    (``contract_book_2050.dpa_energy_terms``; gas CCS at its build-time resource cost,
    hydrogen turbines at ``(bus0 Stage 1 price + link cost) / efficiency``, the model's own
    dispatch cost), then each unit's contract (``contract_book_2050.settle_dpa_book``).

    ``bm_strike_table`` (FES 2050 decision 41; None: ``strike_table``) values the curtailment
    compensation at the strikes the solved BM bid on, so it still reconciles against the
    accepted bids when the bill's contract strikes differ. Each unit's ``cfd_paid_mwh`` is the
    energy its difference payment accrues on (the change in its payment per GBP of strike:
    the hours the negative-pricing rule leaves paid, after any collar), which the break-even
    calibration divides by.

    Returns:
        ``{"cfd": per-unit frame, "dpa": per-unit frame, "period_share": float}``.
    """
    from scripts.public_ownership import contract_book_2050 as cb
    from scripts.public_ownership.support_regime_2050 import _value, load_parameters

    params = load_parameters()
    reference, used_prices = _dpa_reference(dpa_prices, params)
    breakdown_mc = (pd.to_numeric(marginal_cost_breakdown.drop_duplicates("generator").set_index("generator")
                                  ["marginal_cost"], errors="coerce")
                    if marginal_cost_breakdown is not None else None)
    margin = _value(params["dpa"]["variable_payment_margin_gbp_per_mwh"])
    dpa_book = book.loc[book["scheme"].eq(cb.SCHEME_DPA)]
    dpa_generators = pd.Index(dpa_book.loc[dpa_book["component_type"].eq("Generator"), "network_asset"].unique())
    dpa_links = pd.Index(dpa_book.loc[dpa_book["component_type"].eq("Link"), "network_asset"].unique())
    cfd_parts, terms = [], []
    for w, b, frame, audit in zip(wholesale_paths, balancing_paths, frames, audits):
        price = frame["price_gbp_per_mwh"]
        with xr.open_dataset(w, engine="netcdf4") as dw, xr.open_dataset(b, engine="netcdf4") as db:
            snapshots = pd.DatetimeIndex(db["snapshots_snapshot"].values)
            if not snapshots.equals(pd.DatetimeIndex(price.index)):
                raise ValueError("Price series and network snapshots do not match")
            hours = pd.Series(db["snapshots_generators"].values, index=snapshots, dtype=float)
            names = pd.Index(db["generators_i"].values)
            labelled = _text(_static(db, "generators", "support_type", names))
            if labelled.isin(["CfD", "DPA", "ROC"]).any():
                raise ValueError("the network carries support labels and a contract book was given: one source only")
            static = pd.DataFrame({"carrier": _text(_static(db, "generators", "carrier", names)),
                                   "bus": _text(_static(db, "generators", "bus", names))}, index=names)
            assets = pd.Index(book.loc[book["component_type"].eq("Generator"), "network_asset"].unique())
            p_b = _table(db, "generators_t_p", assets.intersection(names), snapshots)
            p_w = _table(dw, "generators_t_p", assets.intersection(names), snapshots)
            units, units_b = cb.cfd_units(book, static, p_b)
            if len(units):
                _, units_w = cb.cfd_units(book, static, p_w)
                network = SimpleNamespace(generators=units, snapshots=snapshots,
                                          generators_t=SimpleNamespace(p=units_b),
                                          snapshot_weightings=pd.DataFrame({"generators": hours}))
                month_overrides = ({c: s.loc[snapshots] for c, s in overrides.items()} if overrides else None)
                details = compute_cfd_payments(network, strike_table, reference_price=price,
                                               reference_price_overrides=month_overrides, volume_caps=volume_caps,
                                               negative_price_rule=negative_price_rule).set_index("generator")
                if len(details) != len(units):
                    raise ValueError(f"{len(units) - len(details)} contract-book CfD units found no strike")
                strikes = details["strike_price"].reindex(units.index)
                # Decision 41: the change in each payment per GBP of strike (it is linear in the strike).
                probe = compute_cfd_payments(network, strike_table.adjusted({k: -1.0 for k in strike_table.strikes},
                                                                            "paid_energy_probe"),
                                             reference_price=price, reference_price_overrides=month_overrides,
                                             volume_caps=volume_caps,
                                             negative_price_rule=negative_price_rule).set_index("generator")
                paid_mwh = (probe["net_cfd_cost"] - details["net_cfd_cost"]).reindex(units.index).to_numpy()
                if bm_strike_table is not None:
                    keys = [bm_strike_table.resolve(c, r, t) for c, r, t in
                            zip(units["carrier"], units["cfd_round"], contract_technology(units))]
                    if any(k is None or fb for k, fb in keys):
                        raise ValueError("a contract-book CfD unit has no entry in the BM strike table")
                    bid_strikes = pd.Series([bm_strike_table.strikes[k] for k, _ in keys], index=units.index)
                else:
                    bid_strikes = strikes
                turn_down = (units_w - units_b).clip(lower=0.0).mul(hours, axis=0)
                bid_basis = bid_basis_from_audit(audit)
                premium = (np.broadcast_to(bid_strikes.to_numpy()[None, :], turn_down.shape).copy()
                           if bid_basis == "strike"
                           else contract_value_rate(bid_strikes.to_numpy()[None, :], price.to_numpy()[:, None],
                                                    contract_value_floor_from_audit(audit)))
                bid_rule = negative_price_rule_from_audit(audit)
                if bid_rule != "none":
                    premium = premium * negative_price_eligibility(price.to_numpy(), units["cfd_round"], bid_rule)
                cfd_parts.append(pd.DataFrame({
                    "unit": units.index, "cfd_support_gbp": details["net_cfd_cost"].reindex(units.index).to_numpy(),
                    "cfd_curtailment_compensation_gbp": (turn_down.to_numpy() * premium).sum(axis=0),
                    "cfd_settled_mwh": units_b.mul(hours, axis=0).sum().to_numpy(),
                    "cfd_paid_mwh": paid_mwh,
                    "strike_gbp_per_mwh_2024": strikes.to_numpy()}))
            # DPA energy terms, per network asset.
            if len(dpa_generators):
                present = dpa_generators.intersection(names)
                energy = _table(db, "generators_t_p", present, snapshots).clip(lower=0.0).mul(hours, axis=0)
                mc = pd.to_numeric(_static(db, "generators", "marginal_cost", present, 0.0), errors="coerce").fillna(0.0)
                # The build-time marginal cost (fuel + carbon + VOM: what the plant dispatches
                # on), never a balancing floor; the network's where the breakdown has no row.
                srmc_rate = (breakdown_mc.reindex(present).fillna(mc) if breakdown_mc is not None else mc)
                srmc = pd.DataFrame(np.broadcast_to(srmc_rate.to_numpy()[None, :], energy.shape),
                                    index=snapshots, columns=present)
                terms.append(cb.dpa_energy_terms(energy, price, srmc, reference, margin))
            if len(dpa_links):
                lk = pd.Index(db["links_i"].values)
                present = dpa_links.intersection(lk)
                bus0 = _text(_static(db, "links", "bus0", present))
                eff = pd.to_numeric(_static(db, "links", "efficiency", present, 1.0), errors="coerce")
                mc = pd.to_numeric(_static(db, "links", "marginal_cost", present, 0.0), errors="coerce").fillna(0.0)
                if not (eff > 0).all():
                    raise ValueError("DPA hydrogen turbines need a positive efficiency")
                p0 = _table(db, "links_t_p0", present, snapshots)
                delivery = p0.clip(lower=0.0).mul(eff, axis=1).mul(hours, axis=0)
                h2 = _table_strict(dw, "buses_t_marginal_price", pd.Index(bus0.unique()), snapshots,
                                   "the Stage 1 hydrogen bus price")
                lam = pd.DataFrame(h2[bus0.to_numpy()].to_numpy(), index=snapshots, columns=present)
                srmc = lam.add(mc, axis=1).div(eff, axis=1)
                terms.append(cb.dpa_energy_terms(delivery, price, srmc, reference, margin))
    cfd = (pd.concat(cfd_parts).groupby("unit", as_index=False).agg(
        cfd_support_gbp=("cfd_support_gbp", "sum"), cfd_curtailment_compensation_gbp=("cfd_curtailment_compensation_gbp", "sum"),
        cfd_settled_mwh=("cfd_settled_mwh", "sum"), cfd_paid_mwh=("cfd_paid_mwh", "sum"),
        strike_gbp_per_mwh_2024=("strike_gbp_per_mwh_2024", "first"))
           if cfd_parts else pd.DataFrame(columns=["unit", "cfd_support_gbp", "cfd_curtailment_compensation_gbp",
                                                   "cfd_settled_mwh", "cfd_paid_mwh", "strike_gbp_per_mwh_2024"]))
    annual_terms = (pd.concat(terms).groupby(level=0).sum() if terms
                    else pd.DataFrame(columns=list(cb.ENERGY_TERMS), dtype=float))
    share = _period_share(frames)
    dpa = cb.settle_dpa_book(book, annual_terms, period_share=share)
    if len(dpa):
        dpa["reference_ccgt_srmc_gbp_per_mwh"] = reference
        dpa["dpa_prices"] = str(used_prices)
    return {"cfd": cfd, "dpa": dpa, "period_share": share}


def register_row_settlement(network_rows: pd.DataFrame, register: pd.DataFrame, cfd_units: pd.DataFrame,
                            dpa_units: pd.DataFrame) -> pd.DataFrame:
    """One settlement row per register row: its vintaged share of its network asset, its own support.

    Every amount is the network asset's x the row's :func:`vintaged_settlement_share`
    (what :func:`attach_settlement` does for a vintaged register), except
    :data:`BOOK_SUPPORT_COLUMNS`, which are the row's own from the book (zero for a row
    holding no contract: a merchant cohort of a CfD asset is paid no CfD).

    Raises:
        ValueError: The network rows carry CfD or DPA support already (one source only).
    """
    for column in BOOK_SUPPORT_COLUMNS:
        if network_rows[column].abs().sum() > 0.0:
            raise ValueError(f"network-level {column} is non-zero and a contract book was given: one source only")
    share = vintaged_settlement_share(register)
    base = register["asset_id"].astype(str).str.split("::", n=1).str[0]
    share = share.where(share.notna(), 1.0)
    keyed = network_rows.set_index(["component_type", "asset_id"])
    joined = keyed.reindex(pd.MultiIndex.from_arrays([register["component_type"].astype(str), base]))
    matched = joined["wholesale_revenue_gbp"].notna().to_numpy()
    out = pd.DataFrame({"component_type": register["component_type"].astype(str).to_numpy(),
                        "asset_id": register["asset_id"].astype(str).to_numpy(),
                        "carrier": register["carrier"].astype(str).to_numpy(),
                        "support_type": register["support_type"].fillna("").astype(str).to_numpy(),
                        "network_asset_id": base.to_numpy(), "settlement_share": share.to_numpy()})
    for column in DESCRIPTIVE_COLUMNS:
        out[column] = joined[column].to_numpy()
    numeric = ["wholesale_mwh", "balancing_mwh", "wholesale_revenue_gbp", "balancing_settlement_gbp",
               "balancing_bid_gbp", "roc_support_gbp", "fuel_carbon_gbp", "cfd_support_gbp",
               "cfd_curtailment_compensation_gbp", "dpa_support_gbp"]
    for column in numeric:
        out[column] = pd.to_numeric(joined[column], errors="coerce").fillna(0.0).to_numpy() * share.to_numpy()
    out["months"] = joined["months"].to_numpy()
    if "state_share" in joined.columns:
        out["state_share"] = pd.to_numeric(joined["state_share"], errors="coerce").fillna(0.0).to_numpy()
    out = out.loc[matched].reset_index(drop=True)
    by_unit = out.set_index("asset_id")
    for column in ("cfd_support_gbp", "cfd_curtailment_compensation_gbp"):
        by_unit[column] = cfd_units.set_index("unit")[column].reindex(by_unit.index).fillna(0.0) \
            if len(cfd_units) else 0.0
    # Decision 41 (b): the energy each row's difference payment accrues on (zero off contract).
    by_unit["cfd_paid_mwh"] = (cfd_units.set_index("unit")["cfd_paid_mwh"].reindex(by_unit.index).fillna(0.0)
                               if len(cfd_units) and "cfd_paid_mwh" in cfd_units.columns else 0.0)
    by_unit["dpa_support_gbp"] = (dpa_units.set_index("unit")["net_support_cost_gbp"].reindex(by_unit.index).fillna(0.0)
                                  if len(dpa_units) else 0.0)
    missing_units = set(cfd_units["unit"]) - set(by_unit.index) if len(cfd_units) else set()
    missing_units |= set(dpa_units["unit"]) - set(by_unit.index) if len(dpa_units) else set()
    if missing_units:
        raise ValueError(f"{len(missing_units)} contract units have no settled register row (e.g. {sorted(missing_units)[:3]})")
    out = by_unit.reset_index()
    out[SETTLEMENT_LEVEL_COLUMN] = SETTLEMENT_LEVEL_REGISTER_ROW
    return out


def combine_register_rows(rows: pd.DataFrame, book: pd.DataFrame, *, priced_on_contract_value: bool) -> pd.DataFrame:
    """:func:`combine_settlement_revenue` for register rows, with the compensation check on whole-CfD assets.

    Under contract-value bids the 2050 BM bids each network asset the capacity-weighted
    blend of its cohorts' contracts (``support_bids_2050``), so an asset's accepted bid cost
    carries its merchant and DPA cohorts' bids too and equals the CfD units' compensation
    only where the asset is all CfD. The check runs on those assets (``energy_factor`` of
    its CfD units summing to one); the rest are reported, not checked (declared).
    """
    out = rows.copy()
    out["market_revenue_gbp"] = out["wholesale_revenue_gbp"] + out["balancing_settlement_gbp"]
    support = out["cfd_support_gbp"] + out["roc_support_gbp"] + out["dpa_support_gbp"]
    if priced_on_contract_value:
        cfd = book.loc[book["scheme"].eq("CfD")]
        factor = pd.to_numeric(cfd["energy_factor"], errors="coerce").groupby(cfd["network_asset"]).sum()
        whole = set(factor.index[factor >= 1.0 - 1e-6])
        check = out.loc[out["network_asset_id"].isin(whole) & out["support_type"].eq("CfD")]
        if len(check):
            _check_compensation_matches_balancing(check)
        out["support_revenue_gbp"] = support
    else:
        out["support_revenue_gbp"] = support + out["cfd_curtailment_compensation_gbp"]
    return out


def attach_register_row_settlement(assets: pd.DataFrame, settlement: pd.DataFrame) -> pd.DataFrame:
    """Join a register-row settlement (:func:`register_row_settlement`) onto the register, row for row."""
    if not settlement[SETTLEMENT_LEVEL_COLUMN].eq(SETTLEMENT_LEVEL_REGISTER_ROW).all():
        raise ValueError("a settlement mixes register rows and network assets")
    keyed = settlement.set_index(["component_type", "asset_id"])
    if keyed.index.has_duplicates:
        raise ValueError("the register-row settlement repeats a row")
    descriptive = [c for c in DESCRIPTIVE_COLUMNS if c in settlement.columns]
    skip = {"component_type", "asset_id", "carrier", "support_type", "months", "network_asset_id",
            "settlement_share", SETTLEMENT_LEVEL_COLUMN, *descriptive}
    columns = [c for c in settlement.columns if c not in skip]
    assets = assets.copy()
    joined = keyed.reindex(pd.MultiIndex.from_arrays([assets["component_type"].astype(str),
                                                      assets["asset_id"].astype(str)]))
    matched = joined["market_revenue_gbp"].notna().to_numpy()
    for column in columns:
        assets[column] = pd.to_numeric(joined[column], errors="coerce").fillna(0.0).to_numpy()
    for column in descriptive:
        assets[column] = joined[column].to_numpy()
    assets["settlement_status"] = np.where(matched, "settled", "no_settlement_record")
    return assets
