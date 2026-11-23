"""Decompose a public-ownership balancing run against its matched baseline (item BM1).

**This is the deliverable, not a check to run afterwards.** When a state-owned unit
on an at-cost contract bids its avoided short-run cost instead of its contract
value, the balancing mechanism turns it down in preference to expensive private
contracts and stops paying it much to do so.

The obvious guess is that most of the fall in accepted bid cost is therefore the
state paying itself, which ``surplus_recycling: public_cfd_at_cost`` would route
straight back to the bill. **The seven-day probe of 22 September says otherwise**:
only 18.7% of the fall was a transfer to the state, and 81.3% was money that
stopped reaching private counterparties, because the state absorbed the
curtailment those units would have been paid for. Do not assume either way --
that is what this script is for.

The second term is the one that decides how much of it is real. A private unit
turned down less DELIVERS more, and a CfD unit earns ``strike - reference`` per
delivered MWh, the very rate it was bidding. So part of what leaves BSUoS
re-appears on the CfD line, and the consumer saving is the residue. On the week
that offset gave back 4.71m of an 11.75m fall.

That is the shape the WP-D double count had on 19 September, so the fall is split
before any of it is called a saving:

* **transfer to the state** -- the change in bid cost on the state share of each
  unit. It leaves BSUoS and returns through the public account; it is not a
  consumer saving and must never be quoted as one.
* **payment to private counterparties** -- the change in bid cost on the private
  share. This one *is* a consumer saving (or cost), because the money leaves the
  consumer and reaches someone outside the public account.
* **resource cost** -- the change in fuel, carbon and variable O&M from any
  change in the physical dispatch. Renewable-for-renewable substitution moves no
  resource cost; thermal substitution does.

A CfD unit is close to indifferent between being turned down and generating,
because it is paid ``strike - reference`` either way, so the volume check below
matters: if total turn-down barely moves, the exercise is about *who* is paid,
not *how much* energy is spilled.

    python scripts/market/decompose_public_bm_bids.py BASE_SCENARIO PUBLIC_SCENARIO \\
        --shares resources/public_ownership/cp30_2030_cp30_network_generator_state_shares.csv \\
        --case public_ownership
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.ownership_shares import load_shares  # noqa: E402

# F20 parity (14 Oct 2026): a parity run reads the solved months from another tree
# (PYPSA_GB_INPUT_RESOURCES, set by scripts/public_ownership/f20_parity/run_parity.py); unset, this tree's.
RESOURCES_ROOT = Path(os.environ.get("PYPSA_GB_INPUT_RESOURCES") or PROJECT_ROOT / "resources")
RESOURCES = RESOURCES_ROOT / "market"
#: constraint_costs.csv carries a TOTAL row as well as the per-carrier rows.
#: Summing the file without dropping it double counts the whole table, which is
#: the 18 September audit finding; the per-generator file has no such row.
TOTAL_ROW = "TOTAL"


def _redispatch(scenario: str) -> pd.DataFrame:
    path = RESOURCES / f"{scenario}_redispatch_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"no redispatch summary for {scenario}: {path}")
    # On the month's turn-down basis (pro rata zonal re-sharing where its network's headline basis is that;
    # scripts/market/turn_down_allocation.py): the base and public months of a group share their network.
    from scripts.market.turn_down_allocation import read_redispatch_in_force

    frame = read_redispatch_in_force(path)
    frame = frame.loc[~frame["component"].astype(str).eq(TOTAL_ROW)]
    return frame.set_index("component")


def _constraint_costs(scenario: str) -> pd.DataFrame:
    path = RESOURCES / f"{scenario}_constraint_costs.csv"
    frame = pd.read_csv(path)
    return frame.loc[~frame["carrier"].astype(str).eq(TOTAL_ROW)]


def _contract_value_rate(scenario: str, units) -> pd.Series:
    """Mean ``max(strike - reference, 0)`` per CfD unit, from the baseline audit.

    This is both the balancing bid a CfD unit makes and the difference payment it
    earns per delivered MWh, which is why a CfD unit is close to indifferent
    between the two -- and why the offset below exists at all.
    """
    path = RESOURCES / f"{scenario}_contract_value_bids.csv"
    if not path.exists():
        return pd.Series(0.0, index=units, dtype=float)
    audit = pd.read_csv(path).set_index("generator")
    audit = audit.loc[audit["pricing"].astype(str).eq("cfd_contract_value")]
    return pd.to_numeric(audit["mean_bid_gbp_per_mwh"], errors="coerce").reindex(units).fillna(0.0)


def _tie_break_steps(scenario: str, units) -> pd.Series:
    """Decision 20: the share-weighted tie step inside each unit's bid, 0 where none."""
    path = RESOURCES / f"{scenario}_contract_value_bids.csv"
    if not path.exists():
        return pd.Series(0.0, index=units, dtype=float)
    audit = pd.read_csv(path)
    if "tie_break_step_gbp_per_mwh" not in audit.columns:
        return pd.Series(0.0, index=units, dtype=float)
    return pd.to_numeric(audit.set_index("generator")["tie_break_step_gbp_per_mwh"],
                         errors="coerce").reindex(units).fillna(0.0)


def _net_bid_costs(scenario: str, frame: pd.DataFrame, prefix: str) -> float:
    """Take the tie steps out of accepted bid cost in place; return what they came to.

    The steps only order a tie that the state is indifferent to, so they are an
    ordering device, not money: every total is reported net of them.
    """
    steps = _tie_break_steps(scenario, frame.index)
    step_cost = steps * frame[f"{prefix}_decrease_MWh"]
    frame[f"{prefix}_bid_cost"] = frame[f"{prefix}_bid_cost"] - step_cost
    return float(step_cost.sum())


def decompose(base_scenario: str, public_scenario: str, shares: pd.Series) -> dict:
    """Split the change in accepted bid cost into transfer, private payment and volume."""
    base, public = _redispatch(base_scenario), _redispatch(public_scenario)
    units = base.index.union(public.index)
    frame = pd.DataFrame(index=units)
    for name, source in (("base", base), ("public", public)):
        for column in ("bid_cost", "offer_cost", "decrease_MWh", "increase_MWh"):
            frame[f"{name}_{column}"] = pd.to_numeric(
                source[column], errors="coerce").reindex(units).fillna(0.0)
    base_steps = _net_bid_costs(base_scenario, frame, "base")
    public_steps = _net_bid_costs(public_scenario, frame, "public")
    frame["carrier"] = base["carrier"].reindex(units).fillna(public["carrier"].reindex(units))
    frame["state_share"] = shares.reindex(units).fillna(0.0).clip(0.0, 1.0)

    frame["delta_bid_cost"] = frame["public_bid_cost"] - frame["base_bid_cost"]
    frame["delta_offer_cost"] = frame["public_offer_cost"] - frame["base_offer_cost"]
    frame["delta_decrease_MWh"] = frame["public_decrease_MWh"] - frame["base_decrease_MWh"]
    # The split: money moving on the state share never leaves the public sector.
    frame["delta_to_state"] = frame["delta_bid_cost"] * frame["state_share"]
    frame["delta_to_private"] = frame["delta_bid_cost"] * (1.0 - frame["state_share"])

    total_delta = float(frame["delta_bid_cost"].sum())
    to_state = float(frame["delta_to_state"].sum())
    to_private = float(frame["delta_to_private"].sum())
    result = {
        "base_scenario": base_scenario,
        "public_scenario": public_scenario,
        "base_bid_cost_gbp": float(frame["base_bid_cost"].sum()),
        "public_bid_cost_gbp": float(frame["public_bid_cost"].sum()),
        "delta_bid_cost_gbp": total_delta,
        "delta_offer_cost_gbp": float(frame["delta_offer_cost"].sum()),
        # The whole point of the exercise.
        "transfer_to_the_state_gbp": to_state,
        "payment_to_private_counterparties_gbp": to_private,
        "transfer_share_of_delta": (to_state / total_delta) if total_delta else None,
        # If this is near zero the network set the turn-down and only its
        # composition moved, which is what B1 found for the contract-value stack.
        "base_decrease_MWh": float(frame["base_decrease_MWh"].sum()),
        "public_decrease_MWh": float(frame["public_decrease_MWh"].sum()),
        "delta_decrease_MWh": float(frame["delta_decrease_MWh"].sum()),
        "state_owned_units": int((frame["state_share"] > 0).sum()),
        "units": int(len(frame)),
        # Decision 20: what the tie steps came to, already taken out of the bid costs.
        "tie_break_step_cost_gbp": public_steps + base_steps,
    }
    # ── the offset that decides whether any of this is a consumer saving ──
    # A private CfD unit turned down LESS delivers more, and on that extra
    # delivery it earns strike - reference through the CfD line: the same rate it
    # was bidding. So what leaves BSUoS partly re-appears as policy cost, and the
    # consumer saving is the residue. Under an at-cost contract the state side has
    # no matching entry, because its required revenue is recovered whatever the
    # volume -- a smaller volume simply raises the unit strike. Under a fixed strike
    # (cfd_strike_basis: auction_clearing, as in the treasury_first preset) the
    # state is settled on its delivered volume like any CfD holder, so it loses
    # strike - reference on every MWh the stack moves off it: the CfD line falls by
    # that much and the state's own balance takes it. Reported, never netted here:
    # the bill merge picks the terms that apply to each case's strike basis.
    frame["delta_delivered_MWh"] = (
        (frame["public_increase_MWh"] - frame["public_decrease_MWh"])
        - (frame["base_increase_MWh"] - frame["base_decrease_MWh"]))
    rate = _contract_value_rate(base_scenario, frame.index)
    frame["delta_cfd_settlement_private"] = (
        (1.0 - frame["state_share"]) * frame["delta_delivered_MWh"] * rate)
    cfd_offset = float(frame["delta_cfd_settlement_private"].sum())
    result["offsetting_private_cfd_settlement_gbp"] = cfd_offset
    result["net_consumer_change_private_side_gbp"] = to_private + cfd_offset
    result["private_delivered_MWh_change"] = float(
        ((1.0 - frame["state_share"]) * frame["delta_delivered_MWh"]).sum())
    result["state_delivered_MWh_change"] = float(
        (frame["state_share"] * frame["delta_delivered_MWh"]).sum())
    frame["delta_cfd_settlement_state_fixed_strike"] = (
        frame["state_share"] * frame["delta_delivered_MWh"] * rate)
    result["state_cfd_settlement_at_fixed_strike_gbp"] = float(
        frame["delta_cfd_settlement_state_fixed_strike"].sum())

    by_carrier = frame.groupby("carrier")[
        ["delta_bid_cost", "delta_to_state", "delta_to_private", "delta_decrease_MWh"]].sum()
    by_carrier = by_carrier.loc[by_carrier.abs().sum(axis=1) > 1.0].sort_values("delta_bid_cost")
    result["by_carrier"] = by_carrier.round(2).to_dict(orient="index")
    return result, frame


# The household conversion, and it is VAT-basis-dependent, which is easy to get
# wrong. The projector recovers 98,879,614 GBP of system cost per 1 GBP on the
# ex-VAT household bill, so 1m GBP is 0.010113 ex VAT and 0.010619 inc VAT. Every
# headline in this programme is quoted inc VAT, so that is the default here, but
# the ex-VAT figure is kept because the bid cost itself is an ex-VAT system flow.
SYSTEM_GBP_PER_HOUSEHOLD_GBP_EX_VAT = 98_879_614.0
VAT_RATE = 0.05
GBP_PER_HOUSEHOLD_PER_MILLION_EX_VAT = 1e6 / SYSTEM_GBP_PER_HOUSEHOLD_GBP_EX_VAT
GBP_PER_HOUSEHOLD_PER_MILLION = GBP_PER_HOUSEHOLD_PER_MILLION_EX_VAT * (1.0 + VAT_RATE)
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
SUMMABLE = (
    "base_bid_cost_gbp", "public_bid_cost_gbp", "delta_bid_cost_gbp", "delta_offer_cost_gbp",
    "transfer_to_the_state_gbp", "payment_to_private_counterparties_gbp",
    "offsetting_private_cfd_settlement_gbp", "net_consumer_change_private_side_gbp",
    "base_decrease_MWh", "public_decrease_MWh", "delta_decrease_MWh",
    "private_delivered_MWh_change", "state_delivered_MWh_change", "tie_break_step_cost_gbp",
    "state_cfd_settlement_at_fixed_strike_gbp",
)


#: Carriers that bid and offer close to zero under P3, so that turning one unit
#: down and another up in the same hour costs nothing the solver can see.
SWAP_CARRIERS = ("wind_offshore", "wind_onshore", "solar_pv", "marine")
ANALYSIS = RESOURCES_ROOT / "analysis"


def renewable_swap_mwh(scenario: str, wholesale_scenario: str, carriers: pd.Series) -> float | None:
    """MWh turned up and down at once among the swap carriers, summed hour by hour.

    ``min(renewables turned up, renewables turned down)`` in each hour against the
    wholesale schedule the balancing run anchored to. ``None`` when either hourly
    dispatch file is missing.
    """
    balancing = RESOURCES / f"{scenario}_balancing_dispatch.csv"
    wholesale = RESOURCES / f"{wholesale_scenario}_wholesale_dispatch.csv"
    if not (balancing.exists() and wholesale.exists()):
        return None
    units = set(carriers.index[carriers.astype(str).isin(SWAP_CARRIERS)])
    frames = []
    for path in (balancing, wholesale):
        header = pd.read_csv(path, nrows=0).columns
        columns = [c for c in header[1:] if c in units]
        frame = pd.read_csv(path, usecols=[header[0], *columns], index_col=0)
        frame.index = pd.DatetimeIndex(frame.index)
        frames.append(frame)
    physical, schedule = frames
    delta = physical - schedule.reindex(index=physical.index, columns=physical.columns).fillna(0.0)
    steps = physical.index.to_series().diff().dt.total_seconds().dropna()
    hours = float(steps.median()) / 3600.0 if len(steps) else 1.0
    up = delta.clip(lower=0.0).sum(axis=1)
    down = (-delta).clip(lower=0.0).sum(axis=1)
    return float(np.minimum(up, down).sum() * hours)


def _retrading_cost(scenario: str) -> float:
    path = ANALYSIS / f"{scenario}_market_summary.json"
    if not path.exists():
        return 0.0
    accounting = json.loads(path.read_text(encoding="utf-8")).get("accounting", {})
    return float(accounting.get("interconnector_retrading_cost_GBP") or 0.0)


def compare_tie_break_orderings(first: str, second: str, rel_tol: float = 1e-3,
                                abs_tol_gbp: float = 1000.0, wholesale: str | None = None) -> dict:
    """Decision 20's gate: the same case solved under two tie orderings.

    The tie is among state-owned units at the same at-cost bid, so the ordering may
    move *which* units are turned down but must not move what the balancing stage
    costs or how much it has to move. Two things are compared, each within
    ``rel_tol`` of its size (or ``abs_tol_gbp`` / 1 MWh):

    * **money**: the cost the solver minimises, accepted bids net of the tie steps
      plus offers plus interconnector re-trading. The components are reported but
      not gated, because they trade against one another inside the solver's
      tolerance (on the P3 week offers moved 0.11% while the total moved 0.03%);
    * **volume net of renewable swaps**: turn-down and turn-up less the MWh turned
      up and down at once among zero-cost renewables (``renewable_swap_mwh``). Such
      a swap costs about 0.02 GBP/MWh under P3, far inside the solver's tolerance,
      so its size is arbitrary: on the P3 week it was the whole 1.3% gross gap
      (14.7 of 14.0 GWh), and net of it the two orderings agreed to 0.07%.

    ``wholesale`` names the scenario whose schedule both runs anchored to; without
    it the swaps cannot be netted and the gross volumes are gated instead.
    """
    totals = {}
    for scenario in (first, second):
        frame = _redispatch(scenario)
        carriers = frame["carrier"] if "carrier" in frame.columns else pd.Series(dtype=str)
        frame = pd.DataFrame({column: pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
                              for column in ("bid_cost", "offer_cost", "decrease_MWh", "increase_MWh")},
                             index=frame.index).rename(columns=lambda c: f"run_{c}")
        steps = _net_bid_costs(scenario, frame, "run")
        swap = renewable_swap_mwh(scenario, wholesale, carriers) if wholesale else None
        bid, offer, retrading = (float(frame["run_bid_cost"].sum()), float(frame["run_offer_cost"].sum()),
                                 _retrading_cost(scenario))
        decrease, increase = float(frame["run_decrease_MWh"].sum()), float(frame["run_increase_MWh"].sum())
        totals[scenario] = {
            "total_cost_net_gbp": bid + offer + retrading,
            "bid_cost_net_gbp": bid, "offer_cost_gbp": offer, "retrading_cost_gbp": retrading,
            "decrease_MWh": decrease, "increase_MWh": increase,
            "renewable_swap_MWh": swap,
            "decrease_net_of_swaps_MWh": decrease - swap if swap is not None else None,
            "increase_net_of_swaps_MWh": increase - swap if swap is not None else None,
            "tie_break_step_cost_gbp": steps,
        }
    netted = all(totals[s]["renewable_swap_MWh"] is not None for s in (first, second))
    gated = ["total_cost_net_gbp"] + (["decrease_net_of_swaps_MWh", "increase_net_of_swaps_MWh"] if netted
                                      else ["decrease_MWh", "increase_MWh"])
    result = {"first": first, "second": second, "rel_tol": rel_tol, "swaps_netted": netted, "gated": gated}
    passed = True
    for key in totals[first]:
        a, b = totals[first][key], totals[second][key]
        result[key] = (a, b)
        if key in gated and abs(a - b) > max(rel_tol * max(abs(a), abs(b)),
                                             abs_tol_gbp if key.endswith("gbp") else 1.0):
            passed = False
    result["passed"] = passed
    return result


def aggregate(pairs, shares: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Sum the decomposition over a list of ``(base, public)`` scenario pairs.

    Twelve months summed is the year, because every quantity here is a flow over
    the solved period and the months tile it exactly. The per-month rows come back
    too: the seven-day probe suggested the effect tracks the size of the bid stack,
    and the only way to see whether it does is month by month.
    """
    rows, total = [], {key: 0.0 for key in SUMMABLE}
    for label, base_scenario, public_scenario in pairs:
        result, _ = decompose(base_scenario, public_scenario, shares)
        for key in SUMMABLE:
            total[key] += float(result.get(key) or 0.0)
        rows.append({"period": label, **{k: float(result.get(k) or 0.0) for k in SUMMABLE}})
    total["transfer_share_of_delta"] = (
        total["transfer_to_the_state_gbp"] / total["delta_bid_cost_gbp"]
        if total["delta_bid_cost_gbp"] else None)
    total["net_consumer_change_gbp_per_household_inc_vat"] = (
        total["net_consumer_change_private_side_gbp"] / 1e6 * GBP_PER_HOUSEHOLD_PER_MILLION)
    total["net_consumer_change_gbp_per_household_ex_vat"] = (
        total["net_consumer_change_private_side_gbp"] / 1e6 * GBP_PER_HOUSEHOLD_PER_MILLION_EX_VAT)
    total["periods"] = len(rows)
    return total, pd.DataFrame(rows).set_index("period")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_scenario", nargs="?",
                        help="baseline scenario (omit when using --months)")
    parser.add_argument("public_scenario", nargs="?",
                        help="public-bid scenario (omit when using --months)")
    parser.add_argument("--gate", nargs=2, metavar=("FIRST", "SECOND"), default=None,
                        help="decision 20's gate: compare two tie orderings of one case and exit")
    parser.add_argument("--wholesale", default=None,
                        help="with --gate: the scenario whose schedule both runs anchored to (nets renewable swaps)")
    parser.add_argument("--shares", default=None, help="per-generator state share table")
    parser.add_argument("--case", default=None, help="the ownership case the public run priced")
    parser.add_argument("--out", default=None, help="write the decomposition as JSON")
    parser.add_argument("--months", nargs=2, metavar=("BASE_TEMPLATE", "PUBLIC_TEMPLATE"), default=None,
                        help="two templates containing {month}; decompose all twelve and sum them")
    args = parser.parse_args()

    if args.gate:
        result = compare_tie_break_orderings(*args.gate, wholesale=args.wholesale)
        print(json.dumps(result, indent=1, default=str))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        raise SystemExit(0 if result["passed"] else 1)
    if not (args.shares and args.case):
        raise SystemExit("--shares and --case are required unless --gate is given")
    shares = load_shares(args.shares, args.case)
    if args.months:
        base_template, public_template = args.months
        pairs = [(m, base_template.format(month=m), public_template.format(month=m)) for m in MONTHS]
        total, rows = aggregate(pairs, shares)
        pretty = rows[["delta_bid_cost_gbp", "transfer_to_the_state_gbp",
                       "payment_to_private_counterparties_gbp",
                       "offsetting_private_cfd_settlement_gbp",
                       "net_consumer_change_private_side_gbp"]].div(1e6).round(2)
        pretty.columns = ["d bid cost", "to state", "to private", "CfD offset", "NET"]
        print(pretty.to_string())
        print("")
        print(f"YEAR   accepted bid cost {total['base_bid_cost_gbp']/1e6:,.0f} m -> "
              f"{total['public_bid_cost_gbp']/1e6:,.0f} m  ({total['delta_bid_cost_gbp']/1e6:+,.0f} m)")
        print(f"       transfer to the state {total['transfer_to_the_state_gbp']/1e6:+,.0f} m "
              f"({total['transfer_share_of_delta']:.1%})")
        print(f"       paid to private counterparties {total['payment_to_private_counterparties_gbp']/1e6:+,.0f} m")
        print(f"       offsetting private CfD settlement {total['offsetting_private_cfd_settlement_gbp']/1e6:+,.0f} m")
        print(f"       NET consumer change {total['net_consumer_change_private_side_gbp']/1e6:+,.0f} m "
              f"= {total['net_consumer_change_gbp_per_household_inc_vat']:+.2f} GBP/household/year inc VAT")
        print(f"       turn-down volume {total['delta_decrease_MWh']/1e3:+,.1f} GWh on "
              f"{total['base_decrease_MWh']/1e3:,.0f} GWh")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump({"year": total, "by_period": rows.to_dict(orient="index")}, handle, indent=2)
            print("")
            print(f"wrote {args.out}")
        return

    if not (args.base_scenario and args.public_scenario):
        raise SystemExit("give a base and public scenario, or use --months")
    result, frame = decompose(args.base_scenario, args.public_scenario, shares)

    delta = result["delta_bid_cost_gbp"]
    print(f"Accepted bid cost  {result['base_bid_cost_gbp']/1e6:10.2f} m -> "
          f"{result['public_bid_cost_gbp']/1e6:10.2f} m   ({delta/1e6:+.2f} m)")
    print(f"  of which transfer to the state       {result['transfer_to_the_state_gbp']/1e6:+10.2f} m"
          f"   ({result['transfer_share_of_delta']:.1%} of the change)" if result["transfer_share_of_delta"]
          is not None else "")
    print(f"  of which paid to private counterparties {result['payment_to_private_counterparties_gbp']/1e6:+8.2f} m")
    print(f"Offer cost change                      {result['delta_offer_cost_gbp']/1e6:+10.2f} m")
    print(f"Turn-down volume   {result['base_decrease_MWh']/1e3:10.1f} GWh -> "
          f"{result['public_decrease_MWh']/1e3:10.1f} GWh ({result['delta_decrease_MWh']/1e3:+.1f} GWh)")
    print("")
    print(f"Delivery moves from the state to private plant: "
          f"state {result['state_delivered_MWh_change']/1e3:+.1f} GWh, "
          f"private {result['private_delivered_MWh_change']/1e3:+.1f} GWh")
    print(f"Offsetting CfD settlement on that private delivery "
          f"{result['offsetting_private_cfd_settlement_gbp']/1e6:+10.2f} m")
    print(f"NET consumer change, private side              "
          f"{result['net_consumer_change_private_side_gbp']/1e6:+10.2f} m")
    print(f"\n{result['state_owned_units']} of {result['units']} units at least partly state-owned\n")
    if result["by_carrier"]:
        print(pd.DataFrame(result["by_carrier"]).T.div(1e6).round(3).to_string())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
