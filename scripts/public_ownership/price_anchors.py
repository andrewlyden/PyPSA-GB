"""The legacy fleet's price path anchors: today, and every solved model year (user decision, 10 Nov 2026).

Each solved year's anchor is its asset settlement's wholesale revenue per MWh by carrier (wholesale revenue over
wholesale MWh, the private case's settled dispatch), so a technology's capture price in that year. Today's anchor
has no hourly shape of its own (the user chose the 2026 forward price, a level): it is the earliest solved year's
capture price per carrier scaled by today's forward level over the model's reference price for that year, i.e.
each technology keeps the earliest solved year's capture ratio. Between anchors the valuation runs a straight
line; it holds flat after the last (``legacy_valuation.anchored_price_ratios``).

    python scripts/public_ownership/price_anchors.py --settlement 2030=resources/.._asset_settlement.csv \\
        --settlement 2050=resources/.._asset_settlement.csv --today-year 2026 --today-gbp-per-mwh 95.4 \\
        --today-deflator 0.943 --reference-gbp-per-mwh 51.6 --out resources/public_ownership/legacy_price_anchors.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

#: Carriers with too little settled energy to give a stable capture price are left out (held flat).
MIN_MWH = 1.0e5

# ---- every stream per MW (user, 11 Nov 2026) ----
# With ``--register YEAR=PARQUET`` each solved year also gives, per technology across its whole fleet (the private
# case's rows), every stream of the legacy valuation per MW of capacity: output (MWh per MW, so load factor and
# curtailment), wholesale revenue, balancing income, fuel and carbon cost and Capacity Market income. Per MW, not per
# MWh: a technology whose role changes (biomass as a 2050 peaker) then earns and burns less per MW, rather than
# carrying a price per MWh no existing plant earns. The legacy valuation scales each plant's own model-year streams
# by its technology's path (legacy_valuation.anchored_stream_ratios).
#: The streams, as (anchor column, register column); the register column is summed and divided by capacity.
STREAMS = (("mwh_per_mw", "wholesale_mwh"), ("wholesale_gbp_per_mw", "wholesale_revenue_gbp"),
           ("balancing_gbp_per_mw", "balancing_settlement_gbp"), ("fuel_carbon_gbp_per_mw", "fuel_carbon_gbp"),
           ("capacity_market_gbp_per_mw", "capacity_market_revenue_gbp"))
#: A technology with less capacity than this in a solved year gives that year no stream anchor (too few plant).
MIN_MW = 50.0
#: One name per technology across the builds (the CP30 2030 build and the FES long-range builds name pumped
#: storage differently).
CANONICAL_CARRIER = {"Pumped Storage Hydroelectricity": "pumped_hydro"}


def canonical(carrier) -> str:
    return CANONICAL_CARRIER.get(str(carrier), str(carrier))


def stream_anchor(register: pd.DataFrame, year: int) -> pd.DataFrame:
    """Every stream per MW by technology in one solved year (the register's private case, every row)."""
    rows = register.loc[register["case_id"].astype(str).eq("private")] if "case_id" in register.columns else register
    rows = rows.assign(carrier=rows["carrier"].map(canonical))
    num = {c: pd.to_numeric(rows[c], errors="coerce").fillna(0.0) for _, c in STREAMS if c in rows.columns}
    frame = pd.DataFrame({"carrier": rows["carrier"], "capacity_mw": pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0), **num})
    g = frame.groupby("carrier").sum()
    g = g.loc[g["capacity_mw"] >= MIN_MW]
    out = pd.DataFrame({"carrier": g.index, "year": int(year), "capacity_mw": g["capacity_mw"].to_numpy()})
    for col, src in STREAMS:
        out[col] = (g[src] / g["capacity_mw"]).to_numpy() if src in g.columns else float("nan")
    out["stream_basis"] = f"solved {year}: the register's private case, every row of the technology, per MW"
    return out


def solved_anchor(settlement: pd.DataFrame, year: int) -> pd.DataFrame:
    """Wholesale revenue per MWh by carrier in one solved year's settlement."""
    need = {"carrier", "wholesale_mwh", "wholesale_revenue_gbp"}
    if missing := need - set(settlement.columns):
        raise ValueError(f"the {year} settlement lacks {sorted(missing)}")
    g = settlement.groupby(settlement["carrier"].astype(str)).agg(
        mwh=("wholesale_mwh", "sum"), revenue=("wholesale_revenue_gbp", "sum"))
    g = g.loc[g["mwh"] >= MIN_MWH]
    return pd.DataFrame({"carrier": g.index, "year": int(year),
                         "wholesale_gbp_per_mwh": (g["revenue"] / g["mwh"]).to_numpy(),
                         "wholesale_mwh": g["mwh"].to_numpy(), "basis": f"solved {year}: settled wholesale revenue / MWh"})


def today_anchor(first: pd.DataFrame, year: int, level_nominal: float, deflator: float, reference: float) -> pd.DataFrame:
    """Today's anchor: the earliest solved year's capture prices scaled to today's forward level (real 2024)."""
    if reference <= 0:
        raise ValueError("--reference-gbp-per-mwh must be positive")
    factor = float(level_nominal) * float(deflator) / float(reference)
    out = first[["carrier"]].copy()
    out["year"] = int(year)
    out["wholesale_gbp_per_mwh"] = first["wholesale_gbp_per_mwh"].to_numpy() * factor
    out["wholesale_mwh"] = float("nan")
    out["basis"] = (f"today ({year}): forward level {level_nominal:.2f} nominal x {deflator:.3f} to 2024 prices over the "
                    f"model's {int(first['year'].iloc[0])} reference {reference:.2f} (factor {factor:.4f}), on the "
                    f"{int(first['year'].iloc[0])} capture ratios")
    return out


def build(settlements: dict[int, pd.DataFrame], today: dict | None,
          registers: dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
    frames = [solved_anchor(s.assign(carrier=s["carrier"].map(canonical)), y) for y, s in sorted(settlements.items())]
    if not frames:
        raise ValueError("at least one solved settlement is needed")
    if today:
        frames.insert(0, today_anchor(frames[0], today["year"], today["level"], today["deflator"], today["reference"]))
    out = pd.concat(frames, ignore_index=True)
    if registers:
        # The registers of the years whose groups never read this table (the vintaged years): a register that
        # read it would be a cycle. The valued register gives its own model year's streams
        # (legacy_valuation.with_model_year_streams).
        if not set(registers) <= set(settlements):
            raise ValueError(f"registers {sorted(registers)} name years with no settlement {sorted(settlements)}")
        streams = pd.concat([stream_anchor(r, y) for y, r in sorted(registers.items())], ignore_index=True)
        out = out.merge(streams, on=["carrier", "year"], how="outer")
    return out.sort_values(["carrier", "year"]).reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--settlement", action="append", default=[], metavar="YEAR=CSV", required=True)
    ap.add_argument("--register", action="append", default=[], metavar="YEAR=PARQUET",
                    help="each solved year's asset finance register: every stream per MW by technology (11 Nov 2026)")
    ap.add_argument("--today-year", type=int, default=None)
    ap.add_argument("--today-gbp-per-mwh", type=float, default=None, help="today's forward level, nominal GBP/MWh")
    ap.add_argument("--today-deflator", type=float, default=1.0, help="nominal today to real 2024 prices")
    ap.add_argument("--reference-gbp-per-mwh", type=float, default=None,
                    help="the model's reference price in the earliest solved year (the same blend as today's level)")
    ap.add_argument("--reference-price-files", nargs="*", default=None,
                    help="the earliest solved year's hourly price files (column wholesale_price); their mean is "
                         "the reference when --reference-gbp-per-mwh is not given")
    ap.add_argument("--carriers", nargs="*", default=None, help="carriers on the path (default: every carrier)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    settlements = {}
    for item in a.settlement:
        year, _, path = item.partition("=")
        settlements[int(year)] = pd.read_csv(path, low_memory=False)
    if a.reference_gbp_per_mwh is None and a.reference_price_files:
        prices = pd.concat([pd.read_csv(f, index_col=0)["wholesale_price"] for f in a.reference_price_files])
        a.reference_gbp_per_mwh = float(prices.mean())
        print(f"reference price: {a.reference_gbp_per_mwh:.2f} GBP/MWh over {len(prices)} hours")
    today = None
    if a.today_year is not None:
        if a.today_gbp_per_mwh is None or a.reference_gbp_per_mwh is None:
            raise SystemExit("--today-year needs --today-gbp-per-mwh and --reference-gbp-per-mwh")
        today = {"year": a.today_year, "level": a.today_gbp_per_mwh, "deflator": a.today_deflator,
                 "reference": a.reference_gbp_per_mwh}
    registers = {}
    if a.register:
        from scripts.public_ownership.register_io import read_register

        cols = ["case_id", "carrier", "capacity_mw"] + [c for _, c in STREAMS]
        for item in a.register:
            year, _, path = item.partition("=")
            registers[int(year)] = read_register(path, cases=["private"], columns=cols)
    out = build(settlements, today, registers or None)
    if a.carriers:
        out = out.loc[out["carrier"].isin(a.carriers)].reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.pivot_table(index="carrier", columns="year", values="wholesale_gbp_per_mwh").round(1).to_string())
    if registers:
        print(out.pivot_table(index="carrier", columns="year", values="mwh_per_mw").round(0).to_string())


if __name__ == "__main__":
    main()
