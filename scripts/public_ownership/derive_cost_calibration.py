"""Derive, or check, the private cost calibration table (user decision 1 and its extensions, 29 Sep 2026).

Trust the auction: a cohort that settles at a cleared CfD strike is taken to break
even there, so its private required revenue equals its contract revenue (market +
support - fuel and carbon) at that strike. The capital cost is the unidentified
part (DESNZ capex at the workbook's private WACC), so one capex factor k per
cohort solves

    k x capital charge (private) + other costs (fixed O&M incl. TNUoS and per-kW
    insurance, variable O&M) = contract revenue

Each round is its own cohort, at the strike it settles at: offshore AR7, tidal AR7a,
the earlier offshore rounds and tidal AR4 to AR6. Those cohorts are calibrated only
where the modelled private cost is above the strike (k < 1).  F19 decision 2 (user,
6 Oct 2026) adds onshore wind and solar PV AR7a, calibrated **both ways**: their
DESNZ cost is below the AR7a strike, so k > 1 and the cohort no longer carries a
rent the state would capture as its own (the auction is trusted symmetrically).

The modelled AR8 round (``AR8_modelled``, F18, user decision 27 Oct 2026) was the
``future`` cohort, pooled with AR7 (offshore) and AR7a (onshore, solar, tidal)
because it settled at a copy of their strike. It now settles at its own modelled
strike (``scripts/market/build_ar8_modelled_strikes.py``), so it is its own cohort,
anchored both ways at that strike for every technology: its private rent is zero
by construction, as an at-cost modelled strike implies.
**One factor per round, whatever the finance vintage** (user, 6 Nov 2026). Under F28 a row's
``finance_vintage`` is set by the register's legislation year, not by the plant: the same AR7 farm is
``legacy`` with legislation in 2028 and ``new_build`` in the 2025 bound, and a split round's two vintages
are pro rata pieces of the same capacity. The capex factor is a property of the plant at its cleared
strike, so each cohort pools every vintage of its round and the table is keyed on (carrier, round): it
applies unchanged in every register, whatever its legislation year (``vintages_pooled`` records which
vintages the deriving register held).
Offshore insurance is a per-kW rate (BVG 11.22 GBP/kW) and does not scale with k;
tidal insurance is 0.4% of capex and does.

**The register it reads is calibrated.** ``calculate_financing_scenarios`` applies
this table, and records the factor on every row (``capex_calibration_factor``), so
the capital charge and the capex-based insurance are divided by it to recover the
uncalibrated cohort. That makes the derivation reproducible from the live register,
but the table cannot be a rule output (it is the financing step's input), so:

* ``--out``: write the table (a deliberate re-derivation; rerun the financing after);
* ``--check``: compare the committed table with a fresh derivation and write a stamp,
  or fail naming the drifted cohorts. Rule ``verify_public_ownership_cost_calibration``
  runs this on every financing rebuild.

Usage::

    python scripts/public_ownership/derive_cost_calibration.py REGISTER --out TABLE
    python scripts/public_ownership/derive_cost_calibration.py REGISTER --check TABLE --stamp STAMP
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.market.cfd_strike_table import MODELLED_ROUND  # noqa: E402
from scripts.public_ownership.register_io import read_register  # noqa: E402

#: The strike basis of the modelled AR8 cohorts (F18).
AR8_STRIKE_BASIS = "AR8 modelled strike (scripts/market/build_ar8_modelled_strikes.py; its own cohort, F18)"

#: (carrier, finance technology, rounds pooled at one strike, strike basis[, both ways]).
#: Every finance vintage of a round is in its cohort (6 Nov 2026).
#: A cohort is calibrated where its cost is above the strike; one marked both ways
#: (F19 decision 2) also where it is below (k > 1).
COHORTS = [
    ("wind_offshore", "Offshore wind", ("AR7",), "AR7 cleared strike"),
    # F18 (user, 27 Oct 2026): the modelled AR8 round at its own modelled strike, both ways
    # (its private rent is zero), for every technology it holds.
    ("wind_offshore", "Offshore wind", (MODELLED_ROUND,), AR8_STRIKE_BASIS, True),
    # User decision (29 Sep 2026, second round): the earlier offshore rounds whose
    # modelled private cost is above their cleared strike, each at its own strike.
    ("wind_offshore", "Offshore wind", ("AR2",), "AR2 cleared strike"),
    ("wind_offshore", "Offshore wind", ("AR3",), "AR3 cleared strike"),
    ("wind_offshore", "Offshore wind", ("AR4",), "AR4 cleared strike"),
    ("wind_offshore", "Offshore wind", ("AR6",), "AR6 cleared strike"),
    ("marine", "Tidal stream", ("AR4",), "AR4 cleared strike"),
    ("marine", "Tidal stream", ("AR5",), "AR5 cleared strike"),
    ("marine", "Tidal stream", ("AR6",), "AR6 cleared strike"),
    ("marine", "Tidal stream", ("AR7a",), "AR7a cleared strike"),
    ("marine", "Tidal stream", (MODELLED_ROUND,), AR8_STRIKE_BASIS, True),
    # F19 decision 2 (user, 6 Oct 2026): trust the auction for onshore and solar too, both ways.
    ("wind_onshore", "Onshore wind", ("AR7a",), "AR7a cleared strike", True),
    ("wind_onshore", "Onshore wind", (MODELLED_ROUND,), AR8_STRIKE_BASIS, True),
    ("solar_pv", "Solar PV", ("AR7a",), "AR7a cleared strike", True),
    ("solar_pv", "Solar PV", (MODELLED_ROUND,), AR8_STRIKE_BASIS, True),
]
#: The status of a cohort calibrated in each direction.
STATUS_DOWN = "calibrated: trust the auction (user decision 29 Sep 2026)"
STATUS_BOTH_WAYS = "calibrated both ways: trust the auction (F19 decision 2, 6 Oct 2026)"
#: Technologies whose insurance is a per-kW rate, so it does not scale with capex.
PER_KW_INSURANCE = {"Offshore wind"}
REGISTER_COLUMNS = ["case_id", "evidence_status", "carrier", "cfd_round", "finance_vintage", "capacity_mw",
                    "dispatch_mwh", "market_revenue_gbp", "support_revenue_gbp", "fuel_carbon_gbp",
                    "annual_required_revenue_gbp", "annual_capital_recovery_gbp", "annual_insurance_gbp",
                    "capex_calibration_factor"]
#: Follow-up F6 (4 Oct 2026): the auction margin's capex scale, divided out with the factor.
#: Registers written before it have no such column and read without it.
MARGIN_COLUMN = "auction_margin_capex_scale"
#: The table's key (6 Nov 2026: one factor per round, every finance vintage pooled).
KEYS = ["carrier", "cfd_round"]
#: A committed factor further than this from a fresh derivation is drift.
FACTOR_TOLERANCE = 1e-4


def register_columns(path) -> list[str]:
    """:data:`REGISTER_COLUMNS`, plus the auction-margin scale where the register has it."""
    columns = list(REGISTER_COLUMNS)
    try:
        import pyarrow.parquet as pq

        if str(path).endswith(".parquet") and MARGIN_COLUMN in pq.ParquetFile(path).schema_arrow.names:
            columns.append(MARGIN_COLUMN)
    except (ImportError, OSError):
        pass
    return columns


def uncalibrated_private_rows(register: pd.DataFrame) -> pd.DataFrame:
    """The resolved private rows with the calibration divided back out of capital and scaling insurance."""
    rows = register.loc[register["case_id"].astype(str).eq("private")
                        & register["evidence_status"].astype(str).eq("resolved")].copy()
    num = lambda column: pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
    factor = pd.to_numeric(rows["capex_calibration_factor"], errors="coerce").fillna(1.0)
    if MARGIN_COLUMN in rows.columns:
        # The private case at the central auction margin carries capex below break-even;
        # the table is the margin-free (k(0)) derivation, so divide the margin out too.
        factor = factor * pd.to_numeric(rows[MARGIN_COLUMN], errors="coerce").fillna(1.0)
    if (factor <= 0).any():
        raise ValueError("register carries a non-positive capex_calibration_factor")
    rows["contract_revenue_gbp"] = num("market_revenue_gbp") + num("support_revenue_gbp") - num("fuel_carbon_gbp")
    capital_cal, insurance_cal = num("annual_capital_recovery_gbp"), num("annual_insurance_gbp")
    rows["_capital"] = capital_cal / factor
    rows["_insurance"] = insurance_cal / factor
    # Required revenue less the calibrated capital charge and capex-based insurance is
    # untouched by k; add the uncalibrated ones back (per-kW insurance is left as it is).
    rows["_required_fixed"] = num("annual_required_revenue_gbp") - capital_cal
    return rows


def derive(register: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """The calibration table from a (calibrated or uncalibrated) register."""
    private = uncalibrated_private_rows(register)
    rows = []
    for carrier, technology, rounds, strike_basis, *flags in COHORTS:
        both_ways = bool(flags and flags[0])
        # One cohort per round, every finance vintage pooled (6 Nov 2026).
        cohort = private.loc[private["carrier"].astype(str).eq(carrier)
                             & private["cfd_round"].astype(str).isin(rounds)]
        if cohort.empty:
            continue
        vintages = "+".join(sorted(set(cohort["finance_vintage"].astype(str))))
        per_kw = technology in PER_KW_INSURANCE
        insurance_cal = pd.to_numeric(cohort["annual_insurance_gbp"], errors="coerce").fillna(0.0)
        scaling_insurance = 0.0 if per_kw else float(cohort["_insurance"].sum())
        capital = float(cohort["_capital"].sum())
        # Other costs, uncalibrated: required less capital, with capex-based insurance at its uncalibrated value.
        other = float(cohort["_required_fixed"].sum()) - (0.0 if per_kw else float(insurance_cal.sum())) + scaling_insurance
        required = capital + other
        revenue = float(cohort["contract_revenue_gbp"].sum())
        if revenue - required >= 0 and not both_ways:
            continue  # the cohort already recovers its cost at the strike: nothing to trust away
        mwh = float(pd.to_numeric(cohort["dispatch_mwh"], errors="coerce").sum())
        k = (revenue - (other - scaling_insurance)) / (capital + scaling_insurance)
        for rnd in rounds:
            rows.append({
                "carrier": carrier, "finance_technology": technology, "cfd_round": rnd, "vintages_pooled": vintages,
                "capex_factor": round(k, 6), "pooled_rounds": "+".join(rounds), "strike_basis": strike_basis,
                "cohort_capacity_mw": round(float(pd.to_numeric(cohort["capacity_mw"], errors="coerce").sum()), 3),
                "cohort_dispatch_mwh": round(mwh, 1),
                "cohort_contract_revenue_gbp": round(revenue, 0), "cohort_other_costs_gbp": round(other, 0),
                "cohort_capital_charge_private_gbp": round(capital, 0),
                "cohort_capex_scaled_insurance_gbp": round(scaling_insurance, 0),
                "cohort_cost_before_gbp_per_mwh": round(required / mwh, 3),
                "cohort_revenue_gbp_per_mwh": round(revenue / mwh, 3),
                "cohort_private_rent_before_gbp": round(revenue - required, 0),
                "price_year": 2024,
                "derivation": ("k = (contract revenue - (required revenue - capital charge - capex-scaled insurance)) / "
                               "(capital charge + capex-scaled insurance) on the private rows of the cohort; contract "
                               "revenue = market + support - fuel and carbon at the settled strike; "
                               f"register {source_label} (private case, calibration divided back out)"),
                "status": STATUS_BOTH_WAYS if both_ways else STATUS_DOWN,
            })
    return pd.DataFrame(rows)


def compare(committed: pd.DataFrame, fresh: pd.DataFrame) -> dict:
    """Factor-by-factor comparison of the committed table with a fresh derivation."""
    left = committed.set_index(KEYS)["capex_factor"]
    right = fresh.set_index(KEYS)["capex_factor"] if not fresh.empty else pd.Series(dtype=float)
    joined = pd.concat({"committed": left, "derived": right}, axis=1)
    joined["abs_diff"] = (joined["committed"] - joined["derived"]).abs()
    drift = joined.loc[joined["abs_diff"].isna() | joined["abs_diff"].gt(FACTOR_TOLERANCE)]
    return {"cohorts": int(len(joined)), "max_abs_diff": float(joined["abs_diff"].max()) if len(joined) else 0.0,
            "drifted": [dict(zip(KEYS, key), **{k: (None if pd.isna(v) else float(v)) for k, v in row.items()})
                        for key, row in drift.iterrows()]}


def _is_vintaged(register: pd.DataFrame) -> bool:
    """A vintaged (FES long-range) register: its model year is past the CP30 year."""
    if "model_year" not in register.columns or register.empty:
        return False
    from scripts.public_ownership.model_year import is_vintaged_model_year

    return is_vintaged_model_year(int(pd.to_numeric(register["model_year"]).iloc[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("register", help="the market group's asset finance register (.parquet)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", help="write the derived table here")
    mode.add_argument("--check", help="the committed table to check against a fresh derivation")
    parser.add_argument("--stamp", help="with --check: the JSON stamp written when the table reproduces")
    args = parser.parse_args()

    register = read_register(args.register, columns=register_columns(args.register) + ["model_year"], cases=["private"])
    if args.check and _is_vintaged(register):
        # 9 Nov 2026: a vintaged (2050) register never takes this table (calculate_financing_scenarios passes None;
        # its strikes are the vintage cost at the private rate), so there is nothing to reproduce: the check passes
        # and says so. A cohort absent from the register counted as drift, and every 2050 group failed.
        print(f"{args.check}: not applied to {Path(args.register).name} (a vintaged register; the financing does not "
              "use the table)")
        if args.stamp:
            Path(args.stamp).parent.mkdir(parents=True, exist_ok=True)
            Path(args.stamp).write_text(json.dumps({"table": args.check, "register": args.register,
                                                    "applies": False, "reason": "vintaged register"}, indent=1),
                                        encoding="utf-8")
        return
    fresh = derive(register, Path(args.register).name)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fresh.to_csv(args.out, index=False, lineterminator="\r\n")
        print(fresh[KEYS + ["capex_factor", "cohort_cost_before_gbp_per_mwh", "cohort_revenue_gbp_per_mwh"]]
              .to_string(index=False))
        return
    report = compare(pd.read_csv(args.check), fresh)
    print(f"{args.check}: {report['cohorts']} cohorts, max factor difference {report['max_abs_diff']:.2e}")
    if report["drifted"]:
        print(json.dumps(report["drifted"], indent=1), file=sys.stderr)
        raise SystemExit(
            "The cost calibration no longer reproduces from the register: the cohorts above have moved. Re-derive it "
            f"(python scripts/public_ownership/derive_cost_calibration.py {args.register} --out {args.check}) and "
            "rerun the financing, or find the change that moved them.")
    if args.stamp:
        Path(args.stamp).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stamp).write_text(json.dumps({"table": args.check, "register": args.register, **report}, indent=1),
                                    encoding="utf-8")


if __name__ == "__main__":
    main()
