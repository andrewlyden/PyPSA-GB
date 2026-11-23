"""Show how public-ownership finance savings relate to consumer-bill categories.

This is a reporting bridge, not a tariff recalculation.  The core policy model
does not specify a recovery/allocation rule, so existing consumer-bill
components remain unchanged and the financing saving is shown separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership import fingerprints  # noqa: E402  (F20 phase 2)


BILL_GROUPS_HELD_CONSTANT = (
    "Wholesale & hedging",
    "System operation, balancing & losses",
    "Policy costs",
    "Network costs",
    "Supplier, payment, margin & headroom",
    "VAT",
)


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} lacks columns: {sorted(missing)}")


#: The register columns the bridge reads.
REGISTER_COLUMNS = ("case_id", "asset_id", "finance_technology", "finance_vintage", "capacity_mw",
                    "dispatch_mwh", "annual_required_revenue_gbp", "annual_capital_recovery_gbp", "support_type")
SUMMARY_COLUMNS = {"case_id", "annual_financing_saving_vs_private_gbp", "consumer_equivalent_saving_gbp_per_mwh_demand"}


def _is_private(case_id) -> bool:
    return str(case_id).split("@", 1)[0] == "private"


def _technology_savings(case_id: str, rows: pd.DataFrame, private: pd.DataFrame) -> pd.DataFrame:
    """One case's saving against its private comparator, by technology, vintage and support type."""
    scenario = rows.set_index("asset_id")
    aligned = private.join(
        scenario[["annual_required_revenue_gbp", "annual_capital_recovery_gbp"]],
        how="inner", lsuffix="_private", rsuffix="_case", validate="one_to_one",
    )
    aligned["annual_financing_saving_gbp"] = (
        aligned["annual_required_revenue_gbp_private"] - aligned["annual_required_revenue_gbp_case"]
    )
    aligned["annual_capital_recovery_saving_gbp"] = (
        aligned["annual_capital_recovery_gbp_private"] - aligned["annual_capital_recovery_gbp_case"]
    )
    aligned["support_type"] = aligned["support_type"].fillna("unclassified").astype(str)
    grouped = aligned.groupby(
        ["finance_technology", "finance_vintage", "support_type"], dropna=False, as_index=False
    ).agg(
        assets=("capacity_mw", "size"),
        capacity_mw=("capacity_mw", "sum"),
        dispatch_mwh=("dispatch_mwh", "sum"),
        annual_financing_saving_gbp=("annual_financing_saving_gbp", "sum"),
        annual_capital_recovery_saving_gbp=("annual_capital_recovery_saving_gbp", "sum"),
    )
    grouped.insert(0, "case_id", case_id)
    grouped["saving_source"] = "Lower capital recovery from finance structure; fixed and variable operating costs unchanged"
    grouped["consumer_bill_treatment"] = grouped["support_type"].map(
        lambda value: "new_cfd_contract_revenue_requirement_reduction"
        if value.strip().lower() == "cfd" else "unallocated_non_cfd_financing_saving"
    )
    return grouped


class _SavingsMemo:
    """:func:`_technology_savings` once per distinct (comparator, case rows) (F20 phase 1, 13 Oct 2026).

    A case's technology table depends on its comparator's rows and on its own
    ``asset_id``, required revenue and capital recovery alone; most cases share
    their register rows with another (89% on the 6 Oct run), and cases that share
    them are neighbours in register order.  The last ``size`` distinct row sets
    are kept with their tables; a case whose three columns equal one of them
    (bit for bit) takes its table relabelled, the rest are computed as before.
    """

    COLUMNS = ("annual_required_revenue_gbp", "annual_capital_recovery_gbp")

    def __init__(self, size: int = 8):
        self.size = int(size)
        self._entries: list[tuple] = []          # (comparator rows, asset ids, numbers as bits, table), newest last
        self.hits = self.misses = 0

    @classmethod
    def _key_arrays(cls, rows: pd.DataFrame):
        """The columns the table depends on, or None when they are not plain text ids and floats (no reuse)."""
        from pandas._libs import lib

        ids = rows["asset_id"].to_numpy()
        if ids.dtype != object or not lib.is_string_array(ids, skipna=False):
            return None
        numbers = []
        for column in cls.COLUMNS:
            values = rows[column].to_numpy()
            if values.dtype != np.float64:
                return None
            numbers.append(values.view(np.int64).copy())
        return ids.copy(), numbers

    def savings(self, case_id: str, rows: pd.DataFrame, private: pd.DataFrame) -> pd.DataFrame:
        arrays = self._key_arrays(rows)
        if arrays is not None:
            ids, numbers = arrays
            for i in range(len(self._entries) - 1, -1, -1):
                stored_private, stored_ids, stored_numbers, table = self._entries[i]
                if (stored_private is private and len(stored_ids) == len(ids) and np.array_equal(stored_ids, ids)
                        and all(np.array_equal(a, b) for a, b in zip(stored_numbers, numbers))):
                    self._entries.append(self._entries.pop(i))
                    self.hits += 1
                    out = table.copy()
                    out["case_id"] = case_id
                    return out
        self.misses += 1
        table = _technology_savings(case_id, rows, private)
        if arrays is not None:
            self._entries.append((private, arrays[0], arrays[1], table.copy()))
            del self._entries[:-self.size]
        return table


def build_consumer_bill_financing_bridge(
    register: pd.DataFrame, summary: pd.DataFrame, *, benchmark_kwh: float = 3100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return technology savings, bill-category bridge and concise audit summary."""
    _require(register, set(REGISTER_COLUMNS), "Asset finance register")
    _require(summary, SUMMARY_COLUMNS, "Financing summary")
    comparators = (summary.set_index("case_id")["comparator_case_id"].to_dict()
                   if "comparator_case_id" in summary.columns else {})
    private_by_case = {
        case_id: rows.set_index("asset_id").copy()
        for case_id, rows in register.groupby("case_id") if _is_private(case_id)
    }
    output = []
    memo = _SavingsMemo()        # F20 phase 1: a case sharing its rows and comparator with a neighbour reuses its table
    for case_id, rows in register.groupby("case_id"):
        if _is_private(case_id):
            continue
        private = private_by_case.get(comparators.get(case_id, "private"), private_by_case["private"])
        output.append(memo.savings(case_id, rows, private))
    return _bridge(output, summary, benchmark_kwh)


def build_consumer_bill_financing_bridge_from_file(
    register_path, summary: pd.DataFrame, *, benchmark_kwh: float = 3100.0, batch_cases: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """:func:`build_consumer_bill_financing_bridge` reading the register a batch of cases at a time (30 Sep 2026).

    Only the bridge's columns are read; the private-family cases are read once,
    then the other cases ``batch_cases`` at a time, so memory is flat in the
    number of cases.  Each case's rows depend on that case and its comparator
    alone, and the output is sorted by case, so it equals the whole-register result.
    """
    from scripts.public_ownership.register_io import iter_register_cases, register_cases

    _require(summary, SUMMARY_COLUMNS, "Financing summary")
    comparators = (summary.set_index("case_id")["comparator_case_id"].to_dict()
                   if "comparator_case_id" in summary.columns else {})
    cases = register_cases(register_path)
    private_cases = [case for case in cases if _is_private(case)]
    private_by_case = {}
    for rows in iter_register_cases(register_path, cases=private_cases, columns=list(REGISTER_COLUMNS),
                                    batch_cases=batch_cases):
        _require(rows, set(REGISTER_COLUMNS), "Asset finance register")
        private_by_case.update({case_id: frame.set_index("asset_id").copy()
                                for case_id, frame in rows.groupby("case_id")})
    output = []
    others = [case for case in cases if not _is_private(case)]
    memo = _SavingsMemo()        # F20 phase 1: a case sharing its rows and comparator with a neighbour reuses its table
    if fingerprints.sharing_enabled() and fingerprints.register_is_keyed(register_path):
        # F20 phase 2: one table per bridge key, read and computed for its first case only.
        output = _shared_tables(register_path, others, comparators, private_by_case, memo, batch_cases)
        return _bridge(output, summary, benchmark_kwh)
    for rows in iter_register_cases(register_path, cases=others, columns=list(REGISTER_COLUMNS),
                                    batch_cases=batch_cases):
        for case_id, frame in rows.groupby("case_id"):
            private = private_by_case.get(comparators.get(case_id, "private"), private_by_case["private"])
            output.append(memo.savings(case_id, frame, private))
    return _bridge(output, summary, benchmark_kwh)


def _shared_tables(register_path, cases, comparators, private_by_case, memo, batch_cases) -> list[pd.DataFrame]:
    """Each case's technology table, computed once per bridge key (F20 phase 2, 27 Oct 2026).

    A case's table depends on its comparator and on its own rows' required revenue and capital
    recovery, which are the register body's (:func:`fingerprints.bridge_key`).  Only the first
    case of each key is read and computed; every other case takes that table with its own
    ``case_id``, and a sample (:class:`fingerprints.SampleCheck`) is read and computed in full
    as well.  The tables come back in ``cases`` order; :func:`_bridge` sorts them by case, so
    the output is the one computing every case gives.
    """
    from scripts.public_ownership.cfd_repricing import register_settings_by_case
    from scripts.public_ownership.register_io import frames_identical, iter_register_cases, read_register

    settings = register_settings_by_case(register_path)
    vintaged = fingerprints.register_vintaged(register_path)

    def comparator_of(case_id):
        return comparators.get(case_id, "private")

    def private_of(case_id):
        return private_by_case.get(comparator_of(case_id), private_by_case["private"])

    key_of = {case_id: (fingerprints.bridge_key(case_id, settings[case_id], comparator_of(case_id), vintaged=vintaged)
                        if case_id in settings else ("case", case_id)) for case_id in cases}
    first = {}
    for case_id in cases:
        first.setdefault(key_of[case_id], case_id)
    computed = {}
    for rows in iter_register_cases(register_path, cases=list(first.values()), columns=list(REGISTER_COLUMNS),
                                    batch_cases=batch_cases):
        for case_id, frame in rows.groupby("case_id"):
            computed[key_of[case_id]] = memo.savings(case_id, frame, private_of(case_id))
    sample = fingerprints.SampleCheck("bridge._technology_savings")
    output = []
    for case_id in cases:
        table = computed[key_of[case_id]]
        if first[key_of[case_id]] != case_id:
            table = table.copy()
            table["case_id"] = case_id
            if sample.due():
                own = read_register(register_path, columns=list(REGISTER_COLUMNS), cases=[case_id])
                sample.check(case_id, frames_identical(_technology_savings(case_id, own, private_of(case_id)), table))
        output.append(table)
    print(sample.report())
    return output


def _new_cfd_saving_by_case(by_technology: pd.DataFrame):
    """``case_id -> float``: the case's saving on new-build CfD rows of the technology table (F20 phase 1).

    The rows ``by_technology.loc[case & new_build & cfd, saving].sum()`` selected for each
    case, in the same order and summed the same way, found with one pass over the table;
    a case with no such rows, or an empty table, is 0.0 as before.
    """
    if by_technology.empty:
        return lambda case_id: 0.0
    mask = (by_technology["finance_vintage"].eq("new_build")
            & by_technology["support_type"].astype(str).str.lower().eq("cfd"))
    rows = np.flatnonzero(mask.to_numpy())
    saving = by_technology["annual_financing_saving_gbp"]
    # Each case's rows as positions in the table (in table order), taken and summed as the boolean filter was.
    groups = pd.Series(rows).groupby(by_technology["case_id"].to_numpy()[rows], sort=False).indices
    sums = {case_id: float(saving.take(rows[positions]).sum()) for case_id, positions in groups.items()}
    return lambda case_id: sums.get(case_id, 0.0)


def _new_cfd_saving_reference(by_technology: pd.DataFrame, case_id) -> float:
    """One case's new-CfD saving as :func:`_bridge` read it before F20 (a scan of the whole table; the tests' reference)."""
    return (
        float(by_technology.loc[
            by_technology["case_id"].eq(case_id)
            & by_technology["finance_vintage"].eq("new_build")
            & by_technology["support_type"].astype(str).str.lower().eq("cfd"),
            "annual_financing_saving_gbp",
        ].sum())
        if not by_technology.empty else 0.0
    )


def _bridge(output: list[pd.DataFrame], summary: pd.DataFrame, benchmark_kwh: float):
    """The technology table, the bill-category bridge and the summary from each case's technology rows."""
    by_technology = pd.concat(output, ignore_index=True) if output else pd.DataFrame()
    if not by_technology.empty:
        by_technology = by_technology.sort_values(
            ["case_id", "annual_financing_saving_gbp"], ascending=[True, False]
        ).reset_index(drop=True)

    # F20 phase 1: each case's new-CfD saving read once (it was two scans of the whole table per case).
    new_cfd_saving = _new_cfd_saving_by_case(by_technology)
    bridge_rows = []
    for row in summary.itertuples(index=False):
        case_id = row.case_id
        saving = float(row.annual_financing_saving_vs_private_gbp)
        per_mwh = float(row.consumer_equivalent_saving_gbp_per_mwh_demand)
        cfd_saving = new_cfd_saving(case_id)
        non_cfd_saving = saving - cfd_saving
        for group in BILL_GROUPS_HELD_CONSTANT:
            bridge_rows.append({
                "case_id": case_id,
                "consumer_bill_group": group,
                "annual_system_change_gbp": 0.0,
                "equivalent_gbp_per_mwh_demand": 0.0,
                "equivalent_gbp_per_3100kwh_ex_vat": 0.0,
                "treatment": "held_unchanged",
                "note": "Existing wholesale, support, network, supplier and VAT recovery rules are unchanged; new-CfD savings are shown separately.",
            })
        bridge_rows.append({
            "case_id": case_id,
            "consumer_bill_group": "Policy costs - new CfD contracts",
            "annual_system_change_gbp": -cfd_saving,
            "equivalent_gbp_per_mwh_demand": -cfd_saving / saving * per_mwh if saving else 0.0,
            "equivalent_gbp_per_3100kwh_ex_vat": -cfd_saving / saving * per_mwh * benchmark_kwh / 1000.0 if saving else 0.0,
            "treatment": "new_cfd_contract_revenue_requirement_reduction",
            "note": "Lower required revenue for post-2025 assets identified as CfD. Existing CfD and ROC contracts are unchanged; this is the new-contract cost reduction before a levy-recovery calculation.",
        })
        bridge_rows.append({
            "case_id": case_id,
            "consumer_bill_group": "Generation capital financing (non-CfD, unallocated)",
            "annual_system_change_gbp": -non_cfd_saving,
            "equivalent_gbp_per_mwh_demand": -non_cfd_saving / saving * per_mwh if saving else 0.0,
            "equivalent_gbp_per_3100kwh_ex_vat": -non_cfd_saving / saving * per_mwh * benchmark_kwh / 1000.0 if saving else 0.0,
            "treatment": "unallocated_financing_saving",
            "note": "Merchant, storage and unclassified assets have no specified bill-recovery route; this is a volumetric equivalent only.",
        })
    bill_bridge = pd.DataFrame(bridge_rows)
    report = {
        "scope": "Fixed-dispatch post-2025 generation/storage finance comparison; existing assets, CfD/ROC contracts and network costs unchanged.",
        "interpretation": "Savings arise wholly from lower capital recovery. Post-2025 CfD assets reduce the new CfD-contract revenue requirement; non-CfD savings remain unallocated without a recovery rule.",
        "benchmark_kwh": benchmark_kwh,
        "cases": {
            row.case_id: {
                "annual_financing_saving_gbp": float(row.annual_financing_saving_vs_private_gbp),
                "new_cfd_contract_saving_gbp": new_cfd_saving(row.case_id),
                "consumer_equivalent_gbp_per_mwh_demand": float(row.consumer_equivalent_saving_gbp_per_mwh_demand),
                "consumer_equivalent_gbp_per_3100kwh_ex_vat": float(row.consumer_equivalent_saving_gbp_per_mwh_demand) * benchmark_kwh / 1000.0,
            }
            for row in summary.itertuples(index=False)
        },
    }
    return by_technology, bill_bridge, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_finance_register")
    parser.add_argument("financing_summary")
    parser.add_argument("technology_output")
    parser.add_argument("bill_bridge_output")
    parser.add_argument("summary_output")
    parser.add_argument("--benchmark-kwh", type=float, default=3100.0)
    parser.add_argument("--batch-cases", type=int, default=16, help="cases read at a time")
    args = parser.parse_args()
    # F20 phase 0: opt-in stage timing (PYPSA_GB_STAGE_TIMING); without it nothing is wrapped.
    from scripts.public_ownership import stage_timing

    stage_timing.instrument_register_io()
    stage_timing.instrument(globals(), ("_technology_savings", "_bridge", "_new_cfd_saving_by_case", "_shared_tables"),
                            prefix="bridge")
    stage_timing.instrument(_SavingsMemo, ("savings",), prefix="bridge._SavingsMemo")
    # 30 Sep 2026: the register is read a batch of cases at a time (memory flat in the case count).
    technology, bridge, report = build_consumer_bill_financing_bridge_from_file(
        args.asset_finance_register, pd.read_csv(args.financing_summary), benchmark_kwh=args.benchmark_kwh,
        batch_cases=args.batch_cases,
    )
    for output in (args.technology_output, args.bill_bridge_output, args.summary_output):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    technology.to_csv(args.technology_output, index=False)
    bridge.to_csv(args.bill_bridge_output, index=False)
    Path(args.summary_output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
