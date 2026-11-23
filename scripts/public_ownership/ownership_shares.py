"""Per-generator state ownership shares, for the balancing bid stack (item BM1).

The finance register is **finer than the LP**. It splits a network generator into
``<generator>::legacy`` and ``<generator>::new_build`` cohorts and owns them
separately, so under the CP30 2030 fleet 1,902 of 12,233 units -- 70.0 GW of
221.4 GW -- are part legacy and part new build. The balancing LP has one bid per
generator, so ownership has to reach it as a **capacity-weighted share**:

    share(g) = sum over cohorts of (ownership_share x capacity_mw) / capacity_mw

which is exactly how ``ownership_share`` already works everywhere else in the
finance layer (a 30% stake carries 30% of everything). The alternative, a 50%
threshold, is discontinuous and would flip almost every mixed unit anyway: the
median new-build share inside one is 0.976.

Nothing here decides *who* owns what. That is
``calculate_financing_scenarios._state_owned``, reading the levers, and this
module only aggregates the register it wrote -- so the bid stack and the bill
cannot disagree about the ownership boundary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: The register's cohort suffix separator (``<generator>::<vintage>``).
COHORT_SEPARATOR = "::"
#: Columns the aggregation needs; the register is ~600 MB, so never read it whole.
REGISTER_COLUMNS = ("case_id", "asset_id", "capacity_mw", "ownership_share")
#: Read when present: the balancing tie-break (decision 20) ranks on it.
OPTIONAL_REGISTER_COLUMNS = ("commissioning_year",)
OUTPUT_COLUMNS = ("case_id", "generator", "capacity_mw", "state_capacity_mw", "state_share",
                  "state_commissioning_year")


def generator_name(asset_id) -> str:
    """``WIND_X::new_build`` -> ``WIND_X``: the LP generator a cohort belongs to."""
    return str(asset_id).split(COHORT_SEPARATOR, 1)[0]


def generator_state_shares(register: pd.DataFrame, case_id: str | None = None) -> pd.DataFrame:
    """Capacity-weighted state share per network generator, per case.

    A row whose ``ownership_share`` is missing counts as privately owned, which is
    the only safe reading: an unresolved cohort must not be handed to the state by
    accident.
    """
    missing = set(REGISTER_COLUMNS) - set(register.columns)
    if missing:
        raise ValueError(f"register lacks columns {sorted(missing)}")
    frame = register.copy()
    if case_id is not None:
        frame = frame.loc[frame["case_id"].astype(str).eq(str(case_id))]
        if frame.empty:
            raise ValueError(f"register has no rows for case {case_id!r}")
    frame["generator"] = frame["asset_id"].map(generator_name)
    capacity = pd.to_numeric(frame["capacity_mw"], errors="coerce").fillna(0.0)
    share = pd.to_numeric(frame["ownership_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    year = pd.to_numeric(frame.get("commissioning_year", pd.Series(float("nan"), index=frame.index)),
                         errors="coerce")
    state = share * capacity
    # The state-owned capacity's commissioning year, capacity-weighted across the
    # generator's cohorts: what the P3 tie-break ranks on (decision 20).
    dated = state.where(year.notna(), 0.0)
    frame = frame.assign(_capacity=capacity, _state=state, _dated=dated, _year_mw=(year * dated).fillna(0.0))
    grouped = frame.groupby(["case_id", "generator"], sort=False)[["_capacity", "_state", "_dated", "_year_mw"]].sum()
    grouped = grouped.reset_index().rename(columns={"_capacity": "capacity_mw", "_state": "state_capacity_mw"})
    grouped["state_commissioning_year"] = grouped["_year_mw"].div(grouped["_dated"]).where(grouped["_dated"] > 0.0)
    # A zero-capacity generator carries no ownership to blend with, so it is private.
    grouped["state_share"] = (
        grouped["state_capacity_mw"].div(grouped["capacity_mw"]).where(grouped["capacity_mw"] > 0.0, 0.0)
    ).clip(0.0, 1.0)
    return grouped[list(OUTPUT_COLUMNS)]


def load_shares(path, case_id: str) -> pd.Series:
    """Read a saved share table and return ``generator -> state_share`` for one case.

    Used by the balancing stage, which must not read the 600 MB register.
    """
    table = pd.read_csv(path)
    rows = table.loc[table["case_id"].astype(str).eq(str(case_id))]
    if rows.empty:
        available = sorted(table["case_id"].astype(str).unique())
        raise ValueError(
            f"ownership share table {path} has no rows for case {case_id!r}; it carries {available}")
    return pd.Series(
        pd.to_numeric(rows["state_share"], errors="coerce").fillna(0.0).to_numpy(),
        index=rows["generator"].astype(str).to_numpy(), dtype=float)


def load_commissioning_years(path, case_id: str) -> pd.Series:
    """``generator -> state_commissioning_year`` for one case, NaN where nothing dated is owned."""
    table = pd.read_csv(path)
    if "state_commissioning_year" not in table.columns:
        raise ValueError(f"{path} has no state_commissioning_year; regenerate it with ownership_shares.py")
    rows = table.loc[table["case_id"].astype(str).eq(str(case_id))]
    if rows.empty:
        raise ValueError(f"ownership share table {path} has no rows for case {case_id!r}")
    return pd.Series(pd.to_numeric(rows["state_commissioning_year"], errors="coerce").to_numpy(),
                     index=rows["generator"].astype(str).to_numpy(), dtype=float)


#: Lever L4 of the FES 2050 plan (decision 13): the P2 shares whose balancing solves
#: each need their own bid stack (decision 16), as percentages.
DEPLOYMENT_SHARE_PERCENTS = (25, 50, 75, 100)


def deployment_share_case_id(percent: int, scenario: str = "public_ownership") -> str:
    """``50`` -> ``public_ownership@deployment_share_50``: the case lever L4 at that share makes under P2."""
    from scripts.public_ownership.levers import GATED_LEVERS, LEVER_STATE_DEPLOYMENT_SHARE, VARIANT_SEPARATOR

    value = f"{int(percent) / 100:.2f}"
    lever = GATED_LEVERS[LEVER_STATE_DEPLOYMENT_SHARE]
    if value not in lever.values:
        raise ValueError(f"no state_deployment_share value for {percent}%; supported: {list(lever.values[1:])}")
    return f"{scenario}{VARIANT_SEPARATOR}{lever.label(value)}"


def deployment_share_table_path(output_prefix, percent: int) -> Path:
    """``<prefix>_deployment_share_<pct>.csv``: one share table per L4 share."""
    prefix = Path(output_prefix)
    return prefix.with_name(f"{prefix.name}_deployment_share_{int(percent)}.csv")


def write_deployment_share_tables(register_path, output_prefix, percents=DEPLOYMENT_SHARE_PERCENTS,
                                  scenario: str = "public_ownership") -> dict[int, Path]:
    """Write one BM1 share table per L4 deployment share, for P2's per-share balancing solves (task Y9).

    Each table is :func:`generator_state_shares` of the one case lever L4 makes at
    that share (``public_ownership@deployment_share_<pct>``), in the format
    :func:`load_shares` reads, so a balancing month bids on exactly that fleet.
    The ownership boundary is the register's (``calculate_financing_scenarios``),
    never recomputed here.

    Args:
        register_path: The group's finance register (Parquet or CSV).
        output_prefix: Path prefix; tables are ``<prefix>_deployment_share_<pct>.csv``.
        percents: The shares, in percent.
        scenario: The P2 scenario id.

    Returns:
        ``{percent: path}``.

    Raises:
        ValueError: The register carries no case for a requested share.
    """
    from scripts.public_ownership.register_io import read_register

    cases = {int(p): deployment_share_case_id(p, scenario) for p in percents}
    columns = list(REGISTER_COLUMNS) + list(OPTIONAL_REGISTER_COLUMNS)
    register = read_register(register_path, columns=columns, cases=list(cases.values()))
    found = set(register["case_id"].astype(str))
    missing = sorted(case for case in cases.values() if case not in found)
    if missing:
        raise ValueError(f"register {register_path} has no rows for {missing}; configure state_deployment_share "
                         "for the run group")
    out = {}
    for percent, case in cases.items():
        path = deployment_share_table_path(output_prefix, percent)
        path.parent.mkdir(parents=True, exist_ok=True)
        generator_state_shares(register, case).to_csv(path, index=False)
        out[percent] = path
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("register", help="the per-case asset finance register CSV")
    parser.add_argument("output", help="where to write the per-generator share table")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="restrict to these case ids (default: every case in the register)")
    parser.add_argument("--deployment-shares", nargs="*", type=int, default=None,
                        help="FES 2050 lever L4: write one table per share (percent), "
                             "<OUTPUT without .csv>_deployment_share_<pct>.csv, instead of the full table")
    args = parser.parse_args()

    if args.deployment_shares is not None:
        written = write_deployment_share_tables(args.register, Path(args.output).with_suffix(""),
                                                args.deployment_shares or DEPLOYMENT_SHARE_PERCENTS)
        for percent, path in written.items():
            print(f"{percent}%: {path}")
        return

    from scripts.public_ownership.register_io import iter_register_cases, read_register

    columns = list(REGISTER_COLUMNS) + list(OPTIONAL_REGISTER_COLUMNS)
    if args.cases:
        register = read_register(args.register, columns=columns, cases=args.cases)
        register = register.loc[register["case_id"].astype(str).isin(args.cases)]
        found = set(register["case_id"].astype(str))
        unknown = sorted(set(args.cases) - found)
        if unknown:
            raise SystemExit(f"register has no rows for {unknown}")
        chunks = [generator_state_shares(register)]
    else:
        # 30 Sep 2026: whole register keys at a time, in register order (memory flat in the
        # case count); a (case, generator) pair never spans two keys, so the order is the same.
        chunks = (generator_state_shares(frame)
                  for frame in iter_register_cases(args.register, columns=columns, batch_cases=None))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = {}
    for i, shares in enumerate(chunks):
        # Written as it is made: the table is about a row per generator per case.
        shares.to_csv(args.output, index=False, mode="w" if i == 0 else "a", header=i == 0)
        by_case = shares.groupby("case_id").apply(
            lambda g: g.state_capacity_mw.sum() / g.capacity_mw.sum(), include_groups=False)
        for case, value in by_case.items():
            report[case] = (value, shares.loc[shares.case_id.eq(case), "capacity_mw"].sum())
    for case in sorted(report):
        value, capacity = report[case]
        print(f"{case}: {value:.1%} of {capacity / 1e3:.1f} GW state-owned")


if __name__ == "__main__":
    main()
