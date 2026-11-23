"""Hourly boundary derate factors from NESO's day-ahead constraint limits (9 Oct 2026).

NESO's thermal constraint tool takes "maintenance outage durations by boundary - historic data, TO" (tCSNP2
Issue 10.1, Table 1). The model's secured boundaries otherwise carry only the NOA seasonal factors (NOA
methodology Table 2.3), one profile for every boundary. This builds the historic equivalent: for each ETYS
boundary NESO maps to a day-ahead constraint group (CP30 Data Workbook sheet NA1, "Equivalent ETYS name"),
the group's day-ahead limit in each hour as a fraction of its intact winter limit (the given quantile of its
November to February limits), averaged over every year of the window by month, day and hour. The fraction carries the season and the outages
NESO actually planned for: it replaces the NOA factor on those boundaries
(``secured_capability.historic_derate`` in scripts/solve/secured_boundaries.py).

Forward look: NESO's tool uses historic outage data, but CP30's NA1 sheet also gives each boundary's
construction outage days by financial year to 2030/31. ``na1_ratio`` is the target calendar year's outage days
(3/12 of the financial year it starts in, 9/12 of the next) over the reference financial year's; the solve can
scale the part of the historic derate beyond the NOA seasonal allowance by it (``mode: forward``).

Outage days (``--method outage_days``, NESO's own representation): NESO's constraint models apply the
seasonal rating factors to the intact capability all year and an outage multiplier on a boundary-specific
number of days set from historic TO maintenance-outage durations ("Final Boundary capability (hourly) = Min(...
x Thermal profile (e.g. seasonal)...) x Outage profile", NG Long-term Market and Network Constraint Modelling,
Jan 2017, Fig. A1; tCSNP2 Inputs annex Table 1). Construction outages are left out of the long-term cost model.
This method flags a day as an outage day when its lowest day-ahead limit is more than ``--threshold`` below the
intact seasonal level, and writes the factor as the intact seasonal level (thermal: winter 1.00, spring and
autumn 0.85, summer May-Aug 0.80; voltage and stability 1.00) on ordinary days and that level times NOA's
prior-outage ratio on outage days. NOA's "summer outage" factor is summer ratings with a circuit on prior
outage (NOA methodology 4.1, Appendix B), so the ratio is 0.70 / 0.80 = 0.875 for thermal limits (0.875 in
winter, about 0.74 in spring and autumn, 0.70 in summer), 0.90 for voltage limits and 1.00 for stability. Averaged over several years, a day's factor is the mean of its years.
The limit type comes from the capability file. Use a year with little construction (2022/23) for maintenance
and a construction-heavy year (2024/25) as the sensitivity.

Generic outages (``--method outage_generic``): the same outage days, spread evenly through each month, so a
boundary's factor in a month is its intact seasonal level less the month's outage share times the drop to the
prior-outage level. NESO's zonal PLEXOS model applies "a seasonal profile ... Further scaling of the maximum
flows ... to represent generic maintenance outages throughout the year" (NESO SSEP methodology, May 2025,
p.60). Unlike ``outage_days``, the result does not depend on whether a historic outage day falls in a windy
week of the modelled weather year.

Boundaries without day-ahead data: both outage methods also write rows ``_default_thermal``,
``_default_voltage`` and ``_default_stability``: the intact seasonal level less the mean monthly outage share of
the mapped boundaries with the same limit type (stability takes the thermal share) times the drop to the
prior-outage level. ``historic_derate.unmapped: default`` applies
them to every other boundary; the default (``noa``) leaves those on the NOA factors.

Output: one row per boundary, month, day and hour (no 29 February), columns boundary, group, month, day, hour,
factor, na1_ratio, plus a per-boundary summary on stdout. Hours with no published limit take that month's mean
factor.

usage: python scripts/solve/build_boundary_historic_derate.py [--start 2024-10-01] [--end 2025-09-30]
       [--quantile 0.9] [--floor 0.25] [--target-year 2030] [--na1-reference-fy 2024/25]
       [--out data/network/neso_boundary_historic_derate_2024_25.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = "resources/neso_cache/day_ahead_constraint_flows_limits.csv"
# CP30 Data Workbook V003, sheet NA1 ("Boundary name" -> "Equivalent ETYS name"). NESO renamed SEIMP to
# SEIMPPR23, SSE-SP to SSE-SP2 and SSHARN to SSHARN3 on 21 Apr 2024, and published SSHARN as SHARN from
# Jul 2022 to Apr 2023: a boundary's series joins its groups, each normalised to its own intact winter limit
# (its winter limits in the window, or over its whole history when the window holds too few). SWALEX (SW1) and ESTEX (SC3) stop in 2023 and SSENWEX9
# (B1) is not in the day-ahead dataset, so none of the three is derated.
NA1_GROUPS = {"B6": ["SCOTEX"], "B4": ["SSE-SP", "SSE-SP2"], "B9": ["FLOWSTH"], "LE1": ["SEIMP", "SEIMPPR23"],
              "B7a": ["SHARN", "SSHARN", "SSHARN3"]}
WINTER = (11, 12, 1, 2)
DEFAULT_CAPABILITY = "data/network/etys_boundary_capability.csv"
# Intact seasonal level and NOA prior-outage level, both as fractions of winter capability (NOA methodology
# Issue 8, seasonal fallback factors; NOA methodology 4.1 Appendix B: "summer outage" is capability with a
# circuit on prior outage).
INTACT = {"thermal": {1: 1.0, 2: 1.0, 3: 0.85, 4: 0.85, 5: 0.80, 6: 0.80, 7: 0.80, 8: 0.80, 9: 0.85, 10: 0.85,
                      11: 1.0, 12: 1.0},
          "voltage": {m: 1.0 for m in range(1, 13)},
          "stability": {m: 1.0 for m in range(1, 13)}}
# Capability with a circuit on prior outage over intact capability, same season: NOA's summer outage factor
# over its summer factor for thermal limits (0.70 / 0.80), and its voltage summer outage factor (0.90).
OUTAGE_RATIO = {"thermal": 0.70 / 0.80, "voltage": 0.90, "stability": 1.0}
DEFAULT_NA1 = "resources/CP30/CP30_reference_N1_N2_NA1.csv"


def _na1_days(na1: pd.DataFrame, boundary: str, fy: str) -> float:
    """Construction outage days NA1 gives a boundary in financial year ``fy`` (e.g. '2024/25')."""
    cols = [c for c in na1.columns if str(c).startswith("Capacity Used") and fy in str(c)]
    if len(cols) != 1:
        raise ValueError(f"NA1 has {len(cols)} 'Capacity Used' columns for {fy}")
    row = na1[na1["Equivalent ETYS name"].astype(str).str.strip() == boundary]
    if len(row) != 1:
        raise ValueError(f"NA1 has {len(row)} rows for boundary {boundary}")
    return float(row[cols[0]].iloc[0])


def na1_ratio(na1_path: str | Path, boundary: str, reference_fy: str, target_year: int) -> float:
    """Target calendar year's NA1 outage days over the reference financial year's (e.g. '2024/25').

    NA1's per-boundary columns start at 2024/25, so a multi-year history is still scaled against 2024/25.
    """
    na1 = pd.read_csv(na1_path, low_memory=False)
    na1 = na1[na1["source_sheet"] == "NA1"]
    ref = _na1_days(na1, boundary, reference_fy)
    early = _na1_days(na1, boundary, f"{target_year - 1}/{str(target_year)[-2:]}")
    late = _na1_days(na1, boundary, f"{target_year}/{str(target_year + 1)[-2:]}")
    target = 0.25 * early + 0.75 * late
    return target / ref if ref > 0 else 1.0


def build(source: str | Path, start: str, end: str, quantile: float, floor: float,
          na1_path: str | Path = DEFAULT_NA1, target_year: int = 2030,
          na1_reference_fy: str = "2024/25", method: str = "ratio", threshold: float = 0.10,
          capability_file: str | Path = DEFAULT_CAPABILITY, pathway: str = "HT") -> pd.DataFrame:
    """Hourly factors averaged over every year of [start, end] by month, day and hour."""
    d = pd.read_csv(source, low_memory=False, usecols=["Constraint Group", "Date (GMT/BST)", "Limit (MW)"])
    d["t"] = pd.to_datetime(d["Date (GMT/BST)"], format="ISO8601")
    d = d.dropna(subset=["Limit (MW)"])
    history = d
    d = d[(d["t"] >= start) & (d["t"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    hours = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    hours = hours[~((hours.month == 2) & (hours.day == 29))]
    rows, shares = [], []
    for boundary, groups in NA1_GROUPS.items():
        parts, refs = [], []
        for group in groups:
            x = d[d["Constraint Group"] == group].set_index("t")["Limit (MW)"].astype(float)
            if x.empty:
                continue
            winter = x[x.index.month.isin(WINTER)]
            if len(winter) < 100:
                full = history[history["Constraint Group"] == group].set_index("t")["Limit (MW)"].astype(float)
                winter = full[full.index.month.isin(WINTER)]
                if len(winter) < 100:
                    raise ValueError(f"{group} ({boundary}) has no winter limits to normalise by")
            ref = float(winter.quantile(quantile))
            refs.append(f"{group} {ref:,.0f}")
            parts.append(x.resample("h").mean() / ref)
        if not parts:
            raise ValueError(f"{boundary} ({groups}) has no day-ahead limits between {start} and {end}")
        hourly = pd.concat(parts).groupby(level=0).mean().reindex(hours)
        if method in ("outage_days", "outage_generic"):
            cap = pd.read_csv(capability_file)
            lt = cap[(cap["boundary"] == boundary) & (cap["fes_pathway"] == pathway) & (cap["year"] == target_year)]
            limit_type = str(lt["limit_type"].iloc[0]) if len(lt) else "thermal"
            limit_type = limit_type if limit_type in INTACT else "thermal"
            intact_h = pd.Series([INTACT[limit_type][m] for m in hours.month], index=hours)
            daily_min = hourly.resample("D").min()
            intact_d = pd.Series([INTACT[limit_type][m] for m in daily_min.index.month], index=daily_min.index)
            flag_d = (daily_min < (1.0 - threshold) * intact_d).astype(float).where(daily_min.notna())
            # A day with no published limit takes its month's outage share.
            flag_d = flag_d.fillna(flag_d.groupby(flag_d.index.month).transform("mean")).fillna(0.0)
            flag_h = flag_d.reindex(hours.normalize()).to_numpy()
            factor = pd.Series(intact_h.to_numpy() * (1.0 - flag_h * (1.0 - OUTAGE_RATIO[limit_type])), index=hours)
            outage_days = float(flag_d.sum()) * 365.0 / max(len(flag_d), 1)
            shares.append((limit_type, flag_d.groupby(flag_d.index.month).mean().rename(boundary)))
        else:
            factor = hourly.clip(lower=floor, upper=1.0)
            filled = factor.groupby(factor.index.month).transform("mean")
            factor = factor.fillna(filled).fillna(1.0)
        slot = factor.groupby([factor.index.month, factor.index.day, factor.index.hour]).mean()
        slot.index.names = ["month", "day", "hour"]
        if method == "outage_generic":
            slot = slot.groupby(level="month").transform("mean")
        ratio = na1_ratio(na1_path, boundary, na1_reference_fy, target_year)
        rows.append(slot.round(4).rename("factor").reset_index().assign(
            boundary=boundary, group="+".join(groups), na1_ratio=round(ratio, 4)))
        years = factor.index.to_period("Y-SEP").nunique()
        if method in ("outage_days", "outage_generic"):
            print(f"{boundary:4} {limit_type:8} outage days a year {outage_days:5.0f}  mean factor {slot.mean():.3f}  "
                  f"winter {slot[slot.index.get_level_values('month').isin(WINTER)].mean():.3f}  "
                  f"Jun-Aug {slot[slot.index.get_level_values('month').isin((6, 7, 8))].mean():.3f}  years {years}")
            continue
        print(f"{boundary:4} {'+'.join(groups):18} ref {'; '.join(refs):30} mean {slot.mean():.3f}  "
              f"winter {slot[slot.index.get_level_values('month').isin(WINTER)].mean():.3f}  "
              f"Jun-Aug {slot[slot.index.get_level_values('month').isin((6, 7, 8))].mean():.3f}  "
              f"years {years}  hours missing {int(hourly.isna().sum())}  NA1 {target_year}/{na1_reference_fy} {ratio:.2f}")
    if shares:
        slots = rows[0][["month", "day", "hour"]]
        for limit_type, intact in INTACT.items():
            source = "voltage" if limit_type == "voltage" else "thermal"
            members = [x for t, x in shares if t == source]
            if not members:
                continue
            share = pd.concat(members, axis=1).mean(axis=1)    # mean monthly outage share, same limit type
            m = slots["month"]
            f = m.map(intact) * (1.0 - m.map(share) * (1.0 - OUTAGE_RATIO[limit_type]))
            rows.append(slots.assign(factor=f.round(4).to_numpy(), boundary=f"_default_{limit_type}",
                                     group="mean of " + "+".join(x.name for x in members), na1_ratio=1.0))
            print(f"_default_{limit_type:9} share by month " + " ".join(f"{k}:{v:.2f}" for k, v in share.items())
                  + f"  ({share.mean() * 365:.0f} days a year)  mean factor {f.mean():.3f}")
    cols = ["boundary", "group", "month", "day", "hour", "factor", "na1_ratio"]
    return pd.concat(rows, ignore_index=True)[cols]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--start", default="2024-10-01")
    p.add_argument("--end", default="2025-09-30")
    p.add_argument("--quantile", type=float, default=0.9)
    p.add_argument("--floor", type=float, default=0.25)
    p.add_argument("--na1", default=DEFAULT_NA1)
    p.add_argument("--target-year", type=int, default=2030)
    p.add_argument("--na1-reference-fy", default="2024/25")
    p.add_argument("--method", choices=["ratio", "outage_days", "outage_generic"], default="ratio")
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--capability-file", default=DEFAULT_CAPABILITY)
    p.add_argument("--out", default="data/network/neso_boundary_historic_derate_2024_25.csv")
    a = p.parse_args()
    table = build(a.source, a.start, a.end, a.quantile, a.floor, a.na1, a.target_year, a.na1_reference_fy,
                  a.method, a.threshold, a.capability_file)
    table.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(table):,} rows")


if __name__ == "__main__":
    main()
