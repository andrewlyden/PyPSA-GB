"""Materialise a domestic consumption shape from Ofgem's base/peak allocation.

The bill's wholesale line currently prices a system-shaped MWh: it weights the
hourly PyPSA clearing price by total system load. Ofgem's cap applies a domestic
shaping adjustment because household consumption is peak-heavy, and that
adjustment is presently buried inside the flat retail procurement bridge.

Ofgem's own shaping input is the base/peak demand allocation published in Annex
2 of the cap model and already pinned locally in
``data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv``: for the
2020 cap periods, 70% of the benchmark customer's volume is bought as baseload
product and 30% as peak product. Those are *product* volumes, not consumption
blocks, so they translate into a shape as follows.

Let ``V`` be annual consumption, ``H`` all hours and ``P`` the peak-block hours.
The baseload product covers every hour at a flat rate ``b``, and the peak
product adds ``p`` during peak hours only:

    b * H = base_share  * V        ->  b = base_share  * V / H
    p * P = peak_share  * V        ->  p = peak_share  * V / P

Consumption is therefore ``b + p`` in a peak hour and ``b`` elsewhere, which for
the 2020 shares puts about 55% of household volume into the 36% of hours that
form the peak block. A flat profile would put 36% there, so this is a genuinely
peak-weighted domestic shape rather than a relabelled average.

The peak block is the Elexon/EFA definition: 07:00 to 19:00, Monday to Friday,
in London local time, so British Summer Time shifts it against UTC. England and
Wales bank holidays are not peak days (F18, 27 Oct 2026: Ofgem's cap model, Annex 2
sheet ``3b Demand``, carries the list), read from the committed calendar
``data/market/consumer_bill_sources/england_wales_bank_holidays.csv``
(:mod:`scripts.consumer_bill.bank_holidays`).
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.consumer_bill.bank_holidays import CALENDAR_FILE, is_bank_holiday  # noqa: E402


PEAK_START_HOUR = 7
PEAK_END_HOUR = 19
LOCAL_ZONE = "Europe/London"

OFGEM_SOURCE = {
    "publisher": "Ofgem",
    "dataset": "Default tariff cap model, Annex 2 wholesale, base and peak demand allocation",
    "local_file": "data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv",
    "columns": ["base_share", "peak_share"],
    "cap_periods": ["2020_Jan_Mar", "2020_Apr_Sep", "2020_Oct_Dec"],
    "peak_block_definition": (
        "Elexon/EFA peak block: 07:00-19:00 Monday to Friday, London local time, "
        "England and Wales bank holidays excluded"
    ),
    "bank_holidays": CALENDAR_FILE.relative_to(CALENDAR_FILE.parents[3]).as_posix(),
}


def _shares(hedged_inputs: pd.DataFrame) -> tuple[float, float]:
    """Return the single base/peak allocation the cap periods agree on."""
    for column in ("base_share", "peak_share"):
        if column not in hedged_inputs.columns:
            raise ValueError(f"Ofgem hedged-wholesale inputs lack {column}")
    base = pd.to_numeric(hedged_inputs["base_share"], errors="raise").round(12).unique()
    peak = pd.to_numeric(hedged_inputs["peak_share"], errors="raise").round(12).unique()
    if len(base) != 1 or len(peak) != 1:
        raise ValueError(
            "Cap periods declare different base/peak allocations; a single annual shape cannot "
            f"represent them (base={base}, peak={peak})"
        )
    if not np.isclose(base[0] + peak[0], 1.0, rtol=0, atol=1e-9):
        raise ValueError("Base and peak shares must sum to one")
    return float(base[0]), float(peak[0])


def peak_mask(snapshots: pd.DatetimeIndex, bank_holidays: bool = True) -> pd.Series:
    """Flag snapshots inside the Elexon peak block, evaluated in local time.

    A weekday that is an England and Wales bank holiday (by its local date) is not a
    peak day; ``bank_holidays=False`` gives the plain weekday block (before 27 Oct 2026).
    """
    index = pd.DatetimeIndex(snapshots)
    localised = (
        index.tz_localize("UTC").tz_convert(LOCAL_ZONE) if index.tz is None
        else index.tz_convert(LOCAL_ZONE)
    )
    in_block = ((localised.dayofweek < 5)
                & (localised.hour >= PEAK_START_HOUR)
                & (localised.hour < PEAK_END_HOUR))
    if bank_holidays and len(index):
        in_block = in_block & ~is_bank_holiday(localised.tz_localize(None).normalize()).to_numpy()
    return pd.Series(in_block, index=index)


def build_profile(snapshots: pd.DatetimeIndex, base_share: float, peak_share: float) -> pd.Series:
    """Return relative hourly domestic demand, normalised to sum to one."""
    in_peak = peak_mask(snapshots)
    total_hours = float(len(in_peak))
    peak_hours = float(in_peak.sum())
    if peak_hours <= 0:
        raise ValueError("Snapshot set contains no peak-block hours")
    baseload_rate = base_share / total_hours
    peak_rate = peak_share / peak_hours
    profile = pd.Series(baseload_rate, index=pd.DatetimeIndex(snapshots))
    profile[in_peak.to_numpy()] += peak_rate
    total = float(profile.sum())
    if not np.isclose(total, 1.0, rtol=0, atol=1e-9):
        raise ValueError(f"Domestic profile does not sum to one: {total}")
    return profile


def write_profile(hedged_inputs_path, output_csv, output_summary, profile_year: int):
    hedged = pd.read_csv(hedged_inputs_path)
    base_share, peak_share = _shares(hedged)
    snapshots = pd.date_range(f"{profile_year}-01-01", f"{profile_year + 1}-01-01",
                              freq="h", inclusive="left")
    profile = build_profile(snapshots, base_share, peak_share)
    in_peak = peak_mask(snapshots)

    frame = pd.DataFrame({
        "relative_demand": profile.to_numpy(),
        "in_peak_block": in_peak.to_numpy(),
        "day_of_week": pd.DatetimeIndex(snapshots).dayofweek,
        "local_hour": pd.DatetimeIndex(snapshots).tz_localize("UTC")
                        .tz_convert(LOCAL_ZONE).hour,
    }, index=pd.DatetimeIndex(snapshots))
    frame.index.name = "snapshot_utc"

    summary = {
        "purpose": (
            "Relative hourly domestic electricity consumption for a single-rate customer, used "
            "to weight the hourly PyPSA clearing price into a household-shaped energy cost."
        ),
        "profile_year": profile_year,
        "hours": int(len(frame)),
        "base_share": base_share,
        "peak_share": peak_share,
        "peak_block_hours": int(in_peak.sum()),
        "peak_block_share_of_hours": float(in_peak.mean()),
        "peak_block_share_of_consumption": float(profile[in_peak.to_numpy()].sum()),
        "peak_to_offpeak_rate_ratio": float(
            profile[in_peak.to_numpy()].iloc[0] / profile[~in_peak.to_numpy()].iloc[0]
        ),
        "method": (
            "Ofgem base/peak demand allocation converted to an hourly shape: baseload product "
            "covers every hour at base_share/H, the peak product adds peak_share/P during the "
            "Elexon peak block. Normalised to sum to one."
        ),
        "limitations": [
            "Two blocks only. It reproduces Ofgem's own shaping input, not an Elexon Profile "
            "Class 1 half-hourly regression, so within-block structure such as the domestic "
            "evening peak after 19:00 is absent.",
            "No seasonal variation: the 2020 cap periods declare one base/peak allocation, so "
            "the shape is flat across months and only varies by day type and hour.",
            "Applied to model snapshots by day-of-week and local hour, and by calendar date only "
            "for England and Wales bank holidays (off-peak), because the profile's only structure "
            "is the peak block.",
        ],
        "source": OFGEM_SOURCE,
    }

    for path in (output_csv, output_summary):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hedged-inputs",
        default="data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="data/market/consumer_bill_forecast/domestic_demand_profile_2020.csv",
    )
    parser.add_argument(
        "--output-summary",
        default="data/market/consumer_bill_forecast/domestic_demand_profile_2020_summary.json",
    )
    parser.add_argument("--profile-year", type=int, default=2020)
    args = parser.parse_args()
    result = write_profile(args.hedged_inputs, args.output_csv, args.output_summary,
                           args.profile_year)
    print(f"{result['hours']} hours; peak block holds "
          f"{result['peak_block_share_of_consumption']:.1%} of consumption in "
          f"{result['peak_block_share_of_hours']:.1%} of hours "
          f"(peak/off-peak rate ratio {result['peak_to_offpeak_rate_ratio']:.3f})")
