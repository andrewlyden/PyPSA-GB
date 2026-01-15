"""
Lightweight validation script for renewable profiles.

Writes a simple HTML report, a CSV with basic stats, a JSON with quality flags,
and ensures the validation plots directory exists.
"""

import json
from pathlib import Path
import pandas as pd


def _load_many(paths):
    frames = []
    for p in paths:
        try:
            if Path(p).exists() and Path(p).stat().st_size > 0:
                frames.append(pd.read_csv(p, index_col=0))
        except Exception:
            # Ignore unreadable files
            pass
    if not frames:
        return pd.DataFrame()
    # Align on index if possible
    try:
        return pd.concat(frames, axis=1)
    except Exception:
        # Fallback to simple append
        return pd.concat(frames, axis=0)


def main():
    # Inputs from Snakemake
    wind_on = snakemake.input.wind_onshore_profiles
    wind_off = snakemake.input.wind_offshore_profiles
    solar = snakemake.input.solar_pv_profiles

    # Outputs
    report_html = Path(snakemake.output.validation_report)
    stats_csv = Path(snakemake.output.statistics_summary)
    flags_json = Path(snakemake.output.quality_flags)
    plots_dir = Path(snakemake.output.validation_plots)

    plots_dir.mkdir(parents=True, exist_ok=True)
    report_html.parent.mkdir(parents=True, exist_ok=True)

    # Load profiles
    df_won = _load_many(wind_on)
    df_wof = _load_many(wind_off)
    df_pv = _load_many(solar)

    # Compute basic stats
    def basic_stats(df):
        if df.empty:
            return pd.DataFrame([{"count": 0, "cols": 0, "mean": None, "min": None, "max": None}])
        return pd.DataFrame([
            {
                "count": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "mean": float(pd.to_numeric(df.stack(), errors="coerce").mean()),
                "min": float(pd.to_numeric(df.stack(), errors="coerce").min()),
                "max": float(pd.to_numeric(df.stack(), errors="coerce").max()),
            }
        ])

    stats = {
        "wind_onshore": basic_stats(df_won),
        "wind_offshore": basic_stats(df_wof),
        "solar_pv": basic_stats(df_pv),
    }

    # Combine into a single CSV (with a level to label tech)
    combined = pd.concat({k: v for k, v in stats.items()}, names=["technology"])
    combined.to_csv(stats_csv, index=True)

    # Quality flags (very basic)
    flags = {
        "wind_onshore_has_data": not df_won.empty,
        "wind_offshore_has_data": not df_wof.empty,
        "solar_pv_has_data": not df_pv.empty,
    }
    flags_json.parent.mkdir(parents=True, exist_ok=True)
    flags_json.write_text(json.dumps(flags, indent=2))

    # Minimal HTML report
    html = [
        "<html><head><meta charset='utf-8'><title>Renewable Profiles Validation</title></head><body>",
        "<h1>Renewable Profiles Validation</h1>",
        f"<p>Wind Onshore profiles: {'present' if flags['wind_onshore_has_data'] else 'missing'}</p>",
        f"<p>Wind Offshore profiles: {'present' if flags['wind_offshore_has_data'] else 'missing'}</p>",
        f"<p>Solar PV profiles: {'present' if flags['solar_pv_has_data'] else 'missing'}</p>",
        "<h2>Summary Statistics</h2>",
        combined.to_html(index=True),
        "</body></html>",
    ]
    report_html.write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()


