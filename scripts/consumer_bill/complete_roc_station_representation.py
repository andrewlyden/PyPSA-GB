"""Complete the ROC station ledger with explicit aggregate representations.

This lightweight accounting stage is deliberately downstream of direct PyPSA
station matching. Changing a residual assumption therefore does not require
rewriting the labelled network or repeating the expensive name matcher.
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.consumer_bill.map_pypsa_generators_to_ofgem_roc import (
    complete_station_representation_mapping,
)


def write_complete_mapping(
    direct_mapping_path,
    outturn_path,
    registry_path,
    assumptions_path,
    output_mapping,
    output_summary,
):
    mapping, summary = complete_station_representation_mapping(
        pd.read_csv(direct_mapping_path),
        pd.read_csv(outturn_path),
        pd.read_csv(registry_path),
        pd.read_csv(assumptions_path),
    )
    for output in [output_mapping, output_summary]:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_mapping, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_complete_mapping(
        snakemake.input.direct_mapping_csv,
        snakemake.input.outturn_csv,
        snakemake.input.registry_csv,
        snakemake.input.assumptions_csv,
        snakemake.output.mapping_csv,
        snakemake.output.summary_json,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-mapping", required=True)
    parser.add_argument("--outturn", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--assumptions", required=True)
    parser.add_argument("--output-mapping", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_complete_mapping(
        args.direct_mapping, args.outturn, args.registry, args.assumptions,
        args.output_mapping, args.output_summary,
    )
