"""Build one annual generator/BMU register from monthly scenario outputs."""

import argparse
from pathlib import Path

import pandas as pd


def aggregate_registers(
    generator_paths: list[str | Path],
    bmu_paths: list[str | Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generators = pd.concat(
        [pd.read_csv(path).assign(source_scenario=Path(path).stem) for path in generator_paths],
        ignore_index=True,
    )
    if "name" not in generators.columns:
        raise ValueError("Monthly generator registers need a 'name' column")

    # A project can be absent before commissioning and appear in later months.
    # Keep one representative row once present, while rejecting incompatible
    # carrier or capacity definitions across months.
    for column in ("carrier",):
        if column in generators.columns:
            counts = generators.groupby("name")[column].nunique(dropna=False)
            conflicts = counts[counts > 1]
            if not conflicts.empty:
                raise ValueError(
                    f"Monthly generator register has inconsistent {column}: "
                    f"{conflicts.index.tolist()}"
                )
    generator_union = generators.drop_duplicates("name", keep="last").sort_values("name")

    bmu_frames = [pd.read_csv(path) for path in bmu_paths if Path(path).exists()]
    if bmu_frames:
        bmu_union = pd.concat(bmu_frames, ignore_index=True).drop_duplicates()
    else:
        bmu_union = pd.DataFrame(columns=["generator_name", "bmu_id"])
    return generator_union, bmu_union


def write_registers(generator_paths, bmu_paths, output_generators, output_bmus):
    generators, bmus = aggregate_registers(list(generator_paths), list(bmu_paths))
    Path(output_generators).parent.mkdir(parents=True, exist_ok=True)
    generators.to_csv(output_generators, index=False)
    bmus.to_csv(output_bmus, index=False)


if "snakemake" in globals():
    write_registers(
        snakemake.input.generators,
        snakemake.input.bmus,
        snakemake.output.generators_csv,
        snakemake.output.bmus_csv,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generators", nargs="+", required=True)
    parser.add_argument("--bmus", nargs="+", required=True)
    parser.add_argument("--output-generators", required=True)
    parser.add_argument("--output-bmus", required=True)
    args = parser.parse_args()
    write_registers(
        args.generators, args.bmus, args.output_generators, args.output_bmus
    )
