#!/usr/bin/env python3
"""Write CP30 network validation outputs for a built PyPSA network."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.CP30.energy_validation import (
    build_energy_validation,
    cp30_energy_targets,
    energy_validation_failures,
    format_energy_validation,
    network_energy_tables,
    overlay_carved_into,
)
from scripts.CP30.validation import (
    network_assumption_record,
    summarise_cp30_network_reference,
    validate_cp30_network,
)
from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network
from scripts.CP30.committed_overlay import (
    carved_line_by_carrier, overlay_capacity_by_carrier, resolve_overlay_carve_out,
)


def _run(
    network_path: str,
    es1_path: str,
    ed1_path: str,
    reference_path: str,
    pathway: str,
    modelled_year: int,
    scenario_config: dict,
    validation_output: str,
    reference_output: str,
    assumption_output: str,
    electrolysis_full_load_hours: float = 4000.0,
    energy_output: str | None = None,
) -> None:
    logger = setup_logging("validate_cp30_network")
    network = load_network(network_path, custom_logger=logger)

    validation = validate_cp30_network(
        network,
        es1_path=es1_path,
        ed1_path=ed1_path,
        pathway=pathway,
        modelled_year=modelled_year,
        electrolysis_full_load_hours=electrolysis_full_load_hours,
        overlay_capacity=overlay_capacity_by_carrier(
            ((scenario_config.get("fleet") or {}).get("spatial_proxy") or {}).get("committed_overlay")),
        overlay_carved_into=carved_line_by_carrier(resolve_overlay_carve_out(
            ((scenario_config.get("fleet") or {}).get("spatial_proxy") or {}).get("committed_overlay_carve_out"))),
    )
    reference = summarise_cp30_network_reference(reference_path)
    assumption = network_assumption_record(scenario_config)

    for output in [validation_output, reference_output, assumption_output]:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    validation.to_csv(validation_output, index=False)
    reference.to_csv(reference_output, index=False)
    assumption.to_csv(assumption_output, index=False)

    # Energy on the build: AVAILABLE energy (before curtailment, which is a
    # model output) against CP30 GROSS energy, and demand against CP30 demand.
    # Energy gaps warn; a demand mismatch or an unmapped ES1 line fails.
    energy_failures = None
    if energy_output:
        energy = build_energy_validation(
            cp30_energy_targets(es1_path, pathway, modelled_year),
            network_energy_tables(network, scenario_config.get("renewables", {}).get("hydro", {})),
            stage="build",
            tolerances=(scenario_config.get("cp30_validation", {}) or {}).get("energy"),
            carved_into=overlay_carved_into(scenario_config),   # NZT's CCS Gas in the Gas line
        )
        Path(energy_output).parent.mkdir(parents=True, exist_ok=True)
        energy.to_csv(energy_output, index=False)
        logger.info("CP30 build energy validation:\n%s", format_energy_validation(energy))
        for _, row in energy[energy["status"].eq("WARN")].iterrows():
            logger.warning("CP30 %s %s: model %.2f TWh vs CP30 gross %.2f TWh (%+.1f%%)", row["component"],
                           row["item"], row["model_gwh"] / 1e3, row["cp30_gwh"] / 1e3,
                           100 * row["relative_difference"])
        energy_failures = energy_validation_failures(energy)

    failures = validation[validation["status"] != "PASS"]
    if not failures.empty:
        raise ValueError(
            "CP30 validation failed: "
            + failures[["component", "item", "target", "actual", "difference"]].to_string(index=False)
        )
    if energy_failures is not None and not energy_failures.empty:
        raise ValueError(
            "CP30 energy validation failed: "
            + energy_failures[["component", "item", "cp30_gwh", "model_gwh", "note"]].to_string(index=False)
        )


if "snakemake" in globals():
    scenario_config = dict(snakemake.params.scenario_config)
    hydrogen_config = scenario_config.get("hydrogen", {})
    _run(
        network_path=snakemake.input.network,
        es1_path=snakemake.input.cp30_es1,
        ed1_path=snakemake.input.cp30_ed1,
        reference_path=snakemake.input.cp30_reference,
        pathway=snakemake.params.cp30_pathway,
        modelled_year=int(snakemake.params.modelled_year),
        scenario_config=scenario_config,
        validation_output=snakemake.output.validation,
        reference_output=snakemake.output.reference_summary,
        assumption_output=snakemake.output.network_assumption,
        electrolysis_full_load_hours=float(hydrogen_config.get("electrolysis_full_load_hours", 4000.0)),
        energy_output=getattr(snakemake.output, "energy_validation", None),
    )
else:
    parser = argparse.ArgumentParser(description="Validate a built CP30 network against CP30 totals")
    parser.add_argument("--network", required=True)
    parser.add_argument("--cp30-es1", required=True)
    parser.add_argument("--cp30-ed1", required=True)
    parser.add_argument("--cp30-reference", required=True)
    parser.add_argument("--pathway", required=True)
    parser.add_argument("--modelled-year", type=int, required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--reference-output", required=True)
    parser.add_argument("--assumption-output", required=True)
    parser.add_argument("--electrolysis-full-load-hours", type=float, default=4000.0)
    parser.add_argument("--energy-output", default=None)
    args = parser.parse_args()
    _run(
        network_path=args.network,
        es1_path=args.cp30_es1,
        ed1_path=args.cp30_ed1,
        reference_path=args.cp30_reference,
        pathway=args.pathway,
        modelled_year=args.modelled_year,
        scenario_config={"scenario_source": "CP30"},
        validation_output=args.validation_output,
        reference_output=args.reference_output,
        assumption_output=args.assumption_output,
        electrolysis_full_load_hours=args.electrolysis_full_load_hours,
        energy_output=args.energy_output,
    )