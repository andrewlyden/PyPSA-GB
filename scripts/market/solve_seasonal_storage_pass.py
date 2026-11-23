"""Snakemake entry for rule solve_seasonal_storage_pass (plan decision 33, task Y8a).

Loads a full-year parent network (``resources/network/{scenario}.nc``), runs the annual
copperplate pass on its aggregated fleet (scripts/market/seasonal_storage.py) and writes:

  - ``resources/market/{scenario}_seasonal_trajectory.csv``: the hydrogen store's hourly
    level (start and end of each hour) and value (store-balance dual), with the hydrogen
    bus and GB prices, load shedding and the hydrogen link flows;
  - ``resources/market/{scenario}_seasonal_pass.json``: solver status, LP size, runtime,
    aggregation report and annual diagnostics, with provenance.
"""

import time

from scripts.market.market_utils import solver_io_kwargs
from scripts.market.seasonal_storage import run_annual_pass, write_outputs
from scripts.solve.solve_network import configure_solver
from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network


def main(smk) -> None:
    """Run the annual pass for one parent scenario.

    Args:
        smk: the ``snakemake`` object.
    """
    logger = setup_logging(smk.log[0] if smk.log else "solve_seasonal_storage_pass")
    started = time.time()
    scenario_config = smk.params.scenario_config
    # The decision-29 price-variant twin rule names its pass ``pass_id``.
    scenario = getattr(smk.wildcards, "scenario", None) or smk.wildcards.pass_id
    logger.info("=" * 80)
    logger.info(f"SEASONAL STORAGE ANNUAL PASS (decision 33): {scenario}")
    logger.info("=" * 80)
    solver_name, solver_options = configure_solver(None, smk.params.solver, dict(smk.params.solver_options), logger)
    optimize_kwargs = solver_io_kwargs(scenario_config, solver_name, logger)
    accepted = ((scenario_config.get("solve", {}) or {}).get("solver", {}) or {}).get("accepted_conditions")
    if accepted:
        optimize_kwargs["accepted_conditions"] = tuple(str(c) for c in accepted)
    provenance = {"scenario": scenario, "network": str(smk.input.network)}
    # Handed over in a list so the pass holds the only reference and can release it.
    holder = [load_network(smk.input.network, custom_logger=logger)]
    # Decision 29 (2050 only): a TYNDP price-sensitivity pass re-prices the shared
    # network's interconnectors first; a no-op unless market.foreign_price_override is set.
    from scripts.interconnectors.foreign_price_override import apply_foreign_price_override
    override = apply_foreign_price_override(holder[0], scenario_config, smk.input, logger)
    if override:
        provenance["foreign_price_override"] = override["tyndp"]
    # 2050 price calibration (25 Oct 2026): the pass sees the hydrogen supply price and the interconnector
    # hurdle Stage 1 solves on (each a no-op unless set).
    from scripts.market.market_utils import apply_hydrogen_supply_price
    from scripts.market.interconnector_hurdle import apply_interconnector_hurdle_override
    h2 = apply_hydrogen_supply_price(holder[0], scenario_config.get("market", {}), logger)
    hurdle = apply_interconnector_hurdle_override(
        holder[0], (scenario_config.get("market", {}) or {}).get("wholesale", {}), logger)
    if h2:
        provenance["hydrogen_supply_price"] = h2
    if hurdle:
        provenance["interconnector_hurdle_override"] = hurdle
    trajectory, summary = run_annual_pass(holder, scenario_config, solver_name, solver_options, logger,
                                          optimize_kwargs=optimize_kwargs, provenance=provenance)
    summary["runtime_seconds"]["rule_total"] = round(time.time() - started, 1)
    write_outputs(trajectory, summary, smk.output.trajectory, smk.output.summary)
    logger.info(f"Saved {smk.output.trajectory} and {smk.output.summary} ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main(snakemake)  # noqa: F821
