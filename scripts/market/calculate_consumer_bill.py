"""Snakemake entry point for Ofgem-structured electricity consumer bills."""

import json
import logging
from pathlib import Path

import pandas as pd

from scripts.market.ofgem_price_cap import (
    OfgemPriceCapError,
    calculate_bills,
    calculate_tariffs,
    demand_weighted_price,
    load_snapshot,
    require_full_calendar_year,
)
from scripts.utilities.network_io import load_network
from scripts.consumer_bill.bill_scope import bill_scope


def _snapshot_hours(network) -> pd.Series:
    snapshots = pd.DatetimeIndex(network.snapshots)
    for column in ("generators", "objective"):
        if column in network.snapshot_weightings.columns:
            hours = pd.to_numeric(network.snapshot_weightings[column], errors="coerce")
            if hours.notna().all() and (hours >= 0).all():
                return hours.reindex(snapshots).astype(float)
    return pd.Series(1.0, index=snapshots)


def _demand(network) -> pd.Series:
    snapshots = pd.DatetimeIndex(network.snapshots)
    try:
        frame = network.get_switchable_as_dense("Load", "p_set", snapshots)
    except (AttributeError, KeyError):
        frame = network.loads_t.p_set.reindex(index=snapshots).fillna(0.0)
    return frame.sum(axis=1).astype(float)


def calculate_modelled_wholesale_price(price_csv: str, wholesale_network: str, require_full_year: bool) -> float:
    prices = pd.read_csv(price_csv, index_col=0, parse_dates=True)
    if "wholesale_price" not in prices:
        raise OfgemPriceCapError(f"{price_csv} does not contain wholesale_price")
    network = load_network(wholesale_network)
    snapshots = pd.DatetimeIndex(network.snapshots)
    price = prices["wholesale_price"].reindex(snapshots)
    hours = _snapshot_hours(network)
    if require_full_year:
        require_full_calendar_year(snapshots, hours)
    if price.isna().any():
        missing = int(price.isna().sum())
        raise OfgemPriceCapError(f"Wholesale price CSV is missing {missing} network snapshots")
    return demand_weighted_price(price, _demand(network), hours)


def _main(snakemake) -> None:
    logger = logging.getLogger(__name__)
    scenario_config = snakemake.params.scenario_config
    # Consumer-bill price-cap settings: ``consumer_bill.price_cap``.
    bill_config = scenario_config.get("consumer_bill", {}).get("price_cap", {})
    rates, metadata = load_snapshot(Path(snakemake.input.snapshot_rates_csv).parent)
    regional_weights_input = snakemake.input.get("regional_weights_csv", [])
    regional_weights = pd.read_csv(regional_weights_input[0]) if regional_weights_input else None

    mode = bill_config.get("mode", "official")
    scope = bill_scope(scenario_config, metadata, mode)
    modelled_price = None
    reference_price = None
    if mode == "counterfactual":
        modelled_price = calculate_modelled_wholesale_price(
            snakemake.input.wholesale_price_csv,
            snakemake.input.wholesale_network,
            bool(bill_config.get("counterfactual", {}).get("require_full_year", True)),
        )
        reference_price = bill_config.get("counterfactual", {}).get(
            "reference_wholesale_price_gbp_per_mwh",
            metadata.get("reference_wholesale_price_gbp_per_mwh"),
        )

    meter_types = bill_config.get("meter_types", ["single_rate"])
    if isinstance(meter_types, str):
        meter_types = [meter_types]
    if set(meter_types) != {"single_rate"}:
        raise OfgemPriceCapError(
            "The implemented Ofgem snapshot supports meter_types: [single_rate] only"
        )
    tariffs = calculate_tariffs(
        rates,
        regions=bill_config.get("regions", ["GB_average"]),
        payment_methods=bill_config.get("payment_methods", ["direct_debit"]),
        mode=mode,
        vat_rate=float(bill_config.get("vat_rate", metadata["vat_rate"])),
        modelled_wholesale_price_gbp_per_mwh=modelled_price,
        reference_wholesale_price_gbp_per_mwh=reference_price,
        region_weights=regional_weights,
    )
    bills = calculate_bills(tariffs, bill_config.get("annual_consumption_kwh", [2500]))
    scenario_id = scenario_config.get("scenario_id", snakemake.wildcards.scenario)
    for frame in (tariffs, bills):
        frame.insert(0, "scenario", scenario_id)
        frame.insert(1, "cap_snapshot", metadata["snapshot_id"])
        for key, value in scope.items():
            frame[key] = value

    for path in (snakemake.output.tariffs_csv, snakemake.output.bills_csv, snakemake.output.summary_json):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    tariffs.to_csv(snakemake.output.tariffs_csv, index=False)
    bills.to_csv(snakemake.output.bills_csv, index=False)
    summary = {
        "scenario": scenario_id,
        "mode": mode,
        **scope,
        "cap_snapshot": metadata["snapshot_id"],
        "effective_from": metadata["effective_from"],
        "effective_to": metadata["effective_to"],
        "modelled_wholesale_price_gbp_per_mwh": modelled_price,
        "reference_wholesale_price_gbp_per_mwh": reference_price,
        "rows": int(len(bills)),
        "gb_average_weighting": "domestic-meter weighted" if regional_weights is not None else "equal regional weight",
        "limitations": metadata.get("limitations", []),
    }
    with open(snakemake.output.summary_json, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    logger.info("Wrote Ofgem consumer-bill outputs for %s in %s mode", scenario_id, mode)


if "snakemake" in globals():
    _main(snakemake)
