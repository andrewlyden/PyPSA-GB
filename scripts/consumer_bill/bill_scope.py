"""Shared year and provenance labels for consumer-bill outputs."""

import pandas as pd


def bill_scope(scenario: dict, metadata: dict, mode: str = "official") -> dict:
    config = scenario.get("consumer_bill", {}).get("price_cap", {})
    year = int(scenario["modelled_year"])
    input_mode = config.get("input_mode", "historical_backtest" if year <= 2024 else "future_projection")
    if input_mode not in {"historical_backtest", "future_projection"}:
        raise ValueError("consumer_bill.price_cap.input_mode must be historical_backtest or future_projection")
    start, end = pd.Timestamp(metadata["effective_from"]), pd.Timestamp(metadata["effective_to"])
    if start.year != year or end.year != year:
        raise ValueError(f"Bill bundle covers {start.date()} to {end.date()}, not modelled_year {year}. "
                         "Provide a same-year bundle; future allowances need explicit projected inputs.")
    annual = bool(metadata.get("annualisation")) or (end - start).days > 100
    projected = input_mode == "future_projection"
    return {
        "modelled_year": year,
        "input_mode": input_mode,
        "bill_basis": "future_projection" if projected else (
            "wholesale_sensitivity" if mode == "counterfactual" else
            "annualised_historical_benchmark" if annual else "published_tariff_snapshot"
        ),
        "is_statutory_cap_reproduction": bool(
            mode == "official" and not projected and not annual
            and metadata.get("is_statutory_cap_reproduction", False)
        ),
        "currency": metadata.get("currency", "GBP; price basis unspecified"),
    }
