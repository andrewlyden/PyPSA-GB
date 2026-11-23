"""Create a transparent annual Ofgem/PyPSA electricity-bill decomposition."""

import json
import logging
from pathlib import Path

import pandas as pd

from scripts.market.ofgem_hedging import direct_fuel_components, load_hedge_periods
from scripts.consumer_bill.bill_scope import bill_scope

BENCHMARK_MWH = 3.1


def _period_weights(periods: pd.DataFrame) -> pd.Series:
    days = (periods["delivery_end"] - periods["delivery_start"]).dt.days + 1
    return days / days.sum()


def _modelled_prices(periods: pd.DataFrame, price_csv: str, network_path: str) -> pd.DataFrame:
    # Keep the published-stack helper importable without PyPSA so historical
    # bundle validation and component-register building do not require a
    # network installation. These imports are needed only for a model result.
    from scripts.market.calculate_consumer_bill import _demand, _snapshot_hours
    from scripts.market.ofgem_price_cap import demand_weighted_price, require_full_calendar_year
    from scripts.utilities.network_io import load_network

    prices = pd.read_csv(price_csv, index_col=0, parse_dates=True)["wholesale_price"]
    network = load_network(network_path)
    snapshots = pd.DatetimeIndex(network.snapshots)
    demand, hours, prices = _demand(network), _snapshot_hours(network), prices.reindex(snapshots)
    require_full_calendar_year(snapshots, hours)
    if set(snapshots.year) != set(pd.DatetimeIndex(periods["delivery_start"]).year):
        raise ValueError("PyPSA snapshots and bill delivery periods must cover the same year")
    rows = []
    for _, period in periods.iterrows():
        mask = (snapshots >= period["delivery_start"]) & (
            snapshots < period["delivery_end"] + pd.Timedelta(days=1)
        )
        if not mask.any():
            raise ValueError(f"PyPSA result contains no snapshots for {period['cap_period']}")
        index = snapshots[mask]
        demand_mwh = float((demand.reindex(index) * hours.reindex(index)).sum())
        rows.append({
            "cap_period": period["cap_period"],
            "modelled_demand_mwh": demand_mwh,
            "modelled_demand_weighted_clearing_price_gbp_per_mwh": demand_weighted_price(
                prices.reindex(index), demand.reindex(index), hours.reindex(index)
            ),
        })
    result = pd.DataFrame(rows)
    result["modelled_delivery_demand_share"] = result["modelled_demand_mwh"] / result["modelled_demand_mwh"].sum()
    return result


def _retail_stack(periods: pd.DataFrame, vat_rate: float = 0.05) -> pd.DataFrame:
    weights = _period_weights(periods)
    direct = direct_fuel_components(periods, BENCHMARK_MWH).pivot(
        index="cap_period", columns="component", values="gbp_per_benchmark_customer_ex_vat"
    )
    direct = direct.reindex(periods["cap_period"])
    detailed_wholesale = periods.get("wholesale_detail_level", pd.Series(index=periods.index, dtype=str)).eq(
        "published_3a_direct_fuel_components"
    ).all()
    wholesale_items = (
        [
            ("hedged_wholesale", "direct_fuel_energy_cost", "Published direct fuel energy cost"),
            ("hedge_uplifts", "backwardation_allowance", "Published backwardation allowance"),
            ("hedged_wholesale", "cfd_allowance_embedded_in_direct_fuel", "Published CfD allowance embedded in direct fuel"),
        ] if detailed_wholesale else [
            ("hedged_wholesale", "seasonal_base_forward_index", "Published seasonal base forward allowance"),
            ("hedged_wholesale", "seasonal_peak_forward_index", "Published seasonal peak forward allowance"),
        ]
    )
    legacy_hedge_items = [] if detailed_wholesale else [
        ("hedge_uplifts", "seasonal_to_monthly_shaping", "Seasonal-to-monthly shaping"),
        ("hedge_uplifts", "monthly_to_hourly_shaping", "Monthly peak/base to hourly shaping"),
        ("hedge_uplifts", "day_ahead_rehedging", "Day-ahead re-hedging / forecast error"),
        ("system_balancing", "imbalance", "Imbalance uplift"),
        ("hedge_uplifts", "transaction_cost", "Transaction-cost uplift"),
        ("hedge_uplifts", "additional_risk_and_basis", "Additional risk / basis allowance"),
        ("losses", "regional_losses_gb_average", "GB-average regional electricity losses"),
    ]
    items = wholesale_items + legacy_hedge_items + [
        ("system_balancing", "capacity_market_gbp_per_3100kwh", "Capacity Market"),
        ("system_balancing", "adjustment_allowance_gbp_per_3100kwh", "Adjustment allowance"),
        ("policy", "policy_gbp_per_3100kwh", "Policy costs"),
        ("network", "network_gbp_per_3100kwh", "Network costs"),
        ("supplier_operating", "operating_gbp_per_3100kwh", "Supplier operating costs"),
        ("supplier_operating", "smart_metering_gbp_per_3100kwh", "Smart-metering net cost change"),
        ("payment", "payment_additional_admin_gbp_per_3100kwh", "Payment-method additional administration"),
        ("payment", "payment_adjustment_gbp_per_3100kwh", "Payment-method adjustment percentage"),
        ("bad_debt", None, "Not separately published in this bundle (included within operating allowance)"),
        ("margin", "ebit_gbp_per_3100kwh", "EBIT allowance"),
        ("headroom", "headroom_gbp_per_3100kwh", "Headroom allowance"),
    ]
    rows = []
    for group, component, label in items:
        if component is None:
            amount = 0.0
        else:
            source = direct[component] if component in direct else periods[component]
            amount = float((source * weights.to_numpy()).sum())
        rows.append({
            "stack_group": group, "component": label,
            "annual_gbp_per_3100kwh_ex_vat": amount,
            "is_separately_published": component is not None,
        })
    subtotal = sum(row["annual_gbp_per_3100kwh_ex_vat"] for row in rows)
    published_total = float((periods["total_cap_gbp_per_3100kwh"] * weights).sum())
    if abs(subtotal - published_total) > 1e-6:
        raise ValueError(f"Cost-stack components do not reconcile: {subtotal} != {published_total}")
    rows.extend([
        {"stack_group": "vat", "component": f"VAT at {vat_rate * 100:g}%", "annual_gbp_per_3100kwh_ex_vat": subtotal * vat_rate, "is_separately_published": True},
        {"stack_group": "total", "component": "Total cap benchmark including VAT", "annual_gbp_per_3100kwh_ex_vat": subtotal * (1 + vat_rate), "is_separately_published": True},
    ])
    return pd.DataFrame(rows)


def _main(snakemake) -> None:
    logger = logging.getLogger(__name__)
    periods = load_hedge_periods(snakemake.input.hedge_inputs_csv)
    scenario_config = snakemake.params.scenario_config
    metadata = json.loads(Path(snakemake.input.snapshot_metadata_json).read_text(encoding="utf-8"))
    bill_config = scenario_config.get("consumer_bill", {}).get("price_cap", {})
    scope = bill_scope(scenario_config, metadata, bill_config.get("mode", "official"))
    forward_treatment = (
        "Projected procurement inputs retained; PyPSA prices are diagnostic unless a separate recovery bridge is selected"
        if scope["input_mode"] == "future_projection" else
        "Observed Ofgem forward inputs retained; PyPSA spot prices are diagnostic only"
    )
    has_model = bool(snakemake.input.get("wholesale_network"))
    model = (_modelled_prices(periods, snakemake.input.wholesale_price_csv, snakemake.input.wholesale_network)
             if has_model else pd.DataFrame(columns=["cap_period", "modelled_demand_mwh",
                 "modelled_demand_weighted_clearing_price_gbp_per_mwh", "modelled_delivery_demand_share"]))
    wholesale = periods.merge(model, on="cap_period", validate="one_to_one")
    wholesale["published_forward_index_gbp_per_mwh"] = wholesale["forward_index_gbp_per_mwh"]
    wholesale["spot_less_forward_gbp_per_mwh"] = (
        wholesale["modelled_demand_weighted_clearing_price_gbp_per_mwh"]
        - wholesale["published_forward_index_gbp_per_mwh"]
    )
    wholesale["price_definition"] = "PyPSA demand-weighted day-ahead model clearing price; not a retail wholesale allowance"
    direct_fuel = direct_fuel_components(periods, BENCHMARK_MWH)
    published_direct_fuel = direct_fuel.loc[
        direct_fuel["component"] == "direct_fuel_allowance",
        ["cap_period", "gbp_per_mwh"],
    ].rename(columns={"gbp_per_mwh": "ofgem_direct_fuel_allowance_gbp_per_mwh"})
    wholesale = wholesale.merge(published_direct_fuel, on="cap_period", validate="one_to_one")
    wholesale["modelled_spot_less_observed_forward_gbp_per_mwh"] = (
        wholesale["modelled_demand_weighted_clearing_price_gbp_per_mwh"]
        - wholesale["published_forward_index_gbp_per_mwh"]
    )
    wholesale["forward_price_treatment"] = forward_treatment
    procurement = direct_fuel.merge(
        periods[[
            "cap_period", "observation_start", "observation_end", "delivery_start", "delivery_end",
            "delivery_seasons", "base_share", "peak_share",
        ]],
        on="cap_period", validate="many_to_one",
    )
    retail = _retail_stack(periods, float(bill_config.get("vat_rate", metadata["vat_rate"])))
    for path in snakemake.output:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    wholesale.to_csv(snakemake.output.wholesale_impact_csv, index=False)
    procurement.to_csv(snakemake.output.forward_procurement_csv, index=False)
    retail.to_csv(snakemake.output.retail_stack_csv, index=False)

    shares = wholesale["modelled_delivery_demand_share"]
    modelled = float((wholesale["modelled_demand_weighted_clearing_price_gbp_per_mwh"] * shares).sum()) if has_model else None
    forward = float((wholesale["published_forward_index_gbp_per_mwh"] * shares).sum()) if has_model else None
    scenario_config = snakemake.params.scenario_config
    bill_config = scenario_config.get("consumer_bill", {}).get("price_cap", {})
    consumption = float(bill_config.get("annual_consumption_kwh", [2500])[0])
    modelled_year = int(scenario_config.get("modelled_year"))
    summary = {
        "scenario": snakemake.params.scenario_config["scenario_id"],
        "benchmark": "GB-average, Other Payment Method, single-rate electricity, 3,100 kWh annual benchmark",
        "modelled_year": modelled_year,
        "pypsa_demand_weighted_clearing_price_gbp_per_mwh": modelled,
        "ofgem_demand_weighted_forward_index_gbp_per_mwh": forward,
        "annual_consumption_kwh": consumption,
        "forward_price_treatment": forward_treatment,
        **scope,
        "limitations": [
            "PyPSA provides a delivery-period price diagnostic, not historic forward prices observed in Ofgem's hedge windows.",
            "The retail stack is an annualised published 3,100 kWh benchmark, not a component allocation of a different household-consumption assumption.",
            "Bundle component definitions must document whether bad debt is separately published or included in the supplier operating allowance.",
        ],
    }
    with open(snakemake.output.summary_json, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    logger.info("Wrote annual forward-procurement and retail cost-stack analysis for %s", summary["scenario"])


if "snakemake" in globals():
    _main(snakemake)
