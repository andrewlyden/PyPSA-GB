"""Build tariff-ready network-cost layers and a future allowed-revenue projection.

The input is an audited regional annual network-charge replay.  It converts annual
representative-household amounts into a tariff-class/time-band table and keeps
losses and future allowed-revenue assumptions as separate, visible layers.
Direct DNO time-band tariffs can replace the default ``all_hours`` rows without
changing the calculation engine.
"""


import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VOLUMETRIC_COMPONENTS = {"TNUoS", "DUoS unit charges", "BSUoS"}
FIXED_COMPONENT = "DUoS fixed charges"
BENCHMARK_KWH = 3100.0
NESO_NHH_RATE_COLUMN = "NHHTariff(Floored)_p/kWh"


def _normalise_tnuos_region(value: object) -> str:
    """Make the NESO zone names match the Ofgem/PES regional names."""
    return str(value).strip().replace("N Wales & Mersey", "N Wales and Mersey").replace(
        "South Western", "Southern Western"
    )


def build_neso_tnuos_comparator(
    neso_tariffs: pd.DataFrame,
    weights: pd.DataFrame,
    financial_year: int,
) -> tuple[pd.DataFrame, dict]:
    """Select the latest published NESO NHH rate for every domestic PES area.

    This is deliberately a comparator, not a replacement for the historical
    Ofgem TNUoS allowance.  A direct consumer-cost calculation needs the
    applicable charging window and domestic settlement/profile mix; the raw
    NESO rate alone does not provide that mix.
    """
    required = {"Publication", "Year_FY", "Published_Date", "Zone_Name", NESO_NHH_RATE_COLUMN}
    if missing := required.difference(neso_tariffs.columns):
        raise ValueError(f"NESO TNUoS data is missing: {sorted(missing)}")
    weight_required = {"pes_id", "ofgem_region", "domestic_meter_weight"}
    if missing := weight_required.difference(weights.columns):
        raise ValueError(f"Domestic weights are missing: {sorted(missing)}")

    source = neso_tariffs.copy()
    source["Year_FY"] = pd.to_numeric(source["Year_FY"], errors="raise").astype(int)
    source["Published_Date"] = pd.to_datetime(source["Published_Date"], dayfirst=True, errors="raise")
    source["ofgem_region"] = source["Zone_Name"].map(_normalise_tnuos_region)
    source[NESO_NHH_RATE_COLUMN] = pd.to_numeric(source[NESO_NHH_RATE_COLUMN], errors="raise")
    selected = source.loc[source["Year_FY"].eq(int(financial_year))].copy()
    if selected.empty:
        raise ValueError(f"No NESO NHH TNUoS rows found for financial year {financial_year}")
    # Prefer a final tariff; otherwise take the most recently published forecast.
    selected["_final_rank"] = selected["Publication"].eq("Final").astype(int)
    selected = (
        selected.sort_values(["ofgem_region", "_final_rank", "Published_Date"])
        .groupby("ofgem_region", as_index=False)
        .tail(1)
    )
    output = weights[["pes_id", "ofgem_region", "domestic_meter_weight"]].merge(
        selected[["ofgem_region", "Publication", "Year_FY", "Published_Date", "Zone_No", NESO_NHH_RATE_COLUMN]],
        on="ofgem_region", how="left", validate="one_to_one",
    )
    if output[NESO_NHH_RATE_COLUMN].isna().any():
        missing_regions = output.loc[output[NESO_NHH_RATE_COLUMN].isna(), "ofgem_region"].tolist()
        raise ValueError(f"NESO TNUoS rows missing domestic regions: {missing_regions}")
    output = output.rename(columns={
        "Publication": "publication", "Year_FY": "financial_year", "Published_Date": "published_date",
        "Zone_No": "neso_zone_no", NESO_NHH_RATE_COLUMN: "nhh_tariff_p_per_kwh",
    })
    output["nhh_tariff_gbp_per_kwh"] = output["nhh_tariff_p_per_kwh"] / 100.0
    output["tariff_role"] = "direct_NESO_NHH_rate_comparator"
    output["consumer_bill_treatment"] = (
        "not substituted into the Ofgem TNUoS allowance; requires charging-window and domestic profile/settlement mix"
    )
    summary = {
        "financial_year": int(financial_year),
        "selection": "final tariff where published; otherwise most recently published forecast",
        "source": "NESO demand NHH TNUoS tariff dataset",
        "treatment": output["consumer_bill_treatment"].iloc[0],
    }
    return output.sort_values("pes_id").reset_index(drop=True), summary


def build_tariff_layers(regional: pd.DataFrame, weights: pd.DataFrame, transmission_loss_rate: float) -> tuple[pd.DataFrame, dict]:
    """Convert annual regional allowances to explicit single-rate tariff rows."""
    required = {"pes_id", "ofgem_region", "component", "annual_gbp_per_3100kwh_ex_vat"}
    if missing := required.difference(regional.columns):
        raise ValueError(f"Regional replay is missing: {sorted(missing)}")
    if missing := {"pes_id", "domestic_meter_weight"}.difference(weights.columns):
        raise ValueError(f"Domestic weights are missing: {sorted(missing)}")
    if not 0 <= transmission_loss_rate < 1:
        raise ValueError("Transmission loss rate must be in [0, 1)")
    rows = regional.loc[regional["component"].isin(VOLUMETRIC_COMPONENTS | {FIXED_COMPONENT})].copy()
    if rows.duplicated(["pes_id", "component"]).any():
        raise ValueError("Regional replay must contain one annual value per PES/component")
    rows["annual_gbp_per_3100kwh_ex_vat"] = pd.to_numeric(rows["annual_gbp_per_3100kwh_ex_vat"], errors="raise")
    rows["tariff_class"] = "single_rate"
    rows["time_band"] = "all_hours"
    rows["time_band_demand_share"] = 1.0
    rows["unit_gbp_per_kwh_ex_vat"] = np.where(
        rows["component"].isin(VOLUMETRIC_COMPONENTS),
        rows["annual_gbp_per_3100kwh_ex_vat"] / BENCHMARK_KWH,
        0.0,
    )
    rows["standing_gbp_per_day_ex_vat"] = np.where(
        rows["component"].eq(FIXED_COMPONENT),
        rows["annual_gbp_per_3100kwh_ex_vat"] / 366.0,
        0.0,
    )
    if "domestic_meter_weight" not in rows:
        rows = rows.merge(
            weights[["pes_id", "domestic_meter_weight"]], on="pes_id", how="left", validate="many_to_one"
        )
    if rows["domestic_meter_weight"].isna().any():
        raise ValueError("Regional tariff rows are missing domestic meter weights")
    rows["regional_loss_factor"] = 1.0
    rows["transmission_loss_factor"] = 1.0 / (1.0 - transmission_loss_rate)
    rows["tariff_input_status"] = "Ofgem regional representative-household benchmark; replace all_hours with published DNO tariff-class/time-band rows when supplied"
    summary = {
        "purpose": "Tariff-ready annual regional network-cost layers",
        "tariff_class": "single-rate domestic",
        "time_band_treatment": "all_hours baseline; calculation engine supports explicit time-band demand shares",
        "transmission_loss_rate": transmission_loss_rate,
        "transmission_loss_factor": 1.0 / (1.0 - transmission_loss_rate),
        "regional_distribution_loss_factor": "1.0 pending archived domestic LLFC/DLF inputs by PES area",
        "loss_boundary": "The loss factor is an energy-volume adjustment. It is not substituted for Ofgem's separately published historical loss-cost allowance.",
    }
    return rows.sort_values(["component", "pes_id"]).reset_index(drop=True), summary


def apply_tariff_to_profile(tariffs: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """Price a regional domestic demand profile with tariff-class/time-band rows.

    ``profile`` must contain kWh by PES, tariff class and time band. This makes
    a future detailed-DUoS input a data replacement, rather than a code change.
    """
    tariff_required = {"pes_id", "tariff_class", "time_band", "component", "unit_gbp_per_kwh_ex_vat", "standing_gbp_per_day_ex_vat"}
    profile_required = {"pes_id", "tariff_class", "time_band", "consumption_kwh", "customer_days"}
    if missing := tariff_required.difference(tariffs.columns):
        raise ValueError(f"Tariff table is missing: {sorted(missing)}")
    if missing := profile_required.difference(profile.columns):
        raise ValueError(f"Demand profile is missing: {sorted(missing)}")
    priced = profile.merge(
        tariffs, on=["pes_id", "tariff_class", "time_band"], how="left", validate="many_to_many"
    )
    if priced["component"].isna().any():
        raise ValueError("Demand profile includes a PES/tariff-class/time-band with no tariff input")
    priced["charge_gbp_ex_vat"] = (
        pd.to_numeric(priced["consumption_kwh"], errors="raise") * priced["unit_gbp_per_kwh_ex_vat"]
        + pd.to_numeric(priced["customer_days"], errors="raise") * priced["standing_gbp_per_day_ex_vat"]
    )
    return priced


def project_allowed_revenue(tariffs: pd.DataFrame, assumptions: dict) -> tuple[pd.DataFrame, dict]:
    """Project tariff layers using explicit revenue, sales and account indices."""
    required = {
        "target_year", "tnuos_allowed_revenue_multiplier", "duos_unit_allowed_revenue_multiplier",
        "duos_fixed_allowed_revenue_multiplier", "bsuos_recovery_multiplier",
        "domestic_sales_multiplier", "domestic_accounts_multiplier",
    }
    if missing := required.difference(assumptions):
        raise ValueError(f"Future network assumptions are missing: {sorted(missing)}")
    positive = {key: float(assumptions[key]) for key in required.difference({"target_year"})}
    if any(value <= 0 for value in positive.values()):
        raise ValueError("Allowed-revenue, sales and account multipliers must be positive")
    output = tariffs.copy()
    multiplier = pd.Series(1.0, index=output.index)
    multiplier.loc[output["component"].eq("TNUoS")] = positive["tnuos_allowed_revenue_multiplier"] / positive["domestic_sales_multiplier"]
    multiplier.loc[output["component"].eq("DUoS unit charges")] = positive["duos_unit_allowed_revenue_multiplier"] / positive["domestic_sales_multiplier"]
    multiplier.loc[output["component"].eq(FIXED_COMPONENT)] = positive["duos_fixed_allowed_revenue_multiplier"] / positive["domestic_accounts_multiplier"]
    multiplier.loc[output["component"].eq("BSUoS")] = positive["bsuos_recovery_multiplier"] / positive["domestic_sales_multiplier"]
    output["projection_multiplier"] = multiplier
    output["target_year"] = int(assumptions["target_year"])
    output["projected_unit_gbp_per_kwh_ex_vat"] = output["unit_gbp_per_kwh_ex_vat"] * multiplier
    output["projected_standing_gbp_per_day_ex_vat"] = output["standing_gbp_per_day_ex_vat"] * multiplier
    output["projection_status"] = "assumption_driven_allowed_revenue_projection"
    summary = {
        "target_year": int(assumptions["target_year"]),
        "method": "Component-specific allowed-revenue/recovery index divided by the relevant domestic sales or account index",
        "required_external_inputs": "NESO TNUoS revenue/tariff outlook, DNO DUoS allowed revenue, domestic sales forecast and domestic account forecast",
        "not_modelled_from_pypsa": "DNO allowed revenue and tariff design",
    }
    return output, summary


def write_layers(
    regional_path, weights_path, loss_summary_path, future_assumptions_path, neso_tnuos_path,
    output_tariffs, output_projection, output_historic_tnuos, output_future_tnuos, output_summary,
    historic_tnuos_financial_year,
):
    loss_summary = json.loads(Path(loss_summary_path).read_text(encoding="utf-8")) if loss_summary_path else {}
    loss_rate = float(loss_summary.get("i2r_loss_proxy_rate", 0.0))
    tariffs, tariff_summary = build_tariff_layers(pd.read_csv(regional_path), pd.read_csv(weights_path), loss_rate)
    assumptions = json.loads(Path(future_assumptions_path).read_text(encoding="utf-8"))
    projection, projection_summary = project_allowed_revenue(tariffs, assumptions)
    neso_tnuos = pd.read_csv(neso_tnuos_path)
    weights = pd.read_csv(weights_path)
    historic_tnuos, historic_tnuos_summary = build_neso_tnuos_comparator(
        neso_tnuos, weights, int(historic_tnuos_financial_year)
    )
    future_tnuos, future_tnuos_summary = build_neso_tnuos_comparator(
        neso_tnuos, weights, int(assumptions["target_year"])
    )
    for output in (output_tariffs, output_projection, output_historic_tnuos, output_future_tnuos, output_summary):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    tariffs.to_csv(output_tariffs, index=False)
    projection.to_csv(output_projection, index=False)
    historic_tnuos.to_csv(output_historic_tnuos, index=False)
    future_tnuos.to_csv(output_future_tnuos, index=False)
    Path(output_summary).write_text(json.dumps({
        **tariff_summary,
        "future_projection": projection_summary,
        "neso_tnuos_historic_comparator": historic_tnuos_summary,
        "neso_tnuos_future_outlook": future_tnuos_summary,
    }, indent=2), encoding="utf-8")


if "snakemake" in globals():
    paths = list(snakemake.input.loss_proxy_summary)
    write_layers(
        snakemake.input.regional_replay, snakemake.input.domestic_weights,
        paths[0] if paths else None, snakemake.input.future_assumptions, snakemake.input.neso_tnuos,
        snakemake.output.tariffs_csv, snakemake.output.projection_csv,
        snakemake.output.historic_tnuos_csv, snakemake.output.future_tnuos_csv, snakemake.output.summary_json,
        snakemake.params.historic_tnuos_financial_year,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regional-replay", required=True)
    parser.add_argument("--domestic-weights", required=True)
    parser.add_argument("--loss-summary")
    parser.add_argument("--future-assumptions", required=True)
    parser.add_argument("--neso-tnuos", required=True)
    parser.add_argument("--output-tariffs", required=True)
    parser.add_argument("--output-projection", required=True)
    parser.add_argument("--output-historic-tnuos", required=True)
    parser.add_argument("--output-future-tnuos", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--historic-tnuos-financial-year", type=int, required=True)
    args = parser.parse_args()
    write_layers(args.regional_replay, args.domestic_weights, args.loss_summary, args.future_assumptions,
                 args.neso_tnuos, args.output_tariffs, args.output_projection,
                 args.output_historic_tnuos, args.output_future_tnuos, args.output_summary,
                 args.historic_tnuos_financial_year)
