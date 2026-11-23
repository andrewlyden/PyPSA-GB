"""Create a like-for-like monthly alignment scorecard for a historic run.

ESPENI validates physical system operation; LCCC validates the mapped CfD
contract cohort; and Ofgem validates the station-labelled ROC cohort.  The
three scopes are deliberately kept separate so a policy eligibility gap cannot
be mistaken for a dispatch-model error.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


TIME_COLUMN = "ELEC_elex_startTime[utc](datetime)"
ESPENI_SUPPORT_COMPARISON_COLUMNS = {
    "all wind": "ELEC_POWER_TOTAL_WIND[MW](float32)",
    "solar": "ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)",
    "biomass": "ELEC_POWER_ELEX_BIOMASS_POSTCALC[MW](float32)",
    "non-pumped hydro": "ELEC_POWER_ELEX_NPSHYD[MW](float32)",
}


def _weights(network) -> pd.Series:
    return network.snapshot_weightings.get(
        "generators", pd.Series(1.0, index=network.snapshots)
    )


def _load_espeni(path: str | Path, start, end) -> pd.DataFrame:
    raw = pd.read_csv(path)
    timestamps = pd.to_datetime(raw[TIME_COLUMN], utc=True).dt.tz_convert(None)
    raw.index = timestamps
    raw = raw.select_dtypes(include="number").sort_index()
    # ESPENI is half-hourly MW.  The mean is hourly MW and, for the hourly
    # model, also hourly MWh when summed.
    return raw.loc[(raw.index >= start) & (raw.index <= end)].resample("h").mean()


def _series(network, carrier_dispatch: pd.DataFrame, carrier: str, index) -> pd.Series:
    if carrier not in carrier_dispatch.columns:
        return pd.Series(0.0, index=index)
    return carrier_dispatch[carrier].reindex(index, fill_value=0.0)


def _row(
    period_start, period_end, layer, metric, model_value, reference_value, unit,
    comparison_kind, driver, model_series=None, reference_series=None, tolerance=None,
) -> dict:
    model_value = float(model_value)
    reference_value = float(reference_value)
    difference = model_value - reference_value
    ratio = model_value / reference_value if reference_value else None
    correlation = None
    mae = None
    if model_series is not None and reference_series is not None:
        aligned = pd.concat([model_series, reference_series], axis=1).dropna()
        if len(aligned):
            correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            mae = float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().mean())
    if comparison_kind in {
        "direct_validation", "accounting_check", "cross_source_consistency",
    }:
        status = "aligned" if ratio is not None and abs(ratio - 1.0) <= float(tolerance) else "investigate"
    else:
        status = "diagnostic_only"
    return {
        "period_start": str(period_start),
        "period_end": str(period_end),
        "layer": layer,
        "metric": metric,
        "model_value": model_value,
        "reference_value": reference_value,
        "difference_model_minus_reference": difference,
        "ratio_model_to_reference": ratio,
        "unit": unit,
        "comparison_kind": comparison_kind,
        "expected_tolerance": tolerance,
        "status": status,
        "difference_driver": driver,
        "hourly_correlation": correlation,
        "hourly_mae_mw": mae,
    }


def build_scorecard(
    network_path, espeni_path, cfd_summary_path, roc_summary_path,
    roc_reconciliation_path,
):
    import pypsa

    network = pypsa.Network(network_path)
    start = pd.Timestamp(network.snapshots.min())
    end = pd.Timestamp(network.snapshots.max())
    espeni = _load_espeni(espeni_path, start, end)
    weights = _weights(network)
    dispatch = network.generators_t.p.multiply(weights, axis=0)
    carrier_dispatch = dispatch.T.groupby(network.generators.carrier).sum().T
    # MW time series used for correlation/MAE; snapshot weighting belongs only
    # in the energy totals above.
    carrier_mw = network.generators_t.p.T.groupby(network.generators.carrier).sum().T
    hourly_index = espeni.index

    def model_mw(carrier):
        return _series(network, carrier_mw, carrier, hourly_index)

    def model_mwh(carrier):
        return float(_series(network, carrier_dispatch, carrier, network.snapshots).sum())

    rows = []
    physical_pairs = [
        ("CCGT", "CCGT", "ELEC_POWER_ELEX_CCGT[MW](float32)", 0.05, "Thermal availability, marginal costs and system balance."),
        ("nuclear", "nuclear", "ELEC_POWER_ELEX_NUCLEAR[MW](float32)", 0.05, "Nuclear availability and dispatch profile."),
        ("coal", "coal", "ELEC_POWER_ELEX_COAL[MW](float32)", 0.05, "Historical anchor where configured; not independent validation."),
        ("biomass", "biomass", "ELEC_POWER_ELEX_BIOMASS_POSTCALC[MW](float32)", 0.10, "Biomass availability and dispatch profile."),
        ("OCGT", "OCGT", "ELEC_POWER_ELEX_OCGT[MW](float32)", 0.20, "Peaking dispatch and carrier classification."),
        ("large_hydro", "large hydro", "ELEC_POWER_ELEX_NPSHYD[MW](float32)", 0.10, "Hydro resource, storage representation and dispatch."),
    ]
    for carrier, label, column, tolerance, driver in physical_pairs:
        if column not in espeni:
            continue
        model = model_mw(carrier)
        reference = espeni[column]
        # Coal is deliberately anchored to historical dispatch in this historic
        # configuration. Its exact agreement is useful as a data-integrity
        # check, but it cannot validate the optimiser.
        comparison_kind = "diagnostic" if carrier == "coal" else "direct_validation"
        rows.append(_row(
            start, end, "physical_system", label,
            model_mwh(carrier), reference.sum(), "MWh", comparison_kind, driver,
            model, reference, tolerance,
        ))

    metered_wind = model_mw("wind_onshore") + model_mw("wind_offshore")
    rows.append(_row(
        start, end, "physical_system", "metered wind",
        model_mwh("wind_onshore") + model_mwh("wind_offshore"),
        espeni["ELEC_POWER_ELEX_WIND[MW](float32)"].sum(), "MWh", "direct_validation",
        "Renewable profiles, availability factors and network curtailment.",
        metered_wind, espeni["ELEC_POWER_ELEX_WIND[MW](float32)"], 0.05,
    ))
    all_wind = metered_wind + model_mw("embedded_wind")
    rows.append(_row(
        start, end, "physical_system", "all wind",
        model_mwh("wind_onshore") + model_mwh("wind_offshore") + model_mwh("embedded_wind"),
        espeni["ELEC_POWER_TOTAL_WIND[MW](float32)"].sum(), "MWh", "direct_validation",
        "Whole-system wind check, including the separately represented embedded residual.",
        all_wind, espeni["ELEC_POWER_TOTAL_WIND[MW](float32)"], 0.05,
    ))
    all_solar = model_mw("solar_pv") + model_mw("embedded_solar")
    rows.append(_row(
        start, end, "physical_system", "all solar",
        model_mwh("solar_pv") + model_mwh("embedded_solar"),
        espeni["ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)"].sum(), "MWh", "direct_validation",
        "ESPENI's NGEM solar series is the available all-solar reference.",
        all_solar, espeni["ELEC_POWER_NGEM_EMBEDDED_SOLAR_GENERATION[MW](float32)"], 0.05,
    ))
    demand = network.loads_t.p_set.sum(axis=1).reindex(hourly_index)
    rows.append(_row(
        start, end, "physical_system", "demand served",
        demand.sum(), espeni["ELEC_POWER_TOTAL_ESPENI[MW](float32)"].sum(), "MWh", "direct_validation",
        "Demand integration and time alignment.", demand,
        espeni["ELEC_POWER_TOTAL_ESPENI[MW](float32)"], 0.03,
    ))

    cfd = json.loads(Path(cfd_summary_path).read_text(encoding="utf-8"))
    rows.extend([
        _row(start, end, "CfD_contracts", "mapped contract physical generation",
             cfd["modelled_physical_dispatch_mwh_before_contract_date_filter"],
             cfd["actual_generation_mwh_mapped_contracts"], "MWh", "direct_validation",
             "Mapped contract cohort: availability/profile difference plus modelled curtailment.", tolerance=0.05),
        _row(start, end, "CfD_contracts", "mapped contract available energy",
             cfd["modelled_available_energy_mwh_before_contract_date_filter"],
             cfd["actual_generation_mwh_mapped_contracts"], "MWh", "diagnostic",
             "Separates renewable availability/profile bias from curtailment."),
        _row(start, end, "CfD_contracts", "independent CfD payment formula",
             cfd["modelled_payment_independent_formula_gbp"],
             cfd["actual_payment_gbp_mapped_contracts"], "GBP", "direct_validation",
             "Mapped physical dispatch x published daily strike-minus-reference-price terms.", tolerance=0.05),
        _row(start, end, "CfD_contracts", "published payment arithmetic reproduction",
             cfd["lccc_payment_reproduced_from_published_terms_gbp"],
             cfd["actual_payment_gbp_mapped_contracts"], "GBP", "accounting_check",
             "Checks LCCC input terms and arithmetic, independently of model dispatch.", tolerance=0.0001),
    ])

    roc = json.loads(Path(roc_summary_path).read_text(encoding="utf-8"))
    rows.extend([
        _row(start, end, "ROC_certificates", "station-evidenced direct certificates",
             roc["modelled_labelled_roc_certificates"], roc["ofgem_rocs_issued"],
             "ROC certificates", "direct_validation",
             "Station mapping, generation representation, eligibility and ROC bands.", tolerance=0.05),
        _row(start, end, "ROC_certificates", "direct certificates plus recovery bridge",
             roc["reconciled_roc_certificates"], roc["ofgem_rocs_issued"],
             "ROC certificates", "calibration_only",
             "Bridge closes known historical gaps by construction; not independent validation."),
    ])

    # This is the cross-source test behind the "most renewables are ROC or
    # CfD" hypothesis.  The official sources describe complementary support
    # cohorts: LCCC reports CfD output; Ofgem reports ROC-eligible output.  The
    # ESPENI aggregate intentionally includes the corresponding broad physical
    # technologies.  It is a strong coverage test, but not a licence to treat
    # every individual station as mapped.
    roc_detail = pd.read_csv(roc_reconciliation_path)
    roc_eligible_mwh = float(pd.to_numeric(
        roc_detail.get("ofgem_eligible_generation_mwh"), errors="coerce"
    ).sum())
    espeni_support_columns = list(ESPENI_SUPPORT_COMPARISON_COLUMNS.values())
    missing_espeni = set(espeni_support_columns).difference(espeni.columns)
    if missing_espeni:
        raise ValueError(
            "ESPENI is missing support-bridge columns: "
            f"{sorted(missing_espeni)}"
        )
    espeni_supported_renewable_mwh = float(espeni[espeni_support_columns].sum().sum())
    external_supported_renewable_mwh = (
        float(cfd["actual_generation_mwh_mapped_contracts"]) + roc_eligible_mwh
    )
    modelled_mapped_supported_renewable_mwh = (
        float(cfd["modelled_physical_dispatch_mwh_before_contract_date_filter"])
        + float(pd.to_numeric(
            roc_detail.get("modelled_labelled_roc_dispatch_mwh"), errors="coerce"
        ).sum())
    )
    rows.extend([
        _row(
            start, end, "support_scheme_aggregate",
            "external CfD plus ROC output versus ESPENI renewable aggregate",
            external_supported_renewable_mwh, espeni_supported_renewable_mwh,
            "MWh", "cross_source_consistency",
            "LCCC CfD output plus Ofgem ROC-eligible output; ESPENI total wind, solar, biomass and non-pumped hydro. Residual scope is merchant, FiT, embedded and minor technology classification differences.",
            tolerance=0.05,
        ),
        _row(
            start, end, "support_scheme_aggregate",
            "modelled mapped CfD plus ROC output versus external supported output",
            modelled_mapped_supported_renewable_mwh,
            external_supported_renewable_mwh, "MWh", "direct_validation",
            "Combined station mapping, support labels, renewable profiles and eligibility representation.",
            tolerance=0.05,
        ),
    ])
    scorecard = pd.DataFrame(rows)
    summary = {
        "period_start": str(start),
        "period_end": str(end),
        "physical_rows_aligned": int(((scorecard["layer"] == "physical_system") & (scorecard["status"] == "aligned")).sum()),
        "physical_rows_investigate": int(((scorecard["layer"] == "physical_system") & (scorecard["status"] == "investigate")).sum()),
        "cfd_generation_coverage": float(cfd["dispatch_coverage_against_actual_mapped_contracts"]),
        "cfd_available_energy_coverage": float(cfd["available_energy_coverage_against_actual_mapped_contracts"]),
        "roc_direct_certificate_coverage": float(roc["direct_model_coverage_of_ofgem_issue"]),
        "external_cfd_plus_roc_mwh": external_supported_renewable_mwh,
        "espeni_renewable_aggregate_mwh": espeni_supported_renewable_mwh,
        "external_support_coverage_of_espeni_aggregate": (
            external_supported_renewable_mwh / espeni_supported_renewable_mwh
            if espeni_supported_renewable_mwh else None
        ),
        "modelled_mapped_support_coverage_of_external_support": (
            modelled_mapped_supported_renewable_mwh / external_supported_renewable_mwh
            if external_supported_renewable_mwh else None
        ),
        "scope_rule": "A direct-validation row should converge on one. Rows marked diagnostic or calibration_only are intentionally different scopes and must not be used to tune the physical dispatch.",
    }
    return scorecard, summary


def write_scorecard(
    network_path, espeni_path, cfd_summary_path, roc_summary_path,
    roc_reconciliation_path, output_csv, output_summary,
):
    scorecard, summary = build_scorecard(
        network_path, espeni_path, cfd_summary_path, roc_summary_path,
        roc_reconciliation_path,
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_scorecard(
        snakemake.input.network_nc, snakemake.input.espeni_csv,
        snakemake.input.cfd_summary_json, snakemake.input.roc_summary_json,
        snakemake.input.roc_reconciliation_csv,
        snakemake.output.scorecard_csv, snakemake.output.summary_json,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--espeni", required=True)
    parser.add_argument("--cfd-summary", required=True)
    parser.add_argument("--roc-summary", required=True)
    parser.add_argument("--roc-reconciliation", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_scorecard(
        args.network, args.espeni, args.cfd_summary, args.roc_summary,
        args.roc_reconciliation,
        args.output_csv, args.output_summary,
    )
