"""Map direct PyPSA generators to stations that received calendar-year ROCs."""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market.revenue_tracking import compute_roc_income
from scripts.utilities.network_io import load_network


MODEL_TO_OFGEM = {
    "wind_onshore": "onshore_wind", "wind_offshore": "offshore_wind", "solar_pv": "solar_pv",
    "small_hydro": "hydro", "large_hydro": "hydro", "tidal_stream": "tidal_stream", "shoreline_wave": "wave_power",
    "biomass": "fuelled", "biogas": "fuelled", "advanced_biofuel": "fuelled", "waste_to_energy": "fuelled",
    "landfill_gas": "landfill_gas", "sewage_gas": "sewage_gas",
}

REPD_SITE_CARRIERS = {
    "biomass", "biogas", "advanced_biofuel", "waste_to_energy",
    "landfill_gas", "sewage_gas", "large_hydro",
}

# A normalised station-name score of 90+ together with a capacity match is
# sufficiently discriminating for a direct ROC mapping.  The previous 93
# threshold left clear cases such as "Invergarry" / "Invergarry Power Station"
# in the manual-review queue solely because Ofgem adds station descriptors.
HIGH_CONFIDENCE_NAME_AND_CAPACITY_SCORE = 90.0


def _normalise(value: object) -> str:
    text = str(value or "").lower().replace("_", " ").replace("&", " and ")
    # Keep meaningful qualifiers in brackets (for example North/Central) but
    # remove common naming boilerplate.  REPD often says "Biomass CHP Plant"
    # where Ofgem says simply "CHP".
    text = text.replace("(", " ").replace(")", " ")
    text = re.sub(r"\b(wind\s*farm|solar\s*farm|power\s*station|energy\s*from\s*waste|biomass|chp|plant)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _ofgem_category(technology: object) -> str | None:
    value = str(technology or "").lower()
    for text, category in [("offshore wind", "offshore_wind"), ("onshore wind", "onshore_wind"), ("photovoltaic", "solar_pv"), ("landfill gas", "landfill_gas"), ("sewage gas", "sewage_gas"), ("tidal", "tidal_stream"), ("wave", "wave_power"), ("hydro", "hydro"), ("fuelled", "fuelled")]:
        if text in value:
            return category
    return None


def _capacity_ok(model_mw, station_kw) -> tuple[float | None, bool]:
    if pd.isna(model_mw) or pd.isna(station_kw) or model_mw <= 0 or station_kw <= 0:
        return None, False
    station_mw = station_kw / 1000.0
    difference = abs(model_mw - station_mw) / max(model_mw, station_mw)
    return difference, difference <= 0.20 or abs(model_mw - station_mw) <= 1.0


def _clean_reference(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\.0$", "", str(value).strip())


def transfer_repd_metadata(network, source_network) -> tuple[object, dict]:
    """Transfer REPD identity from an integrated network to an older solved run.

    This allows a historic dispatch result to be re-accounted without repeating
    the expensive solve.  Only exact carrier/bus/capacity fingerprints are used;
    ambiguous groups are reported and left untouched.
    """
    metadata = [
        "source_site_name", "repd_ref_id", "operational_date", "ro_banding",
        "cfd_round", "data_source",
    ]
    source = source_network.generators.copy()
    target = network.generators.copy()
    source = source.loc[source.get("data_source", pd.Series(index=source.index, dtype=object)).astype(str).str.upper().eq("REPD")]
    target = target.loc[target.get("data_source", pd.Series(index=target.index, dtype=object)).astype(str).str.upper().eq("REPD")]

    def fingerprint(frame: pd.DataFrame) -> pd.Series:
        capacity = pd.to_numeric(frame["p_nom"], errors="coerce").round(6).astype(str)
        return frame["carrier"].astype(str) + "|" + frame["bus"].astype(str) + "|" + capacity

    source["_fingerprint"] = fingerprint(source)
    target["_fingerprint"] = fingerprint(target)
    transferred = 0
    ambiguous_target_generators = 0
    pairs = []
    for key, source_group in source.groupby("_fingerprint", sort=False):
        target_group = target.loc[target["_fingerprint"].eq(key)]
        if len(source_group) != len(target_group):
            ambiguous_target_generators += len(target_group)
            continue
        pairs.extend(zip(target_group.index, source_group.index))

    for target_name, source_name in pairs:
        for attribute in metadata:
            if attribute in source_network.generators.columns:
                value = source_network.generators.at[source_name, attribute]
                if pd.notna(value) and str(value).strip():
                    network.generators.at[target_name, attribute] = value
        # Keep the solved component index stable; expose its real source name
        # separately so every dispatch time series remains aligned.
        network.generators.at[target_name, "source_site_name"] = source_network.generators.at[
            source_name, "source_site_name"
        ] if "source_site_name" in source_network.generators.columns else source_name
        network.generators.at[target_name, "roc_metadata_source"] = "integrated_network_exact_fingerprint"
        transferred += 1

    return network, {
        "repd_metadata_transferred": transferred,
        "repd_metadata_ambiguous_target_generators": ambiguous_target_generators,
    }


def transfer_repd_site_metadata(network, sites: pd.DataFrame) -> tuple[object, dict]:
    """Recover source identity on legacy anonymous thermal generators.

    Historical solved networks pre-date the real-name integration fix, but
    retain source capacity and coordinates.  A strict carrier/capacity/lat/lon
    fingerprint reconstructs the identity without altering dispatch columns.
    """
    if sites.empty:
        return network, {"repd_site_metadata_transferred": 0, "repd_site_metadata_ambiguous": 0}
    sites = sites.copy()
    required = {"category", "capacity_mw", "lat", "lon"}
    if missing := required.difference(sites.columns):
        raise ValueError(f"REPD site metadata is missing required columns: {sorted(missing)}")
    targets = network.generators.loc[
        network.generators["carrier"].astype(str).isin(REPD_SITE_CARRIERS)
        & network.generators.get("data_source", pd.Series(index=network.generators.index, dtype=object)).astype(str).str.upper().eq("REPD")
    ].copy()

    def key(carrier, capacity, lat, lon):
        values = pd.to_numeric(pd.Series([capacity, lat, lon]), errors="coerce")
        if values.isna().any() or not all(pd.Series(values).map(lambda value: abs(value) != float("inf"))):
            return None
        return f"{carrier}|{values.iloc[0]:.6f}|{values.iloc[1]:.6f}|{values.iloc[2]:.6f}"

    sites["_fingerprint"] = [
        key(carrier, capacity, lat, lon) for carrier, capacity, lat, lon in
        zip(sites["category"], sites["capacity_mw"], sites["lat"], sites["lon"])
    ]
    targets["_fingerprint"] = [
        key(carrier, capacity, lat, lon) for carrier, capacity, lat, lon in
        zip(
            targets["carrier"], targets["p_nom"],
            targets.get("lat", pd.Series(index=targets.index, dtype=float)),
            targets.get("lon", pd.Series(index=targets.index, dtype=float)),
        )
    ]
    site_groups = {fingerprint: group for fingerprint, group in sites.dropna(subset=["_fingerprint"]).groupby("_fingerprint")}
    transferred = 0
    ambiguous = 0
    for fingerprint, target_group in targets.dropna(subset=["_fingerprint"]).groupby("_fingerprint"):
        site_group = site_groups.get(fingerprint)
        if site_group is None or len(site_group) != len(target_group):
            ambiguous += len(target_group)
            continue
        for target_name, (_, site) in zip(target_group.index, site_group.iterrows()):
            for attribute in (
                "source_site_name", "repd_ref_id", "operational_date",
                "ro_banding", "cfd_round", "data_source",
            ):
                value = site.get(attribute)
                if pd.notna(value) and str(value).strip():
                    network.generators.at[target_name, attribute] = value
            network.generators.at[target_name, "roc_metadata_source"] = "repd_site_exact_fingerprint"
            transferred += 1
    return network, {
        "repd_site_metadata_transferred": transferred,
        "repd_site_metadata_ambiguous": ambiguous,
    }


def build_mapping(
    network,
    outturn: pd.DataFrame,
    registry: pd.DataFrame,
    repd_crosswalk: pd.DataFrame | None = None,
    station_overrides: pd.DataFrame | None = None,
    source_evidence: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    registry = registry.copy()
    registry["Accreditation reference"] = registry["Accreditation reference"].map(_clean_reference)
    registry["station_capacity_kw"] = pd.to_numeric(
        registry["Station DNC in kW"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    registry = registry.drop_duplicates("Accreditation reference").copy()

    outturn = outturn.copy()
    outturn["Accreditation reference"] = outturn["Accreditation reference"].map(_clean_reference)
    outturn = outturn.merge(
        registry[["Accreditation reference", "station_capacity_kw"]],
        on="Accreditation reference",
        how="left",
    ).copy()
    outturn["category"] = outturn["Technology"].map(_ofgem_category)
    outturn["normalised_name"] = outturn["Station name"].map(_normalise)
    generators = network.generators.copy()
    generators["generator"] = generators.index.astype(str)
    generators["p_nom_mw"] = pd.to_numeric(generators["p_nom"], errors="coerce").fillna(0.0)
    generators["category"] = generators["carrier"].astype(str).map(MODEL_TO_OFGEM)
    generators = generators.loc[generators["category"].notna() & ~generators["generator"].str.contains("__agg", regex=False)].copy()

    crosswalk_by_ref = {}
    if repd_crosswalk is not None and not repd_crosswalk.empty:
        crosswalk = repd_crosswalk.copy()
        crosswalk = crosswalk.loc[crosswalk["mapping_status"].astype(str).str.startswith("high_confidence")]
        crosswalk["_repd_ref"] = crosswalk["repd_ref_id"].map(_clean_reference)
        crosswalk_by_ref = crosswalk.drop_duplicates("_repd_ref").set_index("_repd_ref").to_dict("index")
    outturn_by_ref = outturn.drop_duplicates("Accreditation reference").set_index("Accreditation reference")
    registry_by_ref = registry.set_index("Accreditation reference")
    outturn_by_category = {
        category: group.copy()
        for category, group in outturn.loc[outturn["category"].notna()].groupby("category", sort=False)
    }

    overrides_by_generator = {}
    if station_overrides is not None and not station_overrides.empty:
        required = {"generator", "ro_accreditation_reference", "evidence"}
        if missing := required.difference(station_overrides.columns):
            raise ValueError(f"ROC station overrides are missing required columns: {sorted(missing)}")
        for row in station_overrides.to_dict("records"):
            generator_name = str(row["generator"]).strip()
            reference = _clean_reference(row["ro_accreditation_reference"])
            if generator_name and reference:
                overrides_by_generator[generator_name] = row

    evidence_by_repd_ref = {}
    evidence_by_name = {}
    if source_evidence is not None and not source_evidence.empty:
        required = {"Accreditation reference", "source", "source_id", "candidate_name", "mapping_status"}
        if missing := required.difference(source_evidence.columns):
            raise ValueError(f"ROC source evidence is missing required columns: {sorted(missing)}")
        evidence = source_evidence.loc[
            source_evidence["mapping_status"].astype(str).str.startswith("high_confidence")
        ].copy()
        for row in evidence.to_dict("records"):
            source = str(row.get("source", ""))
            source_id = _clean_reference(row.get("source_id"))
            if source.startswith("REPD_") and source_id:
                evidence_by_repd_ref.setdefault(source_id, row)
            for name in (row.get("candidate_name"), row.get("source_id")):
                key = _normalise(name)
                if key:
                    evidence_by_name.setdefault(key, row)

    rows = []
    for _, generator in generators.iterrows():
        repd_ref = _clean_reference(generator.get("repd_ref_id"))
        source_site_name = generator.get("source_site_name")
        if pd.isna(source_site_name) or not str(source_site_name).strip():
            source_site_name = generator["generator"]
        crosswalk_match = crosswalk_by_ref.get(repd_ref) if repd_ref else None
        direct_reference = _clean_reference(generator.get("ro_accreditation_reference"))
        override = overrides_by_generator.get(str(generator["generator"]).strip())
        if override is not None:
            override_reference = _clean_reference(override["ro_accreditation_reference"])
            if override_reference in outturn_by_ref.index:
                station = outturn_by_ref.loc[override_reference]
                status = "high_confidence_explicit_override"
                certificates = station["ofgem_roc_certificates"]
                eligible_generation = station["ofgem_eligible_generation_mwh"]
                actual_band = station["implied_roc_band"]
            elif override_reference in registry_by_ref.index:
                # The monthly post-solve audit can encounter a valid, accredited
                # station with no certificates issued in that particular month.
                # Retain the identity in the audit, but do not let it count as
                # a period-evidenced certificate label.
                station = registry_by_ref.loc[override_reference]
                status = "accredited_no_period_rocs"
                certificates = 0.0
                eligible_generation = 0.0
                actual_band = None
            else:
                raise ValueError(
                    f"ROC override for {generator['generator']!r} refers to station "
                    f"{override_reference!r} which is absent from the Ofgem accredited-station register"
                )
            difference, agrees = _capacity_ok(
                generator["p_nom_mw"], station["station_capacity_kw"]
            )
            rows.append({
                "generator": generator["generator"], "source_site_name": source_site_name,
                "repd_ref_id": repd_ref, "carrier": generator["carrier"],
                "p_nom_mw": generator["p_nom_mw"], "existing_support_type": generator.get("support_type", ""),
                "mapping_status": status,
                "mapping_method": "explicit_2020_station_override",
                "mapping_evidence": override["evidence"],
                "name_score": 100.0, "capacity_relative_difference": difference,
                "capacity_agrees": agrees,
                "ro_accreditation_reference": override_reference,
                "ro_station_name": station.get("Station name", ""),
                "ofgem_roc_certificates": certificates,
                "ofgem_eligible_generation_mwh": eligible_generation,
                "actual_roc_band": actual_band,
            })
            continue
        if direct_reference and direct_reference in outturn_by_ref.index:
            station = outturn_by_ref.loc[direct_reference]
            difference, agrees = _capacity_ok(
                generator["p_nom_mw"], station["station_capacity_kw"]
            )
            rows.append({
                "generator": generator["generator"], "source_site_name": source_site_name,
                "repd_ref_id": repd_ref, "carrier": generator["carrier"],
                "p_nom_mw": generator["p_nom_mw"], "existing_support_type": generator.get("support_type", ""),
                "mapping_status": "high_confidence_source_reference",
                "mapping_method": "renewable_support_override_reference",
                "name_score": 100.0, "capacity_relative_difference": difference,
                "capacity_agrees": agrees,
                "ro_accreditation_reference": direct_reference,
                "ro_station_name": station["Station name"],
                "ofgem_roc_certificates": station["ofgem_roc_certificates"],
                "ofgem_eligible_generation_mwh": station["ofgem_eligible_generation_mwh"],
                "actual_roc_band": station["implied_roc_band"],
            })
            continue
        if direct_reference and direct_reference in registry_by_ref.index:
            station = registry_by_ref.loc[direct_reference]
            difference, agrees = _capacity_ok(
                generator["p_nom_mw"], station["station_capacity_kw"]
            )
            rows.append({
                "generator": generator["generator"], "source_site_name": source_site_name,
                "repd_ref_id": repd_ref, "carrier": generator["carrier"],
                "p_nom_mw": generator["p_nom_mw"], "existing_support_type": generator.get("support_type", ""),
                "mapping_status": "accredited_no_period_rocs",
                "mapping_method": "renewable_support_override_reference",
                "mapping_evidence": "Accredited station identity; no certificates issued in this audit period.",
                "name_score": 100.0, "capacity_relative_difference": difference,
                "capacity_agrees": agrees,
                "ro_accreditation_reference": direct_reference,
                "ro_station_name": station.get("Station name", ""),
                "ofgem_roc_certificates": 0.0,
                "ofgem_eligible_generation_mwh": 0.0,
                "actual_roc_band": None,
            })
            continue
        source_match = evidence_by_repd_ref.get(repd_ref) if repd_ref else None
        if source_match is None:
            for candidate_name in (source_site_name, generator["generator"]):
                source_match = evidence_by_name.get(_normalise(candidate_name))
                if source_match is not None:
                    break
        if source_match is not None:
            evidence_reference = _clean_reference(source_match["Accreditation reference"])
            if evidence_reference in outturn_by_ref.index:
                station = outturn_by_ref.loc[evidence_reference]
                status = "high_confidence_source_register_evidence"
                certificates = station["ofgem_roc_certificates"]
                eligible_generation = station["ofgem_eligible_generation_mwh"]
                actual_band = station["implied_roc_band"]
            elif evidence_reference in registry_by_ref.index:
                station = registry_by_ref.loc[evidence_reference]
                status = "accredited_no_period_rocs"
                certificates = eligible_generation = 0.0
                actual_band = None
            else:
                raise ValueError(f"ROC source evidence refers to unaccredited station {evidence_reference!r}")
            difference, agrees = _capacity_ok(generator["p_nom_mw"], station["station_capacity_kw"])
            rows.append({
                "generator": generator["generator"], "source_site_name": source_site_name,
                "repd_ref_id": repd_ref, "carrier": generator["carrier"],
                "p_nom_mw": generator["p_nom_mw"], "existing_support_type": generator.get("support_type", ""),
                "mapping_status": status, "mapping_method": "roc_source_register_evidence",
                "mapping_evidence": source_match.get("mapping_evidence", "REPD/DUKES source-register evidence"),
                "name_score": source_match.get("name_score"),
                "capacity_relative_difference": difference, "capacity_agrees": agrees,
                "ro_accreditation_reference": evidence_reference,
                "ro_station_name": station.get("Station name", ""),
                "ofgem_roc_certificates": certificates,
                "ofgem_eligible_generation_mwh": eligible_generation,
                "actual_roc_band": actual_band,
            })
            continue
        if crosswalk_match is not None:
            accreditation_reference = crosswalk_match["ro_accreditation_reference"]
            if accreditation_reference in outturn_by_ref.index:
                station = outturn_by_ref.loc[accreditation_reference]
                status = "high_confidence_repd_crosswalk"
            else:
                station = None
                status = "accredited_no_calendar_rocs"
            rows.append({
                "generator": generator["generator"], "source_site_name": source_site_name,
                "repd_ref_id": repd_ref, "carrier": generator["carrier"],
                "p_nom_mw": generator["p_nom_mw"], "existing_support_type": generator.get("support_type", ""),
                "mapping_status": status, "mapping_method": "repd_to_ofgem_crosswalk",
                "name_score": crosswalk_match.get("name_score"),
                "capacity_relative_difference": crosswalk_match.get("capacity_relative_difference"),
                "capacity_agrees": crosswalk_match.get("capacity_agrees"),
                "ro_accreditation_reference": accreditation_reference,
                "ro_station_name": station["Station name"] if station is not None else crosswalk_match.get("ro_station_name"),
                "ofgem_roc_certificates": station["ofgem_roc_certificates"] if station is not None else 0.0,
                "ofgem_eligible_generation_mwh": station["ofgem_eligible_generation_mwh"] if station is not None else 0.0,
                "actual_roc_band": station["implied_roc_band"] if station is not None else None,
            })
            continue

        candidates = outturn_by_category.get(generator["category"], outturn.iloc[0:0]).copy()
        query = _normalise(source_site_name)
        # A name-only nearest neighbour can select a similarly described but
        # physically incompatible station (e.g. Markinch CHP versus Ridham
        # Dock biomass plant). Prefer the best name match among candidates
        # whose Ofgem DNC and model capacity agree, while retaining the full
        # candidate set when capacity itself is unavailable or genuinely differs.
        if not candidates.empty:
            model_capacity_mw = float(generator["p_nom_mw"])
            station_capacity_mw = pd.to_numeric(
                candidates["station_capacity_kw"], errors="coerce"
            ) / 1000.0
            denominator = np.maximum(station_capacity_mw, model_capacity_mw)
            relative_difference = (
                (station_capacity_mw - model_capacity_mw).abs() / denominator
            )
            candidates["_capacity_agrees"] = (
                station_capacity_mw.gt(0)
                & (model_capacity_mw > 0)
                & (
                    relative_difference.le(0.20)
                    | (station_capacity_mw - model_capacity_mw).abs().le(1.0)
                )
            )
            compatible = candidates.loc[candidates["_capacity_agrees"]]
            if not compatible.empty:
                candidates = compatible
        choice = process.extractOne(query, candidates["normalised_name"], scorer=fuzz.ratio) if query and not candidates.empty else None
        station = candidates.loc[choice[2]] if choice else None
        score = float(choice[1]) if choice else None
        capacity_difference, capacity_agrees = _capacity_ok(generator["p_nom_mw"], station["station_capacity_kw"] if station is not None else None)
        status = "high_confidence_exact_name" if score == 100.0 else (
            "high_confidence_name_and_capacity"
            if score is not None
            and score >= HIGH_CONFIDENCE_NAME_AND_CAPACITY_SCORE
            and capacity_agrees
            else ("review_required" if station is not None else "unmatched")
        )
        rows.append({"generator": generator["generator"], "source_site_name": source_site_name,
                     "repd_ref_id": repd_ref, "carrier": generator["carrier"], "p_nom_mw": generator["p_nom_mw"],
                     "existing_support_type": generator.get("support_type", ""), "mapping_status": status,
                     "mapping_method": "direct_station_name", "name_score": score,
                     "capacity_relative_difference": capacity_difference, "capacity_agrees": capacity_agrees,
                     "ro_accreditation_reference": station["Accreditation reference"] if station is not None else None,
                     "ro_station_name": station["Station name"] if station is not None else None,
                     "ofgem_roc_certificates": station["ofgem_roc_certificates"] if station is not None else None,
                     "ofgem_eligible_generation_mwh": station["ofgem_eligible_generation_mwh"] if station is not None else None,
                     "actual_roc_band": station["implied_roc_band"] if station is not None else None})
    result = pd.DataFrame(rows)
    high = result["mapping_status"].str.startswith("high_confidence")
    covered = result.loc[high].drop_duplicates("ro_accreditation_reference")
    overrides = result["mapping_method"].eq("explicit_2020_station_override")
    return result.sort_values(["mapping_status", "generator"]).reset_index(drop=True), {"high_confidence_generator_matches": int(high.sum()), "high_confidence_capacity_mw": float(result.loc[high, "p_nom_mw"].sum()), "unique_ofgem_stations_covered": int(covered["ro_accreditation_reference"].nunique()), "ofgem_rocs_covered": float(covered["ofgem_roc_certificates"].sum()), "explicit_station_overrides_applied": int(overrides.sum()), "explicit_station_override_capacity_mw": float(result.loc[overrides, "p_nom_mw"].sum()), "rule": "Calendar-year accounting starts from no ROC labels. Only high-confidence station matches or evidence-backed 2020 station overrides can add a label; inherited model labels are demoted and mapped CfD generators remain CfD."}


def complete_station_representation_mapping(
    mapping: pd.DataFrame,
    outturn: pd.DataFrame,
    registry: pd.DataFrame,
    assumptions: pd.DataFrame | None,
    small_station_threshold_mw: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """Give every issuing Ofgem station an explicit model representation.

    Direct generator identities remain the validation-grade mapping. Stations
    without a defensible identity are assigned to reviewed technology-level
    aggregates. These synthetic rows are accounting metadata only: they never
    label or alter a PyPSA generator and therefore cannot double-count dispatch.
    """
    result = mapping.loc[
        ~mapping["mapping_status"].astype(str).str.startswith("assumed_")
    ].copy()
    for column, default in {
        "representation_id": "",
        "representation_confidence": "candidate_only",
        "representation_assumption": "",
        "is_direct_station_identity": False,
    }.items():
        if column not in result.columns:
            result[column] = default

    high = result["mapping_status"].astype(str).str.startswith("high_confidence")
    result.loc[high, "representation_id"] = result.loc[high, "generator"].astype(str)
    result.loc[high, "representation_confidence"] = "station_evidenced"
    result.loc[high, "is_direct_station_identity"] = True

    official = outturn.copy()
    official["Accreditation reference"] = official["Accreditation reference"].map(_clean_reference)
    official = official.drop_duplicates("Accreditation reference")
    registry_capacity = registry[["Accreditation reference", "Station DNC in kW"]].copy()
    registry_capacity["Accreditation reference"] = registry_capacity["Accreditation reference"].map(_clean_reference)
    registry_capacity["station_capacity_mw"] = pd.to_numeric(
        registry_capacity["Station DNC in kW"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ) / 1000.0
    official = official.merge(
        registry_capacity.drop_duplicates("Accreditation reference")[[
            "Accreditation reference", "station_capacity_mw"
        ]],
        on="Accreditation reference",
        how="left",
    )

    direct_references = set(
        result.loc[high, "ro_accreditation_reference"].map(_clean_reference).loc[lambda values: values.ne("")]
    )
    residual = official.loc[
        ~official["Accreditation reference"].isin(direct_references)
    ].copy()

    if assumptions is None or assumptions.empty:
        assumption_lookup = {}
    else:
        required = {
            "ofgem_technology", "representation_id", "model_carriers",
            "representation_confidence", "assumption",
        }
        if missing := required.difference(assumptions.columns):
            raise ValueError(f"ROC representation assumptions are missing: {sorted(missing)}")
        if assumptions["ofgem_technology"].duplicated().any():
            raise ValueError("ROC representation assumptions must have one row per Ofgem technology")
        assumption_lookup = assumptions.set_index("ofgem_technology").to_dict("index")

    missing_technologies = sorted(
        set(residual["Technology"].dropna()) - set(assumption_lookup)
    )
    if missing_technologies:
        raise ValueError(
            "No aggregate ROC representation assumption exists for issuing technologies: "
            + ", ".join(missing_technologies)
        )

    assumed_rows = []
    for station in residual.to_dict("records"):
        assumption = assumption_lookup[station["Technology"]]
        capacity_mw = pd.to_numeric(
            pd.Series([station.get("station_capacity_mw")]), errors="coerce"
        ).iloc[0]
        is_small = pd.notna(capacity_mw) and float(capacity_mw) < small_station_threshold_mw
        representation_id = str(assumption["representation_id"]).strip()
        assumed_rows.append({
            "generator": f"__roc_aggregate__{representation_id}",
            "source_site_name": station["Station name"],
            "repd_ref_id": "",
            "carrier": str(assumption["model_carriers"]).strip(),
            "p_nom_mw": float(capacity_mw) if pd.notna(capacity_mw) else None,
            "existing_support_type": "aggregate_accounting_representation",
            "mapping_status": (
                "assumed_distributed_aggregate" if is_small
                else "assumed_technology_aggregate"
            ),
            "mapping_method": "reviewed_technology_aggregate_assumption",
            "mapping_evidence": str(assumption["assumption"]).strip(),
            "name_score": None,
            "capacity_relative_difference": None,
            "capacity_agrees": None,
            "ro_accreditation_reference": station["Accreditation reference"],
            "ro_station_name": station["Station name"],
            "ofgem_roc_certificates": station["ofgem_roc_certificates"],
            "ofgem_eligible_generation_mwh": station["ofgem_eligible_generation_mwh"],
            "actual_roc_band": station["implied_roc_band"],
            "representation_id": representation_id,
            "representation_confidence": str(assumption["representation_confidence"]).strip(),
            "representation_assumption": str(assumption["assumption"]).strip(),
            "is_direct_station_identity": False,
        })
    if assumed_rows:
        result = pd.concat([result, pd.DataFrame(assumed_rows)], ignore_index=True, sort=False)

    high = result["mapping_status"].astype(str).str.startswith("high_confidence")
    assumed = result["mapping_status"].astype(str).str.startswith("assumed_")
    represented = high | assumed
    represented_stations = result.loc[represented].drop_duplicates("ro_accreditation_reference")
    official_rocs = float(pd.to_numeric(official["ofgem_roc_certificates"], errors="coerce").fillna(0.0).sum())
    direct_stations = result.loc[high].drop_duplicates("ro_accreditation_reference")
    direct_rocs = float(pd.to_numeric(direct_stations["ofgem_roc_certificates"], errors="coerce").fillna(0.0).sum())
    assumed_stations = result.loc[assumed].drop_duplicates("ro_accreditation_reference")
    assumed_rocs = float(pd.to_numeric(assumed_stations["ofgem_roc_certificates"], errors="coerce").fillna(0.0).sum())
    represented_rocs = direct_rocs + assumed_rocs
    official_count = int(official["Accreditation reference"].nunique())
    represented_count = int(represented_stations["ro_accreditation_reference"].nunique())
    summary = {
        "official_issuing_station_count": official_count,
        "official_roc_certificates": official_rocs,
        "direct_identity_station_count": int(direct_stations["ro_accreditation_reference"].nunique()),
        "direct_identity_roc_certificates": direct_rocs,
        "direct_identity_certificate_coverage": direct_rocs / official_rocs if official_rocs else None,
        "assumed_aggregate_station_count": int(assumed_stations["ro_accreditation_reference"].nunique()),
        "assumed_aggregate_roc_certificates": assumed_rocs,
        "assumed_aggregate_certificate_share": assumed_rocs / official_rocs if official_rocs else None,
        "total_represented_station_count": represented_count,
        "total_station_representation_coverage": represented_count / official_count if official_count else None,
        "total_represented_roc_certificates": represented_rocs,
        "total_certificate_representation_coverage": represented_rocs / official_rocs if official_rocs else None,
        "unrepresented_station_count": official_count - represented_count,
        "unrepresented_roc_certificates": official_rocs - represented_rocs,
        "representation_rule": (
            "Direct identities are evidence-backed. Residual stations are mapped only to reviewed "
            "technology aggregates and do not alter generator labels or physical dispatch."
        ),
    }
    return result.sort_values(["mapping_status", "ro_accreditation_reference", "generator"]).reset_index(drop=True), summary


def apply_labels(
    network, mapping: pd.DataFrame, buyout_price: float,
    outturn: pd.DataFrame | None = None, registry: pd.DataFrame | None = None,
    *, demote_inherited_roc: bool = True,
) -> tuple[object, dict]:
    """Apply calendar-year, station-evidenced ROC labels.

    The post-solve accounting audit sets ``demote_inherited_roc=True``: only
    direct calendar-year evidence is then allowed to count as a certificate.
    Pre-solve enrichment sets it false: pre-existing, date-inferred ROC labels
    remain available to the dispatch model, while stronger Ofgem evidence
    replaces their banding wherever it exists.
    """
    if "support_type" not in network.generators.columns:
        network.generators["support_type"] = "merchant"
    if "ro_banding" not in network.generators.columns:
        network.generators["ro_banding"] = 0.0
    if "roc_eligible_share" not in network.generators.columns:
        network.generators["roc_eligible_share"] = 1.0
    if "roc_label_source" not in network.generators.columns:
        network.generators["roc_label_source"] = ""

    before = compute_roc_income(network, buyout_price)

    # Begin every historical ROC comparison from a conservative baseline.
    # CfD labels are deliberately untouched: a contract cannot simultaneously
    # be recast as a ROC just because its name matches an RO station.
    audit_carriers = network.generators["carrier"].astype(str).isin(MODEL_TO_OFGEM)
    inherited_roc = (
        audit_carriers
        & network.generators["support_type"].astype(str).eq("ROC")
        & ~network.generators["support_type"].astype(str).eq("CfD")
        # Capacity-overlay labels are an explicit, Ofgem-DNC-constrained
        # representation created before the solve. They are not direct station
        # evidence, but must survive post-solve so their dispatch can be
        # reported separately from the strict direct-evidence diagnostic.
        & ~network.generators["roc_label_source"].fillna("").astype(str).str.startswith(
            "ofgem_capacity_overlay_"
        )
    )
    demoted = network.generators.loc[inherited_roc, ["carrier", "p_nom"]].copy()
    if demote_inherited_roc:
        network.generators.loc[inherited_roc, "support_type"] = "merchant"
        network.generators.loc[inherited_roc, "ro_banding"] = 0.0
        network.generators.loc[inherited_roc, "roc_eligible_share"] = 0.0
        network.generators.loc[inherited_roc, "roc_label_source"] = "not_calendar_mapped"

    applied = 0
    for _, row in mapping.loc[mapping["mapping_status"].str.startswith("high_confidence")].iterrows():
        name = row["generator"]
        if network.generators.at[name, "support_type"] == "CfD":
            continue
        network.generators.at[name, "support_type"] = "ROC"
        network.generators.at[name, "ro_banding"] = float(row["actual_roc_band"])
        network.generators.at[name, "roc_eligible_share"] = 1.0
        network.generators.at[name, "ro_accreditation_reference"] = row["ro_accreditation_reference"]
        network.generators.at[name, "roc_label_source"] = "ofgem_calendar_certificate_outturn"
        applied += 1

    embedded_summary = {
        "embedded_wind_roc_share": 0.0,
        "embedded_wind_roc_capacity_mw": 0.0,
        "embedded_wind_bridge_method": "not_applied",
    }
    embedded_all = network.generators.index[
        network.generators["carrier"].astype(str).isin(["embedded_wind", "embedded_solar"])
    ]
    # These are deliberately constructed from ESPENI residuals rather than a
    # station register.  Give them an explicit status, rather than silently
    # leaving a blank that looks like missing support data.
    if len(embedded_all):
        network.generators.loc[embedded_all, "support_type"] = "embedded_unallocated"
        network.generators.loc[embedded_all, "roc_eligible_share"] = 0.0
        network.generators.loc[embedded_all, "roc_label_source"] = "embedded_residual_not_station_identified"
    embedded = network.generators.index[
        network.generators["carrier"].astype(str).eq("embedded_wind")
    ]
    if len(embedded) and outturn is not None and registry is not None:
        # Embedded wind is an aggregate residual, not a station register.  It
        # remains an explicit *candidate* for the recovery bridge, but is not
        # labelled ROC from a capacity residual.  Capacity-based allocation
        # previously combined with inferred site labels and double counted
        # onshore ROC eligibility.
        network.generators.loc[embedded, "support_type"] = "embedded_unallocated"
        network.generators.loc[embedded, "ro_banding"] = 0.0
        network.generators.loc[embedded, "roc_eligible_share"] = 0.0
        network.generators.loc[embedded, "roc_label_source"] = "unlabelled_embedded_candidate"
        issued = outturn.loc[outturn["Technology"].map(_ofgem_category).eq("onshore_wind")].copy()
        reg = registry[["Accreditation reference", "Station DNC in kW"]].drop_duplicates(
            "Accreditation reference"
        ).copy()
        reg["capacity_mw"] = pd.to_numeric(
            reg["Station DNC in kW"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        ) / 1000.0
        issued = issued.merge(reg[["Accreditation reference", "capacity_mw"]], on="Accreditation reference", how="left")
        high = mapping["mapping_status"].astype(str).str.startswith("high_confidence")
        covered_refs = set(mapping.loc[high & mapping["carrier"].eq("wind_onshore"), "ro_accreditation_reference"].dropna())
        issuing_capacity = float(issued.drop_duplicates("Accreditation reference")["capacity_mw"].sum())
        covered_capacity = float(issued.loc[
            issued["Accreditation reference"].isin(covered_refs)
        ].drop_duplicates("Accreditation reference")["capacity_mw"].sum())
        residual_capacity = max(0.0, issuing_capacity - covered_capacity)
        embedded_capacity = float(network.generators.loc[embedded, "p_nom"].sum())
        share = min(1.0, residual_capacity / embedded_capacity) if embedded_capacity else 0.0
        eligible = float(pd.to_numeric(issued["ofgem_eligible_generation_mwh"], errors="coerce").sum())
        certificates = float(pd.to_numeric(issued["ofgem_roc_certificates"], errors="coerce").sum())
        band = certificates / eligible if eligible > 0 else 0.9
        embedded_summary = {
            "embedded_wind_roc_share": 0.0,
            "embedded_wind_roc_capacity_mw": 0.0,
            "embedded_wind_roc_candidate_share": share,
            "embedded_wind_roc_candidate_capacity_mw": residual_capacity,
            "embedded_wind_model_capacity_mw": embedded_capacity,
            "onshore_issuing_station_capacity_mw": issuing_capacity,
            "direct_onshore_capacity_covered_mw": covered_capacity,
            "embedded_wind_observed_roc_band": band,
            "embedded_wind_bridge_method": (
                "Unlabelled candidate only: Ofgem onshore issuing capacity minus directly mapped station capacity, divided by model embedded-wind capacity."
            ),
        }
    after = compute_roc_income(network, buyout_price)
    certificates = lambda frame: float((frame["roc_eligible_dispatch_MWh"] * frame["ro_banding"]).sum()) if not frame.empty else 0.0
    return network, {
        "labels_applied": applied,
        "inherited_roc_labels_demoted": int(inherited_roc.sum()) if demote_inherited_roc else 0,
        "inherited_roc_labels_preserved": int(inherited_roc.sum()) if not demote_inherited_roc else 0,
        "inherited_roc_capacity_demoted_mw": float(pd.to_numeric(demoted["p_nom"], errors="coerce").sum()) if demote_inherited_roc else 0.0,
        "inherited_roc_capacity_demoted_by_carrier_mw": {
            str(carrier): float(value)
            for carrier, value in demoted.groupby("carrier")["p_nom"].sum().items()
        } if demote_inherited_roc else {},
        "proxy_rocs_before": certificates(before),
        "proxy_rocs_after": certificates(after),
        **embedded_summary,
    }


def apply_historical_dispatch_labels(
    network, outturn: pd.DataFrame, registry: pd.DataFrame,
    repd_crosswalk: pd.DataFrame, station_overrides: pd.DataFrame | None,
    buyout_price: float, source_evidence: pd.DataFrame | None = None,
) -> tuple[object, pd.DataFrame, dict]:
    """Enrich a historical build before marginal costs and optimisation.

    The evidence-backed matches add or correct ROC labels.  Existing inferred
    ROC labels are retained for dispatch only; the post-solve audit still
    applies the strict station-evidence rule before comparing certificates.
    """
    mapping, summary = build_mapping(
        network, outturn, registry, repd_crosswalk, station_overrides, source_evidence
    )
    network, labels = apply_labels(
        network, mapping, buyout_price, outturn=outturn, registry=registry,
        demote_inherited_roc=False,
    )
    if "support_label_confidence" not in network.generators.columns:
        network.generators["support_label_confidence"] = ""
    support = network.generators["support_type"].fillna("").astype(str)
    network.generators.loc[support.eq("CfD"), "support_label_confidence"] = "lccc_contract_or_source_record"
    inferred = support.eq("ROC") & network.generators["roc_label_source"].fillna("").eq("")
    network.generators.loc[inferred, "support_label_confidence"] = "date_inferred"
    evidenced = network.generators["roc_label_source"].fillna("").eq("ofgem_calendar_certificate_outturn")
    network.generators.loc[evidenced, "support_label_confidence"] = "ofgem_station_evidenced"
    network.generators.loc[support.eq("merchant"), "support_label_confidence"] = "merchant_or_unresolved"
    network.generators.loc[support.eq("embedded_unallocated"), "support_label_confidence"] = "aggregate_embedded_residual"
    summary.update(labels)
    summary["application"] = (
        "pre_solve_historical_enrichment: Ofgem-evidenced ROC matches are applied "
        "before marginal costs; inferred ROC labels remain separately identifiable."
    )
    return network, mapping, summary


def write_mapping_and_network(
    network_path, outturn_path, registry_path, buyout_price, output_mapping,
    output_network, output_summary, repd_crosswalk_path=None, source_network_path=None,
    repd_site_paths=None, station_overrides_path=None, source_evidence_path=None,
):
    network = load_network(network_path)
    metadata_summary = {}
    if source_network_path:
        source_network = load_network(source_network_path)
        network, metadata_summary = transfer_repd_metadata(network, source_network)
    if repd_site_paths:
        site_frames = [pd.read_csv(path) for path in repd_site_paths]
        network, site_metadata_summary = transfer_repd_site_metadata(
            network, pd.concat(site_frames, ignore_index=True)
        )
        metadata_summary.update(site_metadata_summary)
    crosswalk = pd.read_csv(repd_crosswalk_path) if repd_crosswalk_path else None
    station_overrides = pd.read_csv(station_overrides_path) if station_overrides_path else None
    source_evidence = pd.read_csv(source_evidence_path) if source_evidence_path else None
    outturn = pd.read_csv(outturn_path)
    registry = pd.read_csv(registry_path)
    mapping, summary = build_mapping(
        network, outturn, registry, crosswalk, station_overrides, source_evidence
    )
    network, applied = apply_labels(
        network, mapping, buyout_price, outturn=outturn, registry=registry
    )
    summary.update(metadata_summary)
    summary.update(applied)
    Path(output_mapping).parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_mapping, index=False)
    network.export_to_netcdf(output_network)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_mapping_and_network(
        snakemake.input.network_nc,
        snakemake.input.outturn_csv,
        snakemake.input.registry_csv,
        snakemake.params.buyout_price,
        snakemake.output.mapping_csv,
        snakemake.output.network_nc,
        snakemake.output.summary_json,
        snakemake.input.repd_crosswalk_csv,
        None,
        list(snakemake.input.repd_sites),
        snakemake.input.station_overrides_csv,
        snakemake.input.source_evidence_csv,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True); parser.add_argument("--outturn", required=True); parser.add_argument("--registry", required=True); parser.add_argument("--buyout-price", type=float, required=True)
    parser.add_argument("--repd-crosswalk"); parser.add_argument("--source-network")
    parser.add_argument("--repd-sites", nargs="*")
    parser.add_argument("--station-overrides")
    parser.add_argument("--source-evidence")
    parser.add_argument("--output-mapping", required=True); parser.add_argument("--output-network", required=True); parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_mapping_and_network(
        args.network, args.outturn, args.registry, args.buyout_price,
        args.output_mapping, args.output_network, args.output_summary,
        args.repd_crosswalk, args.source_network, args.repd_sites, args.station_overrides,
        args.source_evidence,
    )
