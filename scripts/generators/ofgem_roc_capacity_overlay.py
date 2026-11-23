"""Allocate the full known Ofgem ROC DNC into the historical PyPSA fleet.

High-confidence station links are immutable.  All other non-CfD renewable
capacity is a technology-matched residual pool: it is split, not duplicated,
until each Ofgem technology's known DNC is represented as ROC capacity.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
import pandas as pd
from pandas.errors import PerformanceWarning


def _ref(value): return re.sub(r"\.0$", "", str(value).strip()) if pd.notna(value) else ""


def _copy_profiles(network, source, target):
    for _, frame in network.generators_t.items():
        if source in frame.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PerformanceWarning)
                frame[target] = frame[source].copy()


def _split_remainder(network, name, roc_mw):
    """Leave ``roc_mw`` on name and create a merchant remainder with its profile."""
    original = float(network.generators.at[name, "p_nom"])
    if roc_mw >= original - 1e-6:
        return
    remainder = f"{name}__non_ROC_capacity_residual"
    n = 2
    while remainder in network.generators.index:
        remainder = f"{name}__non_ROC_capacity_residual_{n}"; n += 1
    attrs = network.generators.loc[name].dropna().to_dict(); attrs.pop("p_nom_opt", None)
    attrs.update({"p_nom": original - roc_mw, "support_type": "merchant", "ro_banding": 0.0,
                  "roc_eligible_share": 0.0, "ro_accreditation_reference": "",
                  "roc_label_source": "ofgem_capacity_overlay_non_roc_remainder"})
    network.add("Generator", remainder, **attrs); _copy_profiles(network, name, remainder)
    network.generators.at[name, "p_nom"] = roc_mw


def _scale_generator_availability(network, name, multiplier):
    """Scale an aggregate residual's feasible output without changing MW."""
    for attribute in ("p_max_pu", "p_min_pu"):
        static = pd.to_numeric(
            pd.Series([network.generators.at[name, attribute]]), errors="coerce"
        ).fillna(0.0).iloc[0]
        frame = getattr(network.generators_t, attribute)
        if name in frame.columns:
            frame[name] = (frame[name] * multiplier).clip(lower=0.0, upper=1.0)
        else:
            network.generators.at[name, attribute] = static * multiplier


def _apply_residual_output_calibration(
    network, calibration_path, modelled_year, carrier_map, logger
):
    """Apply evidence-backed availability to aggregate historical ROC residuals.

    Direct station mappings retain their independent availability evidence.  A
    residual cohort is deliberately different: it represents an Ofgem station
    inventory that cannot yet be connected to distinct PyPSA assets.  Its
    output calibration is therefore recorded separately and is valid only for
    the historical year stated in the input table.
    """
    if not calibration_path or not Path(str(calibration_path)).exists():
        return {}
    calibration = pd.read_csv(calibration_path)
    required = {"year", "ofgem_technology", "availability_multiplier"}
    missing = required - set(calibration.columns)
    if missing:
        raise ValueError(
            "ROC residual output calibration is missing columns: "
            f"{sorted(missing)}"
        )
    calibration = calibration.loc[
        pd.to_numeric(calibration["year"], errors="coerce").eq(int(modelled_year))
    ].copy()
    applied = {}
    for row in calibration.to_dict("records"):
        multiplier = float(row["availability_multiplier"])
        if not 0.0 < multiplier <= 1.0:
            raise ValueError(
                "ROC residual availability_multiplier must be in (0, 1]: "
                f"{multiplier} for {row['ofgem_technology']}"
            )
        source = str(row.get(
            "roc_label_source", "ofgem_capacity_overlay_assumed_residual"
        ))
        carriers = carrier_map.get(str(row["ofgem_technology"]), [])
        names = network.generators.index[
            network.generators.carrier.isin(carriers)
            & network.generators.support_type.astype(str).eq("ROC")
            & network.generators.roc_label_source.astype(str).eq(source)
        ]
        for name in names:
            _scale_generator_availability(network, name, multiplier)
        applied[str(row["ofgem_technology"])] = {
            "availability_multiplier": multiplier,
            "generators": int(len(names)),
            "roc_mw": float(network.generators.loc[names, "p_nom"].sum()),
        }
        logger.info(
            "Applied historical ROC residual output calibration for %s: "
            "%.4f to %d aggregate generators (%.1f MW).",
            row["ofgem_technology"], multiplier, len(names),
            applied[str(row["ofgem_technology"])]["roc_mw"],
        )
    return applied


def apply_ofgem_roc_capacity_overlay(
    network, outturn, registry, mapping, assumptions_path, logger,
    residual_output_calibration_path=None, modelled_year=None,
):
    assumptions = pd.read_csv(assumptions_path)
    carrier_map = {r["ofgem_technology"]: [x.strip() for x in str(r["model_carriers"]).split(";")]
                   for r in assumptions.to_dict("records")}
    registry = registry.copy(); registry["Accreditation reference"] = registry["Accreditation reference"].map(_ref)
    registry["dnc_mw"] = pd.to_numeric(registry["Station DNC in kW"].astype(str).str.replace(",", "", regex=False), errors="coerce") / 1000
    stations = outturn.copy(); stations["Accreditation reference"] = stations["Accreditation reference"].map(_ref)
    stations = stations.merge(registry[["Accreditation reference", "dnc_mw"]].drop_duplicates("Accreditation reference"), on="Accreditation reference", how="left")
    high = mapping["mapping_status"].astype(str).str.startswith("high_confidence")
    locked = set(mapping.loc[high, "generator"].astype(str)).intersection(network.generators.index)
    # Unmatched labels are intentionally reset before allocation so the result is
    # a capacity reconciliation, rather than old inferred labels plus new ROC MW.
    non_locked_roc = network.generators.index[(network.generators.support_type.astype(str).eq("ROC")) & ~network.generators.index.isin(locked)]
    network.generators.loc[non_locked_roc, ["support_type", "ro_banding", "roc_eligible_share"]] = ["merchant", 0.0, 0.0]
    allocated_residual_names = set()
    rows = []
    for technology, group in stations.groupby("Technology"):
        carriers = carrier_map.get(technology, [])
        desired = float(group["dnc_mw"].sum(skipna=True))
        eligible = float(pd.to_numeric(group.get("ofgem_eligible_generation_mwh"), errors="coerce").sum())
        certificates = float(pd.to_numeric(group.get("ofgem_roc_certificates"), errors="coerce").sum())
        residual_band = certificates / eligible if eligible > 0 else 1.0
        missing_dnc_stations = int(group["dnc_mw"].isna().sum())
        station_refs = set(group["Accreditation reference"])
        locked_names = [
            n for n in mapping.loc[
                high & mapping["ro_accreditation_reference"].map(_ref).isin(station_refs), "generator"
            ].astype(str)
            if n in network.generators.index and network.generators.at[n, "support_type"] != "CfD"
        ]
        locked_mw = float(network.generators.loc[locked_names, "p_nom"].sum()) if locked_names else 0.0
        # A good identity match can still cover a larger planning project than
        # the accredited ROC DNC. Keep its identity, but split the excess into
        # an explicitly non-ROC share before filling the residual pool.
        if locked_mw > desired + 1e-6:
            keep = desired
            for name in sorted(locked_names):
                capacity = float(network.generators.at[name, "p_nom"])
                if keep <= 1e-6:
                    network.generators.loc[name, ["support_type", "ro_banding", "roc_eligible_share", "roc_label_source"]] = ["merchant", 0.0, 0.0, "ofgem_capacity_overlay_matched_non_roc_excess"]
                    continue
                if capacity > keep + 1e-6:
                    _split_remainder(network, name, keep)
                keep -= min(capacity, keep)
            locked_mw = desired
        target = max(0.0, desired - locked_mw)
        candidates = network.generators.index[
            network.generators.carrier.isin(carriers)
            & ~network.generators.index.isin(locked)
            & ~network.generators.index.isin(allocated_residual_names)
            & ~network.generators.support_type.astype(str).eq("CfD")
        ].tolist()
        allocated = 0.0
        for name in candidates:
            if target - allocated <= 1e-6: break
            take = min(float(network.generators.at[name, "p_nom"]), target - allocated)
            _split_remainder(network, name, take)
            network.generators.loc[name, ["support_type", "ro_banding", "roc_eligible_share", "roc_label_source"]] = ["ROC", residual_band, 1.0, "ofgem_capacity_overlay_assumed_residual"]
            allocated_residual_names.add(name)
            allocated += take
        shortfall = max(0.0, target - allocated)
        # A positive shortfall means the historical source fleet lacks physical
        # MW that Ofgem says was accredited and issuing in 2020. Add that MW
        # explicitly at an existing technology-matched site/profile; unlike an
        # unlabelled duplicate this has a visible Ofgem-capacity provenance.
        added = 0.0
        if shortfall > 1e-6 and candidates:
            source = candidates[0]
            name = f"Ofgem_ROC_missing_capacity_{technology}_{source}"
            name = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
            attrs = network.generators.loc[source].dropna().to_dict(); attrs.pop("p_nom_opt", None)
            attrs.update({"p_nom": shortfall, "support_type": "ROC", "ro_banding": residual_band,
                          "roc_eligible_share": 1.0, "ro_accreditation_reference": "",
                          "roc_label_source": "ofgem_capacity_overlay_missing_source_fleet"})
            network.add("Generator", name, **attrs); _copy_profiles(network, source, name)
            allocated_residual_names.add(name)
            allocated += shortfall; added = shortfall; shortfall = 0.0
        rows.append({"ofgem_technology": technology, "ofgem_known_dnc_mw": desired, "ofgem_dnc_missing_station_count": missing_dnc_stations,
                     "locked_evidenced_roc_mw": locked_mw, "residual_pool_roc_mw": allocated - added, "explicit_missing_source_fleet_roc_mw": added,
                     "unrepresented_known_dnc_mw": shortfall, "known_dnc_coverage": (locked_mw + allocated) / desired if desired else 1.0,
                     "model_carriers": ";".join(carriers)})
    residual_calibration = _apply_residual_output_calibration(
        network, residual_output_calibration_path, modelled_year,
        carrier_map, logger,
    )
    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        ledger["residual_output_availability_multiplier"] = ledger[
            "ofgem_technology"
        ].map(lambda technology: residual_calibration.get(
            technology, {}
        ).get("availability_multiplier"))
        ledger["residual_output_calibrated_roc_mw"] = ledger[
            "ofgem_technology"
        ].map(lambda technology: residual_calibration.get(
            technology, {}
        ).get("roc_mw", 0.0))
    logger.info("Ofgem ROC capacity overlay represents %.3f%% of known DNC MW.", 100 * ledger.ofgem_known_dnc_mw.sub(ledger.unrepresented_known_dnc_mw).sum() / ledger.ofgem_known_dnc_mw.sum())
    return network, ledger
