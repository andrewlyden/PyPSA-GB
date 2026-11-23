"""Replace one model generator with evidenced separately-accredited units.

Planning and capacity datasets can describe a physical development as one site,
while Ofgem accredits separately-metered parts of it as independent ROC (or
CfD) stations. This module makes that relationship explicit before the market
model is solved. It deliberately does not infer splits: every split must be in
a reviewed CSV input.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pypsa


REQUIRED_COLUMNS = {
    "source_generator_name", "source_capacity_mw", "capacity_treatment",
    "unit_name", "capacity_mw", "support_type", "ro_banding",
    "ro_accreditation_reference", "evidence",
}


def apply_historical_station_unit_splits(
    network: pypsa.Network,
    split_path: str | Path | None,
    logger: logging.Logger,
) -> pypsa.Network:
    """Apply explicit station-unit splits while retaining source profiles.

    ``preserve_total`` scales input unit capacities to the model source
    capacity. ``replace`` uses the source-backed unit capacities verbatim, so
    an historic station register can correct an aggregated planning total.
    """
    if not split_path:
        return network
    path = Path(str(split_path))
    if not path.exists():
        raise FileNotFoundError(f"Station-unit split file not found: {path}")

    splits = pd.read_csv(path)
    if missing := REQUIRED_COLUMNS.difference(splits.columns):
        raise ValueError(f"Station-unit split file is missing: {sorted(missing)}")
    if splits.empty:
        return network
    if splits["unit_name"].duplicated().any():
        duplicates = splits.loc[splits["unit_name"].duplicated(), "unit_name"].tolist()
        raise ValueError(f"Station-unit split unit_name values must be unique: {duplicates}")

    for source_name, units in splits.groupby("source_generator_name", sort=False):
        if source_name not in network.generators.index:
            raise ValueError(
                f"Station-unit split source generator {source_name!r} is not present "
                "in the pre-aggregation renewable network"
            )
        if any(name in network.generators.index and name != source_name for name in units["unit_name"]):
            raise ValueError(f"Station-unit split would overwrite an existing generator for {source_name!r}")

        source = network.generators.loc[source_name].copy()
        source_capacity = float(source["p_nom"])
        expected = pd.to_numeric(units["source_capacity_mw"], errors="raise")
        if expected.nunique() != 1:
            raise ValueError(f"Station-unit split has inconsistent source_capacity_mw for {source_name!r}")
        if abs(source_capacity - float(expected.iloc[0])) > 0.01:
            raise ValueError(
                f"Station-unit split source capacity mismatch for {source_name!r}: "
                f"network={source_capacity:.3f} MW, input={expected.iloc[0]:.3f} MW"
            )

        treatment = str(units["capacity_treatment"].iloc[0]).strip().lower()
        if units["capacity_treatment"].astype(str).str.strip().str.lower().nunique() != 1:
            raise ValueError(f"Station-unit split has inconsistent capacity_treatment for {source_name!r}")
        unit_capacities = pd.to_numeric(units["capacity_mw"], errors="raise").astype(float)
        if (unit_capacities <= 0).any():
            raise ValueError(f"Station-unit split has non-positive capacity for {source_name!r}")
        if treatment == "preserve_total":
            unit_capacities *= source_capacity / unit_capacities.sum()
        elif treatment != "replace":
            raise ValueError(
                f"Station-unit split capacity_treatment for {source_name!r} must be "
                "'preserve_total' or 'replace'"
            )

        profile = None
        if source_name in network.generators_t.p_max_pu.columns:
            profile = network.generators_t.p_max_pu[source_name].copy()

        network.remove("Generator", source_name)
        for (_, unit), unit_capacity in zip(units.iterrows(), unit_capacities):
            unit_name = str(unit["unit_name"]).strip()
            attrs = source.drop(labels=["p_nom"], errors="ignore").dropna().to_dict()
            attrs.update({
                "p_nom": float(unit_capacity),
                "support_type": str(unit["support_type"]).strip(),
                "ro_banding": float(unit["ro_banding"]),
                "ro_accreditation_reference": str(unit["ro_accreditation_reference"]).strip(),
                "source_site_name": unit_name,
                "source_generator_name": source_name,
                "support_override_source": str(path),
                "support_evidence": str(unit["evidence"]),
            })
            network.add("Generator", unit_name, **attrs)
            if profile is not None:
                network.generators_t.p_max_pu[unit_name] = profile

        logger.info(
            "Split %s (%.1f MW) into %d separately-accredited units (%.1f MW, %s)",
            source_name, source_capacity, len(units), unit_capacities.sum(), treatment,
        )
    return network
