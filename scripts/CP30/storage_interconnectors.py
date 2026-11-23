"""Clean Power 2030 storage and interconnector helpers."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CP30StorageMapping:
    subtype: str
    carrier: str
    proxy_technologies: tuple[str, ...]


# Efficiencies are not set here. They come from config/defaults/base/storage.yaml
# storage.round_trip_efficiencies, keyed on the CP30 subtype first (so CP30's
# "LDES" can differ from FES's "LAES") and then on the carrier. A hard-coded
# 0.60 per leg once gave CP30 LDES a 36% round trip and overwrote the value
# add_storage.py had computed; CP30 ES1 itself implies 0.70 for LDES
# (10,441.5 GWh generated from 14,848.7 GWh consumed, FFR 2030).
CP30_STORAGE_MAP: tuple[CP30StorageMapping, ...] = (
    CP30StorageMapping("Battery", "Battery", ("Battery", "Domestic Battery")),
    CP30StorageMapping("LDES", "LAES", ("LAES", "Liquid Air Energy Storage", "Compressed Air Energy Storage", "CAES")),
)


def _read_cp30_es1(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CP30 ES1 file not found: {path}")
    return pd.read_csv(path)


def _mapping_by_subtype() -> dict[str, CP30StorageMapping]:
    return {mapping.subtype: mapping for mapping in CP30_STORAGE_MAP}


def cp30_round_trip_efficiency(
    subtype: str, carrier: str, round_trip_efficiencies: Mapping[str, float]
) -> float:
    """Configured round trip for a CP30 storage subtype, falling back to its carrier."""
    configured = dict(round_trip_efficiencies)
    value = configured.get(subtype, configured.get(carrier))
    if value is None:
        raise ValueError(
            f"No round-trip efficiency for CP30 storage subtype {subtype!r} (carrier {carrier!r}); "
            "add it to storage.round_trip_efficiencies in config/defaults/base/storage.yaml"
        )
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"Round-trip efficiency for {subtype!r} must be in (0, 1], got {value}")
    return value


def load_cp30_storage_targets(
    es1_path: str | Path,
    pathway: str,
    modelled_year: int,
    round_trip_efficiencies: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return CP30 storage power and energy targets by PyPSA carrier.

    With ``round_trip_efficiencies`` each target also carries its round trip
    and the symmetric per-leg split, sqrt(round trip), as add_storage.py uses.
    ``cp30_implied_round_trip`` is CP30's own generation / consumption ratio,
    reported for comparison and never used to set the model.
    """
    df = _read_cp30_es1(es1_path)
    year_col = str(modelled_year)
    required = {"Pathway", "Variable", "Category", "Type", "SubType", year_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CP30 ES1 missing required columns: {sorted(missing)}")

    pathway_storage = df[(df["Pathway"] == pathway) & (df["Category"] == "Storage")].copy()
    pathway_storage[year_col] = pd.to_numeric(pathway_storage[year_col], errors="coerce")
    storage = pathway_storage[
        (pathway_storage["Type"] == "Storage")
        & (pathway_storage["Variable"].isin(["Capacity (MW)", "Storage Capacity (GWh)"]))
    ]
    storage = storage[storage[year_col] > 0.0]

    mapping_by_subtype = _mapping_by_subtype()
    rows = []
    for subtype, group in storage.groupby("SubType"):
        subtype = str(subtype).strip()
        mapping = mapping_by_subtype.get(subtype)
        if mapping is None:
            continue

        power_mw = float(group.loc[group["Variable"] == "Capacity (MW)", year_col].sum())
        energy_gwh = float(group.loc[group["Variable"] == "Storage Capacity (GWh)", year_col].sum())
        if power_mw <= 0.0:
            continue
        duration_h = (energy_gwh * 1000.0 / power_mw) if energy_gwh > 0.0 else None
        flows = pathway_storage[pathway_storage["SubType"].astype(str).str.strip() == subtype]
        generation = float(flows.loc[flows["Variable"] == "Generation (GWh)", year_col].sum())
        consumption = float(flows.loc[flows["Variable"] == "Consumption (GWh)", year_col].sum())
        row = {
            "carrier": mapping.carrier,
            "cp30_subtype": subtype,
            "power_mw": power_mw,
            "energy_gwh": energy_gwh,
            "energy_mwh": energy_gwh * 1000.0,
            "duration_h": duration_h,
            "cp30_implied_round_trip": generation / consumption if consumption > 0.0 else np.nan,
            "pathway": pathway,
            "modelled_year": int(modelled_year),
            "proxy_technologies": mapping.proxy_technologies,
        }
        if round_trip_efficiencies is not None:
            round_trip = cp30_round_trip_efficiency(subtype, mapping.carrier, round_trip_efficiencies)
            row["round_trip_efficiency"] = round_trip
            row["eta_charge"] = row["eta_discharge"] = float(np.sqrt(round_trip))
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "carrier",
                "cp30_subtype",
                "power_mw",
                "energy_gwh",
                "energy_mwh",
                "duration_h",
                "proxy_technologies",
            ]
        )
    return pd.DataFrame(rows).sort_values("carrier").reset_index(drop=True)


def parse_existing_storage_sites(sites) -> list[dict]:
    """Validate the configured existing sites a CP30 storage subtype contains.

    Each site is a mapping with ``name``, ``site_code`` (the four-character
    ETYS substation code the plant connects at), ``lat``, ``lon``,
    ``power_mw`` and ``energy_mwh``; ``subtype`` (default ``LDES``) names the
    CP30 subtype it sits inside and ``enabled`` (default true) switches it.
    """
    out = []
    for site in sites or []:
        if not bool(site.get("enabled", True)):
            continue
        missing = {"name", "site_code", "lat", "lon", "power_mw", "energy_mwh"} - set(site)
        if missing:
            raise ValueError(f"existing storage site {site!r} lacks {sorted(missing)}")
        power, energy = float(site["power_mw"]), float(site["energy_mwh"])
        if not (power > 0.0 and energy > 0.0):
            raise ValueError(f"existing storage site {site['name']!r} needs positive power_mw and energy_mwh")
        code = str(site["site_code"]).strip().upper()
        if len(code) != 4:
            raise ValueError(f"existing storage site {site['name']!r}: site_code must be four characters, got {code!r}")
        out.append({
            "name": str(site["name"]), "site_code": code, "lat": float(site["lat"]), "lon": float(site["lon"]),
            "power_mw": power, "energy_mwh": energy, "subtype": str(site.get("subtype", "LDES")),
            "source": str(site.get("source", "")),
        })
    names = [s["name"].lower() for s in out]
    if len(set(names)) != len(names):
        raise ValueError(f"existing storage sites repeat a name: {sorted(names)}")
    return out


#: Named new-build storage each CP30 pathway counts, at its TEC substation (15 Oct 2026): a
#: declared input of rule add_storage_to_network for CP30 scenarios.
COMMITTED_STORAGE_SITES_FILE = "data/storage/cp30_committed_ldes_sites.csv"
#: Columns of the committed new-build storage file (:data:`COMMITTED_STORAGE_SITES_FILE`).
COMMITTED_SITE_COLUMNS = (
    "name", "unit_label", "pathway", "subtype", "site_code", "lat", "lon", "power_mw", "energy_mwh",
)


def load_committed_storage_sites(path: str | Path, pathway: str) -> list[dict]:
    """Committed new-build storage projects a CP30 pathway counts, from the declared CSV (15 Oct 2026).

    Each row is one project in one CP30 pathway: ``name``, ``unit_label`` (the network unit's
    label; it must not contain an existing site's name, because the public-ownership register
    matches legacy pumped hydro by name), ``pathway``, ``subtype`` (``LDES``), ``site_code`` (the
    four-character ETYS substation the TEC register names, or the nearest same-side substation where
    the TEC substation is not in the network), ``lat``/``lon``, ``power_mw`` and ``energy_mwh``. A
    blank ``energy_mwh`` takes the CP30 subtype's own average duration
    (:func:`scale_storage_proxy_to_cp30_targets`). Returns rows of the given pathway, in
    :func:`parse_committed_storage_sites` form.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CP30 committed storage sites file not found: {path}")
    table = pd.read_csv(path, comment="#")
    missing = set(COMMITTED_SITE_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    rows = table[table["pathway"].astype(str).str.strip().eq(str(pathway).strip())]
    return parse_committed_storage_sites(rows.to_dict("records"))


def parse_committed_storage_sites(sites) -> list[dict]:
    """Validate committed new-build storage sites (see :func:`load_committed_storage_sites`).

    ``energy_mwh`` may be missing or NaN (the subtype's average duration is used).
    """
    out = []
    for site in sites or []:
        if not bool(site.get("enabled", True)):
            continue
        missing = {"name", "site_code", "lat", "lon", "power_mw"} - set(site)
        if missing:
            raise ValueError(f"committed storage site {site!r} lacks {sorted(missing)}")
        power = float(site["power_mw"])
        energy = pd.to_numeric(pd.Series([site.get("energy_mwh")]), errors="coerce").iloc[0]
        energy = float(energy) if np.isfinite(energy) else None
        if not power > 0.0 or (energy is not None and not energy > 0.0):
            raise ValueError(f"committed storage site {site['name']!r} needs positive power_mw (and energy_mwh if given)")
        code = str(site["site_code"]).strip().upper()
        if len(code) != 4:
            raise ValueError(f"committed storage site {site['name']!r}: site_code must be four characters, got {code!r}")
        label = str(site.get("unit_label") or site["name"]).strip().replace(" ", "_")
        out.append({
            "name": str(site["name"]), "unit_label": label, "site_code": code,
            "lat": float(site["lat"]), "lon": float(site["lon"]), "power_mw": power, "energy_mwh": energy,
            "subtype": str(site.get("subtype", "LDES")),
            "source": str(site.get("power_source", site.get("source", "")) or ""),
        })
    labels = [s["unit_label"].lower() for s in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"committed storage sites repeat a unit_label: {sorted(labels)}")
    return out


def _check_committed_labels(committed: list[dict], existing: list[dict]) -> None:
    """A committed unit label naming an existing site would be booked as that site's legacy plant."""
    for site in committed:
        for old in existing:
            if old["name"].lower() in site["unit_label"].lower():
                raise ValueError(
                    f"committed storage unit_label {site['unit_label']!r} contains the existing site name "
                    f"{old['name']!r}; the public-ownership register would book it as legacy plant. Rename it."
                )


def _existing_site_rows(sites: list[dict], target: pd.Series, template_columns) -> pd.DataFrame:
    rows = []
    for site in sites:
        row = {column: np.nan for column in template_columns}
        row.update({
            "site_name": f"CP30_{target['carrier']}_existing_{site['name']}",
            "technology": target["carrier"],
            "power_mw": site["power_mw"],
            "energy_mwh": site["energy_mwh"],
            "duration_h": site["energy_mwh"] / site["power_mw"],
            "lat": site["lat"],
            "lon": site["lon"],
            "site_code": site["site_code"],
            "existing_site": site["name"],
            "existing_site_source": site["source"],
            "spatial_proxy_source": "existing_site",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _committed_site_rows(sites: list[dict], target: pd.Series, template_columns) -> pd.DataFrame:
    """Rows for committed new build; ``energy_mwh`` already resolved (no None)."""
    rows = []
    for site in sites:
        row = {column: np.nan for column in template_columns}
        row.update({
            "site_name": f"CP30_{target['carrier']}_committed_{site['unit_label']}",
            "technology": target["carrier"],
            "power_mw": site["power_mw"],
            "energy_mwh": site["energy_mwh"],
            "duration_h": site["energy_mwh"] / site["power_mw"],
            "lat": site["lat"],
            "lon": site["lon"],
            "site_code": site["site_code"],
            "committed_site": site["name"],
            "committed_site_source": site["source"],
            "spatial_proxy_source": "committed_site",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_site_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows placed at a named site (existing plant or committed new build), never rescaled."""
    mask = pd.Series(False, index=frame.index)
    for column in ("existing_site", "committed_site"):
        if column in frame.columns:
            mask |= frame[column].notna()
    return mask


def _is_existing_site_proxy(proxy: pd.DataFrame, sites: list[dict]) -> pd.Series:
    """Proxy rows that are an existing site's own REPD entry (named after it)."""
    if "site_name" not in proxy.columns or not sites:
        return pd.Series(False, index=proxy.index)
    names = proxy["site_name"].astype(str).str.lower()
    hit = pd.Series(False, index=proxy.index)
    for site in sites:
        hit |= names.str.contains(site["name"].lower(), regex=False)
    return hit


def scale_storage_proxy_to_cp30_targets(
    proxy_df: pd.DataFrame, targets: pd.DataFrame, existing_sites=None, committed_sites=None
) -> pd.DataFrame:
    """Scale storage proxy rows so CP30 power and energy totals are preserved.

    ``existing_sites`` (see :func:`parse_existing_storage_sites`) are plant a
    CP30 subtype already contains, such as the pumped hydro inside CP30's LDES.
    Each becomes one row at its own location with its real power and energy;
    the rest of the subtype's power and energy (the new build) is spread over
    the proxy rows, less the proxy rows that are the existing sites' own REPD
    entries, at the residual duration. The subtype's totals stay exact.

    ``committed_sites`` (:func:`parse_committed_storage_sites`, 15 Oct 2026) are
    named new-build projects the pathway counts (Coire Glas, Cruachan II, ...):
    fixed rows like the existing sites, labelled ``committed_site``. One without
    ``energy_mwh`` takes the subtype's CP30 average duration (ES1 energy / power).
    """
    if proxy_df.empty or targets.empty:
        return pd.DataFrame()
    required = {"technology", "power_mw"}
    missing = required - set(proxy_df.columns)
    if missing:
        raise ValueError(f"Storage proxy dataframe missing columns: {sorted(missing)}")
    sites = parse_existing_storage_sites(existing_sites)
    committed_all = parse_committed_storage_sites(committed_sites)
    _check_committed_labels(committed_all, sites)

    scaled_rows = []
    for _, target in targets.iterrows():
        proxy_technologies = tuple(target.get("proxy_technologies") or (target["carrier"],))
        carrier_proxy = proxy_df[proxy_df["technology"].isin(proxy_technologies)].copy()
        carrier_proxy["power_mw"] = pd.to_numeric(carrier_proxy["power_mw"], errors="coerce").fillna(0.0)
        carrier_proxy = carrier_proxy[carrier_proxy["power_mw"] > 0.0]
        if carrier_proxy.empty:
            continue

        target_power = float(target["power_mw"])
        target_energy = float(target["energy_mwh"])
        target_duration = float(target["duration_h"])
        target_sites = [s for s in sites if s["subtype"] == str(target["cp30_subtype"])]
        committed = [dict(s) for s in committed_all if s["subtype"] == str(target["cp30_subtype"])]
        for site in committed:
            if site["energy_mwh"] is None:      # CP30's own average duration for the subtype
                site["energy_mwh"] = site["power_mw"] * target_energy / target_power
        existing = pd.DataFrame()
        if target_sites or committed:
            fixed_power = sum(s["power_mw"] for s in target_sites + committed)
            fixed_energy = sum(s["energy_mwh"] for s in target_sites + committed)
            if fixed_power >= target_power or fixed_energy >= target_energy:
                raise ValueError(
                    f"Existing and committed {target['cp30_subtype']} sites ({fixed_power:,.0f} MW, "
                    f"{fixed_energy:,.0f} MWh) leave no new build inside the CP30 target "
                    f"({target_power:,.0f} MW, {target_energy:,.0f} MWh)"
                )
            existing = pd.concat(
                [frame for frame in (_existing_site_rows(target_sites, target, carrier_proxy.columns),
                                     _committed_site_rows(committed, target, carrier_proxy.columns))
                 if not frame.empty],
                ignore_index=True,
            )
            named = target_sites + [{"name": s["name"]} for s in committed]
            carrier_proxy = carrier_proxy[~_is_existing_site_proxy(carrier_proxy, named)]
            if carrier_proxy.empty:
                raise ValueError(f"No proxy rows left for the new-build part of CP30 {target['cp30_subtype']}")
            target_power -= fixed_power
            target_duration = (target_energy - fixed_energy) / target_power

        proxy_total = float(carrier_proxy["power_mw"].sum())
        carrier_proxy["power_mw"] = carrier_proxy["power_mw"] / proxy_total * target_power
        carrier_proxy["technology"] = target["carrier"]
        carrier_proxy["duration_h"] = target_duration
        carrier_proxy["energy_mwh"] = carrier_proxy["power_mw"] * target_duration
        if not existing.empty:
            carrier_proxy = pd.concat([existing, carrier_proxy], ignore_index=True)
        # The target's efficiency is the configured CP30 one; without it the
        # proxy rows keep the configured efficiency of their own technology.
        for column in ("eta_charge", "eta_discharge"):
            value = pd.to_numeric(pd.Series([target.get(column)]), errors="coerce").iloc[0]
            if np.isfinite(value):
                carrier_proxy[column] = float(value)
            elif column in carrier_proxy.columns and not existing.empty:
                # existing-site rows take the proxy technology's own efficiency
                own = pd.to_numeric(carrier_proxy[column], errors="coerce").dropna()
                carrier_proxy[column] = carrier_proxy[column].fillna(float(own.iloc[0]) if len(own) else np.nan)
            elif column not in carrier_proxy.columns:
                raise ValueError(
                    f"No {column} for CP30 {target['cp30_subtype']}: pass round_trip_efficiencies "
                    "to load_cp30_storage_targets or supply it on the proxy rows"
                )
        # Keep the FES row type (FES_GSP_* = distribution-connected at a GSP, FES_Direct_*) for placement.
        carrier_proxy["proxy_source"] = (carrier_proxy["source"] if "source" in carrier_proxy.columns
                                         else np.nan)
        carrier_proxy["source"] = "CP30"
        carrier_proxy["data_source"] = "CP30"
        is_existing = fixed_site_mask(carrier_proxy)
        is_committed = (carrier_proxy["committed_site"].notna() if "committed_site" in carrier_proxy.columns
                        else pd.Series(False, index=carrier_proxy.index))
        carrier_proxy["spatial_proxy_source"] = np.where(
            is_committed, "committed_site", np.where(is_existing, "existing_site", "FES"))
        carrier_proxy["cp30_pathway"] = target["pathway"]
        carrier_proxy["cp30_subtype"] = target["cp30_subtype"]
        carrier_proxy["cp30_target_power_mw"] = target_power
        carrier_proxy["cp30_target_energy_mwh"] = float(target["energy_mwh"])
        carrier_proxy["cp30_proxy_technologies"] = ",".join(proxy_technologies)
        if "site_name" in carrier_proxy.columns:
            carrier_proxy["site_name"] = [
                name if existing_row else f"CP30_{target['carrier']}_{idx}"
                for idx, (name, existing_row) in enumerate(zip(carrier_proxy["site_name"], is_existing))
            ]
        scaled_rows.append(carrier_proxy)

    if not scaled_rows:
        return pd.DataFrame()
    return pd.concat(scaled_rows, ignore_index=True)


def load_cp30_interconnector_capacity_mw(es1_path: str | Path, pathway: str, modelled_year: int) -> float:
    """Return CP30 aggregate interconnector capacity in MW."""
    df = _read_cp30_es1(es1_path)
    year_col = str(modelled_year)
    required = {"Pathway", "Variable", "Category", "SubType", year_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CP30 ES1 missing required columns: {sorted(missing)}")

    rows = df[
        (df["Pathway"] == pathway)
        & (df["Variable"] == "Capacity (MW)")
        & (df["Category"] == "Interconnectors")
        & (df["SubType"] == "Interconnectors")
    ].copy()
    if rows.empty:
        raise ValueError(f"No CP30 interconnector capacity found for {pathway} {modelled_year}")
    return float(pd.to_numeric(rows[year_col], errors="coerce").fillna(0.0).sum())


#: Columns of the CP30 interconnector composition file
#: (``data/interconnectors/cp30_interconnector_composition.csv``, 15 Oct 2026).
COMPOSITION_COLUMNS = ("name", "action", "cp30_mw", "first_year")
COMPOSITION_ACTIONS = ("keep", "add", "exclude")
#: Columns an ``add`` row must give so the link can be built like a mapped register cable.
COMPOSITION_ADD_COLUMNS = (
    "landing_point_gb", "gb_bus", "to_bus", "counterparty_country", "international_location",
    "international_latitude", "international_longitude", "gb_latitude", "gb_longitude", "losses_percent",
)


def load_cp30_interconnector_composition(path: str | Path) -> pd.DataFrame:
    """The declared CP30 cable list: each mapped cable kept at CP30's nominal MW, added, or excluded."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CP30 interconnector composition file not found: {path}")
    table = pd.read_csv(path, comment="#")
    missing = set(COMPOSITION_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    table["action"] = table["action"].astype(str).str.strip().str.lower()
    bad = sorted(set(table["action"]) - set(COMPOSITION_ACTIONS))
    if bad:
        raise ValueError(f"{path}: unknown action(s) {bad}; expected one of {COMPOSITION_ACTIONS}")
    if table["name"].duplicated().any():
        raise ValueError(f"{path} repeats a cable: {sorted(table.loc[table['name'].duplicated(), 'name'])}")
    adds = table[table["action"].eq("add")]
    if len(adds):
        missing = set(COMPOSITION_ADD_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f"{path}: an 'add' row needs columns {sorted(missing)}")
        empty = adds[list(COMPOSITION_ADD_COLUMNS)].isna().any(axis=1)
        if empty.any():
            raise ValueError(f"{path}: 'add' rows with blank fields: {sorted(adds.loc[empty, 'name'])}")
    return table


def compose_cp30_interconnectors(
    interconnectors_df: pd.DataFrame,
    target_capacity_mw: float,
    composition: pd.DataFrame,
    modelled_year: int,
    tolerance_mw: float = 1.0,
) -> pd.DataFrame:
    """Build CP30's cable set from a declared composition instead of a pro-rata uplift (15 Oct 2026).

    Every mapped cable must appear in ``composition``: ``keep`` sets its capacity to CP30's nominal
    ``cp30_mw`` (the register figure stays in ``capacity_mw_original``), ``exclude`` removes it
    (outside CP30's count, e.g. the Isle of Man cable), and ``add`` rows are cables CP30's ES1 step
    implies but the operational register feed does not carry yet (MaresConnect). A kept or added
    cable whose ``first_year`` is after ``modelled_year`` is left out. The counted total must equal
    the ES1 target within ``tolerance_mw``: nothing is scaled to close a gap.
    """
    if interconnectors_df.empty:
        return interconnectors_df
    comp = composition.set_index("name")
    unknown = sorted(set(interconnectors_df["name"]) - set(comp.index))
    if unknown:
        raise ValueError(
            f"Mapped interconnectors missing from the CP30 composition: {unknown}. Declare each as keep, add or exclude."
        )
    year = int(modelled_year)
    first_year = pd.to_numeric(comp["first_year"], errors="coerce").fillna(-np.inf)
    in_service = first_year <= year

    action = interconnectors_df["name"].map(comp["action"])
    kept = interconnectors_df[action.eq("keep") & interconnectors_df["name"].map(in_service).astype(bool)].copy()
    kept["capacity_mw_original"] = kept["capacity_mw"]
    kept["capacity_mw"] = kept["name"].map(pd.to_numeric(comp["cp30_mw"], errors="coerce")).astype(float)
    kept["cp30_composition_action"] = "keep"

    adds = comp[comp["action"].eq("add") & in_service].reset_index()
    added = pd.DataFrame()
    if len(adds):
        added = adds.reindex(columns=sorted(set(interconnectors_df.columns) | set(adds.columns)))
        added["capacity_mw"] = pd.to_numeric(adds["cp30_mw"], errors="coerce").astype(float)
        added["capacity_mw_original"] = added["capacity_mw"]
        added["from_bus"] = adds["gb_bus"]
        added["dc"] = True
        added["match_method"] = "cp30_composition"
        added["source"] = "CP30_composition"
        added["cp30_composition_action"] = "add"
        added = added.drop(columns=[c for c in ("action", "cp30_mw", "first_year") if c in added.columns])

    composed = pd.concat([frame for frame in (kept, added) if not frame.empty], ignore_index=True)
    total = float(composed["capacity_mw"].sum())
    if abs(total - float(target_capacity_mw)) > tolerance_mw:
        raise ValueError(
            f"CP30 interconnector composition gives {total:,.1f} MW in {year}, but CP30 ES1 has "
            f"{float(target_capacity_mw):,.1f} MW; update the composition file rather than scaling cables"
        )
    composed["data_source"] = "CP30"
    composed["cp30_target_mw"] = float(target_capacity_mw)
    return composed
