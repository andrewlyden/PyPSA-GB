"""CP30 ES5 regional allocation for storage (9 Oct 2026).

CP30's storage is scaled to its national ES1 totals and placed on FES GSP and Direct proxy rows, but the
regional split was never checked against CP30's ES5 table: the 2030 FFR fleet had 6.1 GW of batteries in North
England against ES5's 9.3, 2.4 GW in South Scotland against 4.7, and long-duration storage spread over every GB
storage site (``load_fes_storage_data`` distributes Direct LDES over all REPD storage). Storage behind the
Scottish and northern boundaries absorbs surplus wind, so the split matters for constraints.

:func:`apply_es5_storage_regions` rescales the new-build rows region by region to ES5's shares of the
national total, keeping existing sites (pumped hydro) and committed new-build projects fixed and every row's duration, so national power and
energy stay exact: batteries follow ES5 ``Batteries`` and the other storage carriers (LAES: CP30's LDES with
the existing pumped hydro) follow ES5 ``Storage`` less ``Batteries``. Regions are the CP30 region of the GSP
polygon each row's coordinates fall in (``spatial_allocation`` uses the same polygons and group map).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

#: ES5 technology rows for each storage carrier: (row, row subtracted).
ES5_STORAGE_ROWS = {"Battery": ("Batteries", None), "LAES": ("Storage", "Batteries")}


def es5_storage_targets(es5_path, pathway: str) -> pd.DataFrame:
    """ES5 regional storage power (MW) by model carrier: columns carrier, cp30_region, capacity_mw.

    ``es5_path`` is the ES5 CSV or the table already read.
    """
    es5 = es5_path.copy() if isinstance(es5_path, pd.DataFrame) else pd.read_csv(es5_path)
    es5 = es5[es5["Scenario"].eq(pathway) & es5["Variable"].eq("Capacity (MW)")].set_index("Technology")
    regions = [c for c in es5.columns if c not in {"source_sheet", "Scenario", "Variable"}]
    rows = []
    for carrier, (tech, less) in ES5_STORAGE_ROWS.items():
        if tech not in es5.index:
            raise ValueError(f"CP30 ES5 has no {tech!r} row for {pathway}")
        values = es5.loc[tech, regions].astype(float)
        if less is not None:
            values = (values - es5.loc[less, regions].astype(float)).clip(lower=0.0)
        rows += [{"carrier": carrier, "cp30_region": r, "capacity_mw": float(v)} for r, v in values.items()]
    return pd.DataFrame(rows)


def point_regions(lat, lon, polygon_file, group_region_file) -> pd.Series:
    """CP30 region of the GSP polygon each WGS84 point falls in (nearest polygon off the coast).

    The polygon join and group map are ``spatial_allocation``'s (``load_gsp_region_map``,
    ``regions_of_xy``), so storage and generation share one regional definition.
    """
    from pyproj import Transformer
    from scripts.CP30.spatial_allocation import load_gsp_region_map, regions_of_xy

    polygons, group_region = load_gsp_region_map(polygon_file, group_region_file)
    index = getattr(lat, "index", None)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True).transform(
        np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    region = regions_of_xy(pd.Series(x, index=index), pd.Series(y, index=index), polygons, group_region)
    if region["cp30_region"].isna().any():
        raise ValueError("storage regional allocation: a row has no coordinates or its GSP group has no CP30 region")
    return region["cp30_region"]


def apply_site_code_regions(regions: pd.Series, frame: pd.DataFrame, site_code_regions=None,
                            logger: logging.Logger | None = None) -> pd.Series:
    """Override the polygon region of rows at a named substation (``site_code``), 15 Oct 2026.

    A transmission-connected plant is counted in its transmission owner's ES5 region, which the
    distribution (GSP) polygon can contradict: Cruachan (CRUA) sits in an SSEN Distribution polygon
    but is SPT transmission, south of B4, so it is South Scotland (FES 2024 counts it in Direct(SPTL)).
    ``site_code_regions`` maps a four-character code to a CP30 region; by default
    ``spatial_allocation.TRANSMISSION_SITE_REGIONS``, the table the locational validation reads too.
    """
    if site_code_regions is None:
        from scripts.CP30.spatial_allocation import TRANSMISSION_SITE_REGIONS as site_code_regions
    mapping = {str(k).strip().upper(): str(v) for k, v in (site_code_regions or {}).items()}
    if not mapping or "site_code" not in frame.columns:
        return regions
    log = logger or logging.getLogger(__name__)
    out = regions.copy()
    codes = frame["site_code"].fillna("").astype(str).str.strip().str.upper().reindex(out.index)
    hit = codes.isin(list(mapping))
    if hit.any():
        changed = hit & (codes.map(mapping) != out)
        out[hit] = codes[hit].map(mapping)
        if changed.any():
            log.info("CP30 ES5 storage regions: site-code overrides " + "; ".join(
                f"{frame.at[i, 'site_name'] if 'site_name' in frame.columns else i} -> {out[i]}" for i in out.index[changed]))
    return out


def storage_bus_regions(storage_df: pd.DataFrame, buses: pd.DataFrame, polygon_file, group_region_file
                        ) -> pd.Series:
    """CP30 region of the bus each storage row sits on (``spatial_allocation.bus_cp30_regions``).

    The rule the renewables reconcile on (the region of the placed bus) and the locational check
    reads; transmission sites (Cruachan, Dalmally) take ``TRANSMISSION_SITE_REGIONS`` by bus name
    and by the row's ``site_code``.
    """
    from scripts.CP30.spatial_allocation import bus_cp30_regions, load_gsp_region_map

    polygons, group_region = load_gsp_region_map(polygon_file, group_region_file)
    used = sorted(set(storage_df["bus"].dropna().astype(str)))
    missing = sorted(set(used) - set(buses.index.astype(str)))
    if missing:
        raise ValueError(f"storage regional reconciliation: rows on buses absent from the network: {missing[:10]}")
    xy = buses.loc[used, ["x", "y"]].astype(float)
    by_bus = bus_cp30_regions(xy, polygons, group_region)["cp30_region"]
    region = storage_df["bus"].astype(str).map(by_bus)
    region = apply_site_code_regions(region, storage_df)
    if region.isna().any():
        raise ValueError("storage regional reconciliation: a storage bus has no CP30 region")
    return region


def reconcile_storage_regions_to_buses(storage_df: pd.DataFrame, network, targets: pd.DataFrame, polygon_file,
                                       group_region_file, logger: logging.Logger | None = None
                                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-reconcile the ES5 storage split on the region of the bus each row ends on (16 Oct 2026).

    :func:`apply_es5_storage_regions` labels rows by the GSP polygon of their coordinates, before
    the network exists. Placement then moves rows: a distribution-size battery goes to its GSP's
    demand node, and 16 GSPs' nodes sit in a neighbouring region's polygon (Kirkby, the node of the
    Kirkby/Washway Farm polygon in Lancashire, is in the Manweb polygon, so Wales; Walpole, the node
    of an East Midlands GSP, is in UKPN Eastern). The renewables reconcile ES5 on the region of the
    placed bus (``assign_cp30_regions`` after the GSP-node move, then ``reconcile_with_es5``) and the
    locational check judges by it, so storage does the same: rows whose bus region differs from
    their label are relabelled and the new-build rows rescaled again (fixed sites and national
    totals unchanged). A no-op when nothing crossed. Returns (storage_df, crossings).
    """
    log = logger or logging.getLogger(__name__)
    if storage_df.empty or "cp30_region" not in storage_df.columns:
        return storage_df, pd.DataFrame()
    region = storage_bus_regions(storage_df, network.buses, polygon_file, group_region_file)
    crossed = region.ne(storage_df["cp30_region"])
    columns = [c for c in ("site_name", "technology", "power_mw", "bus", "gsp_ids", "data_source") if c in storage_df]
    crossings = storage_df.loc[crossed, columns].assign(
        label_region=storage_df.loc[crossed, "cp30_region"], bus_region=region[crossed])
    if crossings.empty:
        return storage_df, crossings
    summary = crossings.groupby(["technology", "label_region", "bus_region"])["power_mw"].sum()
    log.info("CP30 ES5 storage regions by placed bus: %d rows crossed a region in placement (%s); "
             "re-reconciling on the bus region", len(crossings),
             "; ".join(f"{t} {a} -> {b} {mw:,.1f} MW" for (t, a, b), mw in summary.items()))
    for _, row in crossings.sort_values("power_mw", ascending=False).iterrows():
        log.info("  %s %s %.1f MW at %s (GSP %s): %s -> %s", row.get("technology"), row.get("site_name"),
                 row["power_mw"], row["bus"], row.get("gsp_ids"), row["label_region"], row["bus_region"])
    return apply_es5_storage_regions(storage_df, targets, region, log), crossings


def apply_es5_storage_regions(scaled: pd.DataFrame, targets: pd.DataFrame, regions: pd.Series,
                              logger: logging.Logger | None = None) -> pd.DataFrame:
    """Rescale new-build storage rows to ES5's regional shares; existing sites and national totals unchanged."""
    log = logger or logging.getLogger(__name__)
    from scripts.CP30.storage_interconnectors import fixed_site_mask

    out = scaled.copy()
    out["cp30_region"] = regions.reindex(out.index).to_numpy()
    # Existing plant and committed new build (named projects) are fixed; only proxy rows rescale.
    existing = fixed_site_mask(out)
    for carrier in sorted(set(out["technology"]) & set(targets["carrier"])):
        mask = out["technology"].eq(carrier)
        rows = out[mask]
        national = float(rows["power_mw"].sum())
        tgt = targets[targets["carrier"].eq(carrier)].set_index("cp30_region")["capacity_mw"]
        if tgt.sum() <= 0:
            continue
        tgt = tgt / tgt.sum() * national                      # ES5 shares of the CP30 national total
        fixed = rows[existing[mask]].groupby("cp30_region")["power_mw"].sum().reindex(tgt.index).fillna(0.0)
        new = rows[~existing[mask]]
        new_total = national - float(fixed.sum())
        want = (tgt - fixed).clip(lower=0.0)
        have = new.groupby("cp30_region")["power_mw"].sum().reindex(tgt.index).fillna(0.0)
        placeable = want.where(have > 0, 0.0)
        if placeable.sum() <= 0:
            raise ValueError(f"storage regional allocation: no new-build {carrier} rows to rescale")
        placeable = placeable / placeable.sum() * new_total  # regions without proxy rows pass their share on
        factor = (placeable / have.where(have > 0)).fillna(0.0)
        idx = new.index
        out.loc[idx, "power_mw"] = new["power_mw"] * new["cp30_region"].map(factor).to_numpy()
        out.loc[idx, "energy_mwh"] = out.loc[idx, "power_mw"] * pd.to_numeric(new["duration_h"], errors="coerce")
        result = out[mask].groupby("cp30_region")["power_mw"].sum().reindex(tgt.index).fillna(0.0)
        unplaced = sorted(want.index[(want > 1.0) & (have <= 0)])
        log.info(f"CP30 ES5 storage regions, {carrier} (MW, ES5 share of {national:,.0f} -> model): "
                 + "; ".join(f"{r} {tgt[r]:,.0f}->{result[r]:,.0f}" for r in tgt.index)
                 + (f"; regions with no proxy rows (share passed on): {unplaced}" if unplaced else ""))
        if not np.isclose(out.loc[mask, "power_mw"].sum(), national, rtol=1e-9):
            raise ValueError(f"storage regional allocation changed the national {carrier} power")
    # New-build rows in a region ES5 gives nothing are emptied: drop them rather than add zero-MW units.
    emptied = ~existing & out["power_mw"].le(0.0)
    if emptied.any():
        log.info(f"CP30 ES5 storage regions: dropped {int(emptied.sum())} new-build rows scaled to zero")
    return out[~emptied]
