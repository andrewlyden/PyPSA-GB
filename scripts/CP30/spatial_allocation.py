"""CP30 ES5 regional spatial allocation helpers.

CP30 ES1 is the source for national capacity totals, while ES5 supplies the
published 2030 capacity split across eight broad GB regions.  ES5 does not
identify individual projects or connection nodes.  This module therefore keeps
the FES proxy sites (and their weather profiles) but rescales them so that their
capacity equals the CP30 ES5 total in each region.  Regions are assigned from a
small, versioned set of ETYS anchor substations; this is an explicit modelling
crosswalk, not a claim that CP30 publishes site-level locations.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ES5_TECHNOLOGY_TO_CARRIER = {
    "Offshore Wind": "wind_offshore",
    "Onshore Wind": "wind_onshore",
    "Solar": "solar_pv",
    "Hydro": "large_hydro",
    "Marine": "marine",
}


def _read_es5_targets(path, pathway, modelled_year):
    """Read CP30 ES5 capacity targets in the model's carrier names (``path``: the CSV or the table read)."""
    if isinstance(path, pd.DataFrame):
        data = path.copy()
    else:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"CP30 ES5 regional file not found: {path}")
        data = pd.read_csv(path)
    required = {"Scenario", "Technology", "Variable"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CP30 ES5 is missing required columns: {sorted(missing)}")
    selected = data[
        (data["Scenario"] == pathway)
        & (data["Variable"] == "Capacity (MW)")
        & data["Technology"].isin(ES5_TECHNOLOGY_TO_CARRIER)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"CP30 ES5 contains no capacity rows for pathway '{pathway}'"
        )
    region_columns = [
        col for col in selected.columns
        if col not in {"source_sheet", "Scenario", "Technology", "Variable"}
    ]
    long = selected.melt(
        id_vars=["Technology"], value_vars=region_columns,
        var_name="cp30_region", value_name="capacity_mw",
    )
    long["carrier"] = long["Technology"].map(ES5_TECHNOLOGY_TO_CARRIER)
    long["capacity_mw"] = pd.to_numeric(long["capacity_mw"], errors="coerce")
    long = long.dropna(subset=["capacity_mw"])
    long = long[long["capacity_mw"] >= 0.0]
    long["modelled_year"] = int(modelled_year)
    return long[["carrier", "cp30_region", "capacity_mw", "modelled_year"]]


def _read_region_anchors(path, network):
    """Resolve configured ETYS anchor buses and their CP30 region labels."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CP30 regional anchor file not found: {path}")
    anchors = pd.read_csv(path)
    required = {"cp30_region", "bus", "source"}
    missing = required - set(anchors.columns)
    if missing:
        raise ValueError(
            "CP30 regional anchor file is missing required columns: "
            f"{sorted(missing)}"
        )
    anchors = anchors.copy()
    anchors["bus"] = anchors["bus"].astype(str).str.strip()
    unknown = sorted(set(anchors["bus"]) - set(network.buses.index))
    if unknown:
        raise ValueError(
            "CP30 regional anchor file references ETYS buses absent from this "
            f"network: {unknown}"
        )
    if anchors.duplicated("cp30_region").any():
        raise ValueError("CP30 regional anchor file has duplicate CP30 regions")
    return anchors


def _nearest_anchor_regions(sites, anchors, network):
    """Assign each proxy site to its closest anchor in the network coordinate system."""
    site_buses = sites["bus"].astype(str)
    missing = sorted(set(site_buses) - set(network.buses.index))
    if missing:
        raise ValueError(
            "CP30 regional allocation has proxy buses absent from the network: "
            f"{missing}"
        )
    coords = network.buses[["x", "y"]]
    if coords.isna().any().any():
        raise ValueError("CP30 regional allocation requires finite ETYS x/y coordinates")
    anchor_xy = coords.loc[anchors["bus"]].to_numpy(dtype=float)
    anchor_regions = anchors["cp30_region"].to_numpy()
    site_xy = coords.loc[site_buses].to_numpy(dtype=float)
    distances = ((site_xy[:, None, :] - anchor_xy[None, :, :]) ** 2).sum(axis=2)
    return pd.Series(anchor_regions[distances.argmin(axis=1)], index=sites.index)


#: Region methods: the nearest of the configured anchors (the original rule), or the CP30 region of
#: the GSP polygon a site's bus falls in (8 Oct 2026). The anchor rule labelled the Humber landings
#: East England (Bicker Fen is nearer than Harker) and Suffolk/Essex South East (Barking is nearer
#: than Bicker Fen), so East England's ES5 total was filled by Humber plant and contracted East
#: Anglia projects (East Anglia THREE, AR4; East Anglia TWO, AR6) were trimmed out.
REGION_METHODS = ("nearest_anchor", "gsp_polygon")


#: Transmission sites counted in their transmission owner's ES5 region where the
#: GSP (distribution) polygon they sit in says otherwise, by four-character ETYS
#: site code. Cruachan (CRUA) and Dalmally 275 kV (DALL, where Cruachan II is
#: placed) are SP Transmission, south of B4, inside an SSEN Distribution
#: polygon; FES 2024 counts Cruachan in Direct(SPTL). Used by the storage
#: regional split and the CP30 locational validation (15 Oct 2026).
TRANSMISSION_SITE_REGIONS = {"CRUA": "South Scotland", "DALL": "South Scotland"}


#: The GSP group -> region map behind the one region rule (also :class:`LicenceAreas`' default).
DEFAULT_GROUP_REGION_FILE = "data/network/ETYS/cp30_es5_region_by_gsp_group.csv"
#: Transmission licence of a region: North Scotland is SHET (GSP group _P), South Scotland SPT (_N).
LICENCE_OF_REGION = {"North Scotland": "SHETL", "South Scotland": "SPTL"}


class LicenceAreas:
    """Transmission licence areas (SHETL, SPTL, NGET) of points and buses, by GSP polygon (27 Oct 2026).

    FES "Direct(SHETL/SPTL/NGET)" capacity was placed by latitude bands (57 and 55.5 degrees), which cross
    the licence borders: on the 2050 build 2.2 GW of SPT onshore wind sat in North Scotland, 1.9 GW of NGET
    onshore in South Scotland, and Cruachan (SPT) on the SHE side of B4. Here a point's licence is its GSP
    polygon's (:func:`regions_of_xy`) and its bus the nearest bus of the same licence
    (:func:`bus_cp30_regions`, with the transmission-site overrides: Cruachan and Dalmally are SPT), so the
    placed bus cannot cross the border either. The FES long-range chain's Direct renewables and storage.

    Args:
        buses: Network buses with ``x``/``y`` in OSGB36 metres.
    """

    def __init__(self, buses: pd.DataFrame, polygon_file=None, group_region_file=DEFAULT_GROUP_REGION_FILE):
        from pyproj import Transformer
        from scripts.utilities.gsp_nodes import DEFAULT_POLYGON_FILE

        self.polygons, self.group_region = load_gsp_region_map(polygon_file or DEFAULT_POLYGON_FILE,
                                                               group_region_file)
        xy = buses[["x", "y"]].astype(float)
        if float(xy["x"].max() - xy["x"].min()) <= 1000.0:
            raise ValueError("LicenceAreas needs bus x/y in OSGB36 metres")
        regions = bus_cp30_regions(xy, self.polygons, self.group_region)["cp30_region"].dropna()
        self.bus_xy = xy.loc[regions.index]
        self.bus_licence = regions.map(LICENCE_OF_REGION).fillna("NGET")
        self._to_osgb = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

    def site_licence(self, lat, lon) -> pd.Series:
        """Licence of WGS84 points: the GSP polygon each falls in, else the nearest (NaN without coordinates)."""
        lat = pd.to_numeric(pd.Series(lat), errors="coerce")
        lon = pd.to_numeric(pd.Series(lon, index=lat.index), errors="coerce")
        x, y = self._to_osgb.transform(lon.to_numpy(float), lat.to_numpy(float))
        region = regions_of_xy(pd.Series(x, index=lat.index), pd.Series(y, index=lat.index),
                               self.polygons, self.group_region)["cp30_region"]
        return region.map(LICENCE_OF_REGION).fillna("NGET").where(region.notna())

    def bus_latlon(self, bus: str) -> tuple[float, float]:
        """WGS84 (lat, lon) of a bus."""
        from pyproj import Transformer

        lon, lat = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True).transform(
            float(self.bus_xy.at[bus, "x"]), float(self.bus_xy.at[bus, "y"]))
        return float(lat), float(lon)

    def nearest_bus(self, lat: float, lon: float, licence: str):
        """The bus of ``licence`` nearest a WGS84 point (None if the licence has no bus)."""
        candidates = self.bus_xy[self.bus_licence.eq(licence)]
        if candidates.empty:
            return None
        x, y = self._to_osgb.transform(float(lon), float(lat))
        return str(((candidates["x"] - x) ** 2 + (candidates["y"] - y) ** 2).idxmin())


def load_gsp_region_map(polygon_file, group_region_file):
    """GSP polygons (EPSG:27700, licence area ``GSPGroup``) and the GSP group -> CP30 ES5 region map.

    The one reader of the two files behind every CP30 region in the build and the
    checks (``spatial_allocation``, ``storage_regions``, ``locational_validation``).
    NESO's unassigned (Off_NETS) polygons have no licence area and are dropped.
    """
    from scripts.utilities.gsp_nodes import load_gsp_polygons

    map_file = Path(group_region_file)
    if not map_file.is_file():
        raise FileNotFoundError(f"CP30 GSP group region file not found: {map_file}")
    polygons = load_gsp_polygons(polygon_file)
    group_region = pd.read_csv(map_file).set_index("gsp_group")["cp30_region"]
    unknown = sorted(set(polygons["GSPGroup"]) - set(group_region.index))
    if unknown:
        raise ValueError(f"{map_file} has no CP30 region for GSP groups {unknown}")
    return polygons, group_region


def regions_of_xy(x, y, polygons, group_region) -> pd.DataFrame:
    """CP30 ES5 region of points in OSGB36 metres: the GSP polygon each falls in, else the nearest.

    Returns a frame on the index of ``x`` with ``cp30_region``, ``gsp_group``
    and ``distance_m`` (0 inside a polygon). Points without coordinates get NaN.
    """
    import geopandas as gpd

    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y, index=x.index), errors="coerce")
    out = pd.DataFrame({"cp30_region": np.nan, "gsp_group": np.nan, "distance_m": np.nan},
                       index=x.index).astype({"cp30_region": object, "gsp_group": object})
    ok = x.notna() & y.notna()
    if not ok.any():
        return out
    points = gpd.GeoDataFrame({"_row": x.index[ok]}, geometry=gpd.points_from_xy(x[ok], y[ok]), crs="EPSG:27700")
    joined = gpd.sjoin_nearest(points, polygons[["GSPGroup", "geometry"]].to_crs("EPSG:27700"), how="left",
                               distance_col="distance_m")
    joined = joined.drop_duplicates("_row").set_index("_row")
    out.loc[joined.index, "gsp_group"] = joined["GSPGroup"].to_numpy()
    out.loc[joined.index, "cp30_region"] = joined["GSPGroup"].map(group_region).to_numpy()
    out.loc[joined.index, "distance_m"] = joined["distance_m"].to_numpy(dtype=float)
    return out


def bus_cp30_regions(buses: pd.DataFrame, polygons, group_region, overrides=None) -> pd.DataFrame:
    """CP30 ES5 region of each bus (OSGB36 ``x``/``y``): its GSP polygon, with transmission-site overrides.

    The one bus-region rule of the storage reconciliation and the locational checks (16 Oct 2026):
    ``regions_of_xy`` on the bus position, then ``overrides`` (bus-name prefix -> region; by default
    ``TRANSMISSION_SITE_REGIONS``) on buses that have a region. Returns the frame of
    ``regions_of_xy`` (``cp30_region``, ``gsp_group``, ``distance_m``) on ``buses``' index.
    """
    overrides = TRANSMISSION_SITE_REGIONS if overrides is None else overrides
    found = regions_of_xy(buses["x"], buses["y"], polygons, group_region)
    names = pd.Index(buses.index.astype(str))
    for prefix, region in (overrides or {}).items():
        hit = names.str.startswith(str(prefix)) & found["cp30_region"].notna().to_numpy()
        found.loc[hit, "cp30_region"] = region
    return found


def _gsp_polygon_regions(sites, network, config):
    """CP30 region of the GSP polygon each site's bus falls in (or the nearest polygon, for coastal buses).

    ``config['gsp_polygon_file']`` is NESO's GSP regions GeoJSON (property ``GSPGroup``, the DNO licence
    area) and ``config['gsp_group_region_file']`` maps each group to a CP30 ES5 region (``load_gsp_region_map``).
    """
    polygons, group_region = load_gsp_region_map(config.get("gsp_polygon_file", ""),
                                                 config.get("gsp_group_region_file", ""))
    site_buses = sites["bus"].astype(str)
    missing = sorted(set(site_buses) - set(network.buses.index))
    if missing:
        raise ValueError(f"CP30 regional allocation has buses absent from the network: {missing}")
    buses = network.buses.loc[sorted(set(site_buses)), ["x", "y"]].astype(float)
    if buses.isna().any().any():
        raise ValueError("CP30 regional allocation requires finite ETYS x/y coordinates")
    region = regions_of_xy(buses["x"], buses["y"], polygons, group_region)["cp30_region"]
    if region.isna().any():
        raise ValueError(f"CP30 regional allocation found no GSP region for buses {sorted(region.index[region.isna()])}")
    return site_buses.map(region)


def _site_regions(sites, network, config, anchors):
    method = str((config or {}).get("region_method", "nearest_anchor"))
    if method not in REGION_METHODS:
        raise ValueError(f"CP30 regional allocation region_method must be one of {REGION_METHODS}, got {method!r}")
    if method == "gsp_polygon":
        return _gsp_polygon_regions(sites, network, config)
    return _nearest_anchor_regions(sites, anchors, network)


def _make_anchor_proxy(result, carrier, region, target, anchors, network, modelled_year):
    """Create one transparent proxy where ES5 has no FES carrier/region row.

    ES5 is a regional capacity table, while FES is only a spatial/profile proxy.
    A positive ES5 regional total can therefore legitimately have no matching FES
    row. The fallback uses the nearest same-carrier FES proxy record but
    connects the capacity at the configured ETYS regional anchor. It is marked
    explicitly so it cannot be mistaken for a published project connection.
    """
    candidates = result[result["technology"].eq(carrier)]
    if candidates.empty:
        raise ValueError(
            f"CP30 ES5 requests {target:,.1f} MW of {carrier} in {region}, "
            "but the FES proxy has no capacity for that carrier anywhere."
        )
    anchor_bus = str(
        anchors.loc[anchors["cp30_region"].eq(region), "bus"].iloc[0]
    )
    candidate_xy = network.buses.loc[candidates["bus"], ["x", "y"]].to_numpy(dtype=float)
    anchor_xy = network.buses.loc[anchor_bus, ["x", "y"]].to_numpy(dtype=float)
    template = candidates.iloc[((candidate_xy - anchor_xy) ** 2).sum(axis=1).argmin()].copy()
    template["capacity_mw"] = float(target)
    template["bus"] = anchor_bus
    template["cp30_region"] = region
    template["site_name"] = (
        f"CP30_ES5_anchor_proxy_{carrier}_{region.replace(' ', '_')}_{modelled_year}"
    )
    template["spatial_proxy_source"] = (
        "CP30 ES5 regional total; nearest same-carrier FES proxy record; "
        f"allocated directly at ETYS regional anchor {anchor_bus}; no offshore export link"
    )
    template["cp30_spatial_method"] = "es5_anchor_proxy_missing_fes_region"
    template["cp30_anchor_proxy"] = True
    template["data_source"] = "CP30_ES5_regional_anchor_proxy"
    return template


def read_es5_targets(es5_path, pathway, modelled_year):
    """Public reader for the ES5 regional capacity table in model carrier names."""
    return _read_es5_targets(es5_path, pathway, modelled_year)


def assign_cp30_regions(sites, network, region_anchor_file, config=None):
    """Label each site with its CP30 region by the configured region method.

    Exposed so a named REPD fleet can be regionalised with exactly the rule the
    proxy uses; the two must agree or the regional reconciliation between them
    is meaningless. ``config`` is the regional_allocation block (``region_method``).
    """
    anchors = _read_region_anchors(region_anchor_file, network)
    result = sites.copy()
    result["cp30_region"] = _site_regions(result, network, config, anchors)
    return result


def apply_cp30_es5_regional_allocation(
    scaled, es5_path, pathway, modelled_year, network, config, logger, targets_override=None,
    region_buses=None,
):
    """Rescale FES proxy capacity to the published CP30 ES5 regional totals.

    Capacity is conserved at the CP30 ES5 carrier/region level.  The function
    deliberately leaves the number, coordinates, and profiles of FES proxy
    sites intact; a project-level CP30 connection register is not published.

    ``targets_override`` replaces the ES5 table with residual targets already
    net of a named fleet, so the proxy fills only what the named projects do
    not. Its columns are carrier, cp30_region, capacity_mw.

    ``region_buses`` (a Series on ``scaled``'s index, NaN where unused) labels
    a row's region by that bus instead of its own ``bus``: the offshore landing
    bus (``connection_allocation.landing_buses_for_regions``). Only the region
    label changes; the row's bus, coordinates and profile do not.
    """
    config = config or {}
    if not config.get("enabled", False):
        return scaled
    anchors = _read_region_anchors(config.get("region_anchor_file", ""), network)
    if targets_override is not None:
        targets = targets_override[["carrier", "cp30_region", "capacity_mw"]].copy()
        targets["modelled_year"] = int(modelled_year)
    else:
        targets = _read_es5_targets(es5_path, pathway, modelled_year)
    result = scaled.copy()
    labelled = result
    if region_buses is not None:
        region_buses = pd.Series(region_buses, dtype=object).reindex(result.index)
        use = region_buses.notna()
        if use.any():
            labelled = result.copy()
            labelled.loc[use, "bus"] = region_buses[use]
    result["cp30_region"] = _site_regions(labelled, network, config, anchors)
    if labelled is not result:
        source_regions = _site_regions(result, network, config, anchors)
        moved = result["cp30_region"].ne(source_regions)
        if moved.any():
            changes = (
                pd.DataFrame({
                    "carrier": result.loc[moved, "technology"], "source_bus": result.loc[moved, "bus"],
                    "landing_bus": labelled.loc[moved, "bus"], "source_region": source_regions[moved],
                    "landing_region": result.loc[moved, "cp30_region"],
                    "capacity_mw": pd.to_numeric(result.loc[moved, "capacity_mw"], errors="coerce"),
                })
                .groupby(["carrier", "source_bus", "landing_bus", "source_region", "landing_region"])
                ["capacity_mw"].sum()
            )
            for (carrier, source, landing, before, after), mw in changes.items():
                logger.info(
                    "CP30 region by landing bus: %s proxy at %s lands at %s; region %s -> %s "
                    "(%.1f MW before ES5 scaling)", carrier, source, landing, before, after, mw,
                )
    result["capacity_mw"] = pd.to_numeric(result["capacity_mw"], errors="coerce")
    if result["capacity_mw"].isna().any() or (result["capacity_mw"] < 0.0).any():
        raise ValueError("CP30 regional allocation received invalid proxy capacity")

    active_carriers = set(result["technology"])
    targets = targets[targets["carrier"].isin(active_carriers)].copy()
    missing_proxy_policy = str(config.get("missing_proxy_policy", "error")).lower()
    if missing_proxy_policy not in {"error", "anchor_proxy"}:
        raise ValueError(
            "CP30 regional allocation missing_proxy_policy must be 'error' or "
            f"'anchor_proxy'; got {missing_proxy_policy!r}"
        )

    # Fill only genuinely absent positive carrier/region combinations.  Scaling
    # remains unchanged for all regions already represented by FES proxy sites.
    fallback_rows = []
    for row in targets.itertuples(index=False):
        mask = result["technology"].eq(row.carrier) & result["cp30_region"].eq(row.cp30_region)
        if float(row.capacity_mw) > 0.0 and float(result.loc[mask, "capacity_mw"].sum()) <= 0.0:
            if missing_proxy_policy == "error":
                raise ValueError(
                    f"CP30 ES5 requests {row.capacity_mw:,.1f} MW of {row.carrier} in "
                    f"{row.cp30_region}, but the FES proxy has no capacity there."
                )
            fallback_rows.append(
                _make_anchor_proxy(
                    result, row.carrier, row.cp30_region, row.capacity_mw, anchors, network,
                    modelled_year,
                )
            )
    if fallback_rows:
        result = pd.concat([result, pd.DataFrame(fallback_rows)], ignore_index=True)
        for fallback in fallback_rows:
            logger.warning(
                "CP30 ES5 anchor proxy: %.1f MW of %s in %s has no FES regional "
                "proxy; allocated directly at ETYS anchor %s using the nearest "
                "same-carrier FES proxy record (no offshore export link).",
                fallback["capacity_mw"], fallback["technology"],
                fallback["cp30_region"], fallback["bus"],
            )
    for (carrier, region), target in targets.set_index(["carrier", "cp30_region"])["capacity_mw"].items():
        mask = result["technology"].eq(carrier) & result["cp30_region"].eq(region)
        target = float(target)
        if target == 0.0:
            result.loc[mask, "capacity_mw"] = 0.0
            continue
        proxy_total = float(result.loc[mask, "capacity_mw"].sum())
        if proxy_total <= 0.0:
            raise ValueError(
                f"CP30 ES5 requests {target:,.1f} MW of {carrier} in {region}, "
                "but the FES proxy has no capacity there."
            )
        result.loc[mask, "capacity_mw"] *= target / proxy_total

    # ES5 is authoritative for every regional total it supplies.  A carrier
    # may have proxy rows in a region with no ES5 column only if configuration
    # is inconsistent, so stop rather than silently retain capacity there.
    expected = set(targets[["carrier", "cp30_region"]].itertuples(index=False, name=None))
    observed = set(
        result.loc[
            result["technology"].isin(set(targets["carrier"])),
            ["technology", "cp30_region"],
        ].itertuples(index=False, name=None)
    )
    unrepresented = observed - expected
    if unrepresented:
        raise ValueError(
            "FES proxy rows were assigned to carrier/regions not represented in "
            f"CP30 ES5: {sorted(unrepresented)}"
        )

    result = result[result["capacity_mw"] > 0.0].copy()
    # Preserve the distinct provenance of a missing-region anchor fallback.
    # All ordinary rows use a common ES5/FES proxy description.
    anchor_proxy = result.get(
        "cp30_anchor_proxy", pd.Series(False, index=result.index)
    ).fillna(False).astype(bool)
    result.loc[~anchor_proxy, "spatial_proxy_source"] = (
        "CP30 ES5 regional capacity + FES site/profile proxy"
    )
    result.loc[~anchor_proxy, "cp30_spatial_method"] = (
        "nearest_configured_etys_region_anchor"
    )
    summary = result.groupby(["technology", "cp30_region"])["capacity_mw"].sum()
    for (carrier, region), capacity in summary.items():
        logger.info("CP30 ES5 allocation: %s / %s = %.1f MW", carrier, region, capacity)
    return result
