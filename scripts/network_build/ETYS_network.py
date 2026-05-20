"""
Build the ETYS transmission network from preprocessed CSV data.

This script is the MODEL RULE counterpart to process_ETYS_data.py (DATA RULE).
It reads preprocessed CSV files (components + buses) and constructs the final
PyPSA network, including:
  - Coordinate guessing for buses without GSP matches
  - Land boundary validation (offshore buses stay at sea)
  - PyPSA network assembly (buses, lines, transformers, links)
  - ETYS network upgrades (if enabled)

Inputs (via snakemake.input):
  components: preprocessed components CSV
  buses: preprocessed buses CSV (with is_offshore column)
  etys_file: ETYS Appendix B Excel file (for upgrades only)

Outputs (via snakemake.output):
  network: PyPSA network as NetCDF (.nc)
"""

import pandas as pd
import numpy as np
import pypsa
import logging
import time
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Suppress PyPSA warnings about unoptimized networks (expected during network building)
warnings.filterwarnings('ignore', message='The network has not been optimized yet')

from scripts.utilities.logging_config import setup_logging, log_dataframe_info, log_network_info, log_execution_summary
from scripts.network_build.etys_file_registry import VOLTAGE_LEVELS, GSP_REGIONS_FILE

# Land boundary checking constants
LAND_BUFFER_KM = 0.5  # Buffer distance from coastline


def load_land_boundaries(logger: Optional[logging.Logger] = None) -> Optional[gpd.GeoDataFrame]:
    """
    Load Great Britain land boundaries from GSP regions file.

    Args:
        logger: Optional logger instance

    Returns:
        GeoDataFrame with GB land boundaries or None if loading fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        logger.info(f"Loading GB land boundaries from {GSP_REGIONS_FILE}")
        gdf = gpd.read_file(GSP_REGIONS_FILE)

        # Combine all GSP regions into a single land boundary using union_all()
        land_boundary = gdf.geometry.union_all()
        land_gdf = gpd.GeoDataFrame([1], geometry=[land_boundary], crs=gdf.crs)

        logger.info(f"Loaded land boundaries: {len(gdf)} GSP regions combined")
        logger.info(f"Land boundary CRS: {land_gdf.crs}")
        logger.info(f"Land boundary bounds: {land_gdf.total_bounds}")

        return land_gdf

    except Exception as e:
        logger.error(f"Failed to load land boundaries: {e}")
        logger.warning("Proceeding without land boundary checking")
        return None


def check_point_on_land(lat: float, lon: float, land_boundary: gpd.GeoDataFrame,
                       logger: Optional[logging.Logger] = None) -> bool:
    """
    Check if a coordinate point is on land (within GB boundaries).

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        land_boundary: GeoDataFrame with land boundaries
        logger: Optional logger instance

    Returns:
        True if point is on land, False if at sea
    """
    if land_boundary is None:
        return True  # Assume on land if no boundary data

    try:
        point = Point(lon, lat)
        return land_boundary.contains(point).any()
    except Exception as e:
        if logger:
            logger.debug(f"Error checking point ({lat}, {lon}) on land: {e}")
        return True  # Assume on land if check fails


def move_point_to_land(lat: float, lon: float, land_boundary: gpd.GeoDataFrame,
                      logger: Optional[logging.Logger] = None) -> Tuple[float, float]:
    """
    Move a point from sea to the nearest land location.

    Uses the nearest point on the land boundary and nudges slightly inland
    (perpendicular to the boundary toward the interior), avoiding the old
    approach of biasing toward the GB centroid which distorted Scottish locations.
    """
    if pd.isna(lat) or pd.isna(lon):
        if logger:
            logger.warning(f"Cannot move point with NaN coordinates: ({lat}, {lon})")
        return lat, lon

    if land_boundary is None:
        return lat, lon

    try:
        sea_point = Point(lon, lat)
        land_geom = land_boundary.geometry.iloc[0]

        # Find nearest point on land boundary
        _, nearest_point = nearest_points(sea_point, land_geom)

        # Nudge slightly inland: move from the sea point through the nearest
        # boundary point and a bit further (into land)
        buffer_deg = LAND_BUFFER_KM / 111.0
        dx = nearest_point.x - sea_point.x
        dy = nearest_point.y - sea_point.y
        length = np.sqrt(dx**2 + dy**2)

        if length > 0:
            # Continue in the same direction past the boundary point
            final_lon = nearest_point.x + (dx / length) * buffer_deg
            final_lat = nearest_point.y + (dy / length) * buffer_deg
        else:
            final_lon = nearest_point.x
            final_lat = nearest_point.y

        # Verify the nudged point is actually on land; if not, use boundary point
        if not land_geom.contains(Point(final_lon, final_lat)):
            final_lon = nearest_point.x
            final_lat = nearest_point.y

        if logger:
            distance_km = sea_point.distance(nearest_point) * 111.0
            logger.debug(f"Moved point from sea ({lat:.4f}, {lon:.4f}) to land ({final_lat:.4f}, {final_lon:.4f}), distance: {distance_km:.2f} km")

        return final_lat, final_lon

    except Exception as e:
        if logger:
            logger.warning(f"Failed to move point ({lat}, {lon}) to land: {e}")
        return lat, lon


def ensure_buses_on_land(network: pypsa.Network, land_boundary: Optional[gpd.GeoDataFrame],
                        logger: Optional[logging.Logger] = None,
                        skip_buses: Optional[set] = None) -> int:
    """
    Ensure all non-offshore buses in the network are located on land.

    Offshore wind farm buses (identified via skip_buses) are legitimately
    at sea and should NOT be moved to land.

    Args:
        network: PyPSA network with bus coordinates
        land_boundary: GeoDataFrame with land boundaries
        logger: Optional logger instance
        skip_buses: Set of bus IDs to skip (e.g. offshore wind farm buses)

    Returns:
        Number of buses moved from sea to land
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if land_boundary is None:
        logger.warning("No land boundary data available - skipping land validation")
        return 0

    if skip_buses is None:
        skip_buses = set()

    logger.info("Checking bus locations against land boundaries")
    if skip_buses:
        logger.info(f"Skipping {len(skip_buses)} offshore buses")
    buses_moved = 0
    buses_skipped = 0

    for bus_id, bus_data in network.buses.iterrows():
        lat, lon = bus_data.lat, bus_data.lon

        if not check_point_on_land(lat, lon, land_boundary, logger):
            # Check if this is an offshore bus that should stay at sea
            if bus_id in skip_buses:
                buses_skipped += 1
                logger.debug(f"Keeping offshore bus {bus_id} at sea ({lat:.4f}, {lon:.4f})")
                continue

            # Bus is at sea - move to land
            new_lat, new_lon = move_point_to_land(lat, lon, land_boundary, logger)
            network.buses.loc[bus_id, 'lat'] = new_lat
            network.buses.loc[bus_id, 'lon'] = new_lon
            buses_moved += 1

            logger.debug(f"Moved bus {bus_id} from sea ({lat:.4f}, {lon:.4f}) to land ({new_lat:.4f}, {new_lon:.4f})")

    if buses_moved > 0:
        logger.info(f"Moved {buses_moved} buses from sea to land")
    if buses_skipped > 0:
        logger.info(f"Kept {buses_skipped} offshore buses at sea")
    if buses_moved == 0 and buses_skipped == 0:
        logger.info("All buses were already on land")

    return buses_moved


def guess_GSP_location_of_remaining_buses(df: pd.DataFrame, df_buses: pd.DataFrame,
                                          offshore_wf_buses: set,
                                          logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Improved coordinate guessing using distance-weighted estimation and graph connectivity.
    Uses line lengths to better estimate bus positions and ensures all onshore buses are on land.
    Offshore wind farm buses (identified via is_offshore column) are kept at their estimated sea positions.

    Args:
        df: Components DataFrame
        df_buses: Buses DataFrame with partial coordinates
        offshore_wf_buses: Set of bus IDs classified as offshore
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if offshore_wf_buses:
        logger.info(f"Will preserve {len(offshore_wf_buses)} offshore buses at sea positions")

    # Load land boundaries for validation
    land_boundary = load_land_boundaries(logger)

    logger.info("Creating temporary network for location guessing")
    network = pypsa.Network()
    network.set_snapshots(range(1))

    network.add("Bus", df_buses.index, **df_buses)

    # Add lines with length information (drop metadata columns first to avoid PyPSA 1.0.2 warnings)
    # This temporary network is used ONLY for graph connectivity / coordinate guessing,
    # not for power flow — all components (including HVDC links) are added as AC Lines.
    # HVDC links have r=x=0 (correct for Link type), so assign minimum values here to
    # avoid spurious consistency_check warnings about zero-impedance lines.
    df_with_length = df.copy()
    df_with_length = df_with_length.drop(columns=[col for col in ['component', 'lon', 'lat', 'carrier'] if col in df_with_length.columns])
    df_with_length.loc[df_with_length['x'] == 0, 'x'] = 0.02   # HVDC entries: apply line default
    df_with_length.loc[df_with_length['r'] == 0, 'r'] = 0.002  # HVDC entries: apply line default
    network.add("Line", df_with_length.index, **df_with_length)
    network.add("Carrier", ["AC", "DC"])

    network.consistency_check()
    # if lon and lat are 0, set to NaN
    network.buses.loc[network.buses['lon'] == 0, 'lon'] = np.nan
    network.buses.loc[network.buses['lat'] == 0, 'lat'] = np.nan

    logger.info("Starting improved iterative coordinate guessing process")

    # Constants for coordinate estimation
    MAX_ITERATIONS = 50

    prev_count = len(network.buses) + 1
    iteration = 0

    while iteration < MAX_ITERATIONS:
        # Get buses with missing coordinates
        missing_coords = network.buses['lon'].isna() | network.buses['lat'].isna()
        curr_count = missing_coords.sum()

        # Check convergence
        if curr_count == 0 or prev_count == curr_count:
            break

        prev_count = curr_count
        iteration += 1

        logger.info(f"Iteration {iteration}: {curr_count} buses still need coordinates")

        # Strategy 1: Use distance-weighted positioning for buses with one known neighbor
        improved_count = 0

        for bus in network.buses[missing_coords].index:
            # Find all connected lines and their lengths
            connected_lines = network.lines[
                (network.lines['bus0'] == bus) | (network.lines['bus1'] == bus)
            ].copy()

            if connected_lines.empty:
                continue

            # Get connected buses with their distances
            connected_info = []
            for _, line in connected_lines.iterrows():
                other_bus = line['bus1'] if line['bus0'] == bus else line['bus0']
                other_coords = network.buses.loc[other_bus, ['lon', 'lat']]

                # Only use buses with known coordinates
                if not (pd.isna(other_coords['lon']) or pd.isna(other_coords['lat'])):
                    length_km = line.get('length_km', 0)
                    connected_info.append({
                        'bus': other_bus,
                        'lon': other_coords['lon'],
                        'lat': other_coords['lat'],
                        'length_km': max(length_km, 0.1)  # Minimum 100m to avoid division by zero
                    })

            if not connected_info:
                continue

            # Strategy 1a: If only one connected bus with known coordinates
            if len(connected_info) == 1:
                ref = connected_info[0]
                is_offshore = bus in offshore_wf_buses

                lat_deg_per_km = 1 / 111.0
                lon_deg_per_km = 1 / (111.0 * np.cos(np.radians(ref['lat'])))

                if is_offshore:
                    # Offshore bus: place at sea, away from land.
                    # Buses with preserved manual coordinates (from
                    # substation_coordinates.csv) won't reach here since
                    # they already have valid lon/lat and are excluded by
                    # the missing_coords filter above.
                    gb_centroid_lat, gb_centroid_lon = 54.5, -2.0

                    # Direction FROM centroid THROUGH the onshore bus, extending seaward
                    dir_lat = ref['lat'] - gb_centroid_lat
                    dir_lon = ref['lon'] - gb_centroid_lon
                    length = np.sqrt(dir_lat**2 + dir_lon**2)
                    if length > 0:
                        dir_lat /= length
                        dir_lon /= length
                    else:
                        dir_lat, dir_lon = 0, 1  # Default: eastward

                    candidate_lat = ref['lat'] + dir_lat * ref['length_km'] * lat_deg_per_km
                    candidate_lon = ref['lon'] + dir_lon * ref['length_km'] * lon_deg_per_km

                    network.buses.loc[bus, 'lon'] = candidate_lon
                    network.buses.loc[bus, 'lat'] = candidate_lat
                    improved_count += 1
                    logger.debug(f"Placed offshore bus {bus} at ({candidate_lat:.4f}, {candidate_lon:.4f}), "
                                f"{ref['length_km']:.1f} km from {ref['bus']}")
                elif ref['length_km'] < 2.0:
                    # Short onshore connection (transformer or very short line):
                    # place at same location with tiny jitter to avoid overlap
                    jitter = 0.001  # ~100m
                    network.buses.loc[bus, 'lon'] = ref['lon'] + jitter
                    network.buses.loc[bus, 'lat'] = ref['lat'] + jitter
                    improved_count += 1
                else:
                    # Onshore bus with longer connection: try 8 directions,
                    # score each by whether it's on land
                    candidates = []
                    for attempt in range(8):
                        angle = attempt * np.pi / 4

                        dx = ref['length_km'] * np.cos(angle) * lon_deg_per_km
                        dy = ref['length_km'] * np.sin(angle) * lat_deg_per_km

                        candidate_lon = ref['lon'] + dx
                        candidate_lat = ref['lat'] + dy

                        on_land = False
                        dist_to_land = float('inf')
                        if land_boundary is not None:
                            on_land = check_point_on_land(candidate_lat, candidate_lon, land_boundary, logger)
                            if not on_land:
                                try:
                                    point = Point(candidate_lon, candidate_lat)
                                    dist_to_land = point.distance(land_boundary.geometry.iloc[0])
                                except Exception:
                                    pass
                        candidates.append((candidate_lat, candidate_lon, on_land, dist_to_land))

                    # Prefer on-land candidates; among those, pick the first
                    on_land_candidates = [c for c in candidates if c[2]]
                    if on_land_candidates:
                        candidate_lat, candidate_lon = on_land_candidates[0][0], on_land_candidates[0][1]
                    elif candidates:
                        # Pick closest to land
                        best = min(candidates, key=lambda c: c[3])
                        candidate_lat, candidate_lon = best[0], best[1]
                        if land_boundary is not None:
                            candidate_lat, candidate_lon = move_point_to_land(candidate_lat, candidate_lon, land_boundary, logger)
                    else:
                        # Fallback: place at neighbor
                        candidate_lat, candidate_lon = ref['lat'], ref['lon']

                    network.buses.loc[bus, 'lon'] = candidate_lon
                    network.buses.loc[bus, 'lat'] = candidate_lat
                    improved_count += 1

            # Strategy 1b: Multiple connected buses - use distance-weighted centroid
            elif len(connected_info) >= 2:
                # Calculate distance-weighted position
                weights = []
                lon_coords = []
                lat_coords = []

                for info in connected_info:
                    # Weight inversely proportional to distance (closer buses have more influence)
                    weight = 1.0 / info['length_km']
                    weights.append(weight)
                    lon_coords.append(info['lon'])
                    lat_coords.append(info['lat'])

                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize weights

                estimated_lon = np.average(lon_coords, weights=weights)
                estimated_lat = np.average(lat_coords, weights=weights)

                # Validate position is on land (skip for offshore buses)
                if bus not in offshore_wf_buses:
                    if land_boundary is not None and not check_point_on_land(estimated_lat, estimated_lon, land_boundary, logger):
                        estimated_lat, estimated_lon = move_point_to_land(estimated_lat, estimated_lon, land_boundary, logger)

                network.buses.loc[bus, 'lon'] = estimated_lon
                network.buses.loc[bus, 'lat'] = estimated_lat
                improved_count += 1

        logger.debug(f"Improved coordinates for {improved_count} buses in iteration {iteration}")

        # Strategy 2: Simple averaging for remaining buses (fallback)
        remaining_missing = network.buses['lon'].isna() | network.buses['lat'].isna()
        for bus in network.buses[remaining_missing].index:
            connected_lines = network.lines[
                (network.lines['bus0'] == bus) | (network.lines['bus1'] == bus)
            ]

            if connected_lines.empty:
                continue

            # Get connected buses
            connected_buses_list = []
            for _, line in connected_lines.iterrows():
                other_bus = line['bus1'] if line['bus0'] == bus else line['bus0']
                connected_buses_list.append(other_bus)

            connected_buses = network.buses.loc[connected_buses_list]
            connected_buses = connected_buses[connected_buses.index != bus]

            # Only use buses with known coordinates
            known_connected = connected_buses.dropna(subset=['lon', 'lat'])
            if not known_connected.empty:
                # Simple average of connected bus coordinates
                avg_coords = known_connected[['lon', 'lat']].mean()
                estimated_lon, estimated_lat = avg_coords['lon'], avg_coords['lat']

                # Validate position is on land (skip for offshore buses)
                if bus not in offshore_wf_buses:
                    if land_boundary is not None and not check_point_on_land(estimated_lat, estimated_lon, land_boundary, logger):
                        estimated_lat, estimated_lon = move_point_to_land(estimated_lat, estimated_lon, land_boundary, logger)

                network.buses.loc[bus, ['lon', 'lat']] = [estimated_lon, estimated_lat]

    # Final land boundary check for all buses (skip offshore WF buses)
    if land_boundary is not None:
        logger.info("Performing final land boundary validation for all buses")
        buses_moved = ensure_buses_on_land(network, land_boundary, logger, skip_buses=offshore_wf_buses)
        if buses_moved > 0:
            logger.info(f"Final land validation moved {buses_moved} additional buses to land")

    # Log final results
    final_missing = (network.buses['lon'].isna() | network.buses['lat'].isna()).sum()
    logger.info(f"Coordinate guessing complete after {iteration} iterations. {final_missing} buses still without coordinates")

    if final_missing > 0:
        missing_buses = network.buses[network.buses['lon'].isna() | network.buses['lat'].isna()]
        logger.warning(f"Buses without coordinates: {missing_buses.index.tolist()}")

    # Remove buses with NaN coordinates
    initial_count = len(network.buses)
    df_buses_with_locs = network.buses.dropna(subset=['lon', 'lat'])
    dropped_count = initial_count - len(df_buses_with_locs)
    if dropped_count > 0:
        dropped = network.buses[network.buses['lon'].isna() | network.buses['lat'].isna()].index.tolist()

        # Connectivity impact assessment: warn loudly about intermediate buses
        for bus in dropped:
            connected_lines = df[(df['bus0'] == bus) | (df['bus1'] == bus)]
            n_connections = len(connected_lines)
            if n_connections >= 2:
                # This bus connects ≥2 lines/transformers — dropping it may disconnect the network
                peer_buses = set(connected_lines['bus0'].tolist() + connected_lines['bus1'].tolist()) - {bus}
                logger.error(
                    f"Dropping INTERMEDIATE bus '{bus}' with {n_connections} connections "
                    f"(peers: {list(peer_buses)[:6]}) — may disconnect subnetworks!"
                )
            elif n_connections == 1:
                logger.warning(f"Dropping leaf bus '{bus}' with 1 connection")
            else:
                logger.warning(f"Dropping isolated bus '{bus}' with no connections")

        logger.warning(f"DROPPED {dropped_count} buses due to missing coordinates: {dropped}")
    logger.info(f"Final bus count after removing NaN coordinates: {len(df_buses_with_locs)} "
                f"(dropped {dropped_count} of {initial_count})")

    return df_buses_with_locs


def validate_network_topology(network: pypsa.Network,
                               logger: Optional[logging.Logger] = None) -> None:
    """
    Post-build validation checks on network topology.

    Runs warning-only checks (does not block execution):
    - Connectivity: detect disconnected subnetworks
    - Impedance ratios: X/R should typically be 5-20 for transmission
    - Low-connectivity: warn about leaf buses (single connection)
    - Self-loops, zero impedance, missing v_nom
    - Coordinate completeness and bounds
    - Zero/negative ratings, excessive parallel circuits
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Running network topology validation")

    # 1. Connectivity check using networkx
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(network.buses.index)
        for _, line in network.lines.iterrows():
            G.add_edge(line.bus0, line.bus1)
        for _, xfmr in network.transformers.iterrows():
            G.add_edge(xfmr.bus0, xfmr.bus1)
        for _, link in network.links.iterrows():
            G.add_edge(link.bus0, link.bus1)

        components = list(nx.connected_components(G))
        if len(components) == 1:
            logger.info(f"  Connectivity: network is fully connected ({len(G.nodes)} nodes)")
        else:
            logger.warning(f"  Connectivity: network has {len(components)} disconnected subnetworks!")
            for i, comp in enumerate(sorted(components, key=len, reverse=True)):
                if i == 0:
                    logger.info(f"    Main network: {len(comp)} buses")
                elif len(comp) <= 10:
                    logger.warning(f"    Island {i}: {len(comp)} buses: {sorted(comp)}")
                else:
                    logger.warning(f"    Island {i}: {len(comp)} buses")
    except ImportError:
        logger.warning("  networkx not available - skipping connectivity check")

    # 2. Impedance ratio check (X/R) for lines
    if len(network.lines) > 0:
        r_nonzero = network.lines['r'].replace(0, np.nan)
        xr_ratios = network.lines['x'] / r_nonzero
        valid_xr = xr_ratios.dropna()
        low_xr = valid_xr[valid_xr < 2]
        high_xr = valid_xr[valid_xr > 50]
        if len(low_xr) > 0 or len(high_xr) > 0:
            logger.warning(f"  X/R ratios: {len(low_xr)} lines with X/R < 2, "
                          f"{len(high_xr)} lines with X/R > 50")
        else:
            logger.info(f"  X/R ratios: all {len(valid_xr)} lines within normal range (2-50)")

    # 3. Low-connectivity warning (leaf buses)
    if len(network.lines) > 0 or len(network.transformers) > 0:
        bus_connections = pd.Series(0, index=network.buses.index)
        for comp_df in [network.lines, network.transformers]:
            if len(comp_df) > 0:
                bus_connections = bus_connections.add(
                    comp_df['bus0'].value_counts(), fill_value=0)
                bus_connections = bus_connections.add(
                    comp_df['bus1'].value_counts(), fill_value=0)
        if len(network.links) > 0:
            bus_connections = bus_connections.add(
                network.links['bus0'].value_counts(), fill_value=0)
            bus_connections = bus_connections.add(
                network.links['bus1'].value_counts(), fill_value=0)

        leaf_buses = bus_connections[bus_connections == 1]
        if len(leaf_buses) > 0:
            logger.info(f"  Leaf buses (single connection): {len(leaf_buses)}")

    # 4. Summary stats
    logger.info(f"  Network summary: {len(network.buses)} buses, "
               f"{len(network.lines)} lines, "
               f"{len(network.transformers)} transformers, "
               f"{len(network.links)} links")

    # 5. Self-loops (bus0 == bus1)
    try:
        for comp_name, comp_df in [('lines', network.lines),
                                    ('transformers', network.transformers),
                                    ('links', network.links)]:
            if len(comp_df) > 0:
                self_loops = comp_df[comp_df['bus0'] == comp_df['bus1']]
                if len(self_loops) > 0:
                    logger.warning(f"  Self-loops: {len(self_loops)} {comp_name} "
                                   f"where bus0 == bus1: {self_loops.index.tolist()[:10]}")
    except Exception as e:
        logger.debug(f"  Self-loop check failed: {e}")

    # 6. Zero impedance (r==0 AND x==0 for lines/transformers)
    try:
        for comp_name, comp_df in [('lines', network.lines),
                                    ('transformers', network.transformers)]:
            if len(comp_df) > 0 and 'r' in comp_df.columns and 'x' in comp_df.columns:
                zero_z = comp_df[(comp_df['r'] == 0) & (comp_df['x'] == 0)]
                if len(zero_z) > 0:
                    logger.warning(f"  Zero impedance: {len(zero_z)} {comp_name} "
                                   f"with r=0 AND x=0: {zero_z.index.tolist()[:10]}")
    except Exception as e:
        logger.debug(f"  Zero impedance check failed: {e}")

    # 7. Missing v_nom
    try:
        missing_vnom = network.buses['v_nom'].isna()
        if missing_vnom.any():
            logger.warning(f"  Missing v_nom: {missing_vnom.sum()} buses: "
                           f"{network.buses[missing_vnom].index.tolist()[:10]}")
    except Exception as e:
        logger.debug(f"  v_nom check failed: {e}")

    # 8. Coordinate completeness
    try:
        missing_coords = network.buses['x'].isna() | network.buses['y'].isna()
        if missing_coords.any():
            logger.warning(f"  Missing coordinates: {missing_coords.sum()} buses: "
                           f"{network.buses[missing_coords].index.tolist()[:10]}")
    except Exception as e:
        logger.debug(f"  Coordinate completeness check failed: {e}")

    # 9. Coordinate bounds (OSGB36: x ~ -200000 to 800000, y ~ 0 to 1300000)
    try:
        if 'x' in network.buses.columns and 'y' in network.buses.columns:
            valid = network.buses[network.buses['x'].notna() & network.buses['y'].notna()]
            out_of_bounds = valid[
                (valid['x'] < -200000) | (valid['x'] > 800000) |
                (valid['y'] < 0) | (valid['y'] > 1300000)
            ]
            if len(out_of_bounds) > 0:
                logger.warning(f"  Coordinate bounds: {len(out_of_bounds)} buses "
                               f"outside GB OSGB36 bounds: {out_of_bounds.index.tolist()[:10]}")
            else:
                logger.info(f"  Coordinate bounds: all {len(valid)} buses within GB OSGB36 bounds")
    except Exception as e:
        logger.debug(f"  Coordinate bounds check failed: {e}")

    # 10. Zero or negative ratings
    try:
        for comp_name, comp_df, col in [('lines', network.lines, 's_nom'),
                                         ('transformers', network.transformers, 's_nom'),
                                         ('links', network.links, 'p_nom')]:
            if len(comp_df) > 0 and col in comp_df.columns:
                bad_rating = comp_df[comp_df[col] <= 0]
                if len(bad_rating) > 0:
                    logger.warning(f"  Zero/negative {col}: {len(bad_rating)} {comp_name}: "
                                   f"{bad_rating.index.tolist()[:10]}")
    except Exception as e:
        logger.debug(f"  Rating check failed: {e}")

    # 11. Duplicate components (>4 parallel circuits between same bus pair)
    try:
        if len(network.lines) > 0:
            bus_pairs = network.lines.apply(
                lambda r: tuple(sorted([r['bus0'], r['bus1']])), axis=1)
            pair_counts = bus_pairs.value_counts()
            excessive = pair_counts[pair_counts > 4]
            if len(excessive) > 0:
                logger.warning(f"  Excessive parallel circuits: {len(excessive)} bus pairs "
                               f"with >4 lines: {excessive.head(5).to_dict()}")
    except Exception as e:
        logger.debug(f"  Duplicate component check failed: {e}")


def create_network(df: pd.DataFrame,
                   df_buses_with_locs: pd.DataFrame,
                   logger: Optional[logging.Logger] = None,
                   export_path: Optional[str] = None,
                   etys_upgrades_enabled: Optional[bool] = None,
                   upgrade_year: Optional[int] = None,
                   etys_upgrade_file: Optional[str] = None,
                   substation_coords_file: Optional[str] = None) -> pypsa.Network:
    """
    Create the final PyPSA network with all components and export to NetCDF.

    Args:
        df: DataFrame with all network components (lines, transformers, links)
        df_buses_with_locs: DataFrame with buses that have coordinates
        logger: Optional logger instance
        export_path: Optional NetCDF output path. If omitted, falls back to
            snakemake.output[0] when running under Snakemake.
        etys_upgrades_enabled: Whether to apply ETYS upgrades. If omitted,
            falls back to snakemake.params.etys_upgrades_enabled.
        upgrade_year: Year through which to apply upgrades. If omitted,
            falls back to snakemake.params.etys_upgrade_year or modelled_year.
        etys_upgrade_file: Path to the ETYS Appendix B file used for upgrades.
            If omitted, falls back to snakemake.input.etys_file.
        substation_coords_file: Optional path to substation coordinates used
            for upgrade bus placement. If omitted, falls back to
            snakemake.input.substation_coords.

    Returns:
        Complete PyPSA Network object
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Creating final PyPSA network")
    # new network with updated locations of buses
    network2 = pypsa.Network()
    network2.set_snapshots(range(1))

    logger.info(f"Adding {len(df_buses_with_locs)} buses to network")
    network2.add("Bus", df_buses_with_locs.index, **df_buses_with_locs)

    # CRITICAL: Convert WGS84 (lon/lat) to OSGB36 (x/y) for PyPSA plotting
    # PyPSA expects x/y in projected coordinates (meters), not lat/lon (degrees)
    # See: https://docs.pypsa.org/latest/user-guide/plotting/static-map/
    if 'lon' in network2.buses.columns and 'lat' in network2.buses.columns:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)  # WGS84 -> OSGB36
        network2.buses['x'], network2.buses['y'] = transformer.transform(
            network2.buses['lon'].values,
            network2.buses['lat'].values
        )
        logger.info(f"Converted coordinates: lon/lat (WGS84) -> x/y (OSGB36)")
        logger.info(f"  x range: [{network2.buses['x'].min():.0f}, {network2.buses['x'].max():.0f}] meters")
        logger.info(f"  y range: [{network2.buses['y'].min():.0f}, {network2.buses['y'].max():.0f}] meters")

    # filter df with component of line
    df_line = df[df['component'] == 'line']
    # remove lines if bus0 or bus1 is not in buses
    df_line = df_line[df_line['bus0'].isin(df_buses_with_locs.index) & df_line['bus1'].isin(df_buses_with_locs.index)]
    # Drop metadata columns (geolocation only) - keep electrical attributes 'r', 'x', 'b'
    df_line = df_line.drop(columns=[col for col in ['component', 'lon', 'lat', 'carrier'] if col in df_line.columns])
    logger.info(f"Adding {len(df_line)} lines to network")
    network2.add("Line", df_line.index, **df_line)
    # set DC to AC
    network2.lines['carrier'] = 'AC'

    # filter df with component of transformer
    df_transformer = df[df['component'] == 'transformer']
    # remove transformers if bus0 or bus1 is not in buses
    df_transformer = df_transformer[df_transformer['bus0'].isin(df_buses_with_locs.index) & df_transformer['bus1'].isin(df_buses_with_locs.index)]
    # Drop metadata columns (geolocation only) - keep electrical attributes 'r', 'x', 'b'
    df_transformer = df_transformer.drop(columns=[col for col in ['component', 'lon', 'lat', 'carrier'] if col in df_transformer.columns])
    # Use pi-circuit model for transformers since ETYS data provides r, x, b impedance values.
    # This must match the model used for upgrade transformers (ETYS_upgrades.py).
    df_transformer['model'] = 'pi'
    logger.info(f"Adding {len(df_transformer)} transformers to network (model='pi')")
    network2.add("Transformer", df_transformer.index, **df_transformer)

    # filter df with component of link using loc
    df_link = df.loc[df['component'] == 'link']
    # rename s_nom to p_nom
    df_link = df_link.rename(columns={'s_nom': 'p_nom'})
    # Drop metadata columns that aren't PyPSA Link attributes (avoid PyPSA 1.0.2 warnings)
    df_link = df_link.drop(columns=[col for col in ['component', 'x', 'y', 'b', 'r'] if col in df_link.columns])

    # CRITICAL: Make internal HVDC links bidirectional (p_min_pu=-1)
    # Without this, links can only transfer power FROM bus0 TO bus1, which causes
    # infeasibility when power needs to flow the other direction
    df_link['p_min_pu'] = -1.0  # Bidirectional
    df_link['p_max_pu'] = 1.0

    logger.info(f"Adding {len(df_link)} links to network (bidirectional HVDC)")
    network2.add("Link", df_link.index, **df_link)

    # HVDC links should have DC carrier
    network2.links.loc[:, 'carrier'] = 'DC'

    logger.info("Setting network metadata and performing final setup")
    network2.buses['country'] = 'GB'
    # HUCS4- (Hunterston converter) coordinates now handled by substation_coordinates.csv
    network2.add("Carrier", ["AC", "DC"])
    # name network ETYS base
    network2.name = 'ETYS base'
    network2.consistency_check()

    # Run topology validation checks (warning-only, does not block)
    validate_network_topology(network2, logger)

    # ──────────────────────────────────────────────────────────────────────────
    # APPLY ETYS NETWORK UPGRADES (if enabled)
    # ──────────────────────────────────────────────────────────────────────────
    if etys_upgrades_enabled is None:
        if 'snakemake' in globals():
            etys_upgrades_enabled = getattr(snakemake.params, 'etys_upgrades_enabled', False)
        else:
            etys_upgrades_enabled = False

    # NOTE: Land boundary validation ran during coordinate guessing (above).
    # Upgrade buses get coordinates from OSGB36 (x/y) via same-site matching,
    # NOT from lat/lon guessing, so they bypass land boundary checks.
    # If this order changes, ensure offshore upgrade buses are in the skip set.
    if etys_upgrades_enabled:
        from scripts.network_build.ETYS_upgrades import apply_etys_network_upgrades

        if upgrade_year is None:
            if 'snakemake' in globals():
                modelled_year = getattr(snakemake.params, 'modelled_year', 2020)
                etys_upgrade_year = getattr(snakemake.params, 'etys_upgrade_year', None)
                upgrade_year = etys_upgrade_year if etys_upgrade_year else modelled_year
            else:
                raise ValueError("upgrade_year must be provided when using create_network() outside Snakemake with upgrades enabled")

        if etys_upgrade_file is None:
            if 'snakemake' in globals():
                etys_upgrade_file = str(snakemake.input.etys_file)
            else:
                raise ValueError("etys_upgrade_file must be provided when using create_network() outside Snakemake with upgrades enabled")

        if substation_coords_file is None and 'snakemake' in globals():
            substation_coords_file = getattr(snakemake.input, 'substation_coords', None)
            if substation_coords_file:
                substation_coords_file = str(substation_coords_file)

        logger.info("="*70)
        logger.info(f"APPLYING ETYS NETWORK UPGRADES through year {upgrade_year}")
        logger.info("="*70)

        network2 = apply_etys_network_upgrades(
            network2,
            modelled_year=upgrade_year,
            etys_file=etys_upgrade_file,
            substation_coords_file=substation_coords_file,
            logger=logger
        )

        # Ensure all buses (including newly added ones) have country='GB'
        network2.buses['country'] = network2.buses['country'].fillna('GB')
        network2.buses['country'] = network2.buses['country'].replace('', 'GB')

        # Back-convert x/y → lat/lon for any new buses added during upgrades.
        # Upgrade buses only have OSGB36 (x/y) but downstream code may need WGS84 (lat/lon).
        missing_latlon = network2.buses['lat'].isna() & network2.buses['x'].notna()
        if missing_latlon.any():
            from pyproj import Transformer as ProjTransformer
            osgb_to_wgs = ProjTransformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
            new_lon, new_lat = osgb_to_wgs.transform(
                network2.buses.loc[missing_latlon, 'x'].values,
                network2.buses.loc[missing_latlon, 'y'].values
            )
            network2.buses.loc[missing_latlon, 'lon'] = new_lon
            network2.buses.loc[missing_latlon, 'lat'] = new_lat
            logger.info(f"Back-converted {missing_latlon.sum()} upgrade bus coordinates from OSGB36 to WGS84")

        # Re-run consistency check after upgrades
        network2.consistency_check()
        network2.name = f'ETYS base + upgrades ({upgrade_year})'

        logger.info(f"Network upgrades applied successfully (through {upgrade_year})")
    else:
        logger.info("ETYS network upgrades: DISABLED (set etys_upgrades.enabled: true to enable)")

    if export_path is None and 'snakemake' in globals():
        export_path = str(snakemake.output[0])

    if export_path is not None:
        logger.info(f"Exporting network to {export_path}")

        # Set version metadata before export to prevent compatibility warnings
        network2.meta = {"pypsa_version": pypsa.__version__}

        # Suppress PyPSA warnings about unoptimized network during export
        pypsa_logger = logging.getLogger('pypsa.networks')
        original_level = pypsa_logger.level
        pypsa_logger.setLevel(logging.ERROR)
        try:
            network2.export_to_netcdf(export_path)
        finally:
            pypsa_logger.setLevel(original_level)
    else:
        logger.info("No export path provided; returning network without NetCDF export")

    log_network_info(network2, logger)
    logger.info("Network creation completed successfully")

    return network2


if __name__ == "__main__":
    # Initialize timing
    start_time = time.time()

    # Set up logging using centralized system, writing to Snakemake log if available
    log_path = None
    if 'snakemake' in globals() and hasattr(snakemake, 'log') and snakemake.log:
        log_path = snakemake.log[0]
    logger = setup_logging(log_path or "ETYS_network")

    logger.info("="*50)
    logger.info("STARTING ETYS NETWORK CREATION")

    try:
        # Read preprocessed CSV files from process_ETYS_data rule
        logger.info("Step 1: Loading preprocessed components data")
        df = pd.read_csv(str(snakemake.input.components), index_col=0)
        logger.info(f"Loaded {len(df)} components")

        logger.info("Step 2: Loading preprocessed buses data")
        df_buses = pd.read_csv(str(snakemake.input.buses), index_col=0)

        # Extract offshore bus set from is_offshore column
        offshore_wf_buses = set(df_buses[df_buses['is_offshore'] == True].index)
        logger.info(f"Loaded {len(df_buses)} buses ({len(offshore_wf_buses)} offshore)")

        # Drop the is_offshore column before passing to network construction
        # (PyPSA doesn't have a built-in is_offshore attribute; we re-add it after construction)
        df_buses = df_buses.drop(columns=['is_offshore'])

        logger.info("Step 3: Guessing remaining bus locations")
        df_buses = guess_GSP_location_of_remaining_buses(df, df_buses, offshore_wf_buses, logger)

        logger.info("Step 4: Creating final network")
        network = create_network(df, df_buses, logger)

        # Propagate is_offshore flag to the final network for downstream scripts
        # (storage, renewables) that need to know which buses are offshore
        network.buses['is_offshore'] = False
        surviving_offshore = offshore_wf_buses & set(network.buses.index)
        if surviving_offshore:
            network.buses.loc[list(surviving_offshore), 'is_offshore'] = True
            logger.info(f"Propagated is_offshore flag for {len(surviving_offshore)} buses to final network")

        logger.info("ETYS NETWORK CREATION COMPLETED SUCCESSFULLY")

        # Log execution summary
        inputs = [snakemake.input[i] for i in range(len(snakemake.input))] if 'snakemake' in globals() else []
        outputs = [snakemake.output[0]] if 'snakemake' in globals() else []
        log_execution_summary(logger, "ETYS Network Creation", start_time, inputs=inputs, outputs=outputs)

    except Exception as e:
        logger.exception(f"FATAL ERROR in ETYS network creation: {e}")
        raise
