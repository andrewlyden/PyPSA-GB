"""Export an ETYS network for a target year as shapefiles.

This standalone utility reuses the ETYS preprocessing and build pipeline to:
1. Parse raw ETYS data for a chosen ETYS publication year.
2. Build the ETYS base network.
3. Apply ETYS upgrades through a target year.
4. Export the result as shapefiles for GIS use.

Because ESRI Shapefile supports only one geometry type per layer, the export
package contains two shapefiles:
- one point layer for buses
- one line layer for branches (lines, transformers, HVDC links)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.network_build.ETYS_network import (  # noqa: E402
    create_network,
    guess_GSP_location_of_remaining_buses,
)
from scripts.network_build.etys_file_registry import get_etys_input_files  # noqa: E402
from scripts.network_build.process_ETYS_data import (  # noqa: E402
    GSP_locations_from_FES_data,
    add_GSP_location_data,
    buses_from_line_data,
    load_node_to_gsp_mapping,
    load_substation_coordinates,
    repair_disconnected_components,
    sort_raw_ETYS_data,
)
from scripts.utilities.logging_config import setup_logging  # noqa: E402


def _default_output_paths(project_dir: Path, target_year: int) -> tuple[Path, Path, Path]:
    resources_dir = project_dir / "resources" / "network"
    layer_dir = resources_dir / f"ETYS_{target_year}_shapefile"
    archive_path = resources_dir / f"ETYS_{target_year}_shapefile.zip"
    network_path = resources_dir / f"ETYS_{target_year}_network.nc"
    return layer_dir, archive_path, network_path


def preprocess_etys_data(
    etys_year: int,
    data_dir: Path,
    resources_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build or load the ETYS component and bus tables for a publication year."""
    components_path = resources_dir / f"ETYS_{etys_year}_components.csv"
    buses_path = resources_dir / f"ETYS_{etys_year}_buses.csv"

    if components_path.exists() and buses_path.exists():
        logger.info("Loading cached ETYS intermediate CSVs")
        df_components = pd.read_csv(components_path, index_col=0)
        df_buses = pd.read_csv(buses_path, index_col=0)
        return df_components, df_buses

    etys_file, gb_network_file, fes_file = get_etys_input_files(etys_year, str(data_dir))
    substation_coords_file = data_dir / "network" / "ETYS" / "substation_coordinates.csv"

    logger.info("Processing ETYS raw data into intermediate tables")
    df_components, offshore_wf_buses = sort_raw_ETYS_data(etys_file, gb_network_file, logger)
    df_components = repair_disconnected_components(df_components, gb_network_file, logger)

    df_buses = buses_from_line_data(df_components, logger)
    df_gsp = GSP_locations_from_FES_data(fes_file, logger)
    node_to_gsp = load_node_to_gsp_mapping(gb_network_file, logger)
    substation_coords = load_substation_coordinates(str(substation_coords_file), logger)
    df_buses = add_GSP_location_data(df_buses, df_gsp, node_to_gsp, substation_coords, logger)

    df_buses["is_offshore"] = df_buses.index.isin(offshore_wf_buses)

    offshore_with_gsp_coords = (
        df_buses["is_offshore"]
        & df_buses["lat"].notna()
        & df_buses["coord_source"].isin(["gsp_explicit", "gsp_prefix"])
    )
    if offshore_with_gsp_coords.any():
        logger.info(
            "Clearing %s offshore buses with inherited onshore GSP coordinates",
            int(offshore_with_gsp_coords.sum()),
        )
        df_buses.loc[offshore_with_gsp_coords, ["lat", "lon"]] = pd.NA

    if "coord_source" in df_buses.columns:
        df_buses = df_buses.drop(columns=["coord_source"])

    resources_dir.mkdir(parents=True, exist_ok=True)
    df_components.to_csv(components_path)
    df_buses.to_csv(buses_path)
    logger.info("Cached ETYS intermediate CSVs to %s", resources_dir)

    return df_components, df_buses


def build_etys_network(
    etys_year: int,
    target_year: int,
    data_dir: Path,
    resources_dir: Path,
    network_output: Path,
    logger: logging.Logger,
):
    """Build the ETYS network with upgrades through the target year."""
    df_components, df_buses = preprocess_etys_data(etys_year, data_dir, resources_dir, logger)

    offshore_wf_buses = set(df_buses[df_buses["is_offshore"] == True].index)
    df_buses_for_build = df_buses.drop(columns=["is_offshore"])

    logger.info("Guessing unresolved ETYS bus locations")
    df_buses_for_build = guess_GSP_location_of_remaining_buses(
        df_components,
        df_buses_for_build,
        offshore_wf_buses,
        logger,
    )

    etys_file = get_etys_input_files(etys_year, str(data_dir))[0]
    substation_coords_file = data_dir / "network" / "ETYS" / "substation_coordinates.csv"

    logger.info("Creating ETYS network and applying upgrades through %s", target_year)
    network = create_network(
        df_components,
        df_buses_for_build,
        logger=logger,
        export_path=str(network_output),
        etys_upgrades_enabled=True,
        upgrade_year=target_year,
        etys_upgrade_file=etys_file,
        substation_coords_file=str(substation_coords_file),
    )

    network.buses["is_offshore"] = False
    surviving_offshore = offshore_wf_buses & set(network.buses.index)
    if surviving_offshore:
        network.buses.loc[list(surviving_offshore), "is_offshore"] = True

    missing_latlon = network.buses["lat"].isna() & network.buses["x"].notna()
    if missing_latlon.any():
        osgb_to_wgs = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        new_lon, new_lat = osgb_to_wgs.transform(
            network.buses.loc[missing_latlon, "x"].values,
            network.buses.loc[missing_latlon, "y"].values,
        )
        network.buses.loc[missing_latlon, "lon"] = new_lon
        network.buses.loc[missing_latlon, "lat"] = new_lat

    return network


def buses_to_gdf(network) -> gpd.GeoDataFrame:
    """Convert network buses to a GeoDataFrame in OSGB36."""
    buses = network.buses.copy()
    buses["name"] = buses.index
    buses = buses[buses["x"].notna() & buses["y"].notna()].copy()
    geometry = gpd.points_from_xy(buses["x"], buses["y"])
    gdf = gpd.GeoDataFrame(
        buses[
            [col for col in ["name", "v_nom", "carrier", "country", "is_offshore", "lon", "lat"] if col in buses.columns]
        ].rename(columns={"is_offshore": "offshore"}),
        geometry=geometry,
        crs="EPSG:27700",
    )
    gdf.index.name = None
    return gdf


def _branch_frame(component_df: pd.DataFrame, network, component_name: str, rating_col: str) -> gpd.GeoDataFrame:
    rows = []
    bus_xy = network.buses[["x", "y", "v_nom"]].copy()

    for name, row in component_df.iterrows():
        if row["bus0"] not in bus_xy.index or row["bus1"] not in bus_xy.index:
            continue

        start = bus_xy.loc[row["bus0"]]
        end = bus_xy.loc[row["bus1"]]
        if pd.isna(start["x"]) or pd.isna(start["y"]) or pd.isna(end["x"]) or pd.isna(end["y"]):
            continue

        rows.append(
            {
                "name": name,
                "comp": component_name,
                "carrier": row.get("carrier"),
                "bus0": row["bus0"],
                "bus1": row["bus1"],
                "rate_mw": row.get(rating_col),
                "v0_kv": start["v_nom"],
                "v1_kv": end["v_nom"],
                "geometry": LineString([(start["x"], start["y"]), (end["x"], end["y"])]),
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:27700")
    gdf.index.name = None
    return gdf


def branches_to_gdf(network) -> gpd.GeoDataFrame:
    """Convert ETYS branches to a single LineString GeoDataFrame."""
    frames = []
    if len(network.lines) > 0:
        frames.append(_branch_frame(network.lines, network, "line", "s_nom"))
    if len(network.transformers) > 0:
        frames.append(_branch_frame(network.transformers, network, "transformer", "s_nom"))
    if len(network.links) > 0:
        frames.append(_branch_frame(network.links, network, "link", "p_nom"))

    if not frames:
        return gpd.GeoDataFrame(columns=["name", "comp", "carrier", "bus0", "bus1", "rate_mw", "v0_kv", "v1_kv", "geometry"], geometry="geometry", crs="EPSG:27700")

    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:27700")


def write_shapefile_package(network, output_dir: Path, archive_path: Path, logger: logging.Logger) -> None:
    """Write bus and branch shapefiles plus a zip archive."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buses_gdf = buses_to_gdf(network)
    branches_gdf = branches_to_gdf(network)

    buses_path = output_dir / "etys_buses.shp"
    branches_path = output_dir / "etys_branches.shp"
    buses_gdf.to_file(buses_path, driver="ESRI Shapefile", encoding="utf-8")
    branches_gdf.to_file(branches_path, driver="ESRI Shapefile", encoding="utf-8")

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "ETYS shapefile package\n"
        "- etys_buses.shp: bus points\n"
        "- etys_branches.shp: lines, transformers, and HVDC links\n"
        "CRS: EPSG:27700 (OSGB36 / British National Grid)\n",
        encoding="utf-8",
    )

    if archive_path.exists():
        archive_path.unlink()

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output_dir.iterdir():
            archive.write(path, arcname=path.name)

    logger.info("Wrote shapefile package to %s", output_dir)
    logger.info("Wrote shapefile archive to %s", archive_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an ETYS network as shapefiles")
    parser.add_argument("--etys-year", type=int, default=2024, help="ETYS publication year to use (default: 2024)")
    parser.add_argument("--target-year", type=int, default=2031, help="Upgrade year to export (default: 2031)")
    parser.add_argument("--data-dir", type=Path, default=project_root / "data", help="Base data directory")
    parser.add_argument("--resources-dir", type=Path, default=project_root / "resources" / "network", help="Directory for intermediate ETYS CSVs")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for shapefile outputs")
    parser.add_argument("--archive-path", type=Path, default=None, help="Optional zip archive path for the shapefile package")
    parser.add_argument("--network-output", type=Path, default=None, help="Optional NetCDF output path for the built network")
    parser.add_argument("--log-path", type=Path, default=project_root / "logs" / "network_build" / "export_etys_shapefile.log", help="Log file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir, archive_path, network_output = _default_output_paths(project_root, args.target_year)

    if args.output_dir is not None:
        output_dir = args.output_dir
    if args.archive_path is not None:
        archive_path = args.archive_path
    if args.network_output is not None:
        network_output = args.network_output

    logger = setup_logging(args.log_path)

    logger.info("=" * 60)
    logger.info("STARTING ETYS SHAPEFILE EXPORT")
    logger.info("ETYS publication year: %s", args.etys_year)
    logger.info("Target upgrade year: %s", args.target_year)

    network = build_etys_network(
        etys_year=args.etys_year,
        target_year=args.target_year,
        data_dir=args.data_dir,
        resources_dir=args.resources_dir,
        network_output=network_output,
        logger=logger,
    )

    write_shapefile_package(network, output_dir, archive_path, logger)
    logger.info("ETYS shapefile export completed successfully")


if __name__ == "__main__":
    main()