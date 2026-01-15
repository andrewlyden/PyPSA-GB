"""
Diagnostic script to inspect bus coordinates in PyPSA network files.
Checks for coordinate validity, ranges, outliers, and spatial distribution.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pypsa
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if len(sys.argv) < 2:
    print("Usage: python scripts/diagnose_coordinates.py <network_file.nc>")
    sys.exit(1)

net_path = Path(sys.argv[1])
if not net_path.exists():
    logger.error(f"Network file not found: {net_path}")
    sys.exit(1)

logger.info(f"Loading network from {net_path}")
n = pypsa.Network(net_path)

logger.info(f"Network has {len(n.buses)} buses")

# Check for x, y, lon, lat columns
for col in ['x', 'y', 'lon', 'lat']:
    if col in n.buses.columns:
        logger.info(f"  Column '{col}': {n.buses[col].notna().sum()} non-NaN values")
    else:
        logger.warning(f"  Column '{col}' NOT FOUND")

logger.info("\n=== Coordinate Statistics ===")

# X (longitude) analysis
if 'x' in n.buses.columns:
    x_valid = n.buses['x'].notna()
    if x_valid.any():
        x_vals = n.buses.loc[x_valid, 'x']
        logger.info(f"\nX (longitude) statistics:")
        logger.info(f"  Min: {x_vals.min():.6f}, Max: {x_vals.max():.6f}")
        logger.info(f"  Mean: {x_vals.mean():.6f}, Median: {x_vals.median():.6f}")
        logger.info(f"  Std Dev: {x_vals.std():.6f}")
        logger.info(f"  Expected UK range: -8 to 2 degrees")
        
        # Check for outliers (values outside reasonable UK range)
        uk_x_range = (x_vals >= -9) & (x_vals <= 3)
        if not uk_x_range.all():
            logger.warning(f"  {(~uk_x_range).sum()} buses with X outside UK range (-9 to 3):")
            outliers = n.buses.loc[x_valid & ~uk_x_range, ['x', 'y']]
            logger.warning(outliers.head(20))

# Y (latitude) analysis
if 'y' in n.buses.columns:
    y_valid = n.buses['y'].notna()
    if y_valid.any():
        y_vals = n.buses.loc[y_valid, 'y']
        logger.info(f"\nY (latitude) statistics:")
        logger.info(f"  Min: {y_vals.min():.6f}, Max: {y_vals.max():.6f}")
        logger.info(f"  Mean: {y_vals.mean():.6f}, Median: {y_vals.median():.6f}")
        logger.info(f"  Std Dev: {y_vals.std():.6f}")
        logger.info(f"  Expected UK range: 49 to 61 degrees")
        
        # Check for outliers
        uk_y_range = (y_vals >= 48) & (y_vals <= 62)
        if not uk_y_range.all():
            logger.warning(f"  {(~uk_y_range).sum()} buses with Y outside UK range (48 to 62):")
            outliers = n.buses.loc[y_valid & ~uk_y_range, ['x', 'y']]
            logger.warning(outliers.head(20))

# LON (longitude for explore) analysis
if 'lon' in n.buses.columns:
    lon_valid = n.buses['lon'].notna()
    if lon_valid.any():
        lon_vals = n.buses.loc[lon_valid, 'lon']
        logger.info(f"\nLON (for explore) statistics:")
        logger.info(f"  Min: {lon_vals.min():.6f}, Max: {lon_vals.max():.6f}")
        logger.info(f"  Mean: {lon_vals.mean():.6f}, Median: {lon_vals.median():.6f}")

# LAT (latitude for explore) analysis
if 'lat' in n.buses.columns:
    lat_valid = n.buses['lat'].notna()
    if lat_valid.any():
        lat_vals = n.buses.loc[lat_valid, 'lat']
        logger.info(f"\nLAT (for explore) statistics:")
        logger.info(f"  Min: {lat_vals.min():.6f}, Max: {lat_vals.max():.6f}")
        logger.info(f"  Mean: {lat_vals.mean():.6f}, Median: {lat_vals.median():.6f}")

logger.info("\n=== Spatial Distribution ===")

# Check buses by quadrant (rough UK divisions)
if 'x' in n.buses.columns and 'y' in n.buses.columns:
    valid = n.buses[['x', 'y']].notna().all(axis=1)
    if valid.any():
        x = n.buses.loc[valid, 'x']
        y = n.buses.loc[valid, 'y']
        
        # Quadrants
        sw = ((x < -4) & (y < 55)).sum()  # Southwest
        se = ((x >= -4) & (y < 55)).sum()  # Southeast
        nw = ((x < -4) & (y >= 55)).sum()  # Northwest
        ne = ((x >= -4) & (y >= 55)).sum()  # Northeast
        
        logger.info(f"Buses by region (rough quadrants around -4°W, 55°N):")
        logger.info(f"  Southwest (W of -4°, S of 55°): {sw}")
        logger.info(f"  Southeast (E of -4°, S of 55°): {se}")
        logger.info(f"  Northwest (W of -4°, N of 55°): {nw}")
        logger.info(f"  Northeast (E of -4°, N of 55°): {ne}")
        logger.info(f"  Total with valid coords: {valid.sum()}")

logger.info("\n=== Sample Buses ===")
buses_sample = n.buses[['x', 'y', 'lon', 'lat']].head(20)
logger.info("\nFirst 20 buses:")
logger.info(buses_sample.to_string())

logger.info("\n=== Buses with NaN Coordinates ===")
nan_mask = n.buses[['x', 'y']].isna().any(axis=1)
logger.info(f"Total buses with NaN in x or y: {nan_mask.sum()}")
if nan_mask.any():
    logger.info(f"Sample NaN buses:")
    logger.info(n.buses.loc[nan_mask, ['x', 'y', 'v_nom']].head(10))

logger.info("\n✓ Diagnostic complete")

