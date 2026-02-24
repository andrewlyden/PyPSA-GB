"""
ETYS Network Upgrades Application Module

This module applies network upgrades from ETYS Appendix B 2023 to PyPSA networks.

Upgrades include:
- Circuit additions, removals, and modifications (2024-2031)
- Transformer additions, removals, and modifications (2024-2031)
- New HVDC interconnectors (2024-2031)

Usage:
    from scripts.ETYS_upgrades import apply_etys_network_upgrades
    network = apply_etys_network_upgrades(network, modelled_year=2035)
"""

import pandas as pd
import numpy as np
import pypsa
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings

# Add project root to path for imports
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.network_build.etys_file_registry import (
    VOLTAGE_LEVELS, ELECTRICAL_DEFAULTS, ETYS_UPGRADE_SHEETS,
    INVALID_NODE_PATTERNS, DEFAULT_RATINGS,
)


# Realistic per-unit parameter ranges for validation (on 100 MVA base)
PARAM_RANGES = {
    'line': {
        'r': (0.0001, 0.05),    # pu resistance
        'x': (0.0001, 0.10),    # pu reactance
        'b': (0.0, 5.0),        # pu susceptance
        's_nom': (50, 5000),    # MVA
    },
    'transformer': {
        'r': (0.0001, 0.02),    # pu resistance
        'x': (0.02, 0.20),      # pu reactance
        'b': (0.0, 0.10),       # pu susceptance
        's_nom': (50, 2000),    # MVA
    },
}


def validate_electrical_params(component_type, r, x, b, s_nom, component_id,
                                logger=None):
    """
    Warn if electrical parameters are outside realistic ranges.
    Does NOT modify values -- only logs warnings.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    ranges = PARAM_RANGES.get(component_type)
    if ranges is None:
        return

    for param_name, value in [('r', r), ('x', x), ('b', b), ('s_nom', s_nom)]:
        if value is None:
            continue
        lo, hi = ranges[param_name]
        if value < lo or value > hi:
            logger.warning(
                f"  {component_type} {component_id}: {param_name}={value:.6f} "
                f"outside expected range [{lo}, {hi}]"
            )


def load_etys_upgrade_data(etys_file: str, logger: Optional[logging.Logger] = None) -> Dict[str, pd.DataFrame]:
    """
    Load ETYS upgrade data from Appendix B 2023 Excel file.
    
    Args:
        etys_file: Path to ETYS Appendix B 2023.xlsx
        logger: Optional logger instance
        
    Returns:
        Dictionary with DataFrames for each operator/component type:
        - 'circuits': Combined circuit changes (all operators)
        - 'transformers': Combined transformer changes (all operators)
        - 'hvdc': HVDC interconnector data
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading ETYS upgrade data from {etys_file}")
    
    # Sheet-to-operator mappings from central registry
    circuit_sheets = ETYS_UPGRADE_SHEETS['circuits']
    transformer_sheets = ETYS_UPGRADE_SHEETS['transformers']
    
    # Load circuit data
    circuits = []
    for sheet_name, operator in circuit_sheets.items():
        try:
            df = pd.read_excel(etys_file, sheet_name=sheet_name, skiprows=1)
            # Normalize column names to handle ETYS year-to-year naming differences
            df.rename(columns={
                'Node 1': 'Node1', 'Node 2': 'Node2',
                'R (% on 100MVA)': 'R (% on 100 MVA)',
                'X (% on 100MVA)': 'X (% on 100 MVA)',
                'B (% on 100MVA)': 'B (% on 100 MVA)',
                'OHL Length(km)': 'OHL Length (km)',
                'Cable Length(km)': 'Cable Length (km)',
            }, inplace=True)
            df['operator'] = operator
            circuits.append(df)
            logger.info(f"  Loaded {sheet_name} ({operator}): {len(df)} records")
        except Exception as e:
            logger.warning(f"  Failed to load {sheet_name}: {e}")
    
    circuits_df = pd.concat(circuits, ignore_index=True) if circuits else pd.DataFrame()
    
    # Load transformer data
    transformers = []
    for sheet_name, operator in transformer_sheets.items():
        try:
            df = pd.read_excel(etys_file, sheet_name=sheet_name, skiprows=1)
            # Normalize column names to handle ETYS year-to-year naming differences
            df.rename(columns={
                'Node 1': 'Node1', 'Node 2': 'Node2',
                'R (% on 100MVA)': 'R (% on 100 MVA)',
                'X (% on 100MVA)': 'X (% on 100 MVA)',
                'B (% on 100MVA)': 'B (% on 100 MVA)',
                'Rating (MVA)': 'Winter Rating (MVA)',
            }, inplace=True)
            df['operator'] = operator
            transformers.append(df)
            logger.info(f"  Loaded {sheet_name} ({operator}): {len(df)} records")
        except Exception as e:
            logger.warning(f"  Failed to load {sheet_name}: {e}")
    
    transformers_df = pd.concat(transformers, ignore_index=True) if transformers else pd.DataFrame()
    
    # Load HVDC data
    hvdc_sheet = ETYS_UPGRADE_SHEETS['hvdc']
    try:
        hvdc_df = pd.read_excel(etys_file, sheet_name=hvdc_sheet, skiprows=1)
        logger.info(f"  Loaded {hvdc_sheet} (HVDC): {len(hvdc_df)} records")
    except Exception as e:
        logger.warning(f"  Failed to load HVDC data: {e}")
        hvdc_df = pd.DataFrame()
    
    return {
        'circuits': circuits_df,
        'transformers': transformers_df,
        'hvdc': hvdc_df
    }


def filter_upgrades_by_year(upgrades_data: Dict[str, pd.DataFrame], 
                            modelled_year: int,
                            logger: Optional[logging.Logger] = None) -> Dict[str, Dict[str, List]]:
    """
    Filter upgrades by modelled year and categorize by status.
    
    Args:
        upgrades_data: Dictionary with circuits, transformers, hvdc DataFrames
        modelled_year: Target year for upgrades
        logger: Optional logger instance
        
    Returns:
        Dictionary with categorized upgrades:
        {
            'circuits': {'additions': [...], 'removals': [...], 'changes': [...]},
            'transformers': {'additions': [...], 'removals': [...], 'changes': [...]},
            'hvdc': {'additions': [...]}
        }
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Filtering upgrades for year {modelled_year}")
    
    # Ensure modelled_year is an integer
    modelled_year = int(modelled_year)
    
    categorized = {
        'circuits': {'additions': [], 'removals': [], 'changes': []},
        'transformers': {'additions': [], 'removals': [], 'changes': []},
        'hvdc': {'additions': []}
    }
    
    # Process circuits
    if len(upgrades_data['circuits']) > 0:
        # Ensure Year column is numeric
        circuits_df = upgrades_data['circuits'].copy()
        circuits_df['Year'] = pd.to_numeric(circuits_df['Year'], errors='coerce')
        
        # Filter out records with invalid nodes (NaN, empty, etc.)
        circuits_df = circuits_df[
            circuits_df['Node1'].notna() & 
            circuits_df['Node2'].notna() &
            (circuits_df['Node1'].astype(str).str.strip() != '') &
            (circuits_df['Node2'].astype(str).str.strip() != '') &
            (circuits_df['Node1'].astype(str).str.lower() != 'nan') &
            (circuits_df['Node2'].astype(str).str.lower() != 'nan')
        ]
        
        circuits = circuits_df[circuits_df['Year'] <= modelled_year]
        
        for status in ['Addition', 'Removed', 'Change', 'Modify']:
            matching = circuits[circuits['Status'] == status]
            if status in ['Addition']:
                categorized['circuits']['additions'].extend(matching.to_dict('records'))
            elif status == 'Removed':
                categorized['circuits']['removals'].extend(matching.to_dict('records'))
            elif status in ['Change', 'Modify']:
                categorized['circuits']['changes'].extend(matching.to_dict('records'))
        
        logger.info(f"  Circuits: {len(categorized['circuits']['additions'])} additions, "
                   f"{len(categorized['circuits']['removals'])} removals, "
                   f"{len(categorized['circuits']['changes'])} changes")
    
    # Process transformers
    if len(upgrades_data['transformers']) > 0:
        # Ensure Year column is numeric
        transformers_df = upgrades_data['transformers'].copy()
        transformers_df['Year'] = pd.to_numeric(transformers_df['Year'], errors='coerce')
        
        # Filter out records with invalid nodes (NaN, empty, etc.)
        transformers_df = transformers_df[
            transformers_df['Node1'].notna() & 
            transformers_df['Node2'].notna() &
            (transformers_df['Node1'].astype(str).str.strip() != '') &
            (transformers_df['Node2'].astype(str).str.strip() != '') &
            (transformers_df['Node1'].astype(str).str.lower() != 'nan') &
            (transformers_df['Node2'].astype(str).str.lower() != 'nan')
        ]
        
        transformers = transformers_df[transformers_df['Year'] <= modelled_year]
        
        for status in ['Addition', 'Removed', 'Change']:
            matching = transformers[transformers['Status'] == status]
            if status == 'Addition':
                categorized['transformers']['additions'].extend(matching.to_dict('records'))
            elif status == 'Removed':
                categorized['transformers']['removals'].extend(matching.to_dict('records'))
            elif status == 'Change':
                categorized['transformers']['changes'].extend(matching.to_dict('records'))
        
        logger.info(f"  Transformers: {len(categorized['transformers']['additions'])} additions, "
                   f"{len(categorized['transformers']['removals'])} removals, "
                   f"{len(categorized['transformers']['changes'])} changes")
    
    # Process HVDC
    if len(upgrades_data['hvdc']) > 0:
        # Ensure 'Planned from year' column is numeric
        hvdc_df = upgrades_data['hvdc'].copy()
        hvdc_df['Planned from year'] = pd.to_numeric(hvdc_df['Planned from year'], errors='coerce')
        hvdc = hvdc_df[
            (hvdc_df['Existing'].str.lower() == 'no') &
            (hvdc_df['Planned from year'] <= modelled_year)
        ]
        categorized['hvdc']['additions'].extend(hvdc.to_dict('records'))
        
        logger.info(f"  HVDC: {len(categorized['hvdc']['additions'])} new interconnectors")
    
    return categorized


# VOLTAGE_LEVELS imported from etys_file_registry (shared constant)
# Alias for backward compatibility within this module
VOLTAGE_MAP = VOLTAGE_LEVELS


def add_missing_buses_from_upgrades(network: pypsa.Network,
                                    categorized_upgrades: Dict,
                                    substation_coords: Optional[Dict[str, Tuple[float, float]]] = None,
                                    logger: Optional[logging.Logger] = None) -> int:
    """
    Add missing buses to the network that are referenced in upgrade data.

    When ETYS upgrades reference new nodes (e.g., BLYT22 for a new 275kV bus at Blyth),
    these buses need to be added to the network before circuits can be created.

    The function uses a multi-pass approach:
    0. Strategy 0: Exact site lookup in substation_coords dict (lat/lon → OSGB36)
    1. Collects all node names and their connections from circuit/transformer additions
    2. Identifies which nodes don't exist in the network
    3. Infers voltage level from node name (e.g., '2' suffix = 275kV)
    4. First pass: Uses coordinates from existing bus at same site (4-char prefix match)
    5. Second pass: Uses coordinates from connected buses in the upgrade data
    6. Third pass: Uses line length to estimate position from known connected bus
    7. Final fallback: Network centroid (should rarely be needed)

    Args:
        network: PyPSA network object
        categorized_upgrades: Dictionary with 'circuits' and 'transformers' additions
        substation_coords: Optional dict mapping 4-char site code → (lat, lon) in WGS84.
            When provided, used as highest-priority coordinate source (Strategy 0).
        logger: Optional logger instance

    Returns:
        Number of buses added
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Collect all nodes referenced in additions AND their connections
    nodes_needed = set()
    node_connections = {}  # Map: node -> list of (connected_node, length_km)
    
    def is_valid_node(node_name):
        """Check if node name is valid (not a placeholder or invalid pattern)."""
        if not node_name or pd.isna(node_name):
            return False
        node_str = str(node_name).strip().lower()
        return node_str and not any(p in node_str for p in INVALID_NODE_PATTERNS)
    
    def safe_float(val, default=0.0):
        """Safely convert a value to float, handling 'TBC', NaN, etc."""
        if val is None or pd.isna(val):
            return default
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).strip().lower()
        if val_str in ('', 'tbc', 'n/a', 'nan', 'none'):
            return default
        try:
            return float(val_str)
        except ValueError:
            return default
    
    # From circuit additions - collect nodes AND connections
    for record in categorized_upgrades['circuits']['additions']:
        node1 = str(record.get('Node1', '')).strip()
        node2 = str(record.get('Node2', '')).strip()
        
        # Get line length (OHL + cable) - handle 'TBC' and other invalid values
        ohl_length = safe_float(record.get('OHL Length (km)', 0))
        cable_length = safe_float(record.get('Cable Length (km)', 0))
        total_length = ohl_length + cable_length
        
        if is_valid_node(node1):
            nodes_needed.add(node1)
            if node1 not in node_connections:
                node_connections[node1] = []
            if is_valid_node(node2):
                node_connections[node1].append((node2, total_length))
                
        if is_valid_node(node2):
            nodes_needed.add(node2)
            if node2 not in node_connections:
                node_connections[node2] = []
            if is_valid_node(node1):
                node_connections[node2].append((node1, total_length))
    
    # From transformer additions (transformers have no length, use 0)
    for record in categorized_upgrades['transformers']['additions']:
        node1 = str(record.get('Node1', '')).strip()
        node2 = str(record.get('Node2', '')).strip()
        
        if is_valid_node(node1):
            nodes_needed.add(node1)
            if node1 not in node_connections:
                node_connections[node1] = []
            if is_valid_node(node2):
                node_connections[node1].append((node2, 0.0))  # Transformers are co-located
                
        if is_valid_node(node2):
            nodes_needed.add(node2)
            if node2 not in node_connections:
                node_connections[node2] = []
            if is_valid_node(node1):
                node_connections[node2].append((node1, 0.0))
    
    # Remove empty strings
    nodes_needed.discard('')
    
    # Find missing nodes
    existing_buses = set(network.buses.index)
    missing_nodes = nodes_needed - existing_buses
    
    if not missing_nodes:
        logger.info("  No missing buses to add from upgrade data")
        return 0
    
    logger.info(f"  Found {len(missing_nodes)} missing buses referenced in upgrades")
    
    # Calculate network centroid for final fallback
    centroid_x = network.buses['x'].median()
    centroid_y = network.buses['y'].median()

    # Build WGS84 → OSGB36 transformer once (if substation_coords provided)
    wgs_to_osgb = None
    if substation_coords:
        try:
            from pyproj import Transformer as ProjTransformer
            wgs_to_osgb = ProjTransformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        except ImportError:
            logger.warning("  pyproj not available — Strategy 0 (substation_coords) disabled")

    # Track which buses were added and their coordinates
    added_buses = {}  # node -> (x, y)
    buses_at_centroid = []  # Track buses that ended up at centroid
    
    def get_bus_coords(bus_name):
        """Get coordinates for a bus (from network or already-added buses)."""
        if bus_name in network.buses.index:
            x = network.buses.loc[bus_name, 'x']
            y = network.buses.loc[bus_name, 'y']
            # Check if this is NOT at the centroid (valid coordinates)
            if abs(x - centroid_x) > 100 or abs(y - centroid_y) > 100:
                return x, y
        if bus_name in added_buses:
            x, y = added_buses[bus_name]
            if abs(x - centroid_x) > 100 or abs(y - centroid_y) > 100:
                return x, y
        return None, None
    
    def estimate_coords_from_connection(node, connected_node, length_km):
        """
        Estimate coordinates for a node based on a connected node with known coords.
        Uses line length to add a small offset (for visualization clarity).
        """
        x, y = get_bus_coords(connected_node)
        if x is None:
            return None, None
        
        # If length is 0 or very small (transformers, short cables), co-locate
        if length_km < 0.5:
            return x, y
        
        # Add a small offset based on line length for visualization
        # Use 1000 meters per km as rough scale (OSGB36 is in meters)
        # Add offset in a deterministic direction based on node name hash
        angle = (hash(node) % 360) * np.pi / 180
        offset = min(length_km * 500, 5000)  # Cap offset at 5km
        new_x = x + offset * np.cos(angle)
        new_y = y + offset * np.sin(angle)
        
        return new_x, new_y
    
    # Multi-pass approach to resolve coordinates
    max_iterations = 5
    for iteration in range(max_iterations):
        nodes_to_add = [n for n in missing_nodes if n not in added_buses]
        if not nodes_to_add:
            break
            
        added_this_iteration = 0
        
        for node in nodes_to_add:
            # Extract site code and voltage
            site_code = node[:4] if len(node) >= 4 else node
            voltage_suffix = node[4] if len(node) > 4 else '4'
            v_nom = VOLTAGE_MAP.get(voltage_suffix, 400)
            
            x, y = None, None
            coord_source = None

            # Strategy 0: Explicit substation_coords lookup (highest priority)
            if x is None and substation_coords and wgs_to_osgb:
                if site_code in substation_coords:
                    lat, lon = substation_coords[site_code]
                    osgb_x, osgb_y = wgs_to_osgb.transform(lon, lat)
                    x, y = osgb_x, osgb_y
                    coord_source = f"substation_coords ({lat:.4f}, {lon:.4f})"

            # Strategy 1: Same-site bus with valid coordinates (not at centroid)
            site_buses = network.buses[network.buses.index.str.startswith(site_code)]
            for _, site_bus in site_buses.iterrows():
                if abs(site_bus['x'] - centroid_x) > 100 or abs(site_bus['y'] - centroid_y) > 100:
                    x, y = site_bus['x'], site_bus['y']
                    coord_source = f"same-site bus"
                    break
            
            # Strategy 2: Already-added bus at same site
            if x is None:
                for added_node, (ax, ay) in added_buses.items():
                    if added_node[:4] == site_code:
                        if abs(ax - centroid_x) > 100 or abs(ay - centroid_y) > 100:
                            x, y = ax, ay
                            coord_source = f"added same-site bus {added_node}"
                            break
            
            # Strategy 3: Connected bus from upgrade data
            if x is None and node in node_connections:
                for connected_node, length_km in node_connections[node]:
                    cx, cy = estimate_coords_from_connection(node, connected_node, length_km)
                    if cx is not None:
                        x, y = cx, cy
                        coord_source = f"connected bus {connected_node} (len={length_km:.1f}km)"
                        break
            
            # If we found coordinates, add the bus
            if x is not None and y is not None:
                try:
                    network.add('Bus', node,
                               v_nom=v_nom,
                               x=x,
                               y=y,
                               carrier='AC',
                               country='GB')
                    added_buses[node] = (x, y)
                    added_this_iteration += 1
                    logger.debug(f"  Added bus {node} ({v_nom}kV) at ({x:.0f}, {y:.0f}) via {coord_source}")
                except Exception as e:
                    logger.warning(f"  Failed to add bus {node}: {e}")
        
        logger.debug(f"  Iteration {iteration + 1}: added {added_this_iteration} buses")
        
        if added_this_iteration == 0:
            break  # No progress made, remaining nodes have no resolvable coordinates
    
    # Final pass: Add remaining buses at centroid (with warning)
    remaining = [n for n in missing_nodes if n not in added_buses]
    for node in remaining:
        site_code = node[:4] if len(node) >= 4 else node
        voltage_suffix = node[4] if len(node) > 4 else '4'
        v_nom = VOLTAGE_MAP.get(voltage_suffix, 400)
        
        try:
            network.add('Bus', node,
                       v_nom=v_nom,
                       x=centroid_x,
                       y=centroid_y,
                       carrier='AC',
                       country='GB')
            added_buses[node] = (centroid_x, centroid_y)
            buses_at_centroid.append(node)
            logger.warning(f"  Added bus {node} ({v_nom}kV) at CENTROID - no coordinate reference found")
        except Exception as e:
            logger.warning(f"  Failed to add bus {node}: {e}")
    
    buses_added = len(added_buses)
    logger.info(f"  Added {buses_added} new buses from upgrade data")
    if buses_at_centroid:
        logger.warning(f"  WARNING: {len(buses_at_centroid)} buses placed at centroid (no coords): {buses_at_centroid[:5]}{'...' if len(buses_at_centroid) > 5 else ''}")
    
    return buses_added


def apply_circuit_additions(network: pypsa.Network, 
                           additions: List[Dict],
                           logger: Optional[logging.Logger] = None) -> Tuple[int, int, List[Dict]]:
    """
    Add new circuits (lines) to the network.
    
    Args:
        network: PyPSA network object
        additions: List of circuit addition records from ETYS
        logger: Optional logger instance
        
    Returns:
        Tuple of (successful_additions, failed_additions, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    successful, failed = 0, 0
    failure_details = []
    
    # Track line IDs to make parallel circuits unique
    line_id_counter = {}
    
    for record in additions:
        try:
            node1 = str(record['Node1']).strip()
            node2 = str(record['Node2']).strip()
            year = int(record['Year']) if pd.notna(record.get('Year')) else 2024
            
            # Skip invalid nodes (generic names, NaN, etc.)
            if any(p in node1.lower() for p in INVALID_NODE_PATTERNS) or any(p in node2.lower() for p in INVALID_NODE_PATTERNS):
                failure_details.append({'type': 'addition', 'reason': 'invalid_node_name', 'node1': node1, 'node2': node2, 'year': year})
                logger.debug(f"Skipping circuit with generic node name: {node1}-{node2}")
                failed += 1
                continue
            
            # Skip self-loops (bus-section switches, series compensators, etc.)
            # These are "Zero Length" or "SC" circuit types in ETYS that connect
            # a bus to itself.  They cannot be modelled as AC lines (zero
            # impedance → singular susceptance matrix → SVD failure).
            if node1 == node2:
                circuit_type = str(record.get('Circuit Type', '')).strip()
                logger.debug(
                    f"Skipping self-loop circuit {node1}-{node2} "
                    f"(type={circuit_type}): bus-section switch / SC"
                )
                failed += 1
                failure_details.append({
                    'type': 'addition', 'reason': 'self_loop',
                    'node1': node1, 'node2': node2, 'year': year,
                })
                continue
            
            # Check if buses exist
            if node1 not in network.buses.index or node2 not in network.buses.index:
                missing = [n for n in [node1, node2] if n not in network.buses.index]
                failure_details.append({'type': 'addition', 'reason': 'bus_not_found', 'node1': node1, 'node2': node2, 'year': year, 'missing_buses': missing})
                logger.debug(f"Skipping circuit {node1}-{node2}: bus not found ({', '.join(missing)})")
                failed += 1
                continue
            
            # Create unique ID with counter for parallel circuits
            base_line_id = f"{node1}_{node2}_ETYS_UPG_{year}"
            if base_line_id in line_id_counter:
                line_id_counter[base_line_id] += 1
                line_id = f"{base_line_id}_{line_id_counter[base_line_id]}"
            else:
                line_id_counter[base_line_id] = 0
                line_id = base_line_id
            
            # Skip if this exact ID already exists in network
            if line_id in network.lines.index:
                logger.debug(f"Line {line_id} already exists, skipping")
                continue

            # Check for existing parallel circuits (info/warning for visibility)
            existing_parallels = network.lines[
                ((network.lines['bus0'] == node1) & (network.lines['bus1'] == node2)) |
                ((network.lines['bus0'] == node2) & (network.lines['bus1'] == node1))
            ]
            if not existing_parallels.empty:
                n_existing = len(existing_parallels)
                ratings = existing_parallels['s_nom'].tolist()
                logger.info(
                    f"  Adding parallel circuit {node1}–{node2}: "
                    f"{n_existing} circuit(s) already exist (ratings: {ratings} MVA)"
                )
                if n_existing >= 4:
                    logger.warning(
                        f"  High parallel circuit count for {node1}–{node2}: "
                        f"{n_existing} existing + 1 new = {n_existing + 1} total"
                    )

            # Extract electrical parameters (use shared defaults from registry)
            line_defaults = ELECTRICAL_DEFAULTS['line']

            # Convert from "% on 100 MVA" to per-unit (divide by 100)
            # Must match conversion in process_ETYS_data.py
            raw_r = record.get('R (% on 100 MVA)')
            r = float(raw_r) / 100.0 if pd.notna(raw_r) else line_defaults['r']

            raw_x = record.get('X (% on 100 MVA)')
            x = float(raw_x) / 100.0 if pd.notna(raw_x) else line_defaults['x']

            raw_b = record.get('B (% on 100 MVA)')
            b = float(raw_b) / 100.0 if pd.notna(raw_b) else line_defaults['b']

            # ETYS data often has zero impedance for planned circuits where
            # detailed parameters are not yet available.  Zero reactance causes
            # a singular susceptance matrix (B = 1/x → Inf) which makes the
            # post-solve SVD computation of voltage angles fail.
            # Substitute sensible defaults when values are zero.
            if x == 0.0:
                logger.warning(
                    f"  Circuit {line_id}: X=0 in ETYS data (placeholder); "
                    f"using default x={line_defaults['x']}"
                )
                x = line_defaults['x']
            if r == 0.0:
                r = line_defaults['r']
            
            # Use winter rating as conservative estimate
            s_nom = float(record.get('Winter Rating (MVA)', DEFAULT_RATINGS['line'])) if pd.notna(record.get('Winter Rating (MVA)')) else DEFAULT_RATINGS['line']

            # Extract circuit type if available
            circuit_type = str(record.get('Circuit Type', 'AC')).strip() if pd.notna(record.get('Circuit Type')) else 'AC'
            
            # Determine if overhead or cable
            ohl_length = float(record.get('OHL Length (km)', 0)) if pd.notna(record.get('OHL Length (km)')) else 0
            cable_length = float(record.get('Cable Length (km)', 0)) if pd.notna(record.get('Cable Length (km)')) else 0
            
            # Validate electrical parameters
            validate_electrical_params('line', r, x, b, s_nom, line_id, logger)

            # Add line to network
            network.add('Line', line_id,
                       bus0=node1,
                       bus1=node2,
                       r=r,
                       x=x,
                       b=b,
                       s_nom=s_nom,
                       carrier='AC',
                       under_construction=False)
            
            successful += 1
            
        except Exception as e:
            failure_details.append({'type': 'addition', 'reason': 'exception', 'node1': node1, 'node2': node2, 'year': year, 'error': str(e)})
            logger.debug(f"Failed to add circuit {node1}-{node2}: {e}")
            failed += 1
    
    return successful, failed, failure_details


def apply_circuit_removals(network: pypsa.Network,
                          removals: List[Dict],
                          logger: Optional[logging.Logger] = None) -> Tuple[int, int, List[Dict]]:
    """
    Remove circuits (lines) from the network.
    
    Args:
        network: PyPSA network object
        removals: List of circuit removal records from ETYS
        logger: Optional logger instance
        
    Returns:
        Tuple of (successful_removals, failed_removals, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    successful, failed = 0, 0
    failure_details = []
    
    for record in removals:
        try:
            node1 = str(record['Node1']).strip()
            node2 = str(record['Node2']).strip()
            
            # Find line connecting these nodes (check both directions)
            matching_lines = network.lines[
                ((network.lines['bus0'] == node1) & (network.lines['bus1'] == node2)) |
                ((network.lines['bus0'] == node2) & (network.lines['bus1'] == node1))
            ]
            
            if len(matching_lines) == 0:
                failure_details.append({'type': 'removal', 'reason': 'line_not_found', 'node1': node1, 'node2': node2})
                logger.debug(f"No line found for removal: {node1}-{node2}")
                failed += 1
                continue

            # Remove ONE line per removal record (not all parallel circuits).
            # If the removal record specifies a rating, prefer the closest match.
            if len(matching_lines) > 1 and pd.notna(record.get('Winter Rating (MVA)')):
                target_rating = float(record['Winter Rating (MVA)'])
                rating_diff = (matching_lines['s_nom'] - target_rating).abs()
                line_id = rating_diff.idxmin()
            else:
                line_id = matching_lines.index[0]

            network.lines.drop(line_id, inplace=True)
            successful += 1
            if len(matching_lines) > 1:
                logger.debug(f"Removed 1 of {len(matching_lines)} parallel circuits "
                             f"between {node1}-{node2} (line {line_id})")

        except Exception as e:
            failure_details.append({'type': 'removal', 'reason': 'exception', 'node1': node1, 'node2': node2, 'error': str(e)})
            logger.debug(f"Failed to remove circuit {node1}-{node2}: {e}")
            failed += 1

    return successful, failed, failure_details


def apply_circuit_changes(network: pypsa.Network,
                         changes: List[Dict],
                         logger: Optional[logging.Logger] = None) -> Tuple[int, int, List[Dict]]:
    """
    Modify existing circuit parameters.
    
    Args:
        network: PyPSA network object
        changes: List of circuit modification records from ETYS
        logger: Optional logger instance
        
    Returns:
        Tuple of (successful_modifications, failed_modifications, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    successful, failed = 0, 0
    failure_details = []
    
    for record in changes:
        try:
            node1 = str(record['Node1']).strip()
            node2 = str(record['Node2']).strip()
            
            # Find line connecting these nodes
            matching_lines = network.lines[
                ((network.lines['bus0'] == node1) & (network.lines['bus1'] == node2)) |
                ((network.lines['bus0'] == node2) & (network.lines['bus1'] == node1))
            ]
            
            if len(matching_lines) == 0:
                failure_details.append({'type': 'change', 'reason': 'line_not_found', 'node1': node1, 'node2': node2})
                logger.debug(f"No line found for modification: {node1}-{node2}")
                failed += 1
                continue
            
            # Update parameters (convert % on 100 MVA to per-unit, matching process_ETYS_data.py)
            # ETYS change records sometimes have zero impedance as a placeholder.
            # Do NOT overwrite existing (non-zero) values with zeros — that would
            # create a singular susceptance matrix and crash post-solve SVD.
            for line_id in matching_lines.index:
                if pd.notna(record.get('Winter Rating (MVA)')):
                    network.lines.at[line_id, 's_nom'] = float(record['Winter Rating (MVA)'])
                if pd.notna(record.get('R (% on 100 MVA)')):
                    new_r = float(record['R (% on 100 MVA)']) / 100.0
                    if new_r != 0.0:
                        network.lines.at[line_id, 'r'] = new_r
                    else:
                        logger.debug(f"  Skipping zero R for line {line_id} (keeping existing value)")
                if pd.notna(record.get('X (% on 100 MVA)')):
                    new_x = float(record['X (% on 100 MVA)']) / 100.0
                    if new_x != 0.0:
                        network.lines.at[line_id, 'x'] = new_x
                    else:
                        logger.warning(f"  Skipping zero X for line {line_id} (keeping existing value)")
                if pd.notna(record.get('B (% on 100 MVA)')):
                    network.lines.at[line_id, 'b'] = float(record['B (% on 100 MVA)']) / 100.0
                
                successful += 1
                
        except Exception as e:
            failure_details.append({'type': 'change', 'reason': 'exception', 'node1': node1, 'node2': node2, 'error': str(e)})
            logger.debug(f"Failed to modify circuit {node1}-{node2}: {e}")
            failed += 1
    
    return successful, failed, failure_details


def apply_transformer_additions(network: pypsa.Network,
                               additions: List[Dict],
                               logger: Optional[logging.Logger] = None) -> Tuple[int, int, List[Dict]]:
    """
    Add new transformers to the network.
    
    Args:
        network: PyPSA network object
        additions: List of transformer addition records from ETYS
        logger: Optional logger instance
        
    Returns:
        Tuple of (successful_additions, failed_additions, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    successful, failed = 0, 0
    failure_details = []
    
    # Track transformer IDs to make parallel transformers unique
    xfmr_id_counter = {}
    
    for record in additions:
        try:
            node1 = str(record['Node1']).strip()
            node2 = str(record['Node2']).strip()
            
            # Skip invalid nodes (NaN, empty, generic names)
            if (node1.lower() == 'nan' or node2.lower() == 'nan' or not node1 or not node2 or
                any(p in node1.lower() for p in INVALID_NODE_PATTERNS) or
                any(p in node2.lower() for p in INVALID_NODE_PATTERNS)):
                year = int(record['Year']) if pd.notna(record.get('Year')) else 2024
                failure_details.append({'type': 'addition', 'reason': 'invalid_node_name', 'node1': node1, 'node2': node2, 'year': year})
                logger.debug(f"Skipping transformer with invalid nodes: {node1}-{node2}")
                failed += 1
                continue
                
            year = int(record['Year']) if pd.notna(record.get('Year')) else 2024
            
            # Check if buses exist
            if node1 not in network.buses.index or node2 not in network.buses.index:
                missing = [n for n in [node1, node2] if n not in network.buses.index]
                failure_details.append({'type': 'addition', 'reason': 'bus_not_found', 'node1': node1, 'node2': node2, 'year': year, 'missing_buses': missing})
                logger.debug(f"Skipping transformer {node1}-{node2}: bus not found ({', '.join(missing)})")
                failed += 1
                continue
            
            # Create unique ID with counter for parallel transformers
            base_xfmr_id = f"{node1}_{node2}_ETYS_XFMR_{year}"
            if base_xfmr_id in xfmr_id_counter:
                xfmr_id_counter[base_xfmr_id] += 1
                xfmr_id = f"{base_xfmr_id}_{xfmr_id_counter[base_xfmr_id]}"
            else:
                xfmr_id_counter[base_xfmr_id] = 0
                xfmr_id = base_xfmr_id
            
            # Skip if already exists
            if xfmr_id in network.transformers.index:
                logger.debug(f"Transformer {xfmr_id} already exists, skipping")
                continue

            # Check for existing parallel transformers (info/warning for visibility)
            existing_parallels = network.transformers[
                ((network.transformers['bus0'] == node1) & (network.transformers['bus1'] == node2)) |
                ((network.transformers['bus0'] == node2) & (network.transformers['bus1'] == node1))
            ]
            if not existing_parallels.empty:
                n_existing = len(existing_parallels)
                ratings = existing_parallels['s_nom'].tolist()
                logger.info(
                    f"  Adding parallel transformer {node1}–{node2}: "
                    f"{n_existing} transformer(s) already exist (ratings: {ratings} MVA)"
                )
                if n_existing >= 4:
                    logger.warning(
                        f"  High parallel transformer count for {node1}–{node2}: "
                        f"{n_existing} existing + 1 new = {n_existing + 1} total"
                    )

            # Extract electrical parameters (use shared defaults from registry)
            xfmr_defaults = ELECTRICAL_DEFAULTS['transformer']

            # Convert from "% on 100 MVA" to per-unit (divide by 100)
            # Must match conversion in process_ETYS_data.py
            raw_r = record.get('R (% on 100 MVA)')
            r = float(raw_r) / 100.0 if pd.notna(raw_r) else xfmr_defaults['r']

            raw_x = record.get('X (% on 100 MVA)')
            x = float(raw_x) / 100.0 if pd.notna(raw_x) else xfmr_defaults['x']

            raw_b = record.get('B (% on 100 MVA)')
            b = float(raw_b) / 100.0 if pd.notna(raw_b) else xfmr_defaults['b']

            # Substitute defaults for zero impedance (same issue as circuits)
            if x == 0.0:
                logger.warning(
                    f"  Transformer {xfmr_id}: X=0 in ETYS data (placeholder); "
                    f"using default x={xfmr_defaults['x']}"
                )
                x = xfmr_defaults['x']
            if r == 0.0:
                r = xfmr_defaults['r']
            s_nom = float(record.get('Winter Rating (MVA)', DEFAULT_RATINGS['transformer'])) if pd.notna(record.get('Winter Rating (MVA)')) else DEFAULT_RATINGS['transformer']

            # Validate electrical parameters
            validate_electrical_params('transformer', r, x, b, s_nom, xfmr_id, logger)

            # Add transformer (without v_nom_0/v_nom_1 which cause PyPSA warnings)
            # PyPSA infers voltage levels from the connected buses automatically
            network.add('Transformer', xfmr_id,
                       bus0=node1,
                       bus1=node2,
                       model='pi',
                       r=r,
                       x=x,
                       b=b,
                       s_nom=s_nom,
                       under_construction=False)
            
            successful += 1
            
        except Exception as e:
            failure_details.append({'type': 'addition', 'reason': 'exception', 'node1': node1, 'node2': node2, 'year': year, 'error': str(e)})
            logger.debug(f"Failed to add transformer {node1}-{node2}: {e}")
            failed += 1
    
    return successful, failed, failure_details


def apply_transformer_removals(network: pypsa.Network,
                              removals: List[Dict],
                              logger: Optional[logging.Logger] = None) -> Tuple[int, int, List[Dict]]:
    """
    Remove transformers from the network.
    
    Args:
        network: PyPSA network object
        removals: List of transformer removal records from ETYS
        logger: Optional logger instance
        
    Returns:
        Tuple of (successful_removals, failed_removals, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    successful, failed = 0, 0
    failure_details = []
    
    for record in removals:
        try:
            node1 = str(record['Node1']).strip()
            node2 = str(record['Node2']).strip()
            
            # Find transformer connecting these nodes
            matching = network.transformers[
                ((network.transformers['bus0'] == node1) & (network.transformers['bus1'] == node2)) |
                ((network.transformers['bus0'] == node2) & (network.transformers['bus1'] == node1))
            ]
            
            if len(matching) == 0:
                failure_details.append({'type': 'removal', 'reason': 'transformer_not_found', 'node1': node1, 'node2': node2})
                logger.debug(f"No transformer found for removal: {node1}-{node2}")
                failed += 1
                continue

            # Remove ONE transformer per removal record (not all parallel transformers).
            # If the removal record specifies a rating, prefer the closest match.
            if len(matching) > 1 and pd.notna(record.get('Winter Rating (MVA)')):
                target_rating = float(record['Winter Rating (MVA)'])
                rating_diff = (matching['s_nom'] - target_rating).abs()
                xfmr_id = rating_diff.idxmin()
            else:
                xfmr_id = matching.index[0]

            network.transformers.drop(xfmr_id, inplace=True)
            successful += 1
            if len(matching) > 1:
                logger.debug(f"Removed 1 of {len(matching)} parallel transformers "
                             f"between {node1}-{node2} (transformer {xfmr_id})")
                
        except Exception as e:
            failure_details.append({'type': 'removal', 'reason': 'exception', 'node1': node1, 'node2': node2, 'error': str(e)})
            logger.debug(f"Failed to remove transformer {node1}-{node2}: {e}")
            failed += 1
    
    return successful, failed, failure_details


def apply_transformer_changes(network: pypsa.Network,
                              changes: List[Dict],
                              logger: Optional[logging.Logger] = None) -> Tuple[int, int, List[Dict]]:
    """
    Modify existing transformer parameters (rating, impedance).

    Mirrors apply_circuit_changes() but operates on network.transformers.

    Args:
        network: PyPSA network object
        changes: List of transformer modification records from ETYS
        logger: Optional logger instance

    Returns:
        Tuple of (successful_modifications, failed_modifications, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    successful, failed = 0, 0
    failure_details = []

    for record in changes:
        try:
            node1 = str(record['Node1']).strip()
            node2 = str(record['Node2']).strip()

            # Find transformer connecting these nodes (check both directions)
            matching = network.transformers[
                ((network.transformers['bus0'] == node1) & (network.transformers['bus1'] == node2)) |
                ((network.transformers['bus0'] == node2) & (network.transformers['bus1'] == node1))
            ]

            if len(matching) == 0:
                failure_details.append({'type': 'change', 'reason': 'transformer_not_found',
                                       'node1': node1, 'node2': node2})
                logger.debug(f"No transformer found for modification: {node1}-{node2}")
                failed += 1
                continue

            # Update parameters (convert % on 100 MVA to per-unit)
            # Do NOT overwrite existing impedance with zeros (see circuit changes).
            for xfmr_id in matching.index:
                if pd.notna(record.get('Winter Rating (MVA)')):
                    network.transformers.at[xfmr_id, 's_nom'] = float(record['Winter Rating (MVA)'])
                if pd.notna(record.get('R (% on 100 MVA)')):
                    new_r = float(record['R (% on 100 MVA)']) / 100.0
                    if new_r != 0.0:
                        network.transformers.at[xfmr_id, 'r'] = new_r
                    else:
                        logger.debug(f"  Skipping zero R for transformer {xfmr_id} (keeping existing value)")
                if pd.notna(record.get('X (% on 100 MVA)')):
                    new_x = float(record['X (% on 100 MVA)']) / 100.0
                    if new_x != 0.0:
                        network.transformers.at[xfmr_id, 'x'] = new_x
                    else:
                        logger.warning(f"  Skipping zero X for transformer {xfmr_id} (keeping existing value)")
                if pd.notna(record.get('B (% on 100 MVA)')):
                    network.transformers.at[xfmr_id, 'b'] = float(record['B (% on 100 MVA)']) / 100.0

                successful += 1

        except Exception as e:
            failure_details.append({'type': 'change', 'reason': 'exception',
                                   'node1': node1, 'node2': node2, 'error': str(e)})
            logger.debug(f"Failed to modify transformer {node1}-{node2}: {e}")
            failed += 1

    return successful, failed, failure_details


def _resolve_hvdc_bus(network: pypsa.Network, node_name: str,
                     logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Resolve an HVDC node name to an existing network bus.

    ETYS B-5-1 uses generic node references like 'PEHE4-' (Peterhead 400kV)
    or 'WALP4-' (Walpole 400kV) that don't correspond exactly to the bus names
    in the ETYS network (which use suffixes like '41', '4J', '4K', etc.).

    Resolution strategy:
    1. Exact match — return immediately
    2. Prefix match — find buses at the same site with the same voltage digit
       (e.g. PEHE4- → PEHE41, PEHE4J, ...)
    3. Site match at any voltage — last resort, find any bus at the same 4-char site

    Args:
        network: PyPSA network
        node_name: Node name from ETYS B-5-1 (e.g. 'DRAX41', 'PEHE4-')
        logger: Optional logger

    Returns:
        Resolved bus name, or None if unresolvable
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    node_name = str(node_name).strip()
    if not node_name or node_name.lower() in ('nan', '', 'tbc', 'n/a'):
        return None

    # 1. Exact match
    if node_name in network.buses.index:
        return node_name

    # 2. Prefix match: same site + same voltage digit
    #    ETYS convention: first 4 chars = site code, 5th char = voltage digit
    if len(node_name) >= 5:
        site = node_name[:4]
        voltage_digit = node_name[4]  # e.g. '4' for 400kV, '2' for 275kV
        candidates = [b for b in network.buses.index
                      if b.startswith(site) and len(b) > 4 and b[4] == voltage_digit]
        if candidates:
            # Prefer the simplest name (e.g. 'PEHE41' over 'PEHE4J')
            candidates.sort(key=len)
            resolved = candidates[0]
            logger.debug(f"  Resolved HVDC bus '{node_name}' → '{resolved}' (prefix match)")
            return resolved

    # 3. Site match at any voltage (fallback)
    site = node_name[:4] if len(node_name) >= 4 else node_name
    site_buses = [b for b in network.buses.index if b.startswith(site)]
    if site_buses:
        # Prefer 400kV buses
        v400 = [b for b in site_buses if len(b) > 4 and b[4] == '4']
        if v400:
            resolved = sorted(v400, key=len)[0]
            logger.debug(f"  Resolved HVDC bus '{node_name}' → '{resolved}' (site 400kV fallback)")
            return resolved
        resolved = sorted(site_buses, key=len)[0]
        logger.debug(f"  Resolved HVDC bus '{node_name}' → '{resolved}' (site any-voltage fallback)")
        return resolved

    return None


def apply_hvdc_additions(network: pypsa.Network,
                         additions: List[Dict],
                         logger: Optional[logging.Logger] = None) -> Tuple[int, int, List[Dict]]:
    """
    Add new HVDC interconnectors/links to the network.

    These come from B-5-1 rows where Existing='No' and Planned from year <= modelled_year.
    Uses the same column conventions as process_ETYS_data.py for the base HVDC links.

    Args:
        network: PyPSA network object
        additions: List of HVDC addition records from ETYS B-5-1
        logger: Optional logger instance

    Returns:
        Tuple of (successful_additions, failed_additions, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    successful, failed = 0, 0
    failure_details = []

    # Track link IDs to make parallel links unique
    link_id_counter = {}

    for record in additions:
        try:
            # B-5-1 uses 'Node 1'/'Node 2' (may have been normalized to 'Node1'/'Node2')
            node1 = str(record.get('Node 1', record.get('Node1', ''))).strip()
            node2 = str(record.get('Node 2', record.get('Node2', ''))).strip()

            if not node1 or not node2 or node1.lower() == 'nan' or node2.lower() == 'nan':
                failure_details.append({'type': 'hvdc_addition', 'reason': 'invalid_node_name',
                                       'node1': node1, 'node2': node2})
                failed += 1
                continue

            # Skip records with invalid/generic node names (offshore platforms, etc.)
            if (any(p in node1.lower() for p in INVALID_NODE_PATTERNS) or
                    any(p in node2.lower() for p in INVALID_NODE_PATTERNS)):
                failure_details.append({'type': 'hvdc_addition', 'reason': 'invalid_node_name',
                                       'node1': node1, 'node2': node2,
                                       'name': record.get('Interconnector Name', 'Unknown')})
                failed += 1
                continue

            # Resolve node names to actual network buses (handles PEHE4- → PEHE4J etc.)
            bus0 = _resolve_hvdc_bus(network, node1, logger)
            bus1 = _resolve_hvdc_bus(network, node2, logger)

            if bus0 is None or bus1 is None:
                missing = [n for n, b in [(node1, bus0), (node2, bus1)] if b is None]
                failure_details.append({'type': 'hvdc_addition', 'reason': 'bus_not_found',
                                       'node1': node1, 'node2': node2, 'missing_buses': missing})
                failed += 1
                continue

            # Use resolved bus names
            node1, node2 = bus0, bus1

            year = record.get('Planned from year', 2025)
            if pd.notna(year):
                year = int(year)

            # Create unique ID
            base_link_id = f"{node1}_{node2}_HVDC_{year}"
            if base_link_id in link_id_counter:
                link_id_counter[base_link_id] += 1
                link_id = f"{base_link_id}_{link_id_counter[base_link_id]}"
            else:
                link_id_counter[base_link_id] = 0
                link_id = base_link_id

            if link_id in network.links.index:
                logger.debug(f"HVDC link {link_id} already exists, skipping")
                continue

            # Extract capacity — B-5-1 uses 'Winter Rating (MVA)' for power capacity
            p_nom_raw = record.get('Winter Rating (MVA)', record.get('Rating (MVA)', DEFAULT_RATINGS['hvdc']))
            p_nom = float(p_nom_raw) if pd.notna(p_nom_raw) else DEFAULT_RATINGS['hvdc']

            network.add('Link', link_id,
                       bus0=node1,
                       bus1=node2,
                       p_nom=p_nom,
                       p_min_pu=-1.0,  # Bidirectional
                       p_max_pu=1.0,
                       carrier='DC',
                       under_construction=False)

            successful += 1
            logger.debug(f"Added HVDC link {link_id}: {node1}-{node2}, {p_nom:.0f} MW")

        except Exception as e:
            failure_details.append({'type': 'hvdc_addition', 'reason': 'exception',
                                   'node1': node1, 'node2': node2, 'error': str(e)})
            failed += 1

    return successful, failed, failure_details


def remove_orphan_buses(network: pypsa.Network, logger: Optional[logging.Logger] = None) -> int:
    """
    Remove buses that have no attached components (orphan buses).
    
    Orphan buses can occur when:
    - Buses are added from upgrade data but their connecting components failed to add
    - Future infrastructure where only partial data is available
    
    Args:
        network: PyPSA network object
        logger: Optional logger instance
        
    Returns:
        Number of buses removed
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Find buses with no attached components — pre-compute connected bus sets
    # for O(1) membership lookup instead of O(B × C) per-bus Series scans
    connected_buses = set()
    for comp_df in [network.generators, network.loads, network.storage_units]:
        if len(comp_df) > 0 and 'bus' in comp_df.columns:
            connected_buses.update(comp_df['bus'].dropna())
    for comp_df in [network.lines, network.transformers, network.links]:
        if len(comp_df) > 0:
            if 'bus0' in comp_df.columns:
                connected_buses.update(comp_df['bus0'].dropna())
            if 'bus1' in comp_df.columns:
                connected_buses.update(comp_df['bus1'].dropna())

    orphan_buses = [b for b in network.buses.index if b not in connected_buses]
    
    if orphan_buses:
        logger.info(f"  Removing {len(orphan_buses)} orphan buses (no attached components)")
        logger.debug(f"  Orphan buses: {orphan_buses[:20]}" + (" ..." if len(orphan_buses) > 20 else ""))
        network.buses.drop(orphan_buses, inplace=True)
    else:
        logger.debug("  No orphan buses found")
    
    return len(orphan_buses)


def log_failure_summary(failures_by_type: Dict[str, List[Dict]], logger: logging.Logger):
    """
    Log a detailed summary of upgrade failures for investigation.
    
    Args:
        failures_by_type: Dictionary mapping component type to list of failure records
        logger: Logger instance
    """
    total_failures = sum(len(failures) for failures in failures_by_type.values())
    
    if total_failures == 0:
        logger.info("  ✓ No failures - all upgrades applied successfully")
        return
    
    logger.info(f"\n  Failure Analysis ({total_failures} total failures):")
    logger.info("  " + "-" * 68)
    
    for component_type, failures in failures_by_type.items():
        if not failures:
            continue
            
        logger.info(f"\n  {component_type.upper()} FAILURES ({len(failures)} total):")
        
        # Group by failure reason
        by_reason = {}
        for f in failures:
            reason = f.get('reason', 'unknown')
            by_reason.setdefault(reason, []).append(f)
        
        for reason, records in sorted(by_reason.items(), key=lambda x: len(x[1]), reverse=True):
            logger.info(f"    • {reason}: {len(records)} cases")
            
            # Log a few examples
            for record in records[:3]:
                node1 = record.get('node1', 'N/A')
                node2 = record.get('node2', 'N/A')
                year = record.get('year', 'N/A')
                
                if reason == 'bus_not_found':
                    missing = ', '.join(record.get('missing_buses', []))
                    logger.debug(f"      - {node1}-{node2} ({year}): missing buses: {missing}")
                elif reason == 'exception':
                    error = record.get('error', 'unknown')
                    logger.debug(f"      - {node1}-{node2} ({year}): {error}")
                else:
                    logger.debug(f"      - {node1}-{node2} ({year})")
            
            if len(records) > 3:
                logger.debug(f"      ... and {len(records) - 3} more")


def apply_etys_network_upgrades(network: pypsa.Network,
                               modelled_year: int,
                               etys_file: str = None,
                               substation_coords_file: Optional[str] = None,
                               logger: Optional[logging.Logger] = None) -> pypsa.Network:
    """
    Apply ETYS network upgrades from Appendix B to the network.

    This function modifies the network topology by:
    1. Adding new circuits commissioned by modelled_year
    2. Removing circuits decommissioned by modelled_year
    3. Modifying circuit ratings
    4. Adding new transformers
    5. (Future) Adding new HVDC interconnectors

    Args:
        network: PyPSA network object (ETYS network expected)
        modelled_year: Target year for upgrades (e.g., 2030, 2035)
        etys_file: Path to ETYS Appendix B Excel file (required)
        substation_coords_file: Optional path to substation_coordinates.csv.
            When provided, used as the highest-priority coordinate source for
            upgrade buses that have no anchor in the base network (e.g. new
            island substations like KERG, GREM, SYEL).
        logger: Optional logger instance

    Returns:
        Modified PyPSA network object
    """
    if etys_file is None:
        raise ValueError(
            "etys_file must be provided. Pass the ETYS Appendix B file path "
            "from the Snakemake rule input (snakemake.input.etys_file)."
        )
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Applying ETYS network upgrades for year {modelled_year}")

    # Load substation coordinates (Strategy 0 lookup for upgrade buses)
    substation_coords: Optional[Dict[str, Tuple[float, float]]] = None
    if substation_coords_file:
        try:
            coords_df = pd.read_csv(substation_coords_file)
            substation_coords = {
                row['site_code']: (float(row['lat']), float(row['lon']))
                for _, row in coords_df.iterrows()
                if pd.notna(row.get('lat')) and pd.notna(row.get('lon'))
            }
            logger.info(f"  Loaded {len(substation_coords)} substation coordinates from {substation_coords_file}")
        except Exception as e:
            logger.warning(f"  Failed to load substation_coords_file '{substation_coords_file}': {e}")

    # Load upgrade data
    upgrades_data = load_etys_upgrade_data(etys_file, logger)

    # Filter by year and categorize
    categorized = filter_upgrades_by_year(upgrades_data, modelled_year, logger)
    
    # Track summary statistics
    summary = {
        'circuits_added': 0,
        'circuits_removed': 0,
        'circuits_modified': 0,
        'circuits_failed': 0,
        'transformers_added': 0,
        'transformers_removed': 0,
        'transformers_modified': 0,
        'transformers_failed': 0,
        'hvdc_added': 0,
        'hvdc_failed': 0,
        'buses_added': 0,
        'buses_removed': 0,
    }
    
    # Track failure details for logging
    all_failures = {
        'circuits': [],
        'transformers': [],
        'hvdc': [],
    }
    
    # STEP 1: Add missing buses that are referenced in the upgrade data
    # This is critical - many upgrades reference new nodes that don't exist in the base network
    logger.info("Step 1: Adding missing buses from upgrade data...")
    buses_added = add_missing_buses_from_upgrades(network, categorized, substation_coords, logger)
    summary['buses_added'] = buses_added
    
    # STEP 2: Apply circuit additions
    if categorized['circuits']['additions']:
        added, failed, failures = apply_circuit_additions(network, categorized['circuits']['additions'], logger)
        summary['circuits_added'] = added
        summary['circuits_failed'] += failed
        all_failures['circuits'].extend(failures)
        logger.info(f"Added {added} circuits ({failed} failed)")
    
    # Apply circuit removals
    if categorized['circuits']['removals']:
        removed, failed, failures = apply_circuit_removals(network, categorized['circuits']['removals'], logger)
        summary['circuits_removed'] = removed
        summary['circuits_failed'] += failed
        all_failures['circuits'].extend(failures)
        logger.info(f"Removed {removed} circuits ({failed} failed)")
    
    # Apply circuit modifications
    if categorized['circuits']['changes']:
        modified, failed, failures = apply_circuit_changes(network, categorized['circuits']['changes'], logger)
        summary['circuits_modified'] = modified
        summary['circuits_failed'] += failed
        all_failures['circuits'].extend(failures)
        logger.info(f"Modified {modified} circuits ({failed} failed)")
    
    # Apply transformer additions
    if categorized['transformers']['additions']:
        added, failed, failures = apply_transformer_additions(network, categorized['transformers']['additions'], logger)
        summary['transformers_added'] = added
        summary['transformers_failed'] += failed
        all_failures['transformers'].extend(failures)
        logger.info(f"Added {added} transformers ({failed} failed)")
    
    # Apply transformer removals
    if categorized['transformers']['removals']:
        removed, failed, failures = apply_transformer_removals(network, categorized['transformers']['removals'], logger)
        summary['transformers_removed'] = removed
        summary['transformers_failed'] += failed
        all_failures['transformers'].extend(failures)
        logger.info(f"Removed {removed} transformers ({failed} failed)")

    # Apply transformer modifications
    if categorized['transformers']['changes']:
        modified, failed, failures = apply_transformer_changes(
            network, categorized['transformers']['changes'], logger)
        summary['transformers_modified'] = modified
        summary['transformers_failed'] += failed
        all_failures['transformers'].extend(failures)
        logger.info(f"Modified {modified} transformers ({failed} failed)")

    # Apply HVDC additions (new interconnectors)
    if categorized['hvdc']['additions']:
        added, failed, failures = apply_hvdc_additions(network, categorized['hvdc']['additions'], logger)
        summary['hvdc_added'] = added
        summary['hvdc_failed'] += failed
        all_failures['hvdc'].extend(failures)
        logger.info(f"Added {added} HVDC links ({failed} failed)")

    # STEP 3: Remove orphan buses
    logger.info("Step 3: Cleaning up orphan buses...")
    buses_removed = remove_orphan_buses(network, logger)
    summary['buses_removed'] = buses_removed
    
    # Log detailed failure analysis
    log_failure_summary(all_failures, logger)
    
    # Log summary
    logger.info("\n" + "="*70)
    logger.info(f"ETYS Network Upgrades Applied (Year {modelled_year})")
    logger.info("="*70)
    logger.info(f"  Buses Added:          {summary['buses_added']:4d}")
    logger.info(f"  Buses Removed:        {summary['buses_removed']:4d} (orphans)")
    logger.info(f"  Circuits Added:       {summary['circuits_added']:4d}")
    logger.info(f"  Circuits Removed:     {summary['circuits_removed']:4d}")
    logger.info(f"  Circuits Modified:    {summary['circuits_modified']:4d}")
    logger.info(f"  Circuits Failed:      {summary['circuits_failed']:4d}")
    logger.info(f"  Transformers Added:   {summary['transformers_added']:4d}")
    logger.info(f"  Transformers Removed: {summary['transformers_removed']:4d}")
    logger.info(f"  Transformers Modified:{summary['transformers_modified']:4d}")
    logger.info(f"  Transformers Failed:  {summary['transformers_failed']:4d}")
    logger.info(f"  HVDC Links Added:     {summary['hvdc_added']:4d}")
    logger.info(f"  HVDC Links Failed:    {summary['hvdc_failed']:4d}")
    logger.info(f"\nNetwork Summary:")
    logger.info(f"  Buses: {len(network.buses)}")
    logger.info(f"  Lines: {len(network.lines)}")
    logger.info(f"  Transformers: {len(network.transformers)}")
    logger.info(f"  Links: {len(network.links)}")
    logger.info("="*70 + "\n")
    
    return network


if __name__ == "__main__":
    # Example usage
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load a network and apply upgrades
    network = pypsa.Network("resources/network/HT35_network.nc")
    network_upgraded = apply_etys_network_upgrades(network, 2035, logger=logger)
    network_upgraded.export_to_netcdf("resources/network/HT35_network_upgraded.nc")

