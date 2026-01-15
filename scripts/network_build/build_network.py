import pandas as pd
import numpy as np
import pypsa
import logging
import time
from pathlib import Path
from scripts.utilities.logging_config import setup_logging, log_dataframe_info, log_network_info, log_execution_summary


def build_reduced_network(logger=None):
    """
    Build network using the Reduced network model.
    Based on reduced_network data.
    """
    start_time = time.time()
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Building Reduced network model")
    
    # Load buses
    buses_file = Path("data/network/reduced_network/buses.csv")
    if not buses_file.exists():
        raise FileNotFoundError(f"Buses file not found: {buses_file}")
    
    logger.info(f"Loading buses from {buses_file}")
    df_buses = pd.read_csv(buses_file, index_col=0)
    log_dataframe_info(df_buses, logger, "Buses data")
    
    # Load lines
    lines_file = Path("data/network/reduced_network/lines.csv")
    if not lines_file.exists():
        raise FileNotFoundError(f"Lines file not found: {lines_file}")
    
    logger.info(f"Loading lines from {lines_file}")
    df_lines = pd.read_csv(lines_file, index_col=0)
    log_dataframe_info(df_lines, logger, "Lines data")
    
    # Create network
    logger.info("Creating PyPSA network")
    network = pypsa.Network()
    network.set_snapshots(range(1))
    
    # Add buses
    logger.info(f"Adding {len(df_buses)} buses to network")
    for bus_id, bus_data in df_buses.iterrows():
        network.add("Bus", bus_id, **bus_data.to_dict())
    
    # Add lines
    logger.info(f"Adding {len(df_lines)} lines to network")
    for line_id, line_data in df_lines.iterrows():
        network.add("Line", line_id, **line_data.to_dict())
    
    # Set metadata
    network.name = 'Reduced Network'
    if 'country' not in network.buses.columns:
        network.buses['country'] = 'GB'
    
    # Add carrier if not exists
    if len(network.carriers) == 0:
        network.add("Carrier", "AC")
    
    # Consistency check
    try:
        network.consistency_check()
        logger.info("Reduced network passed consistency check")
    except Exception as e:
        logger.warning(f"Reduced network consistency issues: {e}")
    
    # Log execution summary
    log_execution_summary(
        logger, "Build Reduced Network", start_time,
        inputs=[str(buses_file), str(lines_file)],
        outputs=[]
    )
    
    return network

def build_zonal_network(logger=None):
    """
    Build network using the Zonal network model.
    Based on zonal data using links instead of lines.
    """
    start_time = time.time()
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Building Zonal network model")
    
    # Load zones (buses)
    zones_file = Path("data/network/zonal/buses.csv")
    if not zones_file.exists():
        raise FileNotFoundError(f"Zones file not found: {zones_file}")
    
    logger.info(f"Loading zones from {zones_file}")
    df_buses = pd.read_csv(zones_file, index_col=0)
    log_dataframe_info(df_buses, logger, "Zones data")
    
    # Load links
    links_file = Path("data/network/zonal/links.csv")
    if not links_file.exists():
        raise FileNotFoundError(f"Links file not found: {links_file}")
    
    logger.info(f"Loading links from {links_file}")
    df_links = pd.read_csv(links_file, index_col=0)
    log_dataframe_info(df_links, logger, "Links data")
    
    # Create network
    logger.info("Creating PyPSA network")
    network = pypsa.Network()
    network.set_snapshots(range(1))
    
    # Add buses (zones)
    logger.info(f"Adding {len(df_buses)} zones to network")
    valid_bus_ids = set(df_buses.index)
    for bus_id, bus_data in df_buses.iterrows():
        network.add("Bus", bus_id, **bus_data.to_dict())
    
    # Filter links to only include those connecting valid buses (internal zone links)
    # External interconnector links (to non-existent buses) will be added by add_interconnectors_to_network
    internal_links = df_links[
        df_links['bus0'].isin(valid_bus_ids) & df_links['bus1'].isin(valid_bus_ids)
    ]
    external_links = df_links[
        ~(df_links['bus0'].isin(valid_bus_ids) & df_links['bus1'].isin(valid_bus_ids))
    ]
    
    if len(external_links) > 0:
        logger.info(f"Filtering out {len(external_links)} external interconnector links (will be added separately)")
        for link_id in external_links.index:
            logger.debug(f"  Skipping external link: {link_id}")
    
    # Add internal links only
    logger.info(f"Adding {len(internal_links)} internal zonal links to network")
    for link_id, link_data in internal_links.iterrows():
        network.add("Link", link_id, **link_data.to_dict())
    
    # Set metadata
    network.name = 'Zonal Network'
    if 'country' not in network.buses.columns:
        network.buses['country'] = 'GB'
    
    # Add carrier if not exists
    if len(network.carriers) == 0:
        network.add("Carrier", "AC")
    
    # Consistency check
    try:
        network.consistency_check()
        logger.info("Zonal network passed consistency check")
    except Exception as e:
        logger.warning(f"Zonal network consistency issues: {e}")
    
    # Log execution summary
    log_execution_summary(
        logger, "Build Zonal Network", start_time,
        inputs=[str(zones_file), str(links_file)],
        outputs=[]
    )
    
    return network

def build_ETYS_network(logger=None):
    """
    Build network using the ETYS network model.
    For ETYS, we import the network directly since it's already built.
    """
    start_time = time.time()
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Building ETYS network model")
    logger.info("For ETYS networks, the network should be built by the ETYS_network.py script directly")
    logger.info("This function is a placeholder - ETYS network building handled by rule build_ETYS_network")
    
    # This shouldn't be reached in normal operation
    raise NotImplementedError("ETYS network building should use the dedicated ETYS_network.py script")

# Registry of network builders
NETWORK_BUILDERS = {
    'ETYS': build_ETYS_network,
    'Reduced': build_reduced_network,
    'Zonal': build_zonal_network
}

def build_network_by_type(network_model: str, logger=None):
    """
    Build network based on the specified model type.
    
    Parameters
    ----------
    network_model : str
        Network model type ('ETYS', 'Reduced', 'Zonal')
    logger : logging.Logger, optional
        Logger instance
        
    Returns
    -------
    pypsa.Network
        Built network
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if network_model not in NETWORK_BUILDERS:
        raise ValueError(f"Unknown network model: {network_model}. "
                        f"Available models: {list(NETWORK_BUILDERS.keys())}")
    
    logger.info(f"Building network using {network_model} model")
    builder_func = NETWORK_BUILDERS[network_model]
    network = builder_func(logger)
    
    # Set version metadata to prevent compatibility warnings
    network.meta = {"pypsa_version": pypsa.__version__}
    
    return network

if __name__ == "__main__":
    # Set up logging using centralized system, writing to Snakemake log if available
    log_path = None
    if 'snakemake' in globals() and hasattr(snakemake, 'log') and snakemake.log:
        log_path = snakemake.log[0]
    logger = setup_logging(log_path or "build_network")
    
    # Get network model from snakemake params
    network_model = snakemake.params.network_model[0]
    
    logger.info("="*50)
    logger.info(f"STARTING NETWORK CREATION - {network_model} MODEL")
    
    try:
        # Build the network using flexible system (Reduced/Zonal)
        network = build_network_by_type(network_model, logger)
        
        # Export network
        logger.info(f"Exporting network to {snakemake.output[0]}")
        network.export_to_netcdf(snakemake.output[0])
        
        log_network_info(network, logger)
        logger.info(f"{network_model} NETWORK CREATION COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logger.error(f"FATAL ERROR in {network_model} network creation: {e}")
        raise

