"""
Enhanced Renewable Profile Integration for PyPSA-GB

This script ensures proper integration of renewable generation profiles with PyPSA networks,
addressing timestep synchronization and data alignment issues.

Key Features:
1. Automatic timestep alignment between renewable profiles and network snapshots
2. Validation of timeseries data consistency
3. Integration with existing PyPSA-GB renewable profile generation
4. Support for multiple renewable technologies (wind, solar, hydro, marine)

Author: PyPSA-GB Development Team  
Date: September 2025
"""

import pandas as pd
import numpy as np
import pypsa
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

# Import the timeseries synchronization utilities
from scripts.utilities.timeseries_synchronization import TimeseriesSynchronizer, create_standard_snapshots

# Configure logging
try:
    from scripts.utilities.logging_config import setup_logging
    logger = setup_logging("renewable_integration")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging
        logger = setup_logging("renewable_integration")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("renewable_integration")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class RenewableProfileIntegrator:
    """
    Integrate renewable generation profiles with PyPSA networks.
    
    This class handles the complete workflow from profile loading to network integration,
    ensuring proper timestep alignment and data validation.
    """
    
    def __init__(self, network_snapshots: pd.DatetimeIndex = None):
        """
        Initialize the integrator.
        
        Parameters
        ----------
        network_snapshots : pd.DatetimeIndex, optional
            Target network snapshots for alignment
        """
        self.synchronizer = TimeseriesSynchronizer(network_snapshots)
        self.renewable_profiles = {}
        self.technology_mapping = {
            'wind_onshore': 'onwind',
            'wind_offshore': 'offwind', 
            'solar_pv': 'solar',
            'hydro_run_of_river': 'ror',
            'hydro_reservoir': 'hydro',
            'marine_tidal': 'tidal',
            'marine_wave': 'wave'
        }
        
    def load_renewable_profiles(self, profiles_config: Dict[str, str]) -> None:
        """
        Load renewable generation profiles from multiple sources.
        
        Parameters
        ----------
        profiles_config : Dict[str, str]
            Mapping of technology names to profile file paths
        """
        logger.info(f"Loading renewable profiles from {len(profiles_config)} sources")
        
        for tech, file_path in profiles_config.items():
            try:
                logger.info(f"Loading {tech} profiles from {file_path}")
                
                if not Path(file_path).exists():
                    logger.warning(f"Profile file not found: {file_path}")
                    continue
                
                # Load profiles with robust error handling
                profiles_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Validate profiles
                validation_report = self.synchronizer.validate_timeseries_index(profiles_df, tech)
                
                if validation_report['valid']:
                    logger.info(f"Successfully loaded {tech}: {profiles_df.shape}")
                else:
                    logger.warning(f"Validation issues in {tech} profiles:")
                    for issue in validation_report['issues']:
                        logger.warning(f"  - {issue}")
                
                self.renewable_profiles[tech] = profiles_df
                
            except Exception as e:
                logger.error(f"Failed to load {tech} profiles: {e}")
                
        logger.info(f"Loaded profiles for {len(self.renewable_profiles)} technologies")
    
    def synchronize_all_profiles(self) -> Dict[str, pd.DataFrame]:
        """
        Synchronize all loaded profiles to network snapshots.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Synchronized profiles for each technology
        """
        logger.info("Synchronizing all renewable profiles")
        
        synchronized_profiles = {}
        
        for tech, profiles_df in self.renewable_profiles.items():
            try:
                logger.info(f"Synchronizing {tech} profiles")
                
                synchronized = self.synchronizer.synchronize_renewable_profiles(
                    profiles_df, profile_type=tech
                )
                
                synchronized_profiles[tech] = synchronized
                logger.info(f"Successfully synchronized {tech}: {synchronized.shape}")
                
            except Exception as e:
                logger.error(f"Failed to synchronize {tech} profiles: {e}")
        
        return synchronized_profiles
    
    def validate_network_compatibility(self, network: pypsa.Network) -> Dict[str, any]:
        """
        Validate renewable profiles compatibility with PyPSA network.
        
        Parameters
        ----------
        network : pypsa.Network
            PyPSA network to validate against
            
        Returns
        -------
        Dict[str, any]
            Validation report
        """
        logger.info("Validating network compatibility")
        
        # Check network snapshots
        network_consistency = self.synchronizer.check_network_timeseries_consistency(network)
        
        # Check generator alignment
        generator_issues = []
        generator_matches = {}
        
        for tech, profiles_df in self.renewable_profiles.items():
            tech_generators = []
            pypsa_carrier = self.technology_mapping.get(tech, tech)
            
            # Find generators of this technology
            if hasattr(network, 'generators') and not network.generators.empty:
                tech_gens = network.generators[network.generators.carrier == pypsa_carrier]
                tech_generators = tech_gens.index.tolist()
            
            # Check profile coverage
            profile_sites = profiles_df.columns.tolist()
            missing_profiles = set(tech_generators) - set(profile_sites)
            extra_profiles = set(profile_sites) - set(tech_generators)
            
            if missing_profiles:
                generator_issues.append(f"{tech}: Missing profiles for {len(missing_profiles)} generators")
            
            if extra_profiles:
                logger.info(f"{tech}: {len(extra_profiles)} extra profiles (not in network)")
            
            generator_matches[tech] = {
                'network_generators': len(tech_generators),
                'available_profiles': len(profile_sites),
                'missing_profiles': len(missing_profiles),
                'coverage_pct': (len(tech_generators) - len(missing_profiles)) / max(len(tech_generators), 1) * 100
            }
        
        validation_report = {
            'network_consistent': network_consistency['consistent'],
            'network_issues': network_consistency['issues'],
            'generator_issues': generator_issues,
            'technology_coverage': generator_matches,
            'total_renewable_generators': sum([m['network_generators'] for m in generator_matches.values()]),
            'overall_coverage_pct': np.mean([m['coverage_pct'] for m in generator_matches.values()]) if generator_matches else 0
        }
        
        return validation_report
    
    def integrate_with_network(self, network: pypsa.Network, 
                             synchronized_profiles: Dict[str, pd.DataFrame] = None) -> None:
        """
        Integrate synchronized profiles with PyPSA network.
        
        Parameters
        ----------
        network : pypsa.Network
            Target PyPSA network
        synchronized_profiles : Dict[str, pd.DataFrame], optional
            Pre-synchronized profiles. If None, will synchronize loaded profiles.
        """
        logger.info("Integrating renewable profiles with PyPSA network")
        
        if synchronized_profiles is None:
            synchronized_profiles = self.synchronize_all_profiles()
        
        # Set network snapshots if needed
        if len(network.snapshots) <= 1 and self.synchronizer.reference_snapshots is not None:
            network.set_snapshots(self.synchronizer.reference_snapshots)
            logger.info(f"Set network snapshots: {len(network.snapshots)} periods")
        
        # Integrate each technology
        total_added = 0
        
        for tech, profiles_df in synchronized_profiles.items():
            pypsa_carrier = self.technology_mapping.get(tech, tech)
            
            # Find generators of this technology
            if hasattr(network, 'generators') and not network.generators.empty:
                tech_generators = network.generators[network.generators.carrier == pypsa_carrier]
                
                # Add timeseries for each generator
                for gen_name in tech_generators.index:
                    if gen_name in profiles_df.columns:
                        try:
                            # Add p_max_pu timeseries
                            network.generators_t.p_max_pu[gen_name] = profiles_df[gen_name]
                            total_added += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to add timeseries for {gen_name}: {e}")
                    else:
                        logger.debug(f"No profile available for generator {gen_name}")
        
        logger.info(f"Successfully integrated timeseries for {total_added} generators")
        
        # Final validation
        final_validation = self.synchronizer.check_network_timeseries_consistency(network)
        if final_validation['consistent']:
            logger.info("Network timeseries integration completed successfully")
        else:
            logger.warning("Network timeseries consistency issues remain:")
            for issue in final_validation['issues']:
                logger.warning(f"  - {issue}")
    
    def export_synchronized_profiles(self, output_dir: str, 
                                   synchronized_profiles: Dict[str, pd.DataFrame] = None) -> None:
        """
        Export synchronized profiles to CSV files.
        
        Parameters
        ----------
        output_dir : str
            Output directory for synchronized profiles
        synchronized_profiles : Dict[str, pd.DataFrame], optional
            Profiles to export. If None, will synchronize loaded profiles.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if synchronized_profiles is None:
            synchronized_profiles = self.synchronize_all_profiles()
        
        logger.info(f"Exporting synchronized profiles to {output_dir}")
        
        for tech, profiles_df in synchronized_profiles.items():
            output_file = output_path / f"{tech}_synchronized_profiles.csv"
            profiles_df.to_csv(output_file)
            logger.info(f"Exported {tech} profiles: {output_file}")
        
        # Export summary report
        summary_file = output_path / "synchronization_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Renewable Profile Synchronization Summary\\n")
            f.write("=" * 50 + "\\n\\n")
            
            for tech, profiles_df in synchronized_profiles.items():
                f.write(f"{tech}:\\n")
                f.write(f"  Shape: {profiles_df.shape}\\n")
                f.write(f"  Period: {profiles_df.index[0]} to {profiles_df.index[-1]}\\n")
                f.write(f"  Frequency: {profiles_df.index.freq or 'Irregular'}\\n")
                f.write("\\n")
        
        logger.info(f"Exported synchronization summary: {summary_file}")


def integrate_renewable_profiles_with_network(network_file: str, profiles_config: Dict[str, str],
                                            output_network_file: str = None,
                                            snapshots_config: Dict[str, str] = None) -> pypsa.Network:
    """
    Complete workflow to integrate renewable profiles with PyPSA network.
    
    Parameters
    ----------
    network_file : str
        Path to input PyPSA network file
    profiles_config : Dict[str, str]
        Mapping of technology names to profile file paths
    output_network_file : str, optional
        Path to save integrated network
    snapshots_config : Dict[str, str], optional
        Configuration for network snapshots (start_date, end_date, freq)
        
    Returns
    -------
    pypsa.Network
        Network with integrated renewable profiles
    """
    logger.info("Starting renewable profile integration workflow")
    
    # Load network
    logger.info(f"Loading network from {network_file}")
    network = pypsa.Network(network_file)
    
    # Set up snapshots if specified
    if snapshots_config:
        snapshots = create_standard_snapshots(
            start_date=snapshots_config['start_date'],
            end_date=snapshots_config['end_date'],
            freq=snapshots_config.get('freq', '1H')
        )
        network.set_snapshots(snapshots)
    
    # Initialize integrator
    integrator = RenewableProfileIntegrator(network.snapshots)
    
    # Load profiles
    integrator.load_renewable_profiles(profiles_config)
    
    # Validate compatibility
    validation_report = integrator.validate_network_compatibility(network)
    logger.info(f"Network validation: Overall coverage {validation_report['overall_coverage_pct']:.1f}%")
    
    if validation_report['generator_issues']:
        logger.warning("Generator coverage issues:")
        for issue in validation_report['generator_issues']:
            logger.warning(f"  - {issue}")
    
    # Integrate profiles
    integrator.integrate_with_network(network)
    
    # Save integrated network
    if output_network_file:
        network.export_to_netcdf(output_network_file)
        logger.info(f"Saved integrated network to {output_network_file}")
    
    logger.info("Renewable profile integration completed successfully")
    return network


if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    test_profiles_config = {
        'wind_onshore': 'resources/renewable/profiles/onshore_wind_profiles.csv',
        'wind_offshore': 'resources/renewable/profiles/offshore_wind_profiles.csv',
        'solar_pv': 'resources/renewable/profiles/solar_pv_profiles.csv'
    }
    
    test_snapshots_config = {
        'start_date': '2020-01-01',
        'end_date': '2020-01-07',
        'freq': '1H'
    }
    
    # Note: This would need actual files to run
    print("Renewable profile integration module loaded successfully")
    print("Use integrate_renewable_profiles_with_network() for complete workflow")

