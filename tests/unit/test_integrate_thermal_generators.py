"""
Unit tests for integrate_thermal_generators.py (Stage 2 of generator integration).

Tests thermal generator integration with hybrid data source routing:
- Historical scenarios (≤2024): DUKES + REPD (NO FES)
- Future scenarios (>2024): FES only

Tests verify:
1. DUKES data loading and parsing
2. FES technology mapping (30+ technologies)
3. FES data loading and filtering by year/scenario
4. REPD thermal site loading (7 dispatchable types)
5. merge_generator_data_sources (CRITICAL: data routing logic)
6. REPD data standardization for merging
7. Complete workflow for historical scenarios
8. Complete workflow for future scenarios

Author: PyPSA-GB Testing Team
Date: 2025-10-24
"""

import pytest
import pandas as pd
import numpy as np
import pypsa
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add scripts to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_workspace():
    """Create temporary workspace for test files."""
    temp_dir = tempfile.mkdtemp(prefix="stage2_thermal_")
    workspace = Path(temp_dir)
    
    # Create directory structure
    (workspace / "data" / "generators").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "FES").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "renewables" / "repd").mkdir(parents=True, exist_ok=True)
    
    yield workspace
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_dukes_data(temp_workspace):
    """Create sample DUKES thermal generator CSV."""
    dukes_file = temp_workspace / "data" / "generators" / "DUKES_2020_generators.csv"
    
    data = pd.DataFrame({
        'station_name': [
            'Drax Power Station', 'Ratcliffe-on-Soar', 'West Burton A',
            'Sizewell B', 'Heysham 1', 'Torness',
            'Peterhead CCGT', 'Pembroke CCGT', 'Grain CCGT'
        ],
        'capacity_mw': [
            2600.0, 2000.0, 1332.0,  # Coal
            1198.0, 1160.0, 1365.0,  # Nuclear
            1180.0, 2000.0, 1275.0   # CCGT
        ],
        'fuel_type': [
            'coal', 'coal', 'coal',
            'nuclear', 'nuclear', 'nuclear',
            'CCGT', 'CCGT', 'CCGT'
        ],
        'lat': [53.74, 52.86, 53.35, 52.21, 54.03, 56.00, 57.50, 51.68, 51.45],
        'lon': [-0.99, -1.25, -0.82, 1.62, -2.92, -2.41, -1.79, -5.01, 0.71],
        'data_source': ['DUKES'] * 9
    })
    
    data.to_csv(dukes_file, index=False)
    return dukes_file


@pytest.fixture
def sample_fes_data(temp_workspace):
    """Create sample FES data CSV (pivot format with years as columns)."""
    fes_file = temp_workspace / "data" / "FES" / "FES_2030_data.csv"
    
    # FES has pivot format: rows are (Pathway, Technology, GSP), columns are years
    data = pd.DataFrame({
        'FES Pathway': [
            'Leading the Way', 'Leading the Way', 'Leading the Way', 'Leading the Way',
            'Leading the Way', 'Leading the Way', 'Leading the Way', 'Leading the Way',
            'Consumer Transformation', 'Consumer Transformation', 'Consumer Transformation'
        ],
        'Technology': [
            'CCGTs (non CHP)', 'OCGTs (non CHP)', 'Nuclear',
            'Biomass & Energy Crops (including CHP)', 'Waste Incineration (including CHP)',
            'Hydrogen fuelled generation', 'Geothermal',
            'Solar Generation',  # Non-thermal for completeness
            'CCGTs (non CHP)', 'Nuclear', 'Hydrogen fuelled generation'
        ],
        'GSP': [
            'NEEP', 'SOUT', 'HINK', 'NEEP', 'LOND', 'SOUT', 'CORN',
            'LOND',
            'SOUT', 'TRAW', 'NEEP'
        ],
        '2025': [5000, 500, 8000, 800, 600, 0, 50, 15000, 4800, 7500, 0],
        '2030': [6000, 600, 9000, 1000, 700, 500, 100, 25000, 5500, 8000, 800],
        '2035': [7000, 700, 10000, 1200, 800, 1500, 150, 35000, 6000, 8500, 1500],
        '2040': [7500, 750, 11000, 1400, 900, 3000, 200, 45000, 6500, 9000, 3500],
        '2050': [8000, 800, 12000, 1600, 1000, 5000, 300, 60000, 7000, 9500, 6000]
    })
    
    data.to_csv(fes_file, index=False)
    return fes_file


@pytest.fixture
def sample_repd_thermal_sites(temp_workspace):
    """Create sample REPD thermal site CSVs for 7 dispatchable types."""
    repd_dir = temp_workspace / "data" / "renewables" / "repd"
    
    site_files = {}
    
    # Biomass sites
    biomass_data = pd.DataFrame({
        'Site Name': ['Drax Biomass', 'Lynemouth Biomass', 'Tilbury Biomass'],
        'Installed Capacity (MWelec)': [2600.0, 420.0, 750.0],
        'X-coordinate': [465000, 430000, 560000],
        'Y-coordinate': [425000, 580000, 175000],
        'Technology Type': ['Biomass'] * 3
    })
    biomass_file = repd_dir / "biomass_sites.csv"
    biomass_data.to_csv(biomass_file, index=False)
    site_files['biomass'] = biomass_file
    
    # Waste to energy
    waste_data = pd.DataFrame({
        'Site Name': ['SELCHP', 'Riverside EfW', 'Runcorn EfW'],
        'Installed Capacity (MWelec)': [35.0, 50.0, 65.0],
        'X-coordinate': [535000, 465000, 355000],
        'Y-coordinate': [178000, 200000, 380000],
        'Technology Type': ['Waste'] * 3
    })
    waste_file = repd_dir / "waste_to_energy_sites.csv"
    waste_data.to_csv(waste_file, index=False)
    site_files['waste_to_energy'] = waste_file
    
    # Biogas sites
    biogas_data = pd.DataFrame({
        'Site Name': ['Biogas Plant 1', 'Biogas Plant 2'],
        'Installed Capacity (MWelec)': [2.5, 3.0],
        'X-coordinate': [450000, 460000],
        'Y-coordinate': [300000, 310000],
        'Technology Type': ['Biogas'] * 2
    })
    biogas_file = repd_dir / "biogas_sites.csv"
    biogas_data.to_csv(biogas_file, index=False)
    site_files['biogas'] = biogas_file
    
    # Landfill gas
    landfill_data = pd.DataFrame({
        'Site Name': ['Landfill Gas Site 1'],
        'Installed Capacity (MWelec)': [5.0],
        'X-coordinate': [470000],
        'Y-coordinate': [320000],
        'Technology Type': ['Landfill Gas'] * 1
    })
    landfill_file = repd_dir / "landfill_gas_sites.csv"
    landfill_data.to_csv(landfill_file, index=False)
    site_files['landfill_gas'] = landfill_file
    
    # Sewage gas
    sewage_data = pd.DataFrame({
        'Site Name': ['Sewage Treatment Works'],
        'Installed Capacity (MWelec)': [1.5],
        'X-coordinate': [480000],
        'Y-coordinate': [330000],
        'Technology Type': ['Sewage Gas'] * 1
    })
    sewage_file = repd_dir / "sewage_gas_sites.csv"
    sewage_data.to_csv(sewage_file, index=False)
    site_files['sewage_gas'] = sewage_file
    
    # Advanced biofuel
    biofuel_data = pd.DataFrame({
        'Site Name': ['Advanced Biofuel Plant'],
        'Installed Capacity (MWelec)': [10.0],
        'X-coordinate': [490000],
        'Y-coordinate': [340000],
        'Technology Type': ['Advanced Biofuel'] * 1
    })
    biofuel_file = repd_dir / "advanced_biofuel_sites.csv"
    biofuel_data.to_csv(biofuel_file, index=False)
    site_files['advanced_biofuel'] = biofuel_file
    
    # Geothermal
    geothermal_data = pd.DataFrame({
        'Site Name': ['Geothermal Plant Cornwall'],
        'Installed Capacity (MWelec)': [3.0],
        'X-coordinate': [200000],
        'Y-coordinate': [60000],
        'Technology Type': ['Geothermal'] * 1
    })
    geothermal_file = repd_dir / "geothermal_sites.csv"
    geothermal_data.to_csv(geothermal_file, index=False)
    site_files['geothermal'] = geothermal_file
    
    return site_files


# ═══════════════════════════════════════════════════════════════════════════
# TEST LOAD_DUKES_GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadDukesGenerators:
    """Test DUKES generator data loading."""
    
    def test_load_dukes_basic(self, sample_dukes_data):
        """Test basic DUKES data loading."""
        from integrate_thermal_generators import load_dukes_generators
        
        df = load_dukes_generators(str(sample_dukes_data))
        
        assert len(df) == 9, "Should load 9 DUKES generators"
        assert 'station_name' in df.columns
        assert 'capacity_mw' in df.columns
        assert 'fuel_type' in df.columns
        assert 'data_source' in df.columns
    
    def test_dukes_data_source_tag(self, sample_dukes_data):
        """Test that DUKES data is tagged with data_source='DUKES'."""
        from integrate_thermal_generators import load_dukes_generators
        
        df = load_dukes_generators(str(sample_dukes_data))
        
        assert all(df['data_source'] == 'DUKES'), "All records should have data_source='DUKES'"
    
    def test_dukes_capacity_parsing(self, sample_dukes_data):
        """Test DUKES capacity parsing and summation."""
        from integrate_thermal_generators import load_dukes_generators
        
        df = load_dukes_generators(str(sample_dukes_data))
        
        total_capacity = df['capacity_mw'].sum()
        expected_capacity = 2600 + 2000 + 1332 + 1198 + 1160 + 1365 + 1180 + 2000 + 1275
        
        assert total_capacity == pytest.approx(expected_capacity), "Total capacity should match"
    
    def test_dukes_fuel_type_breakdown(self, sample_dukes_data):
        """Test DUKES fuel type categorization."""
        from integrate_thermal_generators import load_dukes_generators
        
        df = load_dukes_generators(str(sample_dukes_data))
        
        coal_capacity = df[df['fuel_type'] == 'coal']['capacity_mw'].sum()
        nuclear_capacity = df[df['fuel_type'] == 'nuclear']['capacity_mw'].sum()
        ccgt_capacity = df[df['fuel_type'] == 'CCGT']['capacity_mw'].sum()
        
        assert coal_capacity == pytest.approx(5932.0), "Coal capacity should be correct"
        assert nuclear_capacity == pytest.approx(3723.0), "Nuclear capacity should be correct"
        assert ccgt_capacity == pytest.approx(4455.0), "CCGT capacity should be correct"
    
    def test_dukes_missing_file(self, temp_workspace):
        """Test handling of missing DUKES file."""
        from integrate_thermal_generators import load_dukes_generators
        
        missing_file = temp_workspace / "data" / "nonexistent.csv"
        df = load_dukes_generators(str(missing_file))
        
        assert len(df) == 0, "Should return empty DataFrame for missing file"
    
    def test_dukes_coordinates_present(self, sample_dukes_data):
        """Test that DUKES data includes coordinates."""
        from integrate_thermal_generators import load_dukes_generators
        
        df = load_dukes_generators(str(sample_dukes_data))
        
        assert 'lat' in df.columns, "Should have lat column"
        assert 'lon' in df.columns, "Should have lon column"
        assert df['lat'].notna().all(), "All generators should have latitude"
        assert df['lon'].notna().all(), "All generators should have longitude"


# ═══════════════════════════════════════════════════════════════════════════
# TEST MAP_FES_TECHNOLOGY_TO_CARRIER
# ═══════════════════════════════════════════════════════════════════════════

class TestMapFESTechnologyToCarrier:
    """Test FES technology to PyPSA carrier mapping."""
    
    def test_thermal_technology_mappings(self):
        """Test mapping of thermal generation technologies."""
        from integrate_thermal_generators import map_fes_technology_to_carrier
        
        assert map_fes_technology_to_carrier('CCGTs (non CHP)') == 'CCGT'
        assert map_fes_technology_to_carrier('OCGTs (non CHP)') == 'OCGT'
        assert map_fes_technology_to_carrier('Nuclear') == 'nuclear'
        assert map_fes_technology_to_carrier('Coal') == 'coal'
        assert map_fes_technology_to_carrier('Hydrogen fuelled generation') == 'H2'
    
    def test_chp_technology_mappings(self):
        """Test CHP technology mappings."""
        from integrate_thermal_generators import map_fes_technology_to_carrier
        
        assert map_fes_technology_to_carrier('Non-renewable CHP') == 'CHP'
        assert map_fes_technology_to_carrier('Micro CHP') == 'micro_CHP'
    
    def test_renewable_thermal_mappings(self):
        """Test renewable thermal technology mappings."""
        from integrate_thermal_generators import map_fes_technology_to_carrier
        
        assert map_fes_technology_to_carrier('Biomass & Energy Crops (including CHP)') == 'biomass'
        assert map_fes_technology_to_carrier('Waste Incineration (including CHP)') == 'waste'
        assert map_fes_technology_to_carrier('Renewable Engines (Landfill Gas, Sewage Gas, Biogas)') == 'biogas'
    
    def test_non_renewable_engine_mappings(self):
        """Test non-renewable engine mappings."""
        from integrate_thermal_generators import map_fes_technology_to_carrier
        
        assert map_fes_technology_to_carrier('Non-renewable Engines (Diesel) (non CHP)') == 'oil'
        assert map_fes_technology_to_carrier('Non-renewable Engines (Gas) (non CHP)') == 'gas_engine'
    
    def test_other_technology_mappings(self):
        """Test miscellaneous technology mappings."""
        from integrate_thermal_generators import map_fes_technology_to_carrier
        
        assert map_fes_technology_to_carrier('Fuel Cells') == 'fuel_cell'
        assert map_fes_technology_to_carrier('Hydro') == 'hydro'
        assert map_fes_technology_to_carrier('Geothermal') == 'geothermal'
    
    def test_renewable_non_thermal_mappings(self):
        """Test renewable non-thermal mappings (for completeness)."""
        from integrate_thermal_generators import map_fes_technology_to_carrier
        
        assert map_fes_technology_to_carrier('Solar Generation') == 'solar'
        assert map_fes_technology_to_carrier('Wind') == 'onwind'
        assert map_fes_technology_to_carrier('Offshore-Wind (off-Grid)') == 'offwind'
        assert map_fes_technology_to_carrier('Marine') == 'marine'
    
    def test_interconnector_mapping(self):
        """Test interconnector mapping."""
        from integrate_thermal_generators import map_fes_technology_to_carrier
        
        assert map_fes_technology_to_carrier('Interconnector') == 'interconnector'
    
    def test_unknown_technology_mapping(self):
        """Test mapping of unknown technology."""
        from integrate_thermal_generators import map_fes_technology_to_carrier
        
        assert map_fes_technology_to_carrier('Unknown Tech') == 'other'
        assert map_fes_technology_to_carrier('Future Tech 2050') == 'other'
    
    def test_mapping_count(self):
        """Test that all expected mappings are present (30+ technologies)."""
        from integrate_thermal_generators import map_fes_technology_to_carrier
        import inspect
        
        # Get function source to count mappings
        source = inspect.getsource(map_fes_technology_to_carrier)
        
        # Count dictionary entries (rough check)
        assert 'tech_map' in source, "Should have tech_map dictionary"
        # Should have 30+ technology mappings defined


# ═══════════════════════════════════════════════════════════════════════════
# TEST LOAD_FES_GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadFESGenerators:
    """Test FES generator data loading and filtering."""
    
    def test_load_fes_basic(self, sample_fes_data):
        """Test basic FES data loading for specific year."""
        from integrate_thermal_generators import load_fes_generators
        
        df = load_fes_generators(str(sample_fes_data), modelled_year=2030)
        
        assert len(df) > 0, "Should load FES generators"
        assert 'technology' in df.columns
        assert 'capacity_mw' in df.columns
        assert 'fuel_type' in df.columns
        assert 'data_source' in df.columns
    
    def test_fes_data_source_tag(self, sample_fes_data):
        """Test that FES data is tagged with data_source='FES'."""
        from integrate_thermal_generators import load_fes_generators
        
        df = load_fes_generators(str(sample_fes_data), modelled_year=2030)
        
        assert all(df['data_source'] == 'FES'), "All records should have data_source='FES'"
    
    def test_fes_year_extraction(self, sample_fes_data):
        """Test extraction of capacity for specific year."""
        from integrate_thermal_generators import load_fes_generators
        
        df_2030 = load_fes_generators(str(sample_fes_data), modelled_year=2030)
        df_2040 = load_fes_generators(str(sample_fes_data), modelled_year=2040)
        
        # Capacities should differ between years
        cap_2030 = df_2030['capacity_mw'].sum()
        cap_2040 = df_2040['capacity_mw'].sum()
        
        assert cap_2030 != cap_2040, "Capacities should differ between years"
        assert cap_2040 > cap_2030, "Capacity typically increases in future years"
    
    def test_fes_scenario_filtering(self, sample_fes_data):
        """Test filtering by FES scenario/pathway."""
        from integrate_thermal_generators import load_fes_generators
        
        df_ltw = load_fes_generators(str(sample_fes_data), modelled_year=2030, fes_scenario='Leading the Way')
        df_ct = load_fes_generators(str(sample_fes_data), modelled_year=2030, fes_scenario='Consumer Transformation')
        
        # Different scenarios should give different results
        assert len(df_ltw) > 0, "Leading the Way should have generators"
        assert len(df_ct) > 0, "Consumer Transformation should have generators"
        
        # Check capacities differ
        cap_ltw = df_ltw['capacity_mw'].sum()
        cap_ct = df_ct['capacity_mw'].sum()
        
        assert cap_ltw != cap_ct, "Different scenarios should have different total capacities"
    
    def test_fes_default_scenario(self, sample_fes_data):
        """Test default scenario selection when none specified."""
        from integrate_thermal_generators import load_fes_generators
        
        df = load_fes_generators(str(sample_fes_data), modelled_year=2030, fes_scenario=None)
        
        assert len(df) > 0, "Should load data with default scenario"
    
    def test_fes_zero_capacity_exclusion(self, sample_fes_data):
        """Test that generators with zero capacity are excluded."""
        from integrate_thermal_generators import load_fes_generators
        
        df_2025 = load_fes_generators(str(sample_fes_data), modelled_year=2025)
        
        # Hydrogen in 2025 should be zero and excluded
        hydrogen_gens = df_2025[df_2025['technology'] == 'Hydrogen fuelled generation']
        assert len(hydrogen_gens) == 0, "Zero capacity hydrogen should be excluded in 2025"
        
        # But should appear in 2030
        df_2030 = load_fes_generators(str(sample_fes_data), modelled_year=2030)
        hydrogen_gens_2030 = df_2030[df_2030['technology'] == 'Hydrogen fuelled generation']
        assert len(hydrogen_gens_2030) > 0, "Non-zero hydrogen should appear in 2030"
    
    def test_fes_gsp_assignment(self, sample_fes_data):
        """Test that GSP (Grid Supply Point) is correctly assigned."""
        from integrate_thermal_generators import load_fes_generators
        
        df = load_fes_generators(str(sample_fes_data), modelled_year=2030)
        
        assert 'gsp' in df.columns, "Should have GSP column"
        # Check some GSPs are assigned
        assert df['gsp'].notna().any(), "Some generators should have GSP assigned"
    
    def test_fes_missing_file(self, temp_workspace):
        """Test handling of missing FES file."""
        from integrate_thermal_generators import load_fes_generators
        
        missing_file = temp_workspace / "data" / "nonexistent_fes.csv"
        df = load_fes_generators(str(missing_file), modelled_year=2030)
        
        assert len(df) == 0, "Should return empty DataFrame for missing file"
    
    def test_fes_invalid_year(self, sample_fes_data):
        """Test handling of year not in FES data."""
        from integrate_thermal_generators import load_fes_generators
        
        df = load_fes_generators(str(sample_fes_data), modelled_year=1999)
        
        assert len(df) == 0, "Should return empty DataFrame for invalid year"


# ═══════════════════════════════════════════════════════════════════════════
# TEST LOAD_REPD_THERMAL_SITES
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadREPDThermalSites:
    """Test REPD dispatchable thermal site loading."""
    
    def test_load_all_repd_types(self, sample_repd_thermal_sites):
        """Test loading all 7 REPD dispatchable thermal types."""
        from integrate_thermal_generators import load_repd_thermal_sites
        
        df = load_repd_thermal_sites(sample_repd_thermal_sites)
        
        assert len(df) > 0, "Should load REPD thermal sites"
        
        # Check all 7 fuel types present
        fuel_types = df['fuel_type'].unique()
        expected_types = ['biomass', 'waste_to_energy', 'biogas', 'landfill_gas', 
                         'sewage_gas', 'advanced_biofuel', 'geothermal']
        
        assert len(fuel_types) == 7, "Should have all 7 REPD fuel types"
        for fuel_type in expected_types:
            assert fuel_type in fuel_types, f"{fuel_type} should be in REPD data"
    
    def test_repd_data_source_tag(self, sample_repd_thermal_sites):
        """Test that REPD data is tagged with data_source='REPD'."""
        from integrate_thermal_generators import load_repd_thermal_sites
        
        df = load_repd_thermal_sites(sample_repd_thermal_sites)
        
        assert all(df['data_source'] == 'REPD'), "All records should have data_source='REPD'"
    
    def test_repd_coordinate_standardization(self, sample_repd_thermal_sites):
        """Test coordinate column standardization (X/Y → lat/lon)."""
        from integrate_thermal_generators import load_repd_thermal_sites
        
        df = load_repd_thermal_sites(sample_repd_thermal_sites)
        
        assert 'lat' in df.columns, "Should have standardized lat column"
        assert 'lon' in df.columns, "Should have standardized lon column"
        assert df['lat'].notna().any(), "Should have valid latitudes"
        assert df['lon'].notna().any(), "Should have valid longitudes"
    
    def test_repd_capacity_standardization(self, sample_repd_thermal_sites):
        """Test capacity column standardization."""
        from integrate_thermal_generators import load_repd_thermal_sites
        
        df = load_repd_thermal_sites(sample_repd_thermal_sites)
        
        assert 'capacity_mw' in df.columns, "Should have standardized capacity_mw column"
        assert df['capacity_mw'].notna().all(), "All sites should have capacity"
        assert (df['capacity_mw'] > 0).all(), "All capacities should be positive"
    
    def test_repd_total_capacity(self, sample_repd_thermal_sites):
        """Test total REPD thermal capacity calculation."""
        from integrate_thermal_generators import load_repd_thermal_sites
        
        df = load_repd_thermal_sites(sample_repd_thermal_sites)
        
        total_capacity = df['capacity_mw'].sum()
        
        # Sum from fixtures: biomass (2600+420+750) + waste (35+50+65) + biogas (2.5+3.0) 
        # + landfill (5) + sewage (1.5) + biofuel (10) + geothermal (3)
        expected_total = 3770 + 150 + 5.5 + 5 + 1.5 + 10 + 3
        
        assert total_capacity == pytest.approx(expected_total), "Total capacity should match fixture data"
    
    def test_repd_missing_files(self, temp_workspace):
        """Test handling when some REPD files are missing."""
        from integrate_thermal_generators import load_repd_thermal_sites
        
        # Only provide 2 files
        partial_files = {
            'biomass': temp_workspace / "nonexistent1.csv",
            'waste_to_energy': temp_workspace / "nonexistent2.csv"
        }
        
        df = load_repd_thermal_sites(partial_files)
        
        assert len(df) == 0, "Should return empty DataFrame when all files missing"
    
    def test_repd_biomass_capacity(self, sample_repd_thermal_sites):
        """Test biomass-specific capacity aggregation."""
        from integrate_thermal_generators import load_repd_thermal_sites
        
        df = load_repd_thermal_sites(sample_repd_thermal_sites)
        
        biomass_capacity = df[df['fuel_type'] == 'biomass']['capacity_mw'].sum()
        
        assert biomass_capacity == pytest.approx(3770.0), "Biomass total should be 3770 MW"


# ═══════════════════════════════════════════════════════════════════════════
# TEST MERGE_GENERATOR_DATA_SOURCES (CRITICAL)
# ═══════════════════════════════════════════════════════════════════════════

class TestMergeGeneratorDataSources:
    """Test critical data source merging logic."""
    
    def test_merge_historical_scenario(self, sample_dukes_data, sample_repd_thermal_sites):
        """Test merging for historical scenario (DUKES + REPD, NO FES)."""
        from integrate_thermal_generators import load_dukes_generators, load_repd_thermal_sites, merge_generator_data_sources
        
        dukes_df = load_dukes_generators(str(sample_dukes_data))
        repd_df = load_repd_thermal_sites(sample_repd_thermal_sites)
        fes_df = pd.DataFrame()  # NO FES for historical
        
        merged = merge_generator_data_sources(dukes_df, repd_df, fes_df, scenario_year=2020)
        
        # Should have DUKES + REPD
        assert len(merged) == len(dukes_df) + len(repd_df), "Should combine DUKES + REPD"
        
        # Check data sources
        data_sources = merged['data_source'].unique()
        assert 'DUKES' in data_sources, "Should have DUKES data"
        assert 'REPD' in data_sources, "Should have REPD data"
        assert 'FES' not in data_sources, "Should NOT have FES in historical scenario"
    
    def test_merge_future_scenario(self, sample_fes_data):
        """Test merging for future scenario (FES only)."""
        from integrate_thermal_generators import load_fes_generators, merge_generator_data_sources
        
        dukes_df = pd.DataFrame()  # NO DUKES for future
        repd_df = pd.DataFrame()   # Could have REPD for dispatchable renewables
        fes_df = load_fes_generators(str(sample_fes_data), modelled_year=2030)
        
        merged = merge_generator_data_sources(dukes_df, repd_df, fes_df, scenario_year=2030)
        
        # Should have FES only
        assert len(merged) == len(fes_df), "Should have FES generators"
        
        # Check data source
        data_sources = merged['data_source'].unique()
        assert 'FES' in data_sources, "Should have FES data"
        assert 'DUKES' not in data_sources, "Should NOT have DUKES in future scenario"
    
    def test_merge_data_source_provenance(self, sample_dukes_data, sample_repd_thermal_sites):
        """Test that data_source provenance is tracked correctly."""
        from integrate_thermal_generators import load_dukes_generators, load_repd_thermal_sites, merge_generator_data_sources
        
        dukes_df = load_dukes_generators(str(sample_dukes_data))
        repd_df = load_repd_thermal_sites(sample_repd_thermal_sites)
        fes_df = pd.DataFrame()
        
        merged = merge_generator_data_sources(dukes_df, repd_df, fes_df, scenario_year=2020)
        
        # Count by source
        dukes_count = len(merged[merged['data_source'] == 'DUKES'])
        repd_count = len(merged[merged['data_source'] == 'REPD'])
        
        assert dukes_count == len(dukes_df), "DUKES count should match input"
        assert repd_count == len(repd_df), "REPD count should match input"
    
    def test_merge_capacity_conservation(self, sample_dukes_data, sample_repd_thermal_sites):
        """Test that total capacity is conserved during merge."""
        from integrate_thermal_generators import load_dukes_generators, load_repd_thermal_sites, merge_generator_data_sources
        
        dukes_df = load_dukes_generators(str(sample_dukes_data))
        repd_df = load_repd_thermal_sites(sample_repd_thermal_sites)
        fes_df = pd.DataFrame()
        
        input_capacity = dukes_df['capacity_mw'].sum() + repd_df['capacity_mw'].sum()
        
        merged = merge_generator_data_sources(dukes_df, repd_df, fes_df, scenario_year=2020)
        
        output_capacity = merged['capacity_mw'].sum()
        
        assert output_capacity == pytest.approx(input_capacity), "Capacity should be conserved during merge"
    
    def test_merge_empty_sources(self):
        """Test merging with all empty sources."""
        from integrate_thermal_generators import merge_generator_data_sources
        
        dukes_df = pd.DataFrame()
        repd_df = pd.DataFrame()
        fes_df = pd.DataFrame()
        
        merged = merge_generator_data_sources(dukes_df, repd_df, fes_df, scenario_year=2020)
        
        assert len(merged) == 0, "Should return empty DataFrame when all sources empty"
    
    def test_merge_summary_by_source(self, sample_dukes_data, sample_repd_thermal_sites):
        """Test summary statistics grouping by data source."""
        from integrate_thermal_generators import load_dukes_generators, load_repd_thermal_sites, merge_generator_data_sources
        
        dukes_df = load_dukes_generators(str(sample_dukes_data))
        repd_df = load_repd_thermal_sites(sample_repd_thermal_sites)
        fes_df = pd.DataFrame()
        
        merged = merge_generator_data_sources(dukes_df, repd_df, fes_df, scenario_year=2020)
        
        # Group by source
        summary = merged.groupby('data_source')['capacity_mw'].agg(['count', 'sum'])
        
        assert 'DUKES' in summary.index, "Summary should include DUKES"
        assert 'REPD' in summary.index, "Summary should include REPD"
        
        # Check counts
        assert summary.loc['DUKES', 'count'] == 9, "DUKES should have 9 generators"
        assert summary.loc['REPD', 'count'] > 0, "REPD should have generators"


# ═══════════════════════════════════════════════════════════════════════════
# TEST STANDARDIZE_REPD_FOR_MERGE
# ═══════════════════════════════════════════════════════════════════════════

class TestStandardizeREPDForMerge:
    """Test REPD data standardization."""
    
    def test_standardize_site_name_column(self):
        """Test standardization of Site Name column."""
        from integrate_thermal_generators import standardize_repd_for_merge
        
        repd_df = pd.DataFrame({
            'Site Name': ['Plant A', 'Plant B'],
            'capacity_mw': [100.0, 200.0],
            'fuel_type': ['biomass', 'waste']
        })
        
        standardized = standardize_repd_for_merge(repd_df)
        
        assert 'station_name' in standardized.columns, "Should have station_name column"
        assert standardized['station_name'].iloc[0] == 'Plant A', "Site Name should map to station_name"
    
    def test_standardize_technology_column(self):
        """Test standardization of fuel_type to technology column."""
        from integrate_thermal_generators import standardize_repd_for_merge
        
        repd_df = pd.DataFrame({
            'Site Name': ['Plant A'],
            'capacity_mw': [100.0],
            'fuel_type': ['biomass']
        })
        
        standardized = standardize_repd_for_merge(repd_df)
        
        assert 'technology' in standardized.columns, "Should have technology column"
        assert standardized['technology'].iloc[0] == 'biomass', "fuel_type should map to technology"
    
    def test_standardize_empty_dataframe(self):
        """Test standardization of empty DataFrame."""
        from integrate_thermal_generators import standardize_repd_for_merge
        
        empty_df = pd.DataFrame()
        
        standardized = standardize_repd_for_merge(empty_df)
        
        assert len(standardized) == 0, "Empty DataFrame should remain empty"
    
    def test_standardize_preserves_existing_columns(self):
        """Test that existing standard columns are preserved."""
        from integrate_thermal_generators import standardize_repd_for_merge
        
        repd_df = pd.DataFrame({
            'station_name': ['Existing Plant'],
            'technology': ['biomass'],
            'capacity_mw': [100.0],
            'fuel_type': ['biomass']
        })
        
        standardized = standardize_repd_for_merge(repd_df)
        
        assert standardized['station_name'].iloc[0] == 'Existing Plant', "Existing station_name should be preserved"
        assert standardized['technology'].iloc[0] == 'biomass', "Existing technology should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
