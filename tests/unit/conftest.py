"""
Pytest fixtures for unit tests

Imports shared fixtures from the fixtures directory.
"""

import pytest
import sys
from pathlib import Path

# Add fixtures directory to path
fixtures_dir = Path(__file__).parent.parent / 'fixtures'
sys.path.insert(0, str(fixtures_dir))

# Import all FES fixtures  
import fes_fixtures
from fes_fixtures import (
    sample_building_blocks_2024,
    sample_building_block_definitions_2024,
    sample_building_blocks_2021,
    sample_building_block_definitions_2021,
    sample_gsp_info_2024,
    sample_node_info_2024,
    fes_config_yaml,
    temp_fes_output,
    mock_requests_session,
    mock_fes_csv_response_2024,
    mock_fes_definitions_response_2024,
    gsp_mapping_with_special_cases,
    fes_data_with_problematic_gsp_names,
    fes_data_2024_with_special_characters,
    fes_data_with_direct_connections,
    fes_scenario_config,
    historical_scenario_config,
    future_scenario_config,
    mock_snakemake_fes,
    patch_yaml_config,
    patch_http_session,
    test_fes_api_urls_file,
    test_gsp_info_file,
    test_network_excel_file
)

# Alias for compatibility with test_fes_data.py
fes_data_with_special_characters = fes_data_2024_with_special_characters

__all__ = [
    'sample_building_blocks_2024',
    'sample_building_block_definitions_2024',
    'sample_building_blocks_2021',
    'sample_building_block_definitions_2021',
    'sample_gsp_info_2024',
    'sample_node_info_2024',
    'fes_config_yaml',
    'temp_fes_output',
    'mock_requests_session',
    'mock_fes_csv_response_2024',
    'mock_fes_definitions_response_2024',
    'gsp_mapping_with_special_cases',
    'fes_data_with_problematic_gsp_names',
    'fes_data_with_special_characters',
    'fes_data_2024_with_special_characters',
    'fes_data_with_direct_connections',
    'fes_scenario_config',
    'historical_scenario_config',
    'future_scenario_config',
    'mock_snakemake_fes',
    'patch_yaml_config',
    'patch_http_session',
    'test_fes_api_urls_file',
    'test_gsp_info_file',
    'test_network_excel_file'
]
