import pytest

@pytest.fixture
def sample_building_blocks_2024():
	return {}

@pytest.fixture
def sample_building_block_definitions_2024():
	return {}

@pytest.fixture
def sample_building_blocks_2021():
	return {}

@pytest.fixture
def sample_building_block_definitions_2021():
	return {}

@pytest.fixture
def sample_gsp_info_2024():
	return {}

@pytest.fixture
def sample_node_info_2024():
	return {}

@pytest.fixture
def fes_config_yaml():
	return ""

@pytest.fixture
def temp_fes_output():
	return ""

@pytest.fixture
def mock_requests_session():
	class DummySession:
		pass
	return DummySession()

@pytest.fixture
def mock_fes_csv_response_2024():
	return ""

@pytest.fixture
def mock_fes_definitions_response_2024():
	return ""

@pytest.fixture
def gsp_mapping_with_special_cases():
	return {}

@pytest.fixture
def fes_data_with_problematic_gsp_names():
	return {}

@pytest.fixture
def fes_data_2024_with_special_characters():
	return {}

@pytest.fixture
def fes_data_with_direct_connections():
	return {}

@pytest.fixture
def fes_scenario_config():
	return {}

@pytest.fixture
def historical_scenario_config():
	return {}

@pytest.fixture
def future_scenario_config():
	return {}

@pytest.fixture
def mock_snakemake_fes():
	return {}

@pytest.fixture
def patch_yaml_config():
	return {}

@pytest.fixture
def patch_http_session():
	return {}

@pytest.fixture
def test_fes_api_urls_file():
	return ""

@pytest.fixture
def test_gsp_info_file():
	return ""

@pytest.fixture
def test_network_excel_file():
	return ""
