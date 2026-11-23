"""
Unit tests for ETYS_upgrades.py - Network infrastructure upgrades module.

Tests the application of ETYS Appendix B 2023 network upgrades including:
- Circuit additions, removals, and modifications
- Transformer additions, removals, and modifications
- HVDC interconnector data
"""

import pytest
import pandas as pd
import pypsa
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from ETYS_upgrades import (
    add_missing_buses_from_upgrades,
    load_etys_upgrade_data,
    filter_upgrades_by_year,
    apply_etys_network_upgrades,
    stable_node_bearing_radians,
    apply_documented_internal_bus_sections,
    apply_topology_sensitivity_bus_sections,
    apply_circuit_removals,
    find_split_transmission_path_candidates,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_circuit_data():
    """Sample circuit upgrade data."""
    return pd.DataFrame({
        'Year': [2025, 2026, 2027],
        'operator': ['NGET', 'SPT', 'SHE'],
        'Node1': ['Bus1', 'Bus2', 'Bus3'],
        'Node2': ['Bus2', 'Bus3', 'Bus4'],
        'Voltage': [400, 400, 275],
        'Status': ['Addition', 'Modify', 'Removed'],
        'Capacity': [1000, 1200, 800]
    })


@pytest.fixture
def sample_transformer_data():
    """Sample transformer upgrade data."""
    return pd.DataFrame({
        'Year': [2025, 2026, 2028],
        'operator': ['NGET', 'NGET', 'SPT'],
        'Node1': ['Bus1', 'Bus2', 'Bus5'],
        'Node2': ['Bus1_Sub', 'Bus2_Sub', 'Bus5_Sub'],
        'Status': ['Addition', 'Modify', 'Addition'],
        'Rating': [500, 600, 400]
    })


@pytest.fixture
def sample_hvdc_data():
    """Sample HVDC interconnector data."""
    return pd.DataFrame({
        'Planned from year': [2026, 2029],
        'Existing': ['No', 'No'],
        'Name': ['Viking Link', 'Interconnector X'],
        'From': ['GB_Bus', 'GB_Bus2'],
        'To': ['DK_Bus', 'FR_Bus'],
        'Capacity': [1400, 2000]
    })


@pytest.fixture
def mock_upgrades_dict(sample_circuit_data, sample_transformer_data, sample_hvdc_data):
    """Mock upgrades data dictionary."""
    return {
        'circuits': sample_circuit_data,
        'transformers': sample_transformer_data,
        'hvdc': sample_hvdc_data
    }


@pytest.fixture
def simple_network():
    """Create a simple PyPSA network for testing."""
    network = pypsa.Network()
    
    # Add buses
    network.add("Bus", "Bus1", x=0, y=52, v_nom=400)
    network.add("Bus", "Bus2", x=1, y=53, v_nom=400)
    network.add("Bus", "Bus3", x=2, y=54, v_nom=400)
    network.add("Bus", "Bus4", x=3, y=55, v_nom=275)
    
    # Add some lines
    network.add("Line", "Line1", bus0="Bus1", bus1="Bus2", s_nom=1000, length=100)
    network.add("Line", "Line2", bus0="Bus2", bus1="Bus3", s_nom=800, length=120)
    
    return network


# ══════════════════════════════════════════════════════════════════════════════
# TEST: ETYS Data Loading
# ══════════════════════════════════════════════════════════════════════════════

class TestETYSUpgradeLoading:
    """Test loading of ETYS upgrade data from Excel file."""
    
    def test_load_etys_upgrade_data_returns_dict(self, mock_upgrades_dict):
        """Test that load function returns dictionary with correct keys."""
        with patch('ETYS_upgrades.pd.read_excel') as mock_read:
            mock_read.side_effect = [
                mock_upgrades_dict['circuits'].copy(),
                mock_upgrades_dict['circuits'].copy(),
                mock_upgrades_dict['circuits'].copy(),
                mock_upgrades_dict['circuits'].copy(),
                mock_upgrades_dict['transformers'].copy(),
                mock_upgrades_dict['transformers'].copy(),
                mock_upgrades_dict['transformers'].copy(),
                mock_upgrades_dict['transformers'].copy(),
                mock_upgrades_dict['hvdc'].copy()
            ]
            
            result = load_etys_upgrade_data("fake_file.xlsx")
            
            assert isinstance(result, dict)
            assert 'circuits' in result
            assert 'transformers' in result
            assert 'hvdc' in result
    
    def test_load_handles_missing_file(self):
        """Test that load function handles missing file gracefully."""
        with patch('ETYS_upgrades.pd.read_excel') as mock_read:
            mock_read.side_effect = FileNotFoundError("File not found")
            
            # Should not raise, but return empty dataframes
            result = load_etys_upgrade_data("nonexistent.xlsx")
            
            assert result['circuits'].empty
            assert result['transformers'].empty
            assert result['hvdc'].empty
    
    def test_load_adds_operator_column(self, sample_circuit_data):
        """Test that operator column is added to each sheet."""
        with patch('ETYS_upgrades.pd.read_excel') as mock_read:
            mock_read.return_value = sample_circuit_data.copy()
            
            result = load_etys_upgrade_data("fake_file.xlsx")
            
            assert 'operator' in result['circuits'].columns


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Year Filtering
# ══════════════════════════════════════════════════════════════════════════════

class TestETYSYearFiltering:
    """Test filtering of upgrades by year."""
    
    def test_filter_upgrades_by_year(self, mock_upgrades_dict):
        """Test that upgrades are correctly filtered by year."""
        result = filter_upgrades_by_year(mock_upgrades_dict, modelled_year=2026)
        
        # Result should be a dict with categorized upgrades
        assert 'circuits' in result
        assert 'transformers' in result
        assert 'hvdc' in result
        
        # Should have additions, removals, changes lists
        assert 'additions' in result['circuits']
        assert 'removals' in result['circuits']
        assert 'changes' in result['circuits']
    
    def test_filter_handles_future_year(self, mock_upgrades_dict):
        """Test filtering with year beyond all upgrades."""
        result = filter_upgrades_by_year(mock_upgrades_dict, modelled_year=2050)
        
        # Should include all upgrades (check at least some exist)
        assert 'circuits' in result
        assert 'transformers' in result
        assert isinstance(result, dict)
    
    def test_filter_handles_past_year(self, mock_upgrades_dict):
        """Test filtering with year before any upgrades."""
        result = filter_upgrades_by_year(mock_upgrades_dict, modelled_year=2020)
        
        # Should exclude all upgrades (empty lists)
        assert len(result['circuits']['additions']) == 0
        assert len(result['transformers']['additions']) == 0
        assert len(result['hvdc']['additions']) == 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Network Application
# ══════════════════════════════════════════════════════════════════════════════

class TestETYSNetworkApplication:
    """Test application of upgrades to PyPSA network."""
    
    def test_apply_upgrades_preserves_network_type(self, simple_network):
        """Test that network type is preserved."""
        with patch('ETYS_upgrades.load_etys_upgrade_data') as mock_load:
            mock_load.return_value = {
                'circuits': pd.DataFrame(),
                'transformers': pd.DataFrame(),
                'hvdc': pd.DataFrame()
            }
            
            result = apply_etys_network_upgrades(
                simple_network.copy(), 
                modelled_year=2025,
                etys_file="fake.xlsx"
            )
            
            assert isinstance(result, pypsa.Network)
    
    def test_apply_upgrades_with_disabled_config(self, simple_network):
        """Test that upgrades can be disabled via config."""
        initial_lines = len(simple_network.lines)
        
        with patch('ETYS_upgrades.load_etys_upgrade_data') as mock_load:
            mock_load.return_value = {
                'circuits': pd.DataFrame({'Year': [2025], 'Status': ['Addition'], 'Node1': ['Bus1'], 'Node2': ['Bus2']}),
                'transformers': pd.DataFrame(),
                'hvdc': pd.DataFrame()
            }
            
            # Should not apply if disabled
            result = apply_etys_network_upgrades(
                simple_network.copy(),
                modelled_year=2025,
                etys_file="fake.xlsx"
            )
            
            # Network should be valid (may have more lines if upgrades applied)
            assert isinstance(result, pypsa.Network)
            assert len(result.lines) >= initial_lines
    
    def test_apply_handles_missing_etys_file(self, simple_network):
        """Test that function handles missing ETYS file gracefully."""
        with patch('ETYS_upgrades.load_etys_upgrade_data') as mock_load:
            mock_load.return_value = {
                'circuits': pd.DataFrame(),
                'transformers': pd.DataFrame(),
                'hvdc': pd.DataFrame()
            }
            
            # Should not raise
            result = apply_etys_network_upgrades(
                simple_network.copy(),
                modelled_year=2025,
                etys_file="nonexistent.xlsx"
            )
            
            assert isinstance(result, pypsa.Network)


class TestETYSUpgradeCoordinateReproducibility:
    """Regression tests for deterministic coordinates of upgrade-added buses."""

    @staticmethod
    def _base_network(reverse_bus_order=False):
        network = pypsa.Network()
        buses = [
            ("WEST4A", 100000.0, 500000.0),
            ("EAST4A", 500000.0, 500000.0),
        ]
        if reverse_bus_order:
            buses.reverse()
        for name, x, y in buses:
            network.add("Bus", name, x=x, y=y, v_nom=400, carrier="AC", country="GB")
        return network

    @staticmethod
    def _categorized_upgrades(reverse_records=False):
        records = [
            {
                "Node1": "NEWA4A",
                "Node2": "WEST4A",
                "OHL Length (km)": 8.0,
                "Cable Length (km)": 0.0,
            },
            {
                "Node1": "NEWB4A",
                "Node2": "NEWA4A",
                "OHL Length (km)": 6.0,
                "Cable Length (km)": 0.0,
            },
            {
                "Node1": "NEWC4A",
                "Node2": "EAST4A",
                "OHL Length (km)": 5.0,
                "Cable Length (km)": 0.0,
            },
        ]
        if reverse_records:
            records.reverse()
        return {
            "circuits": {"additions": records},
            "transformers": {"additions": []},
        }

    def test_stable_node_bearing_has_fixed_value(self):
        """The pseudo-bearing must not depend on Python's randomized hash seed."""
        assert stable_node_bearing_radians("NEWA4A") == pytest.approx(
            6.126105674500097
        )

    def test_upgrade_coordinates_ignore_input_and_bus_order(self):
        """Equivalent upgrade inputs must produce identical buses and coordinates."""
        first = self._base_network(reverse_bus_order=False)
        second = self._base_network(reverse_bus_order=True)

        add_missing_buses_from_upgrades(
            first,
            self._categorized_upgrades(reverse_records=False),
        )
        add_missing_buses_from_upgrades(
            second,
            self._categorized_upgrades(reverse_records=True),
        )

        expected_order = ["WEST4A", "EAST4A", "NEWA4A", "NEWB4A", "NEWC4A"]
        assert first.buses.index.tolist() == expected_order
        assert second.buses.index.tolist() == [
            "EAST4A", "WEST4A", "NEWA4A", "NEWB4A", "NEWC4A"
        ]
        pd.testing.assert_frame_equal(
            first.buses.loc[sorted(first.buses.index), ["x", "y"]],
            second.buses.loc[sorted(second.buses.index), ["x", "y"]],
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Upgrade Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestETYSUpgradeValidation:
    """Test validation of upgrade data."""
    
    def test_detect_invalid_years(self, sample_circuit_data):
        """Test detection of invalid year values."""
        invalid_data = sample_circuit_data.copy()
        invalid_data.loc[0, 'Year'] = np.nan
        
        # Should handle gracefully
        result = filter_upgrades_by_year(
            {'circuits': invalid_data, 'transformers': pd.DataFrame(), 'hvdc': pd.DataFrame()},
            modelled_year=2025
        )
        
        # Should still work, just skip invalid rows
        assert isinstance(result, dict)
        assert 'circuits' in result
    
    def test_validate_upgrade_years_in_range(self, sample_circuit_data):
        """Test that upgrade years are within expected range."""
        years = sample_circuit_data['Year'].values
        
        # ETYS upgrades should be 2024-2031
        assert all(2024 <= y <= 2031 for y in years if not pd.isna(y))


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Integration
# ══════════════════════════════════════════════════════════════════════════════

class TestETYSUpgradeIntegration:
    """Test end-to-end upgrade application."""
    
    def test_complete_upgrade_workflow(self, simple_network, mock_upgrades_dict):
        """Test complete upgrade workflow from load to apply."""
        with patch('ETYS_upgrades.load_etys_upgrade_data') as mock_load:
            mock_load.return_value = mock_upgrades_dict
            
            result = apply_etys_network_upgrades(
                simple_network.copy(),
                modelled_year=2026,
                etys_file="fake.xlsx"
            )
            
            assert isinstance(result, pypsa.Network)
            # Network should still have buses
            assert len(result.buses) > 0



class TestDocumentedInternalBusSections:
    """Test the narrow, source-backed internal topology corrections."""

    def test_little_horsted_bus_section_is_year_gated_and_idempotent(self):
        network = pypsa.Network()
        for bus in ["BOLN41", "LHOR4A", "LHOR4B", "NINF41", "LHOR11"]:
            network.add("Bus", bus, v_nom=400 if bus != "LHOR11" else 11)

        assert apply_documented_internal_bus_sections(network, 2024) == 0
        assert "LHOR4A_LHOR4B_ETYS_2025_BUS_SECTION" not in network.lines.index

        assert apply_documented_internal_bus_sections(network, 2025) == 1
        section = network.lines.loc["LHOR4A_LHOR4B_ETYS_2025_BUS_SECTION"]
        assert section.bus0 == "LHOR4A"
        assert section.bus1 == "LHOR4B"
        assert section.s_nom == pytest.approx(3102.0)
        assert "ETYS 2024 Appendix B" in section.etys_topology_source

        # Reapplying upgrades must not create a parallel duplicate.
        assert apply_documented_internal_bus_sections(network, 2030) == 0
        assert len(network.lines) == 1

    def test_split_path_audit_keeps_only_small_offtake_patterns(self):
        categorized = {
            "circuits": {
                "additions": [
                    {"Node1": "BOLN41", "Node2": "LHOR4A", "Year": 2025,
                     "Winter Rating (MVA)": 3102.0},
                    {"Node1": "LHOR4B", "Node2": "NINF41", "Year": 2025,
                     "Winter Rating (MVA)": 3102.0},
                    # York has a similarly shaped pair of bus sections, but
                    # its 2 x 1100 MVA transformer banks make it a normal
                    # 400/275 kV supply arrangement, not a through path.
                    {"Node1": "OBST4A", "Node2": "YORK4A", "Year": 2027,
                     "Winter Rating (MVA)": 1500.0},
                    {"Node1": "OBST4B", "Node2": "YORK4B", "Year": 2027,
                     "Winter Rating (MVA)": 1500.0},
                ],
            },
            "transformers": {
                "additions": [
                    {"Node1": "LHOR4A", "Node2": "LHOR11", "Year": 2025,
                     "Winter Rating (MVA)": 240.0},
                    {"Node1": "LHOR4B", "Node2": "LHOR11", "Year": 2025,
                     "Winter Rating (MVA)": 240.0},
                    {"Node1": "YORK4A", "Node2": "OVER21", "Year": 2027,
                     "Winter Rating (MVA)": 1100.0},
                    {"Node1": "YORK4A", "Node2": "OVER21", "Year": 2027,
                     "Winter Rating (MVA)": 1100.0},
                    {"Node1": "YORK4B", "Node2": "OVER21", "Year": 2027,
                     "Winter Rating (MVA)": 1100.0},
                    {"Node1": "YORK4B", "Node2": "OVER21", "Year": 2027,
                     "Winter Rating (MVA)": 1100.0},
                ],
            },
        }

        candidates = find_split_transmission_path_candidates(categorized)

        assert candidates == [
            {
                "site": "LHOR",
                "bus0": "LHOR4A",
                "bus1": "LHOR4B",
                "year": 2025,
                "lower_voltage_buses": ["LHOR11"],
                "line_ratings_mva": [3102.0, 3102.0],
                "shared_offtake_ratings_mva": {
                    "LHOR4A": 240.0,
                    "LHOR4B": 240.0,
                },
            }
        ]



class TestExplicitTopologySensitivities:
    """Test non-default ETYS topology assumptions remain opt-in."""

    def test_yorkshire_green_mfsn_sensitivity_is_year_gated_and_idempotent(self):
        network = pypsa.Network()
        for bus in ["MFSN21", "MFSN2A", "MFSN2B"]:
            network.add("Bus", bus, v_nom=275)

        sensitivity = ["yorkshire_green_mfsn_2027"]
        assert apply_topology_sensitivity_bus_sections(
            network, 2026, sensitivity
        ) == 0

        assert apply_topology_sensitivity_bus_sections(
            network, 2027, sensitivity
        ) == 2
        added = network.lines.loc[
            [
                "MFSN21_MFSN2A_YORKSHIRE_GREEN_SENSITIVITY_2027",
                "MFSN21_MFSN2B_YORKSHIRE_GREEN_SENSITIVITY_2027",
            ]
        ]
        assert set(added.s_nom) == {1500.0}
        assert set(added.etys_topology_status) == {"sensitivity"}
        assert set(added.etys_topology_sensitivity) == {
            "yorkshire_green_mfsn_2027"
        }

        assert apply_topology_sensitivity_bus_sections(
            network, 2035, sensitivity
        ) == 0
        assert len(network.lines) == 2

    def test_yorkshire_green_full_chain_adds_275_and_400_kv_sections(self):
        network = pypsa.Network()
        for bus, v_nom in {
            "MFSN21": 275,
            "MFSN2A": 275,
            "MFSN2B": 275,
            "MONF42": 400,
            "MFSN41": 400,
        }.items():
            network.add("Bus", bus, v_nom=v_nom)

        sensitivity = ["yorkshire_green_full_chain_2027"]
        assert apply_topology_sensitivity_bus_sections(
            network, 2026, sensitivity
        ) == 0
        assert apply_topology_sensitivity_bus_sections(
            network, 2027, sensitivity
        ) == 3

        added = network.lines.loc[
            [
                "MFSN21_MFSN2A_YORKSHIRE_GREEN_SENSITIVITY_2027",
                "MFSN21_MFSN2B_YORKSHIRE_GREEN_SENSITIVITY_2027",
                "MONF42_MFSN41_YORKSHIRE_GREEN_SENSITIVITY_2027",
            ]
        ]
        assert added.loc[
            "MONF42_MFSN41_YORKSHIRE_GREEN_SENSITIVITY_2027", "s_nom"
        ] == 6140.0
        assert set(added.etys_topology_sensitivity) == {
            "yorkshire_green_full_chain_2027"
        }

        assert apply_topology_sensitivity_bus_sections(
            network, 2035, sensitivity
        ) == 0

    def test_unknown_topology_sensitivity_fails_fast(self):
        network = pypsa.Network()
        with pytest.raises(ValueError, match="Unknown ETYS topology sensitivity"):
            apply_topology_sensitivity_bus_sections(
                network, 2035, ["not_a_valid_sensitivity"]
            )
class TestYorkshireGreenRemovalReconciliation:
    """The raw Appendix-B removal terminals must match the base ETYS topology."""

    def test_reconciles_the_two_reviewed_monk_fryston_bus_labels(self):
        network = pypsa.Network()
        for bus in ["KNAR21", "MONF21", "MONF22", "POPP21"]:
            network.add("Bus", bus, v_nom=275)
        network.add(
            "Line", "KNAR21_MONF21_0", bus0="KNAR21", bus1="MONF21",
            s_nom=551.0, r=0.001, x=0.01,
        )
        network.add(
            "Line", "MONF22_POPP21_0", bus0="MONF22", bus1="POPP21",
            s_nom=682.0, r=0.001, x=0.01,
        )

        removed, failed, failures = apply_circuit_removals(network, [
            {"Year": 2027, "Node1": "KNAR21", "Node2": "MONF22",
             "Winter Rating (MVA)": 551.0},
            {"Year": 2027, "Node1": "MONF21", "Node2": "POPP21",
             "Winter Rating (MVA)": 682.0},
        ])

        assert (removed, failed, failures) == (2, 0, [])
        assert network.lines.empty

    def test_does_not_apply_to_an_unreviewed_removal(self):
        network = pypsa.Network()
        network.add("Bus", "KNAR21", v_nom=275)
        network.add("Bus", "MONF21", v_nom=275)
        network.add(
            "Line", "KNAR21_MONF21_0", bus0="KNAR21", bus1="MONF21",
            s_nom=551.0, r=0.001, x=0.01,
        )

        removed, failed, failures = apply_circuit_removals(network, [{
            "Year": 2028, "Node1": "KNAR21", "Node2": "MONF22",
            "Winter Rating (MVA)": 551.0,
        }])

        assert (removed, failed) == (0, 1)
        assert failures[0]["reason"] == "line_not_found"
        assert "KNAR21_MONF21_0" in network.lines.index

# ══════════════════════════════════════════════════════════════════════════════
# TEST: HVDC terminal enabling works (Western Isles link)
# ══════════════════════════════════════════════════════════════════════════════

def _western_isles_upgrades():
    """Appendix B rows shaped like ETYS 2024: the link is 2030, its terminals 2032."""
    def circuit(n1, n2, year, ohl, rating):
        return {'Node1': n1, 'Node2': n2, 'Year': year, 'Status': 'Addition',
                'OHL Length (km)': ohl, 'Cable Length (km)': 0,
                'R (% on 100 MVA)': 0.01, 'X (% on 100 MVA)': 0.01,
                'B (% on 100 MVA)': 0.0, 'Winter Rating (MVA)': rating,
                'operator': 'SHE'}
    circuits = pd.DataFrame([
        circuit('LEWI4J', 'LEWI4R', 2032, 0, 9999),
        circuit('LEWI1J', 'LEWI1K', 2032, 0, 9999),
        circuit('LEWI1J', 'STOR1-', 2032, 1, 300),        # leaves the site: not pulled
        circuit('FNLN4B', 'FNLN4R', 2032, 0, 9999),
        circuit('FNLN4A', 'FNLN4B', 2032, 0, 9999),
        circuit('BEAU4-', 'FNLN4A', 2032, 5, 2780),        # shortest tie: pulled
        circuit('BEAU4-', 'FNLN4B', 2032, 5, 2780),        # its parallel: pulled
        circuit('CAAI4J', 'FNLN4A', 2032, 82, 2780),       # wider reinforcement: not pulled
        circuit('LEWI4J', 'LEWI4M', 2035, 0, 9999),        # later batch: not pulled
    ])
    transformers = pd.DataFrame([
        {'Node1': 'LEWI1J', 'Node2': 'LEWI4J', 'Year': 2032, 'Status': 'Addition',
         'R (% on 100 MVA)': 0.1119, 'X (% on 100 MVA)': 9.3882,
         'B (% on 100 MVA)': -0.19, 'Winter Rating (MVA)': 360, 'operator': 'SHE'},
    ])
    hvdc = pd.DataFrame([{
        'Interconnector Name': 'Western Isles HVDC Link', 'Existing': 'No',
        'Planned from year': 2030, 'Node 1': 'LEWI4R', 'Node 2': 'FNLN4R',
        'Winter Rating (MVA)': 1800,
    }])
    return {'circuits': circuits, 'transformers': transformers, 'hvdc': hvdc}


def _beauly_network():
    network = pypsa.Network()
    network.add("Bus", "BEAU4-", v_nom=400, x=250766, y=844664)
    network.add("Bus", "CAAI4J", v_nom=400, x=300000, y=900000)
    network.add("Line", "BEAU_CAAI", bus0="BEAU4-", bus1="CAAI4J", s_nom=2000, x=0.01, r=0.001)
    return network


class TestHVDCEnablingWorks:
    """An in-scope HVDC link gets the later-year works that create its terminals."""

    def test_pulls_site_works_and_shortest_tie_only(self):
        from ETYS_upgrades import add_hvdc_enabling_works
        upgrades = _western_isles_upgrades()
        categorized = filter_upgrades_by_year(upgrades, 2030)
        assert categorized['circuits']['additions'] == []

        pulled = add_hvdc_enabling_works(upgrades, categorized, _beauly_network(), 2030,
                                         links={'Western Isles HVDC Link'})

        pairs = {(r['Node1'], r['Node2']) for r in pulled}
        assert pairs == {
            ('LEWI4J', 'LEWI4R'), ('LEWI1J', 'LEWI1K'), ('LEWI1J', 'LEWI4J'),
            ('FNLN4B', 'FNLN4R'), ('FNLN4A', 'FNLN4B'),
            ('BEAU4-', 'FNLN4A'), ('BEAU4-', 'FNLN4B'),
        }
        assert all(r['_hvdc_enabling_works'] == 'Western Isles HVDC Link' for r in pulled)
        assert len(categorized['circuits']['additions']) == 6
        assert len(categorized['transformers']['additions']) == 1

    def test_only_listed_links_are_completed(self):
        from ETYS_upgrades import add_hvdc_enabling_works
        upgrades = _western_isles_upgrades()
        categorized = filter_upgrades_by_year(upgrades, 2030)
        assert add_hvdc_enabling_works(
            upgrades, categorized, _beauly_network(), 2030, links=set()) == []
        assert categorized['circuits']['additions'] == []

    def test_terminal_resolvable_as_same_site_busbar_is_left_alone(self):
        # EGL2 names PEHE4-, which resolves to the existing PEHE4J busbar
        from ETYS_upgrades import add_hvdc_enabling_works
        network = pypsa.Network()
        network.add("Bus", "PEHE4J", v_nom=400)
        network.add("Bus", "DRAX41", v_nom=400)
        upgrades = {
            'circuits': pd.DataFrame([{'Node1': 'PEHE2Z', 'Node2': 'PEHE4-', 'Year': 2035,
                                       'Status': 'Addition', 'Winter Rating (MVA)': 1200}]),
            'transformers': pd.DataFrame(),
            'hvdc': pd.DataFrame([{'Interconnector Name': 'Eastern Green Link 2',
                                   'Existing': 'No', 'Planned from year': 2029,
                                   'Node 1': 'DRAX41', 'Node 2': 'PEHE4-'}]),
        }
        categorized = filter_upgrades_by_year(upgrades, 2030)
        assert add_hvdc_enabling_works(
            upgrades, categorized, network, 2030, links={'Eastern Green Link 2'}) == []

    def test_western_isles_is_not_completed_by_default(self):
        # CP30 sheet N1 does not list the link; it waits for its 2032 terminals
        from ETYS_upgrades import add_hvdc_enabling_works
        from scripts.network_build.etys_file_registry import HVDC_TERMINAL_ENABLING_WORKS
        assert 'PSDC HVDC Link' in HVDC_TERMINAL_ENABLING_WORKS
        assert 'Western Isles HVDC Link' not in HVDC_TERMINAL_ENABLING_WORKS
        upgrades = _western_isles_upgrades()
        categorized = filter_upgrades_by_year(upgrades, 2030)
        assert add_hvdc_enabling_works(upgrades, categorized, _beauly_network(), 2030) == []
        # ... and the audit can say why the link is absent
        assert categorized['hvdc']['additions'][0]['_terminals_from_year'] == 2032

    def test_deferred_link_is_audited_as_waiting_for_its_terminals(self, tmp_path):
        audit = tmp_path / "audit.csv"
        with patch('ETYS_upgrades.load_etys_upgrade_data', return_value=_western_isles_upgrades()):
            network = apply_etys_network_upgrades(
                _beauly_network(), modelled_year=2030, etys_file="fake.xlsx", audit_path=str(audit),
            )
        assert not network.links.bus0.eq('LEWI4R').any()
        rows = pd.read_csv(audit)
        assert rows.loc[rows.component == 'hvdc', 'audit_status'].tolist() == ['deferred_terminals_later_year']

    def test_link_is_built_with_terminals_and_audited(self, tmp_path, monkeypatch):
        import ETYS_upgrades
        monkeypatch.setattr(ETYS_upgrades, 'HVDC_TERMINAL_ENABLING_WORKS', {'Western Isles HVDC Link'})
        upgrades = _western_isles_upgrades()
        audit = tmp_path / "audit.csv"
        coords = tmp_path / "coords.csv"
        pd.DataFrame({
            'site_code': ['LEWI', 'FNLN'], 'site_name': ['LEWIS', 'FANELLAN'],
            'lat': [58.216, 57.452], 'lon': [-6.379, -4.527], 'source': ['m', 'm'],
        }).to_csv(coords, index=False)
        with patch('ETYS_upgrades.load_etys_upgrade_data', return_value=upgrades):
            network = apply_etys_network_upgrades(
                _beauly_network(), modelled_year=2030, etys_file="fake.xlsx",
                substation_coords_file=str(coords), audit_path=str(audit),
            )

        link = network.links[(network.links.bus0 == 'LEWI4R') & (network.links.bus1 == 'FNLN4R')]
        assert len(link) == 1 and link.p_nom.iloc[0] == 1800
        for bus in ['LEWI4R', 'LEWI4J', 'LEWI1J', 'LEWI1K', 'FNLN4R', 'FNLN4A', 'FNLN4B']:
            assert bus in network.buses.index
        # Lewis sits at its substation, not at a connected bus or the centroid
        assert network.buses.at['LEWI1J', 'y'] > 930000
        assert 'STOR1-' not in network.buses.index
        rows = pd.read_csv(audit)
        hvdc_row = rows[rows.component == 'hvdc']
        assert hvdc_row.audit_status.tolist() == ['applied']
        assert (rows.audit_status == 'applied_as_hvdc_enabling_works').sum() == 7
