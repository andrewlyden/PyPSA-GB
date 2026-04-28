"""
Unit tests for the connectivity repair function in process_ETYS_data.py.

Tests the two-pass repair strategy:
  Pass 1: Cross-reference GB_network.xlsx AC sheet for missing connections
  Pass 2: Same-substation bus section inference for remaining islands
"""

import pytest
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.network_build.process_ETYS_data import repair_disconnected_components


@pytest.fixture
def logger():
    return logging.getLogger("test_connectivity_repair")


@pytest.fixture
def connected_components():
    """Components DataFrame where the network is fully connected."""
    return pd.DataFrame({
        'component': ['line', 'line', 'transformer'],
        'carrier': ['AC', 'AC', 'AC'],
        'bus0': ['A41', 'B11', 'A41'],
        'bus1': ['B41', 'C11', 'B11'],
        'r': [0.002, 0.002, 0.002],
        'x': [0.02, 0.02, 0.08],
        'b': [0.5, 0.5, 0.0],
        's_nom': [1000, 500, 500],
        'length_km': [50, 20, 0],
    }, index=['A41_B41_0', 'B11_C11_0', 'A41_B11_0'])


@pytest.fixture
def disconnected_components():
    """Components DataFrame with one disconnected island (X11-Y11)."""
    return pd.DataFrame({
        'component': ['line', 'transformer', 'line'],
        'carrier': ['AC', 'AC', 'AC'],
        'bus0': ['A41', 'A41', 'X11'],
        'bus1': ['B41', 'A11', 'Y11'],
        'r': [0.002, 0.002, 0.002],
        'x': [0.02, 0.08, 0.02],
        'b': [0.5, 0.0, 0.5],
        's_nom': [1000, 500, 300],
        'length_km': [50, 0, 20],
    }, index=['A41_B41_0', 'A41_A11_0', 'X11_Y11_0'])


class TestRepairDisconnectedComponentsNoIslands:
    """Test that fully connected networks pass through unchanged."""

    def test_connected_returns_same_length(self, connected_components, logger, tmp_path):
        gb_file = tmp_path / "GB_network.xlsx"
        ac = pd.DataFrame({'Node 1': [], 'Node 2': [], 'Circuit Type': [],
                           'Winter Rating (MVA)': [], 'R (% on 100 MVA)': [],
                           'X (% on 100 MVA)': [], 'B (% on 100 MVA)': []})
        with pd.ExcelWriter(gb_file) as w:
            ac.to_excel(w, sheet_name='AC', index=False)

        result = repair_disconnected_components(connected_components, str(gb_file), logger)
        assert len(result) == len(connected_components)


class TestRepairPass1GBNetworkLookup:
    """Test Pass 1: GB_network.xlsx AC sheet cross-reference."""

    def test_adds_missing_transformer(self, disconnected_components, logger, tmp_path):
        """Island X11-Y11 should be repaired by a transformer A41→X11 from GB_network."""
        gb_file = tmp_path / "GB_network.xlsx"
        ac = pd.DataFrame({
            'Node 1': ['A41'],
            'Node 2': ['X11'],
            'Circuit Type': ['Transformer'],
            'Winter Rating (MVA)': [240],
            'R (% on 100 MVA)': [0.13],
            'X (% on 100 MVA)': [7.95],
            'B (% on 100 MVA)': [0.0],
        })
        with pd.ExcelWriter(gb_file) as w:
            ac.to_excel(w, sheet_name='AC', index=False)

        result = repair_disconnected_components(disconnected_components, str(gb_file), logger)
        assert len(result) == len(disconnected_components) + 1
        repair_rows = result[result.index.str.contains('repair')]
        assert len(repair_rows) == 1
        row = repair_rows.iloc[0]
        assert row['bus0'] == 'A41'
        assert row['bus1'] == 'X11'
        assert row['component'] == 'transformer'
        assert row['s_nom'] == 240

    def test_skips_existing_connections(self, disconnected_components, logger, tmp_path):
        """If the GB_network connection already exists in components, don't duplicate it."""
        gb_file = tmp_path / "GB_network.xlsx"
        ac = pd.DataFrame({
            'Node 1': ['A41'],
            'Node 2': ['B41'],  # Already exists in components
            'Circuit Type': ['OHL'],
            'Winter Rating (MVA)': [1000],
            'R (% on 100 MVA)': [0.2],
            'X (% on 100 MVA)': [2.0],
            'B (% on 100 MVA)': [0.5],
        })
        with pd.ExcelWriter(gb_file) as w:
            ac.to_excel(w, sheet_name='AC', index=False)

        result = repair_disconnected_components(disconnected_components, str(gb_file), logger)
        # Island still unrepaired, but no duplicate added
        repair_rows = result[result.index.str.contains('repair')]
        assert len(repair_rows) == 0


class TestRepairPass2BusSectionInference:
    """Test Pass 2: same-substation prefix inference."""

    def test_same_prefix_coupler(self, logger, tmp_path):
        """Island bus NECU1C should couple to NECU1Q (same prefix, same voltage)."""
        components = pd.DataFrame({
            'component': ['line', 'transformer', 'line'],
            'carrier': ['AC', 'AC', 'AC'],
            'bus0': ['MAIN41', 'MAIN41', 'NECU1C'],
            'bus1': ['MAIN42', 'NECU1Q', 'ISOL1-'],
            'r': [0.002, 0.002, 0.002],
            'x': [0.02, 0.08, 0.02],
            'b': [0.5, 0.0, 0.5],
            's_nom': [1000, 500, 134],
            'length_km': [50, 0, 10],
        }, index=['MAIN41_MAIN42_0', 'MAIN41_NECU1Q_0', 'NECU1C_ISOL1-_0'])

        gb_file = tmp_path / "GB_network.xlsx"
        ac = pd.DataFrame({'Node 1': [], 'Node 2': [], 'Circuit Type': [],
                           'Winter Rating (MVA)': [], 'R (% on 100 MVA)': [],
                           'X (% on 100 MVA)': [], 'B (% on 100 MVA)': []})
        with pd.ExcelWriter(gb_file) as w:
            ac.to_excel(w, sheet_name='AC', index=False)

        result = repair_disconnected_components(components, str(gb_file), logger)
        coupler_rows = result[result.index.str.contains('coupler')]
        assert len(coupler_rows) == 1
        row = coupler_rows.iloc[0]
        # NECU1C should couple to NECU1Q (both 132kV, prefix NECU)
        assert 'NECU1C' in [row['bus0'], row['bus1']]
        assert 'NECU1Q' in [row['bus0'], row['bus1']]

    def test_different_voltage_no_coupler(self, logger, tmp_path):
        """Buses at different voltage levels should NOT be coupled."""
        components = pd.DataFrame({
            'component': ['line', 'line'],
            'carrier': ['AC', 'AC'],
            'bus0': ['MAIN41', 'XBUS1-'],
            'bus1': ['MAIN42', 'YBUS1-'],
            'r': [0.002, 0.002],
            'x': [0.02, 0.02],
            'b': [0.5, 0.5],
            's_nom': [1000, 300],
            'length_km': [50, 20],
        }, index=['MAIN41_MAIN42_0', 'XBUS1-_YBUS1-_0'])

        gb_file = tmp_path / "GB_network.xlsx"
        ac = pd.DataFrame({'Node 1': [], 'Node 2': [], 'Circuit Type': [],
                           'Winter Rating (MVA)': [], 'R (% on 100 MVA)': [],
                           'X (% on 100 MVA)': [], 'B (% on 100 MVA)': []})
        with pd.ExcelWriter(gb_file) as w:
            ac.to_excel(w, sheet_name='AC', index=False)

        result = repair_disconnected_components(components, str(gb_file), logger)
        # XBUS (132kV) has no same-prefix, same-voltage bus in main network
        # (MAIN is 400kV) — no coupler should be added
        coupler_rows = result[result.index.str.contains('coupler')]
        assert len(coupler_rows) == 0
