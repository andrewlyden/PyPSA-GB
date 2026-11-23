"""Items F3 and F14 (rebuild batch, 13 Oct 2026).

The twin Sofia HVDC poles merge into one link; a load-free AC island behind DC
is kept out of nearest-bus placement; a rolling Stage 1 saves its link flows on
the network; the bound check accepts at most 1 MW.
"""
import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market.market_utils import MAX_BOUND_VIOLATION_MW, max_bound_violation
from scripts.market.solve_wholesale import write_link_flows
from scripts.network_build.parallel_links import MERGED_COLUMN, merge_parallel_internal_links
from scripts.utilities.spatial_utils import dc_island_buses, generation_placement_exclusions


def _grid():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for bus in ("LACK4A", "SOFW4A", "MAIN1", "SOFWD", "SOFW61", "SHET1", "SHET2", "CDC"):
        n.add("Bus", bus, x=450000.0, y=520000.0, country="GB")
    n.add("Bus", "France External", x=0.0, y=0.0, country="FR")
    n.add("Line", "main_1", bus0="LACK4A", bus1="SOFW4A", x=0.1, s_nom=1000.0)
    n.add("Line", "main_2", bus0="SOFW4A", bus1="MAIN1", x=0.1, s_nom=1000.0)
    n.add("Load", "demand", bus="MAIN1", p_set=100.0)
    # The Sofia converter pair: AC between themselves, DC to the mainland, no load.
    n.add("Transformer", "sofia_t", bus0="SOFWD", bus1="SOFW61", x=0.01, s_nom=9999.0)
    # Shetland: an AC island with demand, kept.
    n.add("Line", "shet", bus0="SHET1", bus1="SHET2", x=0.1, s_nom=100.0)
    n.add("Load", "shetland", bus="SHET2", p_set=20.0)
    for name in ("SOFW4A_SOFWD_ETYS_UPG_2025_HVDC", "SOFW4A_SOFWD_ETYS_UPG_2025_1_HVDC"):
        n.add("Link", name, bus0="SOFW4A", bus1="SOFWD", p_nom=660.0, p_min_pu=-1.0, p_max_pu=1.0,
              efficiency=1.0, carrier="DC")
    n.add("Link", "shetland_hvdc", bus0="MAIN1", bus1="SHET1", p_nom=600.0, p_min_pu=-1.0, efficiency=1.0, carrier="DC")
    n.add("Link", "cdc_link", bus0="MAIN1", bus1="CDC", p_nom=100.0, p_min_pu=-1.0, efficiency=1.0, carrier="DC")
    return n


def test_twin_poles_merge_into_one_link_of_the_summed_rating():
    n = _grid()
    audit = merge_parallel_internal_links(n, logging.getLogger("t"))
    assert "SOFW4A_SOFWD_ETYS_UPG_2025_1_HVDC" not in n.links.index
    kept = n.links.loc["SOFW4A_SOFWD_ETYS_UPG_2025_HVDC"]
    assert kept["p_nom"] == pytest.approx(1320.0)
    assert kept[MERGED_COLUMN] == "SOFW4A_SOFWD_ETYS_UPG_2025_1_HVDC"
    assert len(audit) == 1 and audit.iloc[0]["p_nom_after"] == pytest.approx(1320.0)
    # Unrelated links are untouched.
    assert n.links.at["shetland_hvdc", "p_nom"] == pytest.approx(600.0)


def test_reversed_pair_merges_only_when_symmetric_and_lossless():
    n = _grid()
    n.remove("Link", "SOFW4A_SOFWD_ETYS_UPG_2025_1_HVDC")
    n.add("Link", "reverse_pole", bus0="SOFWD", bus1="SOFW4A", p_nom=660.0, p_min_pu=-1.0, p_max_pu=1.0,
          efficiency=1.0, carrier="DC")
    merge_parallel_internal_links(n)
    assert "reverse_pole" not in n.links.index
    assert n.links.at["SOFW4A_SOFWD_ETYS_UPG_2025_HVDC", "p_nom"] == pytest.approx(1320.0)

    n = _grid()
    n.links.loc["SOFW4A_SOFWD_ETYS_UPG_2025_1_HVDC", "efficiency"] = 0.98   # a different asset
    n.add("Link", "lossy_reverse", bus0="SOFWD", bus1="SOFW4A", p_nom=10.0, p_min_pu=0.0, p_max_pu=1.0,
          efficiency=1.0, carrier="DC")
    audit = merge_parallel_internal_links(n)
    assert audit.empty
    assert {"SOFW4A_SOFWD_ETYS_UPG_2025_1_HVDC", "lossy_reverse"} <= set(n.links.index)


def test_external_and_time_varying_links_are_never_merged():
    n = _grid()
    for name in ("ic_a", "ic_b"):
        n.add("Link", name, bus0="MAIN1", bus1="France External", p_nom=1000.0, p_min_pu=-1.0,
              efficiency=1.0, carrier="DC")
    n.links_t.p_max_pu = pd.DataFrame({"SOFW4A_SOFWD_ETYS_UPG_2025_1_HVDC": [1.0, 0.5]}, index=n.snapshots)
    audit = merge_parallel_internal_links(n)
    assert audit.empty
    assert {"ic_a", "ic_b", "SOFW4A_SOFWD_ETYS_UPG_2025_1_HVDC"} <= set(n.links.index)


def test_load_free_dc_island_is_excluded_from_placement():
    n = _grid()
    assert set(dc_island_buses(n)) == {"SOFWD", "SOFW61"}
    reasons = generation_placement_exclusions(n, land_boundary_file=None)
    assert reasons["SOFWD"] == "dc_island" and reasons["SOFW61"] == "dc_island"
    assert reasons["CDC"] == "dc_only"
    for bus in ("SHET1", "SHET2", "MAIN1", "SOFW4A"):
        assert bus not in reasons.index


def test_stage1_link_flows_are_written_with_the_efficiency_leg():
    n = _grid()
    n.links.loc["shetland_hvdc", "efficiency"] = 0.97
    p0 = pd.DataFrame({"shetland_hvdc": [100.0, -50.0], "cdc_link": [5.0, 0.0]}, index=n.snapshots)
    write_link_flows(n, p0)
    assert n.links_t.p0["shetland_hvdc"].tolist() == [100.0, -50.0]
    assert n.links_t.p1["shetland_hvdc"].tolist() == pytest.approx([-97.0, 48.5])
    # Links absent from the table carry zero flow.
    assert n.links_t.p0["SOFW4A_SOFWD_ETYS_UPG_2025_HVDC"].eq(0.0).all()
    # A solved p1 is taken as given.
    solved_p1 = pd.DataFrame({"shetland_hvdc": [-90.0, 40.0]}, index=n.snapshots)
    write_link_flows(n, p0, {"p1": solved_p1})
    assert n.links_t.p1["shetland_hvdc"].tolist() == [-90.0, 40.0]
    # The saved flows are what the bound check reads.
    value, _ = max_bound_violation(n)
    assert np.isfinite(value)


def test_bound_threshold_is_one_megawatt():
    assert MAX_BOUND_VIOLATION_MW == pytest.approx(1.0)
    n = _grid()
    n.links_t.p0 = pd.DataFrame({"cdc_link": [101.5, 0.0]}, index=n.snapshots).reindex(
        columns=n.links.index, fill_value=0.0)
    value, where = max_bound_violation(n)
    assert value == pytest.approx(1.5) and where == "Link cdc_link"
    assert value > MAX_BOUND_VIOLATION_MW
