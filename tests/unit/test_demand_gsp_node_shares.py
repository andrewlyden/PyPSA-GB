"""GSP -> ETYS node split must keep every GSP's demand on the network.

The CP30 2030 build lost 7.3 TWh (307.886 of 315.163 TWh) because shares of
Dem_per_node nodes absent from the network were dropped silently.
"""
import logging

import pandas as pd
import pypsa
import pytest

from scripts.demand.load import (
    assert_demand_conserved,
    distribute_future_fes_demand_to_etys_nodes,
    resolve_gsp_node_shares,
)

LOGGER = logging.getLogger("test_demand_gsp_node_shares")


def _network(buses):
    n = pypsa.Network()
    for name, (x, y, v) in buses.items():
        n.add("Bus", name, x=x, y=y, v_nom=v)
    return n


def test_all_listed_nodes_present_keeps_dem_per_node_weights():
    n = _network({"AAAA11": (400000, 300000, 132), "AAAA12": (400000, 300000, 132)})
    shares, method = resolve_gsp_node_shares([("AAAA11", 0.3), ("AAAA12", 0.1)], n)
    assert method == "dem_per_node"
    assert dict(shares) == pytest.approx({"AAAA11": 0.75, "AAAA12": 0.25})


def test_partly_missing_gsp_is_renormalised_over_present_nodes():
    # WIMBS1 on the CP30 build: WIMB12 missing, WIMB13 present
    n = _network({"WIMB13": (525000, 170000, 132)})
    shares, method = resolve_gsp_node_shares([("WIMB12", 0.5), ("WIMB13", 0.5)], n)
    assert method == "renormalised"
    assert shares == [("WIMB13", pytest.approx(1.0))]


def test_wimbledon_south_missing_board_goes_to_wimb14_not_all_on_wimb13():
    # Fix S4 27 Sep 2026: WIMB13 carried the whole 2.8 TWh GSP (0.99 of its
    # 500 MVA) while WIMB14 had its own 500 MVA circuit and 98 MW.
    n = _network({"WIMB13": (525000, 170000, 132), "WIMB14": (525000, 170000, 132)})
    shares, method = resolve_gsp_node_shares([("WIMB12", 0.048569), ("WIMB13", 0.048569)], n)
    assert method == "dem_per_node+successor"
    assert dict(shares) == pytest.approx({"WIMB14": 0.5, "WIMB13": 0.5})


def test_wholly_missing_gsp_goes_to_same_substation_same_voltage():
    # HARK11 missing; HARK13 (132 kV) is preferred over HARK41 (400 kV)
    n = _network({
        "HARK13": (340000, 560000, 132),
        "HARK41": (340000, 560000, 400),
        "FOUR71": (320000, 560000, 20),
    })
    shares, method = resolve_gsp_node_shares([("HARK11", 1.0)], n)
    assert method == "substation"
    assert shares == [("HARK13", pytest.approx(1.0))]


def test_retired_node_goes_to_its_successor_board_not_the_orphaned_sibling():
    # Review 26 Sep 2026: Appendix B 2027 removes the HARK21-HARK1x transformers,
    # so HARK13 keeps only 331 MVA of 132 kV lines; the new board is HARG11.
    n = _network({
        "HARK13": (338315, 561188, 132),
        "HARG11": (332787, 574279, 132),
        "RRIG11": (330000, 570000, 132),
    })
    shares, method = resolve_gsp_node_shares([("HARK11", 1.0)], n)
    assert method == "dem_per_node+successor"
    assert shares == [("HARG11", pytest.approx(1.0))]
    # RRIG: its HARK11/HARK12 shares both land on HARG11, merged into one row.
    shares, method = resolve_gsp_node_shares([("RRIG11", 0.9), ("HARK11", 0.05), ("HARK12", 0.05)], n)
    assert method == "dem_per_node+successor"
    assert dict(shares) == pytest.approx({"RRIG11": 0.9, "HARG11": 0.1})


def test_substation_without_same_voltage_uses_any_sibling():
    n = _network({"ALVE4A": (250000, 125000, 400), "LAND11": (240000, 80000, 132)})
    shares, method = resolve_gsp_node_shares([("ALVE11", 1.0)], n)
    assert method == "substation"
    assert shares == [("ALVE4A", pytest.approx(1.0))]


def test_no_bus_at_substation_falls_back_to_nearest_demand_node():
    # OSGB36 network; the GSP centroid is given in WGS84 and converted
    n = _network({
        "NEAR11": (340000, 730000, 132),   # Dundee area
        "FAR011": (530000, 180000, 132),   # London
    })
    shares, method = resolve_gsp_node_shares(
        [("DUDH3-", 1.0)], n, gsp_lonlat=(-2.98, 56.47), candidate_buses=["NEAR11", "FAR011"]
    )
    assert method == "nearest"
    assert shares == [("NEAR11", 1.0)]


def test_unresolvable_gsp_returns_empty():
    n = _network({"FAR011": (530000, 180000, 132)})
    assert resolve_gsp_node_shares([("DUDH3-", 1.0)], n) == ([], "unresolved")


def test_conservation_check_raises_on_lost_energy():
    assert_demand_conserved(315_162.7, 315_162.7 * (1 + 1e-8), "stage", LOGGER)
    with pytest.raises(ValueError, match="not conserved"):
        assert_demand_conserved(315_162.7, 307_885.9, "stage", LOGGER)


def test_future_split_conserves_demand_when_listed_nodes_are_missing(monkeypatch, tmp_path):
    """End to end on a toy network: a GSP whose only listed node is absent."""
    meta = tmp_path / "gsp_info.csv"
    pd.DataFrame({
        "GSP ID": ["HARK_1", "ABHA1"],
        "Name": ["Harker", "Abham"],
        "Latitude": [54.93, 50.47],
        "Longitude": [-2.95, -3.72],
    }).to_csv(meta, index=False)
    dem_per_node = pd.DataFrame({
        "GSP Id": ["HARK_1", "ABHA1"],
        "Node Id": ["HARK11", "ABHA11"],
        "GSP Group ID": ["_G", "_L"],
        "Dem as % of demand within the GSP Group ID per each node": [1.0, 1.0],
    })
    import scripts.demand.load as demand_load
    monkeypatch.setattr(demand_load, "resolve_gsp_metadata_path", lambda *a, **k: meta)
    monkeypatch.setattr(demand_load.pd, "read_excel", lambda *a, **k: dem_per_node)
    monkeypatch.setattr(
        demand_load, "redistribute_demand_by_transformer_capacity",
        lambda node_weights, **kwargs: node_weights,
    )

    n = _network({"HARK13": (340000, 560000, 132), "ABHA11": (280000, 70000, 132)})
    n.add("Line", "l", bus0="HARK13", bus1="ABHA11", x=0.1, s_nom=1000)
    index = pd.date_range("2030-01-01", periods=3, freq="h")
    timeseries = pd.DataFrame({"Harker": [100.0, 110.0, 120.0], "Abham": [50.0, 55.0, 60.0]}, index=index)
    annual = pd.DataFrame({"demand": timeseries.sum() / 1000.0})

    demand, node_ts = distribute_future_fes_demand_to_etys_nodes(annual, timeseries, n, LOGGER, fes_year=2025)

    assert node_ts.to_numpy().sum() == pytest.approx(timeseries.to_numpy().sum())
    assert node_ts["HARK13"].tolist() == pytest.approx([100.0, 110.0, 120.0])
    assert demand.iloc[:, 0].sum() == pytest.approx(annual["demand"].sum())


# --- GSPs with no Dem_per_node row under their FES name (40 GSPs, 21.0 TWh on
# --- the CP30 2030 build) are placed at their own location, not pro rata.

def _unlisted_inputs():
    from scripts.demand.load import canonical_gsp_name
    canonical_codes = {canonical_gsp_name(name): [code] for name, code in [
        ("St John's Wood", "SJOW_1"), ("Sandford", "SAFO_1"), ("Wesley Avenue (Willesden 66kV)", "WESA_H")]}
    gsp_code_to_nodes = {"SJOW_1": [("SJOW11", 0.14)], "WISD_1": [("WISD11", 0.05)]}
    return canonical_codes, gsp_code_to_nodes


def test_fes_label_variant_finds_the_metadata_code():
    from scripts.demand.load import resolve_unlisted_gsp
    n = _network({"SJOW11": (527000, 183000, 132), "FAR011": (330000, 670000, 132)})
    canonical_codes, code_to_nodes = _unlisted_inputs()
    shares, method = resolve_unlisted_gsp(
        "St Johns Wood", None, canonical_codes, code_to_nodes, {}, n, {}, ["SJOW11", "FAR011"])
    assert shares == [("SJOW11", pytest.approx(1.0))]
    assert method.startswith("name_variant:SJOW_1")


def test_code_without_dem_per_node_row_uses_its_site_lowest_voltage():
    from scripts.demand.load import resolve_unlisted_gsp
    n = _network({"SAFO41": (345000, 160000, 400), "SAFO11": (345000, 160000, 132)})
    canonical_codes, code_to_nodes = _unlisted_inputs()
    shares, method = resolve_unlisted_gsp(
        "Sandford", "SAFO_1", canonical_codes, code_to_nodes, {}, n, {}, [])
    assert shares == [("SAFO11", 1.0)]
    assert method == "code_site:SAFO_1"


def test_declared_location_tries_alternatives_in_order():
    from scripts.demand.load import resolve_unlisted_gsp
    overrides = {"harris": {"gsp_name": "Harris", "locations": ["HARI", "LEWI", "ARMO"],
                            "lonlat": (-6.8, 57.9), "evidence": ""}}
    n = _network({"ARMO1J": (122000, 861000, 132), "ARMO3-": (122000, 861000, 33)})
    shares, method = resolve_unlisted_gsp("Harris", None, {}, {}, overrides, n, {}, ["ARMO3-"])
    assert (shares, method) == ([("ARMO3-", 1.0)], "declared:ARMO")
    # Once the Western Isles link builds the Lewis 132 kV buses, Harris goes there
    n.add("Bus", "LEWI1J", x=143000, y=933000, v_nom=132)
    n.add("Bus", "LEWI4R", x=143000, y=933000, v_nom=400)
    shares, method = resolve_unlisted_gsp("Harris", None, {}, {}, overrides, n, {}, ["ARMO3-"])
    assert (shares, method) == ([("LEWI1J", 1.0)], "declared:LEWI")


def test_declared_exact_bus_and_eleven_kv_generator_buses():
    from scripts.demand.load import site_demand_buses
    n = _network({"WISD61": (520000, 183000, 66), "WISD11": (520000, 183000, 132),
                  "KILB3-": (250000, 670000, 33), "KILB5-": (250000, 670000, 11)})
    assert site_demand_buses("WISD61", n) == ["WISD61"]
    assert site_demand_buses("KILB", n) == ["KILB3-"]
    assert site_demand_buses("NONE", n) == []


def test_repo_override_table_is_valid():
    from scripts.demand.load import GSP_LOCATION_OVERRIDES_FILE, load_gsp_location_overrides
    overrides = load_gsp_location_overrides(GSP_LOCATION_OVERRIDES_FILE)
    assert len(overrides) >= 10
    assert all(entry["locations"] for entry in overrides.values())
    assert overrides["stannahsgate"]["locations"] == ["STNG"]


def test_future_split_places_unlisted_gsp_locally_not_pro_rata(monkeypatch, tmp_path):
    meta = tmp_path / "gsp_info.csv"
    pd.DataFrame({
        "GSP ID": ["CONQA1", "ABHA1"],
        "Name": ["Connah's Quay", "Abham"],
        "Latitude": [53.19, 50.47],
        "Longitude": [-3.29, -3.72],
    }).to_csv(meta, index=False)
    dem_per_node = pd.DataFrame({
        "GSP Id": ["CONQA1", "ABHA1"],
        "Node Id": ["CONQ11", "ABHA11"],
        "GSP Group ID": ["_D", "_L"],
        "Dem as % of demand within the GSP Group ID per each node": [1.0, 1.0],
    })
    import scripts.demand.load as demand_load
    monkeypatch.setattr(demand_load, "resolve_gsp_metadata_path", lambda *a, **k: meta)
    monkeypatch.setattr(demand_load.pd, "read_excel", lambda *a, **k: dem_per_node)
    monkeypatch.setattr(demand_load, "load_gsp_location_overrides", lambda *a, **k: {})
    monkeypatch.setattr(
        demand_load, "redistribute_demand_by_transformer_capacity",
        lambda node_weights, **kwargs: node_weights,
    )
    n = _network({"CONQ11": (329000, 369000, 132), "ABHA11": (280000, 70000, 132)})
    n.add("Line", "l", bus0="CONQ11", bus1="ABHA11", x=0.1, s_nom=1000)
    index = pd.date_range("2030-01-01", periods=2, freq="h")
    # FES spells it "Connahs Quay"; the metadata "Connah's Quay"
    timeseries = pd.DataFrame({"Connahs Quay": [200.0, 210.0], "Abham": [50.0, 55.0]}, index=index)
    annual = pd.DataFrame({"demand": timeseries.sum() / 1000.0})

    _, node_ts = distribute_future_fes_demand_to_etys_nodes(annual, timeseries, n, LOGGER, fes_year=2025)

    assert node_ts["CONQ11"].tolist() == pytest.approx([200.0, 210.0])
    assert node_ts["ABHA11"].tolist() == pytest.approx([50.0, 55.0])
