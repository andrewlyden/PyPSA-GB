import logging

import pandas as pd
import pypsa
import pytest

from scripts.CP30.hnd_hybrid import (
    apply_hnd_hybrid_topology,
    apply_hnd_project_allocation,
)
from scripts.generators.integrate_renewable_generators import add_renewable_generators


def _network():
    network = pypsa.Network()
    network.add("Bus", "ETYS_A", v_nom=400.0, x=0.0, y=0.0)
    network.add("Bus", "ETYS_B", v_nom=400.0, x=1.0, y=0.0)
    return network


def test_hnd_topology_adds_documented_assets_and_audits(tmp_path):
    register = tmp_path / "assets.csv"
    audit = tmp_path / "audit.csv"
    pd.DataFrame([
        {
            "asset_id": "HND_BUS", "component_type": "bus", "name": "HND node",
            "bus0": "HND_NODE", "bus1": "", "v_nom": 400.0, "x": 2.0, "y": 0.0,
            "in_service_year": 2030, "etys_overlap_status": "not_in_etys",
            "source_url": "https://example.test/hnd", "evidence_reference": "table 1",
        },
        {
            "asset_id": "HND_LINK", "component_type": "link", "name": "HND HVDC",
            "bus0": "HND_NODE", "bus1": "ETYS_A", "p_nom": 2000.0, "efficiency": 0.98,
            "in_service_year": 2030, "etys_overlap_status": "not_in_etys",
            "source_url": "https://example.test/hnd", "evidence_reference": "table 1",
        },
    ]).to_csv(register, index=False)

    network = _network()
    result = apply_hnd_hybrid_topology(
        network, {"enabled": True, "asset_register": register}, 2030, audit,
        logging.getLogger(__name__),
    )

    assert set(result["audit_status"]) == {"added"}
    assert "HND_NODE" in network.buses.index
    assert network.links.at["HND HVDC", "p_nom"] == 2000.0
    assert audit.is_file()


def test_hnd_topology_rejects_unreviewed_etys_overlap(tmp_path):
    register = tmp_path / "assets.csv"
    pd.DataFrame([{
        "asset_id": "HND_EXISTING", "component_type": "bus", "name": "existing",
        "bus0": "ETYS_A", "bus1": "", "in_service_year": 2030,
        "etys_overlap_status": "not_in_etys", "source_url": "https://example.test/hnd",
        "evidence_reference": "table 1",
    }]).to_csv(register, index=False)

    with pytest.raises(ValueError, match="already exists in ETYS"):
        apply_hnd_hybrid_topology(
            _network(), {"enabled": True, "asset_register": register}, 2030,
            tmp_path / "audit.csv", logging.getLogger(__name__),
        )


def test_hnd_project_allocation_preserves_cp30_regional_capacity_and_profiles(tmp_path):
    portfolio = tmp_path / "portfolio.csv"
    crosswalk = tmp_path / "crosswalk.csv"
    pd.DataFrame([
        {"map_id": "A", "capacity_mw": 1000.0},
        {"map_id": "B", "capacity_mw": 3000.0},
    ]).to_csv(portfolio, index=False)
    pd.DataFrame([
        {"map_id": "A", "carrier": "wind_offshore", "cp30_region": "North", "connection_bus": "HND_A", "allocation_share": 1.0, "source": "test"},
        {"map_id": "B", "carrier": "wind_offshore", "cp30_region": "North", "connection_bus": "HND_B", "allocation_share": 1.0, "source": "test"},
    ]).to_csv(crosswalk, index=False)
    network = _network()
    network.add("Bus", "HND_A", v_nom=400.0, x=2.0, y=0.0)
    network.add("Bus", "HND_B", v_nom=400.0, x=3.0, y=0.0)
    scaled = pd.DataFrame([
        {"technology": "wind_offshore", "cp30_region": "North", "bus": "ETYS_A", "capacity_mw": 40.0, "profile": "p1"},
        {"technology": "wind_offshore", "cp30_region": "North", "bus": "ETYS_B", "capacity_mw": 60.0, "profile": "p2"},
    ])

    result = apply_hnd_project_allocation(
        scaled, network,
        {"enabled": True, "portfolio_file": portfolio, "crosswalk_file": crosswalk},
        logging.getLogger(__name__),
    )

    assert result.groupby("cp30_region")["capacity_mw"].sum().to_dict() == {"North": 100.0}
    assert result.groupby("bus")["capacity_mw"].sum().to_dict() == {"HND_A": 25.0, "HND_B": 75.0}
    assert result.groupby("profile")["capacity_mw"].sum().to_dict() == {"p1": 40.0, "p2": 60.0}


def test_public_portfolio_proxy_preserves_all_cp30_regions(tmp_path):
    portfolio = tmp_path / "portfolio.csv"
    proxy = tmp_path / "proxy.csv"
    pd.DataFrame([
        {"map_id": "EAST", "capacity_mw": 3.0},
        {"map_id": "NORTH", "capacity_mw": 1.0},
    ]).to_csv(portfolio, index=False)
    pd.DataFrame([
        {"map_id": "EAST", "carrier": "wind_offshore", "cp30_region": "East", "connection_bus": "HND_A", "allocation_share": 1.0, "source": "test", "mapping_basis": "interface proxy", "evidence_status": "public_interface_proxy"},
        {"map_id": "NORTH", "carrier": "wind_offshore", "cp30_region": "North", "connection_bus": "HND_B", "allocation_share": 1.0, "source": "test", "mapping_basis": "interface proxy", "evidence_status": "public_interface_proxy"},
    ]).to_csv(proxy, index=False)
    network = _network()
    network.add("Bus", "HND_A", v_nom=400.0, x=2.0, y=0.0)
    network.add("Bus", "HND_B", v_nom=400.0, x=3.0, y=0.0)
    scaled = pd.DataFrame([
        {"technology": "wind_offshore", "cp30_region": "East", "bus": "ETYS_A", "capacity_mw": 40.0},
        {"technology": "wind_offshore", "cp30_region": "North", "bus": "ETYS_B", "capacity_mw": 60.0},
        {"technology": "wind_offshore", "cp30_region": "Untouched", "bus": "ETYS_A", "capacity_mw": 20.0},
    ])

    result = apply_hnd_project_allocation(
        scaled, network,
        {"enabled": True, "allocation_mode": "public_portfolio_proxy", "portfolio_file": portfolio, "crosswalk_file": proxy},
        logging.getLogger(__name__),
    )

    assert result.groupby("cp30_region")["capacity_mw"].sum().to_dict() == {
        "East": 40.0, "North": 60.0, "Untouched": 20.0,
    }
    assert set(result.loc[result["cp30_region"].ne("Untouched"), "data_source"]) == {
        "CP30_HND_public_portfolio_proxy"
    }


def test_profile_split_rows_receive_unique_generator_names():
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    network.add("Bus", "A", v_nom=400.0, x=0.0, y=0.0)
    network.add("Bus", "B", v_nom=400.0, x=1.0, y=0.0)
    network.add("Carrier", "wind_offshore")
    sites = pd.DataFrame([
        {"site_name": "shared_fes_profile", "technology": "wind_offshore", "bus": "A", "capacity_mw": 40.0},
        {"site_name": "shared_fes_profile", "technology": "wind_offshore", "bus": "B", "capacity_mw": 60.0},
    ])

    add_renewable_generators(network, sites, profiles_dir=None)

    assert network.generators.p_nom.sum() == 100.0
    assert set(network.generators.index) == {"shared_fes_profile", "shared_fes_profile_1"}
