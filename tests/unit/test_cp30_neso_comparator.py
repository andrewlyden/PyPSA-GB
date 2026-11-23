import logging

import pandas as pd

from scripts.CP30.reinforcements import audit_cp30_reinforcements
from scripts.CP30.reinforcement_layer import apply_cp30_reinforcement_layer
from scripts.CP30.connection_allocation import _add_explicit_export_links
from scripts.CP30.spatial_allocation import apply_cp30_es5_regional_allocation
from tests.helpers import requires_outputs


class _Network:
    def __init__(self):
        self.buses = pd.DataFrame(
            {
                "x": [0.0, 100.0, 0.0, 100.0],
                "y": [0.0, 0.0, 0.0, 0.0],
            },
            index=["north_proxy", "south_proxy", "north_anchor", "south_anchor"],
        )


def test_es5_allocation_matches_regional_capacity(tmp_path):
    es5 = tmp_path / "es5.csv"
    anchors = tmp_path / "anchors.csv"
    pd.DataFrame(
        [{
            "source_sheet": "ES5", "Scenario": "Further Flex and Renewables",
            "Technology": "Solar", "Variable": "Capacity (MW)",
            "North": 30.0, "South": 70.0,
        }]
    ).to_csv(es5, index=False)
    pd.DataFrame([
        {"cp30_region": "North", "bus": "north_anchor", "source": "test"},
        {"cp30_region": "South", "bus": "south_anchor", "source": "test"},
    ]).to_csv(anchors, index=False)
    scaled = pd.DataFrame([
        {"technology": "solar_pv", "bus": "north_proxy", "capacity_mw": 50.0},
        {"technology": "solar_pv", "bus": "south_proxy", "capacity_mw": 50.0},
    ])

    result = apply_cp30_es5_regional_allocation(
        scaled, es5, "Further Flex and Renewables", 2030, _Network(),
        {"enabled": True, "region_anchor_file": anchors}, logging.getLogger(__name__),
    )

    actual = result.groupby("cp30_region")["capacity_mw"].sum().to_dict()
    assert actual == {"North": 30.0, "South": 70.0}
    assert set(result["cp30_spatial_method"]) == {"nearest_configured_etys_region_anchor"}


def test_es5_anchor_fallback_is_not_converted_to_an_offshore_link():
    import pypsa

    network = pypsa.Network()
    network.add("Bus", "regional_anchor", v_nom=400.0, x=0.0, y=0.0)
    scaled = pd.DataFrame([{
        "technology": "wind_offshore",
        "bus": "regional_anchor",
        "capacity_mw": 1035.0,
        "cp30_anchor_proxy": True,
    }])
    mapping = pd.DataFrame([{
        "carrier": "wind_offshore",
        "source_bus": "an_actual_fes_proxy_bus",
        "landing_bus": "regional_anchor",
        "source": "test mapping",
    }])

    result = _add_explicit_export_links(
        scaled, mapping, network, {}, logging.getLogger(__name__)
    )

    assert result.loc[0, "bus"] == "regional_anchor"
    assert result.loc[0, "data_source"] == "CP30_ES5_regional_anchor_proxy"
    assert network.links.empty


def test_reinforcement_audit_keeps_late_required_project_visible(tmp_path):
    reference = tmp_path / "reference.csv"
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "audit.csv"
    pd.DataFrame([
        {
            "source_sheet": "N1", "Code": "AENC", "Description": "East Anglia circuit",
            "Latest Delivery Year (updated September 2024)": 2031,
            "Required for CP30": "ü",
        },
        {
            "source_sheet": "N1", "Code": "OTHER", "Description": "Later optional work",
            "Latest Delivery Year (updated September 2024)": 2031,
            "Required for CP30": None,
        },
    ]).to_csv(reference, index=False)
    pd.DataFrame([{
        "project_code": "AENC", "representation_status": "requires_electrical_mapping",
        "evidence": "test",
    }]).to_csv(manifest, index=False)

    audit = audit_cp30_reinforcements(
        reference, manifest, 2030, output, logging.getLogger(__name__)
    )

    assert audit["project_code"].tolist() == ["AENC"]
    assert bool(audit.loc[0, "cp30_required"])
    assert not bool(audit.loc[0, "delivery_due_by_model_year"])
    assert audit.loc[0, "coverage_class"] == "unreviewed_transfer_or_connection_asset"
    assert audit.loc[0, "mapping_action"] == "review_etys_appendix_b_or_project_electrical_data"
    assert output.is_file()


class _LayerNetwork:
    def __init__(self):
        self.buses = pd.DataFrame(
            {
                "v_nom": [400.0] * 5,
                "x": [0.0, 1.0, 2.0, 3.0, 4.0],
                "y": [0.0] * 5,
            },
            index=["BRFO41", "NORM41", "TILB41", "RICH41", "SIZE41"],
        )
        self.lines = pd.DataFrame()
        self.links = pd.DataFrame()

    def add(self, component, name, **attributes):
        table = {"Bus": self.buses, "Line": self.lines, "Link": self.links}[component]
        for column in attributes:
            if column not in table.columns:
                table[column] = pd.NA
        table.loc[name, list(attributes)] = list(attributes.values())


def test_accelerated_layer_adds_three_critical_cp30_projects():
    projects = apply_cp30_reinforcement_layer(
        _LayerNetwork(),
        {
            "enabled": True,
            "components_file": "data/network/ETYS/cp30_n1_accelerated_components.csv",
        },
        2030,
        logging.getLogger(__name__),
    )

    assert projects == {"AENC", "ATNC", "SCD1"}


@requires_outputs("resources/CP30/CP30_reference_N1_N2_NA1.csv")
def test_audit_confirms_all_five_prioritised_projects(monkeypatch, tmp_path):
    class _NetworkWithEgl:
        links = pd.DataFrame([
            {"bus0": "BRNX4A", "bus1": "HAWP41", "p_nom": 2000.0},
            # B-5-1 names PEHE4-; the build resolves it to the PEHE4J busbar
            # (manifest fixed 15 Oct 2026, network audit issue 10c).
            {"bus0": "DRAX41", "bus1": "PEHE4J", "p_nom": 2000.0},
        ])

    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: None)
    audit = audit_cp30_reinforcements(
        "resources/CP30/CP30_reference_N1_N2_NA1.csv",
        "data/network/ETYS/cp30_n1_reinforcement_manifest.csv",
        2030,
        tmp_path / "audit.csv",
        logging.getLogger(__name__),
        applied_projects={"AENC", "ATNC", "SCD1"},
        network=_NetworkWithEgl(),
    )

    status = audit.set_index("project_code")["audit_status"]
    assert status[["AENC", "ATNC", "SCD1"]].eq("represented_in_cp30_layer").all()
    assert status[["E2DC", "E4D3"]].eq("represented_in_etys").all()
    coverage = audit.set_index("project_code")["coverage_class"]
    assert coverage[["AENC", "ATNC", "SCD1", "E2DC", "E4D3"]].eq(
        "represented_source_backed"
    ).all()


@requires_outputs("resources/CP30/CP30_reference_N1_N2_NA1.csv")
def test_audit_requires_every_component_of_a_multi_asset_mapping(monkeypatch, tmp_path):
    class _NetworkWithDlup:
        links = pd.DataFrame()
        lines = pd.DataFrame([
            {"bus0": "DENN4-", "bus1": "LAMB4T", "s_nom": 2000.0},
            {"bus0": "LAMB4T", "bus1": "WIYH4Q", "s_nom": 2000.0},
        ])
        transformers = pd.DataFrame([
            {"bus0": "WIYH4Q", "bus1": "WIYH2-", "s_nom": 1000.0},
        ])

    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: None)
    audit = audit_cp30_reinforcements(
        "resources/CP30/CP30_reference_N1_N2_NA1.csv",
        "data/network/ETYS/cp30_n1_reinforcement_manifest.csv",
        2030,
        tmp_path / "audit.csv",
        logging.getLogger(__name__),
        network=_NetworkWithDlup(),
    )
    assert audit.set_index("project_code").at["DLUP", "audit_status"] == "represented_in_etys"

    _NetworkWithDlup.transformers = pd.DataFrame()
    incomplete = audit_cp30_reinforcements(
        "resources/CP30/CP30_reference_N1_N2_NA1.csv",
        "data/network/ETYS/cp30_n1_reinforcement_manifest.csv",
        2030,
        tmp_path / "incomplete.csv",
        logging.getLogger(__name__),
        network=_NetworkWithDlup(),
    )
    assert incomplete.set_index("project_code").at["DLUP", "audit_status"] == "expected_etys_component_missing"
