"""Network delivery cases in the balancing stage (8 Oct 2026): delayed projects and their boundary years."""
import json

import pypsa
import pytest

from scripts.market.network_delivery import apply_network_delivery, network_delivery_config


def _network():
    n = pypsa.Network()
    for bus in ("A", "B", "C", "D"):
        n.add("Bus", bus, v_nom=400.0)
    n.add("Line", "AB_old", bus0="A", bus1="B", x=0.1, s_nom=1000)
    n.add("Line", "CP30_X_AB_new", bus0="A", bus1="B", x=0.1, s_nom=3000)   # a parallel new circuit
    n.add("Line", "CP30_X_BC", bus0="B", bus1="C", x=0.1, s_nom=3000)        # the only path to C
    n.add("Line", "CP30_X_CD", bus0="C", bus1="D", x=0.1, s_nom=3000)        # D is empty
    n.add("Link", "HVDC_AC", bus0="A", bus1="C", p_nom=2000)
    n.add("Generator", "wind_C", bus="C", p_nom=500)
    n.add("Load", "load_A", bus="A", p_set=100)
    n.add("Generator", "shed_D", bus="D", p_nom=10, carrier="load_shedding")   # slack never occupies a bus
    return n


def _scenario():
    return {"network": {"transmission": {"outage_schedule": {"enabled": True, "secured_capability": {
        "fes_pathway": "HT", "capability_year": 2030, "capability_year_overrides": {"B8": 2029, "EC5": 2031},
        "include": {"B4": [{"component": "link", "name": "HVDC_AC", "sign": -1.0}]}}}}}}


def _balancing():
    return {"network_delivery": {
        "case": "test_late",
        "remove_components": [{"component": "link", "name": "HVDC_AC"},
                              {"component": "line", "prefix": "CP30_X_"}],
        "secured_capability": {"capability_year_overrides": {"B8": 2029, "EC5": 2030}}}}


def test_absent_block_changes_nothing():
    n, scenario = _network(), _scenario()
    assert apply_network_delivery(n, scenario, {}, None) is None
    assert len(n.lines) == 4 and scenario == _scenario()


def test_links_are_disabled_and_branches_removed_unless_they_strand_a_bus():
    n, scenario = _network(), _scenario()
    record = apply_network_delivery(n, scenario, _balancing(), None)
    assert n.links.at["HVDC_AC", "p_nom"] == 0.0 and record["links_disabled"] == ["HVDC_AC"]
    # The parallel circuit goes; B-C is kept because C carries generation once the link is off;
    # C-D goes because D carries nothing.
    assert "CP30_X_AB_new" not in n.lines.index and "CP30_X_CD" not in n.lines.index
    assert "CP30_X_BC" in n.lines.index
    assert record["kept_as_connection"] == [{"component": "line", "name": "CP30_X_BC"}]
    assert json.loads(n.meta["network_delivery"])["case"] == "test_late"


def test_capability_keys_replace_and_the_rest_of_the_block_stays():
    n, scenario = _network(), _scenario()
    apply_network_delivery(n, scenario, _balancing(), None)
    secured = scenario["network"]["transmission"]["outage_schedule"]["secured_capability"]
    assert secured["capability_year_overrides"] == {"B8": 2029, "EC5": 2030}
    assert secured["capability_year"] == 2030 and secured["include"]["B4"][0]["name"] == "HVDC_AC"


@pytest.mark.parametrize("block, message", [
    ({"remove_components": []}, "case name"),
    ({"case": "x", "remove_components": [{"component": "bus", "name": "A"}]}, "unknown component"),
    ({"case": "x", "remove_components": [{"component": "line"}]}, "exactly one"),
    ({"case": "x", "secured_capability": {"geometry_file": "y"}}, "unknown keys"),
])
def test_bad_blocks_are_refused(block, message):
    with pytest.raises(ValueError, match=message):
        network_delivery_config({"network_delivery": block})


def test_a_name_not_in_the_network_is_refused():
    with pytest.raises(ValueError, match="not in the network"):
        apply_network_delivery(_network(), _scenario(), {"network_delivery": {
            "case": "x", "remove_components": [{"component": "link", "name": "NOPE"}]}}, None)


def test_outage_and_phase_in_keys_are_accepted():
    from scripts.market.network_delivery import network_delivery_config
    block = {"case": "outage_neso", "secured_capability": {
        "historic_derate": {"enabled": True, "mode": "replay", "file": "x.csv"},
        "phase_in": {"enabled": True, "before_year": 2029, "from": "2030-04-01"}}}
    assert network_delivery_config({"network_delivery": block}) is block
