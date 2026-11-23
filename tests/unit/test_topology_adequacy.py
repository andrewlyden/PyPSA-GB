import pandas as pd
import pypsa

from scripts.analysis.diagnose_topology_adequacy import compare_corridors


def _network_with_corridor(buses, lines):
    network = pypsa.Network()
    for bus in buses:
        network.add("Bus", bus)
    for name, bus0, bus1, capacity in lines:
        network.add("Line", name, bus0=bus0, bus1=bus1, s_nom=capacity, x=0.1)
    return network


def test_clustered_corridor_envelope_reconciles_to_preclustering_cutset():
    unclustered = _network_with_corridor(
        ["a", "b", "c"],
        [("ab", "a", "b", 100.0), ("cb", "c", "b", 25.0)],
    )
    clustered = _network_with_corridor(
        ["cluster_0", "cluster_1"],
        [("corridor", "cluster_0", "cluster_1", 125.0)],
    )
    busmap = pd.Series({"a": "cluster_0", "b": "cluster_1", "c": "cluster_0"})

    result = compare_corridors(
        unclustered, clustered, busmap, {"cluster_0", "cluster_1"}
    )
    corridor = result.iloc[0]

    assert corridor["capacity_MW_unclustered"] == 125.0
    assert corridor["capacity_MW_clustered"] == 125.0
    assert corridor["capacity_ratio_clustered_to_unclustered"] == 1.0
    assert bool(corridor["touches_focus_cluster"])

