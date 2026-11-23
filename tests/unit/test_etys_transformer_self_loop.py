"""A transformer addition from a bus to itself is skipped as a self-loop, as circuits are (15 Oct 2026).

ETYS 2024 Appendix B has a 2029 row HIGM41-HIGM41 that the build had added as a 2,750 MVA transformer.
"""

import pypsa

from scripts.network_build.ETYS_upgrades import apply_transformer_additions


def _network():
    n = pypsa.Network()
    n.add("Bus", "HIGM41", v_nom=400.0)
    n.add("Bus", "HIGM21", v_nom=275.0)
    return n


def test_self_loop_transformer_is_skipped_and_reported():
    n = _network()
    record = {"Node1": "HIGM41", "Node2": "HIGM41", "Year": 2029, "Rating (MVA)": 2750}
    ok, failed, details = apply_transformer_additions(n, [record])
    assert ok == 0 and failed == 1
    assert details[0]["reason"] == "self_loop"
    assert n.transformers.empty
