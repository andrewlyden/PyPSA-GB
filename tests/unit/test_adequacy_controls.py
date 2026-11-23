import pypsa
import pytest

from scripts.market.adequacy_controls import apply_adequacy_relaxation


def network():
    n = pypsa.Network()
    n.add("Bus", "A", v_nom=400)
    n.add("Bus", "B", v_nom=400)
    n.add("Line", "L", bus0="A", bus1="B", r=0.01, x=0.1, s_nom=100)
    n.add("Transformer", "T", bus0="A", bus1="B", r=0.01, x=0.1, s_nom=200)
    return n


def test_all_ac_relaxation_is_audited():
    n = network()
    audit = apply_adequacy_relaxation(n, {"relaxation": {"mode": "all_ac", "capacity_mw": 1000}})
    assert set(audit.component) == {"L", "T"}
    assert n.lines.at["L", "s_nom"] == 1000
    assert n.transformers.at["T", "s_nom"] == 1000
    assert audit.diagnostic_only.all()


def test_component_relaxation_is_targeted_and_validated():
    n = network()
    apply_adequacy_relaxation(n, {"relaxation": {"mode": "components", "capacity_mw": 500, "lines": ["L"]}})
    assert n.lines.at["L", "s_nom"] == 500
    assert n.transformers.at["T", "s_nom"] == 200
    with pytest.raises(ValueError, match="Unknown line"):
        apply_adequacy_relaxation(n, {"relaxation": {"mode": "components", "lines": ["missing"]}})
