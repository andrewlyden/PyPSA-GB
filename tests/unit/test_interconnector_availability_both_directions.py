"""An interconnector outage limits imports and exports (fix S1, 27 Sep 2026).

apply_availability_profiles wrote links_t.p_max_pu only, so a signed cable
(p_min_pu = -1) on outage could not import but could still export at full
rating.
"""
import pandas as pd
import pypsa
import pytest

from scripts.interconnectors.add_to_network import apply_availability_profiles


def _network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=4, freq="h"))
    for bus in ("GB", "FR"):
        n.add("Bus", bus)
    n.add("Link", "IC_Signed", bus0="FR", bus1="GB", p_nom=1000.0, p_min_pu=-1.0, p_max_pu=1.0,
          interconnector_name="Signed")
    n.add("Link", "IC_IMPORT_Pair", bus0="FR", bus1="GB", p_nom=500.0, p_min_pu=0.0, p_max_pu=1.0,
          interconnector_name="Pair")
    return n


def _availability(name, values):
    return pd.DataFrame({"time": pd.date_range("2013-01-01", periods=len(values), freq="h"),
                         "name": name, "p_max_pu": values})


def test_outage_limits_both_directions_of_a_signed_link():
    n = _network()
    av = pd.concat([_availability("Signed", [1.0, 0.0, 0.5, 1.0]), _availability("Pair", [1.0, 0.0, 0.5, 1.0])])
    apply_availability_profiles(n, av)
    assert n.links_t.p_max_pu["IC_Signed"].tolist() == pytest.approx([1.0, 0.0, 0.5, 1.0])
    assert n.links_t.p_min_pu["IC_Signed"].tolist() == pytest.approx([-1.0, 0.0, -0.5, -1.0])
    # A one-way link keeps its zero floor: only its p_max_pu moves.
    assert n.links_t.p_max_pu["IC_IMPORT_Pair"].tolist() == pytest.approx([1.0, 0.0, 0.5, 1.0])
    assert "IC_IMPORT_Pair" not in n.links_t.p_min_pu.columns


def test_link_without_profile_is_fully_available_both_ways():
    n = _network()
    apply_availability_profiles(n, _availability("Pair", [1.0, 1.0, 1.0, 1.0]))
    assert n.links_t.p_max_pu["IC_Signed"].eq(1.0).all()
    assert n.links_t.p_min_pu["IC_Signed"].eq(-1.0).all()
