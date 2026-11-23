"""Bus position overrides reach every network stage; Appendix B lengths are carried as metadata (19 Oct 2026).

The bus position audit found Appendix B buses that ``bus_coordinate_overrides.csv`` could not fix: the
override step ran on the base buses only and skipped every other row silently. These tests hold:

* a base-stage override row sets the position and the offshore flag (both ways), and a row for a bus added
  later is left to the later stages;
* the Appendix B bus placer takes a per-bus override before any site-level source;
* the final pass on the finished network moves a bus added after the base stage, warns about a known bus not
  built in this network and refuses a row naming a bus no stage builds;
* an Appendix B circuit carries its OHL + cable length in ``length_km``, while ``x`` comes from the row and
  PyPSA's own ``length`` is untouched (metadata only).
"""
import logging

import numpy as np
import pandas as pd
import pypsa
import pytest
from pyproj import Transformer

from scripts.network_build.ETYS_network import (
    apply_bus_coordinate_overrides,
    apply_bus_coordinate_overrides_to_network,
    bus_coordinate_override_positions,
    load_bus_coordinate_overrides,
)
from scripts.network_build.ETYS_upgrades import add_missing_buses_from_upgrades, apply_circuit_additions

TO_OSGB = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


def _overrides(tmp_path, rows) -> str:
    path = tmp_path / "bus_coordinate_overrides.csv"
    pd.DataFrame(rows, columns=["bus", "lat", "lon", "is_offshore", "source"]).to_csv(path, index=False)
    return str(path)


def _network(buses: dict) -> pypsa.Network:
    n = pypsa.Network()
    for name, (lat, lon) in buses.items():
        x, y = TO_OSGB.transform(lon, lat)
        n.add("Bus", name, v_nom=400.0, x=x, y=y, carrier="AC")
        n.buses.loc[name, "lat"] = lat
        n.buses.loc[name, "lon"] = lon
    return n


@pytest.mark.unit
def test_override_file_is_validated(tmp_path):
    ok = _overrides(tmp_path, [("AAAA41", 55.0, -3.0, False, "reviewed")])
    assert bus_coordinate_override_positions(ok) == {"AAAA41": (55.0, -3.0)}
    assert load_bus_coordinate_overrides(None).empty
    dup = _overrides(tmp_path, [("AAAA41", 55.0, -3.0, False, "a"), ("AAAA41", 56.0, -3.0, False, "b")])
    with pytest.raises(ValueError, match="more than once"):
        load_bus_coordinate_overrides(dup)
    nosrc = _overrides(tmp_path, [("AAAA41", 55.0, -3.0, False, "")])
    with pytest.raises(ValueError, match="source"):
        load_bus_coordinate_overrides(nosrc)


@pytest.mark.unit
def test_base_stage_sets_positions_and_the_offshore_flag_both_ways(tmp_path):
    df_buses = pd.DataFrame({"v_nom": [400.0, 33.0], "lat": [53.9, 53.1], "lon": [-4.4, -4.2]},
                            index=["WOWE41", "GWAW31"])
    path = _overrides(tmp_path, [("WOWE41", 54.03, -2.89, False, "onshore at Heysham"),
                                 ("GWAW31", 53.47, -3.60, True, "offshore platform"),
                                 ("NEWB41", 52.0, -1.0, True, "an Appendix B bus, placed later")])
    out, offshore = apply_bus_coordinate_overrides(df_buses, {"WOWE41"}, path, logging.getLogger("t"))
    assert out.loc["WOWE41", ["lat", "lon"]].tolist() == [54.03, -2.89]
    assert out.loc["GWAW31", ["lat", "lon"]].tolist() == [53.47, -3.60]
    assert "NEWB41" not in out.index                       # not a base bus: left to the later stages
    assert offshore == {"GWAW31", "NEWB41"}                # WOWE41's sheet flag cleared; later bus flagged


@pytest.mark.unit
def test_appendix_b_placer_takes_the_per_bus_override_first():
    n = _network({"WEST4A": (54.0, -2.0)})
    categorized = {"circuits": {"additions": [{"Node1": "WEST4A", "Node2": "NEWX4A", "OHL Length (km)": 30.0,
                                               "Cable Length (km)": 0.0}]},
                   "transformers": {"additions": []}}
    # The site row says one place, the per-bus override another: the override wins.
    added = add_missing_buses_from_upgrades(n, categorized, {"NEWX": (51.0, -1.0)}, logging.getLogger("t"),
                                            bus_coords={"NEWX4A": (53.5, -1.5)})
    assert added == 1
    x, y = TO_OSGB.transform(-1.5, 53.5)
    assert n.buses.at["NEWX4A", "x"] == pytest.approx(x) and n.buses.at["NEWX4A", "y"] == pytest.approx(y)
    # Without an override the site row (Strategy 0) still places it.
    n2 = _network({"WEST4A": (54.0, -2.0)})
    add_missing_buses_from_upgrades(n2, categorized, {"NEWX": (51.0, -1.0)}, logging.getLogger("t"))
    x2, y2 = TO_OSGB.transform(-1.0, 51.0)
    assert n2.buses.at["NEWX4A", "x"] == pytest.approx(x2) and n2.buses.at["NEWX4A", "y"] == pytest.approx(y2)


@pytest.mark.unit
def test_final_pass_moves_later_buses_and_never_skips_a_row_silently(tmp_path, caplog):
    n = _network({"BASE41": (55.0, -3.0), "LATE41": (57.7, -4.3)})
    path = _overrides(tmp_path, [("LATE41", 58.48, -3.79, False, "reviewed"),
                                 ("OTHR41", 52.0, -1.0, False, "a bus another year builds")])
    with caplog.at_level(logging.WARNING):
        applied, absent = apply_bus_coordinate_overrides_to_network(
            n, path, logging.getLogger("t"), known_buses=lambda: {"BASE41", "LATE41", "OTHR41"})
    assert applied == ["LATE41"] and absent == ["OTHR41"]
    x, y = TO_OSGB.transform(-3.79, 58.48)
    assert n.buses.loc["LATE41", ["x", "y"]].tolist() == pytest.approx([x, y])
    assert n.buses.loc["LATE41", ["lat", "lon"]].tolist() == [58.48, -3.79]
    assert "OTHR41" in caplog.text
    bad = _overrides(tmp_path, [("TYPO41", 52.0, -1.0, False, "a mistyped bus")])
    with pytest.raises(ValueError, match="TYPO41"):
        apply_bus_coordinate_overrides_to_network(n, bad, logging.getLogger("t"), known_buses={"BASE41", "LATE41"})


@pytest.mark.unit
def test_appendix_b_circuit_carries_its_length_as_metadata_only():
    n = _network({"AAAA4A": (55.0, -3.0), "BBBB4A": (55.2, -3.1)})
    record = {"Node1": "AAAA4A", "Node2": "BBBB4A", "Year": 2027, "Status": "Addition", "operator": "SPT",
              "R (% on 100 MVA)": 0.05, "X (% on 100 MVA)": 0.8, "B (% on 100 MVA)": 10.0,
              "Winter Rating (MVA)": 2000.0, "OHL Length (km)": 20.0, "Cable Length (km)": 1.5,
              "Circuit Type": "OHL"}
    added, failed, _ = apply_circuit_additions(n, [record], logging.getLogger("t"))
    assert (added, failed) == (1, 0)
    line = n.lines.iloc[0]
    assert line["length_km"] == pytest.approx(21.5)
    assert line["length"] == 0.0                             # PyPSA's length untouched
    assert line["x"] == pytest.approx(0.008)                 # the row's impedance (pu on 100 MVA), not a length
    # A row without a length leaves length_km empty.
    n2 = _network({"AAAA4A": (55.0, -3.0), "BBBB4A": (55.2, -3.1)})
    apply_circuit_additions(n2, [{**record, "OHL Length (km)": "TBC", "Cable Length (km)": None}],
                            logging.getLogger("t"))
    assert np.isnan(n2.lines.iloc[0]["length_km"])
