"""The secured-boundary capability audit by family (rule audit_secured_capability, 19 Oct 2026)."""
import copy
import json
import logging

import pandas as pd
import pypsa
import pytest

from scripts.market.network_delivery import apply_network_delivery
from scripts.solve.secured_boundaries import secured_boundary_config
from scripts.solve.secured_capability_audit import (
    audit,
    audit_spec,
    delay_case,
    family_of,
    resolve_secured_capability,
    spec_files,
    twin_tag,
)

HEAD = "CP30_FFR_2030_ETYS_market"


@pytest.mark.unit
def test_family_names_and_delay_cases():
    assert family_of(f"{HEAD}_Jan2030_no_upgrades") == (f"{HEAD}_no_upgrades", "no_upgrades", HEAD, "Jan")
    assert family_of("HT35") == ("HT35", "", None, None)
    assert delay_case("seven_critical_late_outage_neso") == "seven_late"
    assert delay_case("outage_neso", "no_acceleration") == "no_acceleration"
    assert delay_case("outage_neso", "cp30_network") is None
    assert twin_tag("no_upgrades_p2_bids") == "cp30_network_p2_bids"
    assert twin_tag("central_dispatch_seven_late") == "central_dispatch"
    assert twin_tag("cp30_network") == "cp30_network"


def _scenario(month: str, capability_file: str, delivery: dict | None = None, enabled: bool = True) -> dict:
    m = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(month) + 1
    sc = {"modelled_year": 2030, "market": {"enabled": True, "balancing": {}},
          "solve_period": {"enabled": True, "start": f"2030-{m:02d}-01 00:00", "end": f"2030-{m:02d}-02 23:00"},
          "network": {"transmission": {"outage_schedule": {
              "enabled": enabled, "source": "secured_capability",
              "secured_capability": {"capability_file": capability_file,
                                     "capability_year": 2030, "fes_pathway": "HT"}}}}}
    if delivery:
        sc["market"]["balancing"]["network_delivery"] = delivery
    return sc


@pytest.mark.unit
def test_resolution_matches_apply_network_delivery():
    """The audit's merge is the one the balancing stage applies to its own scenario config."""
    delivery = {"case": "no_upgrades_outage_neso",
                "secured_capability": {"capability_year": 2025, "capability_year_overrides": {"SW1": 2030},
                                       "historic_derate": {"enabled": True, "mode": "replay", "file": "h.csv"}}}
    sc = _scenario("Jan", "cap.csv", delivery)
    sc["network"]["transmission"]["outage_schedule"]["secured_capability"]["include"] = {"B4": []}
    n = pypsa.Network()
    n.add("Bus", "a")
    applied = copy.deepcopy(sc)
    apply_network_delivery(n, applied, applied["market"]["balancing"], logging.getLogger("t"))
    expected = secured_boundary_config(applied["network"]["transmission"]["outage_schedule"], applied)
    got = resolve_secured_capability(sc)
    assert got["active"] and got["case"] == "no_upgrades_outage_neso"
    for key in ("capability_year", "capability_year_overrides", "capability_file", "historic_derate", "fes_pathway"):
        assert got["secured"][key] == expected[key], key
    assert got["secured"]["month_season"] == {str(k): v for k, v in expected["month_season"].items()}
    assert sc["network"]["transmission"]["outage_schedule"]["secured_capability"]["capability_year"] == 2030  # not mutated
    assert not resolve_secured_capability(_scenario("Jan", "cap.csv", enabled=False))["active"]
    with pytest.raises(ValueError):
        resolve_secured_capability(_scenario("Jan", "cap.csv", {"case": "x", "secured_capability": {"bogus": 1}}))


def _files(tmp_path):
    cap = pd.DataFrame([
        ("B4", "HT", 2025, 4000.0, "thermal"), ("B4", "HT", 2030, 7000.0, "thermal"),
        ("B6", "HT", 2025, 6000.0, "voltage"), ("B6", "HT", 2030, 10000.0, "voltage"),
        ("B6", "HT", 2031, 12000.0, "voltage"),
    ], columns=["boundary", "fes_pathway", "year", "capability_mw", "limit_type"])
    cap.to_csv(tmp_path / "cap.csv", index=False)
    cap.assign(capability_mw=cap["capability_mw"] + 1.0).to_csv(tmp_path / "cap_other.csv", index=False)
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"boundary": b},
         "geometry": {"type": "LineString", "coordinates": [[0, y], [1000, y]]}} for b, y in (("B4", 5), ("B6", 3))]}
    (tmp_path / "geo.geojson").write_text(json.dumps(geo))
    return str(tmp_path / "cap.csv"), str(tmp_path / "cap_other.csv"), str(tmp_path / "geo.geojson")


@pytest.mark.unit
def test_audit_end_to_end(tmp_path):
    cap, cap_other, geo = _files(tmp_path)

    def sc(month, capfile=cap, delivery=None, enabled=True):
        s = _scenario(month, capfile, delivery, enabled)
        s["network"]["transmission"]["outage_schedule"]["secured_capability"]["geometry_file"] = geo
        return s

    scenarios = {
        f"{HEAD}_Jan2030_cp30_network": sc("Jan"), f"{HEAD}_Jul2030_cp30_network": sc("Jul"),
        # Delay family whose twin (fin5) is no scenario: compared with the central family cp30_network. Lower: PASS.
        f"{HEAD}_Jan2030_no_upgrades": sc("Jan", delivery={"case": "no_upgrades_outage_neso",
                                                                "secured_capability": {"capability_year": 2025}}),
        # Delay family above its twin on B6 (override to a year whose capability is higher): WARN.
        f"{HEAD}_Jan2030_p2": sc("Jan"),
        f"{HEAD}_Jan2030_p2_seven_late": sc("Jan", delivery={"case": "seven_critical_late_outage_neso",
                                                             "secured_capability": {"capability_year": 2025,
                                                                                    "capability_year_overrides": {"B6": 2031}}}),
        # Another capability file than the central: WARN on every boundary.
        f"{HEAD}_Jan2030_other": sc("Jan", capfile=cap_other),
        # Not active: one INFO row.
        f"{HEAD}_Jan2030_off": sc("Jan", enabled=False),
    }
    spec = audit_spec(scenarios, list(scenarios), "cp30_network")
    fams = spec["families"]
    assert fams[f"{HEAD}_no_upgrades"]["twin"] == f"{HEAD}_cp30_network"
    assert fams[f"{HEAD}_p2_seven_late"]["twin"] == f"{HEAD}_p2"
    assert fams[f"{HEAD}_p2_seven_late"]["delay_case"] == "seven_late"
    assert spec_files(spec)["capability"] == sorted({cap, cap_other})
    json.dumps(spec)  # the rule params must serialise
    table = audit(spec, root="/")
    t = table.set_index(["family", "boundary"])
    central = t.loc[f"{HEAD}_cp30_network"]
    assert list(central["capability_mw"]) == [7000.0, 10000.0]
    assert list(central["status"]) == ["PASS", "PASS"]
    assert central.at["B4", "months"] == "Jan|Jul"
    # Winter (Jan) and summer outage (Jul) NOA factors: thermal 1.0 / 0.7, voltage 1.0 / 0.9.
    assert central.at["B4", "mean_factor"] == pytest.approx(0.85)
    assert central.at["B6", "mean_factor"] == pytest.approx(0.95)
    nu = t.loc[f"{HEAD}_no_upgrades"]
    assert list(nu["capability_mw"]) == [4000.0, 6000.0] and set(nu["status"]) == {"PASS"}
    assert set(nu["case"]) == {"no_upgrades"} and list(nu["twin_capability_mw"]) == [7000.0, 10000.0]
    sl = t.loc[f"{HEAD}_p2_seven_late"]
    assert sl.at["B4", "status"] == "PASS"
    assert sl.at["B6", "status"] == "WARN" and "ABOVE" not in sl.at["B4", "note"]
    other = t.loc[f"{HEAD}_other"]
    assert set(other["status"]) == {"WARN"} and all("central" in n for n in other["note"])
    off = table[table["family"].eq(f"{HEAD}_off")]
    assert len(off) == 1 and off["status"].iloc[0] == "INFO" and not off["secured_active"].iloc[0]
    # The twin p2 is in the run here, so it is audited as a central family itself.
    assert f"{HEAD}_p2" in set(table["family"])


@pytest.mark.unit
def test_delay_family_on_a_subset_of_boundaries_against_its_twin(tmp_path):
    cap, _, geo = _files(tmp_path)
    s = _scenario("Jan", cap)
    s["network"]["transmission"]["outage_schedule"]["secured_capability"]["geometry_file"] = geo
    d = copy.deepcopy(s)
    d["market"]["balancing"]["network_delivery"] = {"case": "no_acceleration_outage_neso",
                                                    "secured_capability": {"boundaries": ["B6"]}}
    scenarios = {f"{HEAD}_Jan2030_x": s, f"{HEAD}_Jan2030_x_no_acceleration": d}
    table = audit(audit_spec(scenarios, [f"{HEAD}_Jan2030_x_no_acceleration"], "x"), root="/")
    assert list(table["boundary"]) == ["B6"] and list(table["status"]) == ["PASS"]
    assert table["central_twin"].iloc[0] == f"{HEAD}_x"
