"""Hydrogen, storage and interconnector placement fixes for the CP30 2030 rebuild (15 Oct 2026).

1. Electrolysers by FES networked electrolysis (Dem_BB009) GSP rows, not pro rata to H2 turbines.
2. Committed new-build LDES at named projects (data/storage/cp30_committed_ldes_sites.csv); Cruachan in
   South Scotland through ``spatial_allocation.TRANSMISSION_SITE_REGIONS``.
3. Distribution-size batteries on their own GSP polygon's demand node.
4. Interconnectors from a declared composition (data/interconnectors/cp30_interconnector_composition.csv):
   CP30 nominal ratings, MaresConnect added, the Isle of Man cable out, no pro-rata uplift.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.CP30.storage_interconnectors import (
    compose_cp30_interconnectors,
    fixed_site_mask,
    load_committed_storage_sites,
    load_cp30_interconnector_composition,
    parse_committed_storage_sites,
    scale_storage_proxy_to_cp30_targets,
)
from scripts.CP30.storage_regions import apply_es5_storage_regions, apply_site_code_regions
from scripts.hydrogen.add_hydrogen_system import (
    add_hydrogen_system,
    fes_electrolyser_rows,
    place_fes_electrolysers,
    resolve_electrolyser_siting,
)
from config.config_loader import load_defaults

REPO = Path(__file__).resolve().parents[2]
COMMITTED_FILE = REPO / "data/storage/cp30_committed_ldes_sites.csv"
COMPOSITION_FILE = REPO / "data/interconnectors/cp30_interconnector_composition.csv"
FFR, ND = "Further Flex and Renewables", "New Dispatch"
LOG = logging.getLogger("test")

EXISTING = [
    {"name": "Dinorwig", "site_code": "DINO", "lat": 53.1, "lon": -4.1, "power_mw": 1728, "energy_mwh": 9100},
    {"name": "Cruachan", "site_code": "CRUA", "lat": 56.4, "lon": -5.1, "power_mw": 440, "energy_mwh": 7100},
    {"name": "Ffestiniog", "site_code": "FFES", "lat": 53.0, "lon": -4.0, "power_mw": 360, "energy_mwh": 1300},
    {"name": "Foyers", "site_code": "FOYE", "lat": 57.3, "lon": -4.5, "power_mw": 300, "energy_mwh": 6400},
]


# ----------------------------------------------------------------------------- 1. electrolysers
def _fes(rows):
    return pd.DataFrame([{"Building Block ID Number": "Dem_BB009", "FES Pathway": "Holistic Transition",
                          "GSP": g, "Node ID": node, "DNO License Area": dno, "2030": mw}
                         for g, node, dno, mw in rows])


def _network(buses, lines=()):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for name, v, x, y in buses:
        n.add("Bus", name, v_nom=v, x=x, y=y, carrier="AC")
    for name, b0, b1, s in lines:
        n.add("Line", name, bus0=b0, bus1=b1, s_nom=s, x=0.1, r=0.01)
    return n


def test_siting_default_is_the_old_rule_and_validates():
    assert resolve_electrolyser_siting({})["method"] == "h2_turbine_pro_rata"
    assert resolve_electrolyser_siting({"electrolyser_siting": {"method": "fes_gsp"}})["fes_year"] == 2024
    with pytest.raises(ValueError):
        resolve_electrolyser_siting({"electrolyser_siting": {"method": "nearest"}})


def test_direct_rows_split_pro_rata_over_their_licence_area():
    buses = pd.DataFrame({"v_nom": [33.0, 132.0, 132.0]}, index=["ABNE3-", "CAIN1J", "FIDF11"])
    fes = _fes([("Abernethy", "ABNE3-", "SSEN (SHEPD)", 300.0), ("Cairnford", "CAIN1J", "SHETL", 100.0),
                ("Fiddlers Ferry", "FIDF11", "SPEN Manweb", 200.0), ("Direct(SHETL)", "Direct(SHETL)", "SHETL", 400.0),
                ("Direct(NGET)", "Direct(NGET)", "NGET", 50.0)])
    rows = fes_electrolyser_rows(fes, buses, "Holistic Transition", 2030)
    assert rows["power_mw"].sum() == pytest.approx(1050.0)
    by_bus = rows.groupby("bus")["power_mw"].sum()
    # SHETL Direct 400 split 300:100 over the SHETL GSP rows; NGET Direct to Fiddlers Ferry.
    assert by_bus["ABNE3-"] == pytest.approx(300 + 300)
    assert by_bus["CAIN1J"] == pytest.approx(100 + 100)
    assert by_bus["FIDF11"] == pytest.approx(250.0)
    assert set(rows["fes_row"]) == {"gsp", "direct"}


def test_fes_placement_scales_to_cp30_and_relieves_weak_buses():
    # ABNE3- (33 kV) under ABNE1Q (132 kV), which imports over one 100 MVA circuit from PERT1-; PERT4- is the
    # nearest 400 kV node in the region and on the same side of every boundary (import relief, 16 Oct 2026).
    n = _network([("ABNE3-", 33.0, 320000, 717000), ("ABNE1Q", 132.0, 320000, 717000),
                  ("PERT1-", 132.0, 330000, 720000), ("PERT4-", 400.0, 330000, 720000)],
                 [("l1", "ABNE1Q", "PERT1-", 100.0), ("l2", "PERT4-", "PERT1-", 2000.0)])
    n.add("Transformer", "t1", bus0="ABNE1Q", bus1="ABNE3-", s_nom=1000.0, x=0.1, r=0.0)
    fes = _fes([("Abernethy", "ABNE3-", "SSEN (SHEPD)", 500.0), ("Direct(SHETL)", "Direct(SHETL)", "SHETL", 300.0)])
    siting = resolve_electrolyser_siting({"electrolyser_siting": {"method": "fes_gsp"}})
    relief = {"transmission": {"min_bm_constraint_voltage_kv": 0.0},
              "bus_region": pd.Series("North Scotland", index=n.buses.index),
              "bus_sides": pd.DataFrame(True, index=n.buses.index, columns=["B4", "B6", "B8"])}
    by_bus = place_fes_electrolysers(n, fes, "Holistic Transition", 2030, 1000.0, siting, LOG, relief=relief)
    assert by_bus.sum() == pytest.approx(1000.0, rel=1e-12)
    assert by_bus.get("ABNE3-", 0.0) <= 100.0 + 1e-6          # held to the 100 MVA circuit its node imports over
    assert by_bus.get("PERT4-", 0.0) >= 900.0 - 1e-6           # the excess on the nearest 400 kV node


def test_add_hydrogen_system_uses_the_placed_buses_and_keeps_the_total():
    n = _network([("A", 132.0, 0, 0), ("B", 132.0, 1000, 0), ("T", 400.0, 2000, 0)])
    n.add("Generator", "h2gen", bus="T", carrier="H2", p_nom=100.0)
    placed = pd.Series({"A": 600.0, "B": 400.0})
    add_hydrogen_system(n, 1000.0, 100.0, LOG, electrolyser_mw_by_bus=placed)
    el = n.links[n.links.carrier.eq("electrolysis")]
    assert el.groupby("bus0")["p_nom"].sum().to_dict() == {"A": 600.0, "B": 400.0}
    assert "electrolysis_T" not in n.links.index          # not on the turbine bus any more

    n2 = _network([("A", 132.0, 0, 0), ("T", 400.0, 2000, 0)])
    n2.add("Generator", "h2gen", bus="T", carrier="H2", p_nom=100.0)
    with pytest.raises(ValueError, match="totals"):
        add_hydrogen_system(n2, 1000.0, 100.0, LOG, electrolyser_mw_by_bus=pd.Series({"A": 900.0}))


def test_default_hydrogen_placement_unchanged():
    n = _network([("T1", 400.0, 0, 0), ("T2", 400.0, 1000, 0)])
    n.add("Generator", "g1", bus="T1", carrier="H2", p_nom=30.0)
    n.add("Generator", "g2", bus="T2", carrier="H2", p_nom=70.0)
    add_hydrogen_system(n, 1000.0, 100.0, LOG)
    el = n.links[n.links.carrier.eq("electrolysis")].set_index("bus0")["p_nom"]
    assert el.to_dict() == pytest.approx({"T1": 300.0, "T2": 700.0})


# ----------------------------------------------------------------------------- 2. committed LDES
def _target(power=7858.19, energy=98928.2):
    return pd.DataFrame([{"carrier": "LAES", "cp30_subtype": "LDES", "power_mw": power, "energy_mwh": energy,
                          "duration_h": energy / power, "pathway": FFR, "proxy_technologies": ("LAES",),
                          "eta_charge": 0.8367, "eta_discharge": 0.8367}])


def _proxy():
    return pd.DataFrame({"site_name": ["p1", "p2", "Coire Glas PSH"], "technology": ["LAES"] * 3,
                         "power_mw": [100.0, 300.0, 50.0], "lat": [55.9, 53.3, 57.0], "lon": [-4.0, -2.7, -4.8]})


@pytest.mark.parametrize("pathway, totals", [
    (FFR, {"North Scotland": 2046.0, "South Scotland": 1349.8, "South East": 500.0}),
    (ND, {"North Scotland": 1362.0, "South Scotland": 749.8, "South East": 0.0}),
])
def test_committed_sites_match_es5_regions_with_existing_plant(pathway, totals):
    """The declared projects plus Foyers / Cruachan reproduce ES5's Scottish and South East non-battery storage."""
    sites = load_committed_storage_sites(COMMITTED_FILE, pathway)
    region_of = {"COGL": "North Scotland", "KNOC": "North Scotland", "DALL": "South Scotland",
                 "GLGL": "South Scotland", "HUNE": "South Scotland", "KINO": "South East"}
    got = {"North Scotland": 300.0, "South Scotland": 440.0, "South East": 0.0}   # Foyers, Cruachan
    for site in sites:
        got[region_of[site["site_code"]]] += site["power_mw"]
    assert got == pytest.approx(totals)
    # ND South Scotland and both North Scotland totals are ES5 exactly (FFR SS leaves 900 MW to the proxy).
    assert all(len(s["site_code"]) == 4 for s in sites)
    assert not any(old["name"].lower() in s["unit_label"].lower() for s in sites for old in EXISTING)


def test_committed_rows_are_fixed_and_totals_exact():
    committed = load_committed_storage_sites(COMMITTED_FILE, FFR)
    out = scale_storage_proxy_to_cp30_targets(_proxy(), _target(), existing_sites=EXISTING, committed_sites=committed)
    assert out["power_mw"].sum() == pytest.approx(7858.19)
    assert out["energy_mwh"].sum() == pytest.approx(98928.2)
    rows = out[out["committed_site"].notna()].set_index("committed_site")
    assert rows.loc["Coire Glas", "power_mw"] == 1296 and rows.loc["Coire Glas", "energy_mwh"] == 30000
    # blank energy = CP30's own LDES average duration
    assert rows.loc["Cruachan II", "duration_h"] == pytest.approx(98928.2 / 7858.19)
    assert (rows["spatial_proxy_source"] == "committed_site").all()
    assert rows.loc["Cruachan II", "site_name"] == "CP30_LAES_committed_CRUA_expansion"
    # the proxy row named after a committed project is dropped; the remainder has one positive duration
    proxy = out[~fixed_site_mask(out)]
    assert "Coire Glas PSH" not in set(proxy["site_name"])
    assert proxy["duration_h"].nunique() == 1 and proxy["duration_h"].iloc[0] > 0


def test_committed_label_naming_an_existing_site_is_refused():
    bad = parse_committed_storage_sites([{"name": "Cruachan II", "unit_label": "Cruachan_II", "site_code": "CRUA",
                                          "lat": 56.4, "lon": -5.1, "power_mw": 600}])
    with pytest.raises(ValueError, match="legacy"):
        scale_storage_proxy_to_cp30_targets(_proxy(), _target(), existing_sites=EXISTING, committed_sites=bad)


def test_committed_sites_that_overfill_the_target_are_refused():
    with pytest.raises(ValueError, match="leave no new build"):
        scale_storage_proxy_to_cp30_targets(_proxy(), _target(power=3000, energy=40000), existing_sites=EXISTING,
                                            committed_sites=load_committed_storage_sites(COMMITTED_FILE, FFR))


def test_site_code_region_override_and_fixed_committed_rows_in_es5_split():
    frame = pd.DataFrame({
        "site_name": ["CP30_LAES_existing_Cruachan", "CP30_LAES_committed_CoireGlas", "new_ns", "new_ss"],
        "technology": ["LAES"] * 4, "power_mw": [440.0, 1296.0, 500.0, 500.0], "duration_h": [16.1, 23.1, 12.0, 12.0],
        "energy_mwh": [7100.0, 30000.0, 6000.0, 6000.0], "site_code": ["CRUA", "COGL", None, None],
        "existing_site": ["Cruachan", None, None, None], "committed_site": [None, "Coire Glas", None, None]})
    regions = pd.Series(["North Scotland", "North Scotland", "North Scotland", "South Scotland"], index=frame.index)
    regions = apply_site_code_regions(regions, frame)      # the shared table (CRUA, DALL)
    assert regions.iloc[0] == "South Scotland"
    targets = pd.DataFrame({"carrier": "LAES", "cp30_region": ["North Scotland", "South Scotland"],
                            "capacity_mw": [1296.0, 1440.0]})
    out = apply_es5_storage_regions(frame, targets, regions)
    by_region = out.groupby("cp30_region")["power_mw"].sum()
    assert by_region["North Scotland"] == pytest.approx(1296.0)       # committed only: the NS proxy row is emptied
    assert by_region["South Scotland"] == pytest.approx(1440.0)
    assert out.loc[out["committed_site"].eq("Coire Glas"), "power_mw"].iloc[0] == 1296.0
    assert "new_ns" not in set(out["site_name"])
    assert out["energy_mwh"].sum() == pytest.approx(7100 + 30000 + 1000 * 12.0)


# ----------------------------------------------------------------------------- 3. batteries at their GSP node
class _StubLocator:
    """Every row in one polygon whose demand node is GSPN11 (licence area _A, England)."""

    def __init__(self, group="_A"):
        self.group = group

    def assign(self, sites):
        out = sites.copy()
        out["gsp_node"], out["gsp_group"], out["gsp_ids"] = "GSPN11", self.group, "GSPN_1"
        return out


def test_distribution_batteries_move_to_their_gsp_node():
    from scripts.storage.add_storage import place_batteries_at_gsp_nodes
    df = pd.DataFrame({"site_name": ["small", "large", "pumped", "laes"],
                       "technology": ["Battery", "Battery", "LAES", "LAES"],
                       "power_mw": [40.0, 150.0, 300.0, 20.0], "bus": ["SPEN41", "SPEN41", "FOYE2-", "X"],
                       "lat": [52.0] * 4, "lon": [-1.0] * 4, "site_code": [None, None, "FOYE", None],
                       "existing_site": [None, None, "Foyers", None]})
    out, moves = place_batteries_at_gsp_nodes(df, None, log=LOG, locator=_StubLocator())
    assert out.set_index("site_name")["bus"].to_dict() == {"small": "GSPN11", "large": "SPEN41",
                                                           "pumped": "FOYE2-", "laes": "X"}
    assert out["power_mw"].sum() == df["power_mw"].sum()
    assert list(moves["site_name"]) == ["small"]


def test_scottish_thresholds_apply():
    from scripts.storage.add_storage import place_batteries_at_gsp_nodes
    df = pd.DataFrame({"site_name": ["a", "b"], "technology": ["Battery"] * 2, "power_mw": [8.0, 20.0],
                       "bus": ["BUS1", "BUS1"], "lat": [57.0] * 2, "lon": [-4.0] * 2})
    out, _ = place_batteries_at_gsp_nodes(df, None, log=LOG, locator=_StubLocator(group="_P"))
    assert out["bus"].tolist() == ["GSPN11", "BUS1"]       # SHET threshold 10 MW


# ----------------------------------------------------------------------------- 4. interconnectors
REGISTER = {"Britned": ("Netherlands", "GRAI41", 1000), "IFA": ("France", "SELL4A", 1988),
            "IFA2": ("France", "CHIL41", 1000), "Nemo Link": ("Belgium", "RICH41", 1020),
            "NS Link": ("Norway", "BLYT21", 1400), "Moyle": ("Northern Ireland", "AUCH2-", 500),
            "ElecLink": ("France", "SELL4A", 1000), "Viking": ("Denmark", "BICF41", 1400),
            "East West Interconnector": ("Ireland", "FLIB41", 505), "Greenlink": ("Ireland", "PEMB41", 504),
            "Isle of Man Interconnector": ("Isle Of Man", "STAH4A", 74),
            "NeuConnect Interconnector": ("Germany", "GRAI41", 1400)}


def _mapped():
    return pd.DataFrame([{"name": k, "counterparty_country": c, "from_bus": b, "gb_bus": b, "capacity_mw": float(mw),
                          "to_bus": f"HVDC_External_{c.replace(' ', '_')}", "losses_percent": 2.5}
                         for k, (c, b, mw) in REGISTER.items()])


@pytest.mark.parametrize("year, es1", [(2027, 10300.0), (2028, 11700.0), (2029, 11700.0), (2030, 12450.0)])
def test_composition_reproduces_every_cp30_es1_step(year, es1):
    out = compose_cp30_interconnectors(_mapped(), es1, load_cp30_interconnector_composition(COMPOSITION_FILE), year)
    assert out["capacity_mw"].sum() == pytest.approx(es1)
    assert "Isle of Man Interconnector" not in set(out["name"])
    assert ("MaresConnect" in set(out["name"])) == (year >= 2030)
    assert ("NeuConnect Interconnector" in set(out["name"])) == (year >= 2028)


def test_composition_keeps_register_ratings_and_adds_maresconnect_on_the_sem():
    out = compose_cp30_interconnectors(_mapped(), 12450.0, load_cp30_interconnector_composition(COMPOSITION_FILE), 2030)
    out = out.set_index("name")
    assert out.loc["IFA", "capacity_mw"] == 2000 and out.loc["IFA", "capacity_mw_original"] == 1988
    assert out.loc["Viking", "capacity_mw"] == 1400                   # no x1.0559 uplift anywhere
    mares = out.loc["MaresConnect"]
    assert mares["from_bus"] == "BODE41" and mares["capacity_mw"] == 750
    assert mares["counterparty_country"] == "Ireland"
    assert mares["to_bus"] == "HVDC_External_Ireland_Maynooth"
    assert mares["cp30_composition_action"] == "add"


def test_composition_refuses_undeclared_cables_and_gaps():
    comp = load_cp30_interconnector_composition(COMPOSITION_FILE)
    extra = pd.concat([_mapped(), pd.DataFrame([{"name": "LionLink", "counterparty_country": "Netherlands",
                                                 "from_bus": "LEIS41", "capacity_mw": 1800.0}])])
    with pytest.raises(ValueError, match="missing from the CP30 composition"):
        compose_cp30_interconnectors(extra, 12450.0, comp, 2030)
    with pytest.raises(ValueError, match="update the composition"):
        compose_cp30_interconnectors(_mapped(), 13000.0, comp, 2030)


def test_maresconnect_takes_the_sem_settings():
    """SEM tranche market and LCP's Ireland flow limit (0) apply by counterparty country."""
    import yaml
    from scripts.interconnectors.add_to_network import single_price_market_of
    from scripts.market.interconnector_counter_trade import link_shares, resolve_counter_trade_config
    defaults = load_defaults()
    pricing = defaults["interconnectors"]["pricing"]
    assert single_price_market_of("Ireland", pricing) == "sem"
    cfg = resolve_counter_trade_config(defaults["market"]["balancing"])
    n = pypsa.Network()
    n.add("Bus", ["BODE41", "HVDC_External_Ireland_Maynooth"])
    n.add("Link", "IC_MaresConnect", bus0="BODE41", bus1="HVDC_External_Ireland_Maynooth", p_nom=750.0)
    n.links["counterparty_country"] = "Ireland"
    assert float(link_shares(n, ["IC_MaresConnect"], cfg).iloc[0]) == 0.0


def test_availability_appends_added_cables_after_the_register():
    from scripts.interconnectors.availability import append_composition_cables, generate_outage_availability
    names = list(REGISTER)
    extended = append_composition_cables(names, str(COMPOSITION_FILE))
    assert extended[:len(names)] == names and extended[len(names):] == ["MaresConnect"]
    idx = pd.date_range("2030-01-01", periods=24 * 60, freq="h")
    a = generate_outage_availability(names, idx)
    b = generate_outage_availability(extended, idx)
    pd.testing.assert_frame_equal(a, b[b["name"].isin(names)].reset_index(drop=True))


def test_fes_gsp_level_batteries_go_to_their_node_whatever_their_size():
    from scripts.storage.add_storage import place_batteries_at_gsp_nodes
    df = pd.DataFrame({"site_name": ["gsp_sum", "direct_site"], "technology": ["Battery"] * 2,
                       "power_mw": [400.0, 400.0], "bus": ["BUS1", "BUS1"], "lat": [52.0] * 2, "lon": [-1.0] * 2,
                       "proxy_source": ["FES_GSP_Srg_BB001", "FES_Direct_Srg_BB001"]})
    out, moves = place_batteries_at_gsp_nodes(df, None, log=LOG, locator=_StubLocator())
    assert out["bus"].tolist() == ["GSPN11", "BUS1"]
    assert moves["power_mw"].sum() == 400.0


def test_scaled_storage_keeps_the_fes_row_type():
    proxy = _proxy().assign(source=["FES_GSP_Srg_BB004", "FES_Direct_Srg_BB004", "FES_Direct_Srg_BB004"])
    out = scale_storage_proxy_to_cp30_targets(proxy, _target())
    assert set(out["source"]) == {"CP30"}
    assert out["proxy_source"].tolist() == ["FES_GSP_Srg_BB004", "FES_Direct_Srg_BB004", "FES_Direct_Srg_BB004"]


def test_storage_weak_bus_relief_moves_proxies_but_not_named_sites():
    from scripts.storage.add_storage import relieve_storage_weak_buses
    n = _network([("WEAK3-", 33.0, 0, 0), ("WEAK1Q", 132.0, 0, 0), ("NEAR41", 400.0, 5000, 0)],
                 [("t", "WEAK3-", "WEAK1Q", 100.0), ("l", "WEAK1Q", "NEAR41", 3000.0)])
    df = pd.DataFrame({"site_name": ["proxy_laes", "proxy_bat", "named", "gsp_bat"],
                       "technology": ["LAES", "Battery", "LAES", "Battery"],
                       "power_mw": [300.0, 120.0, 200.0, 50.0], "bus": ["WEAK3-"] * 4,
                       "site_code": [None, None, "WEAK", None], "committed_site": [None, None, "Named", None],
                       "data_source": ["CP30", "CP30", "CP30", "battery_gsp_node"]})
    out, moves = relieve_storage_weak_buses(df, n, {"max_loading": 1.5})
    by_name = out.set_index("site_name")["bus"]
    assert by_name["named"] == "WEAK3-" and by_name["gsp_bat"] == "WEAK3-"
    assert by_name["proxy_laes"] != "WEAK3-"                       # largest proxy moved first
    assert set(moves["site_name"]) <= {"proxy_laes", "proxy_bat"}
    assert out["data_source"].tolist() == df["data_source"].tolist()  # labels restored
    assert out["power_mw"].sum() == df["power_mw"].sum()


def test_storage_placement_corrections_have_no_switches():
    """Consolidation of 15 Oct 2026: committed sites, the transmission-site regions, batteries at their
    GSP node and storage weak-bus relief are plain behaviour; their switches are gone from config."""
    import yaml

    from scripts.CP30.spatial_allocation import TRANSMISSION_SITE_REGIONS
    from config.config_loader import defaults_files, load_defaults

    text = "\n".join(path.read_text(encoding="utf-8") for path in defaults_files())
    for key in ("storage_committed_sites:", "storage_weak_bus_relief:", "site_code_regions:", "battery_gsp_node:"):
        assert key not in text, key
    assert TRANSMISSION_SITE_REGIONS == {"CRUA": "South Scotland", "DALL": "South Scotland"}
    defaults = load_defaults()
    assert defaults["network"]["spatial_mapping"]["weak_bus_relief"]["enabled"] is True   # the one relief switch
