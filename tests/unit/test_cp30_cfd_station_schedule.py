"""Named CfD units carved out of the CP30 waste and biogas proxies (rebuild batch, 14 Oct 2026).

K3 CHP (Kemsley, AR1 energy from waste, 45 MW), Energy Works Hull (AR1 advanced
conversion, 25 MW) and the AR5 geothermal plants (12 MW, biogas carrier) replace
that capacity of their carriers' proxies at the nearest bus to the site: carrier
totals unchanged, labels declared, strikes on the contract's own technology, and
the RO register share of the waste proxy still closes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.CP30.cfd_station_schedule import (
    CFD_STATION_SOURCE,
    apply_cfd_station_schedule,
    load_cfd_station_schedule,
)
from scripts.market.cfd_strike_table import load_strike_table, strike_lookup
from scripts.renewables.support_register_labels import apply_ro_register_share_to_proxies

ROOT = Path(__file__).resolve().parents[2]
SCHEDULE = ROOT / "data" / "market" / "CP30" / "cfd_station_schedule_2030.csv"
STRIKES = ROOT / "data" / "market" / "lccc" / "cfd_strike_prices_by_round.csv"


def _network():
    """OSGB36 buses near the three sites, a far-away mainland bus and a DC-only hub at Kemsley."""
    n = pypsa.Network()
    buses = {
        "KEMS1A": (591000.0, 166000.0),    # about 1.4 km from K3
        "CANT1A": (615000.0, 158000.0),    # Canterbury, farther from K3
        "HULL1A": (511000.0, 431000.0),    # about 1 km from Energy Works
        "DRAX1A": (466000.0, 427000.0),
        "INDQ1A": (182000.0, 59000.0),     # Indian Queens, nearest to United Downs
        "LAND1A": (135000.0, 30000.0),
        "MAIN1A": (450000.0, 300000.0),
    }
    for bus, (x, y) in buses.items():
        n.add("Bus", bus, x=x, y=y, v_nom=132.0, country="GB")
    n.add("Bus", "KEMSDC", x=592144.0, y=166704.0, v_nom=400.0, country="GB")   # on the site, DC only
    for i, (a, b) in enumerate([("KEMS1A", "CANT1A"), ("CANT1A", "MAIN1A"), ("HULL1A", "DRAX1A"),
                                ("DRAX1A", "MAIN1A"), ("INDQ1A", "LAND1A"), ("LAND1A", "MAIN1A")]):
        n.add("Line", f"l{i}", bus0=a, bus1=b, x=0.1, s_nom=500.0)
    n.add("Load", "demand", bus="MAIN1A", p_set=100.0)
    n.add("Link", "hub", bus0="MAIN1A", bus1="KEMSDC", p_nom=100.0, p_min_pu=-1.0, carrier="DC")
    return n


def _proxy():
    rows = []
    for carrier, gsps in (("waste_to_energy", {"A": 1200.0, "B": 800.0, "Direct(NGET)": 481.0}),
                          ("biogas", {"C": 600.0, "D": 471.0}),
                          ("CCGT", {"E": 1000.0})):
        for gsp, mw in gsps.items():
            rows.append({"fuel_type": carrier, "technology": carrier, "capacity_mw": mw, "gsp": gsp,
                         "bus": np.nan, "data_source": "CP30_ES1_FES_spatial_proxy",
                         "cp30_pathway": "Further Flex and Renewables", "cp30_target_mw": np.nan})
    return pd.DataFrame(rows)


def _carve(proxy=None):
    return apply_cfd_station_schedule(proxy if proxy is not None else _proxy(), SCHEDULE, _network(),
                                      land_boundary_file=None)


def test_schedule_declares_the_three_contracts():
    table = load_cfd_station_schedule(SCHEDULE)
    by = table.set_index("cfd_technology")
    assert by.at["energy_from_waste", "capacity_mw"] == pytest.approx(45.0)
    assert by.at["advanced_conversion", "capacity_mw"] == pytest.approx(25.0)
    assert by.at["geothermal", "capacity_mw"] == pytest.approx(12.0)
    assert set(table["carrier"]) == {"waste_to_energy", "biogas"}
    assert table["cfd_round"].tolist() == ["AR1", "AR1", "AR5"]
    assert by["load_factor_assumption"].to_dict() == {"energy_from_waste": 0.70, "advanced_conversion": 0.50,
                                                      "geothermal": 0.90}


def test_carve_out_conserves_every_carrier_total():
    before = _proxy().groupby("fuel_type")["capacity_mw"].sum()
    out = _carve()
    after = out.groupby("fuel_type")["capacity_mw"].sum()
    pd.testing.assert_series_equal(before.sort_index(), after.sort_index())
    named = out[out["data_source"].eq(CFD_STATION_SOURCE)]
    assert named.groupby("fuel_type")["capacity_mw"].sum().to_dict() == pytest.approx(
        {"waste_to_energy": 70.0, "biogas": 12.0})
    # The proxy shrinks in proportion; CCGT is untouched.
    proxy = out[~out["data_source"].eq(CFD_STATION_SOURCE)].set_index(["fuel_type", "gsp"])["capacity_mw"]
    assert proxy[("waste_to_energy", "A")] == pytest.approx(1200.0 * (2481.0 - 70.0) / 2481.0)
    assert proxy[("CCGT", "E")] == pytest.approx(1000.0)


def test_units_sit_at_the_nearest_allowed_bus_with_declared_labels():
    out = _carve().set_index("station_name")
    k3 = "K3 CHP Facility (Kemsley EfW)"
    # The DC-only hub sits on the site itself; the nearest AC bus wins.
    assert out.at[k3, "bus"] == "KEMS1A"
    assert out.at["Energy Works (Hull)", "bus"] == "HULL1A"
    assert out.at["AR5 geothermal (United Downs; Manhay; Penhallow)", "bus"] == "INDQ1A"
    named = out[out["data_source"].eq(CFD_STATION_SOURCE)]
    assert named["support_type"].eq("CfD").all()
    assert named["support_label_declared"].eq(True).all()
    assert out.at[k3, "cfd_id"] == "AAA-K3C-180"
    assert out.at["AR5 geothermal (United Downs; Manhay; Penhallow)", "fuel_type"] == "biogas"
    assert out.at["AR5 geothermal (United Downs; Manhay; Penhallow)", "cfd_technology"] == "geothermal"


def test_contract_technology_selects_the_strike():
    unit = {"from_price_year": 2012, "to_price_year": 2012, "from_index": 1.0, "to_index": 1.0,
            "factor": 1.0, "index": "none"}
    table = load_strike_table(STRIKES, unit, 2012)
    out = _carve().set_index("station_name")
    looked = {name: strike_lookup(table, row["fuel_type"], row["cfd_round"], row["cfd_technology"])[0]
              for name, row in out[out["data_source"].eq(CFD_STATION_SOURCE)].iterrows()}
    assert looked["K3 CHP Facility (Kemsley EfW)"] == pytest.approx(80.00)
    assert looked["Energy Works (Hull)"] == pytest.approx((119.89 * 36.0 + 114.39 * 26.0) / 62.0)
    assert looked["AR5 geothermal (United Downs; Manhay; Penhallow)"] == pytest.approx(119.00)


def test_ro_share_of_the_waste_proxy_still_closes():
    out = _carve()
    named = out["data_source"].eq(CFD_STATION_SOURCE)
    register = pd.DataFrame({
        "Station name": ["Riverside EfW", "Ardley EfW"], "capacity_mw": [300.0, 200.0], "banding": [0.5, 0.5],
        "ro_eligible": [True, True], "gb": [True, True], "carrier": ["biomass", "biomass"],
        "last_certificate_year": [2024, 2024]})
    projected, audit = apply_ro_register_share_to_proxies(
        out.loc[~named], register, carriers=("waste_to_energy",),
        exclude_names=out.loc[named, "station_name"].tolist(), classify=lambda name: "waste_to_energy")
    row = audit.set_index("carrier").loc["waste_to_energy"]
    assert row["proxy_mw"] == pytest.approx(2481.0 - 70.0)
    assert row["roc_mw"] == pytest.approx(500.0)          # the register's MW, unchanged by the carve-out
    total = projected.loc[projected["fuel_type"].eq("waste_to_energy"), "capacity_mw"].sum() \
        + out.loc[named & out["fuel_type"].eq("waste_to_energy"), "capacity_mw"].sum()
    assert total == pytest.approx(2481.0)


def test_refuses_a_proxy_smaller_than_its_named_units():
    proxy = _proxy()
    proxy = proxy[~proxy["fuel_type"].eq("biogas")]
    proxy = pd.concat([proxy, pd.DataFrame([{"fuel_type": "biogas", "technology": "biogas", "capacity_mw": 10.0,
                                             "gsp": "C", "data_source": "CP30_ES1_FES_spatial_proxy"}])],
                      ignore_index=True)
    with pytest.raises(ValueError, match="biogas proxy holds"):
        _carve(proxy)
