"""Station-month availability from ELEXON notifications (scripts/generators/build_elexon_monthly_availability.py)
and its ``fixed`` rows (scripts/generators/support_scheme_tranches.apply_generator_monthly_availability)."""

import pandas as pd
import pypsa

from scripts.generators.build_elexon_monthly_availability import monthly_rows, station_bmus
from scripts.generators.support_scheme_tranches import apply_generator_monthly_availability


def _network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-01-30", "2020-02-02 23:00", freq="h"))
    n.add("Bus", "b")
    n.add("Generator", "Peterhead", bus="b", p_nom=1000.0, p_min_pu=0.0)
    n.add("Generator", "Hunterston B", bus="b", p_nom=1000.0, p_min_pu=0.5)
    return n


def test_fixed_row_holds_the_station_and_a_cap_lowers_a_higher_minimum(tmp_path):
    table = pd.DataFrame([
        {"generator": "Peterhead", "year": 2020, "month": 1, "availability_factor": 0.4, "fixed": True},
        {"generator": "Hunterston B", "year": 2020, "month": 1, "availability_factor": 0.0, "fixed": False},
    ])
    path = tmp_path / "availability.csv"
    table.to_csv(path, index=False)
    n = _network()
    assert apply_generator_monthly_availability(n, path, 2020) == 2
    jan = n.snapshots.month == 1
    assert (n.generators_t.p_max_pu.loc[jan, "Peterhead"] == 0.4).all()
    assert (n.generators_t.p_min_pu.loc[jan, "Peterhead"] == 0.4).all()
    assert (n.generators_t.p_min_pu.loc[~jan, "Peterhead"] == 0.0).all()
    assert (n.generators_t.p_max_pu.loc[jan, "Hunterston B"] == 0.0).all()
    assert (n.generators_t.p_min_pu.loc[jan, "Hunterston B"] == 0.0).all()   # cap below the static minimum


def test_heysham_reactors_go_to_their_own_station_and_rows_are_monthly_means():
    mapping = pd.DataFrame({"bmu_id": ["T_HEYM11", "T_HEYM12", "T_HEYM27", "T_HEYM28"],
                            "generator_name": ["Heysham 2"] * 4})
    pn = pd.DataFrame({"T_HEYM11": [300.0, 500.0], "T_HEYM12": [0.0, 0.0],
                       "T_HEYM27": [600.0, 600.0], "T_HEYM28": [600.0, 600.0]},
                      index=pd.to_datetime(["2020-01-01 09:30", "2020-01-01 18:30"]))
    bmus = station_bmus(mapping, pn.columns)
    assert bmus == {"Heysham 1": ["T_HEYM11", "T_HEYM12"], "Heysham 2": ["T_HEYM27", "T_HEYM28"]}
    rows = monthly_rows(pn, bmus, pd.Series({"Heysham 1": 1000.0, "Heysham 2": 1200.0}), 2020,
                        ["Heysham 1", "Heysham 2"], fixed=False).set_index("generator")
    assert rows.loc["Heysham 1", "availability_factor"] == 0.4
    assert rows.loc["Heysham 2", "availability_factor"] == 1.0


def test_random_outages_skip_units_with_observed_availability(tmp_path):
    from scripts.generators.thermal_availability import apply_thermal_outage_availability
    import logging

    table = pd.DataFrame([{"generator": "Peterhead", "year": 2020, "month": 1, "availability_factor": 0.4,
                           "fixed": True, "evidence": "ELEXON physical notifications"}])
    path = tmp_path / "availability.csv"
    table.to_csv(path, index=False)
    n = _network()
    n.generators["carrier"] = "CCGT"
    apply_generator_monthly_availability(n, path, 2020)
    config = {"enabled": True, "carriers": {"CCGT": {"forced_outage_rate": 0.5, "planned_outages_per_year": 0}}}
    apply_thermal_outage_availability(n, config, logging.getLogger("t"))
    jan = n.snapshots.month == 1
    assert (n.generators_t.p_max_pu.loc[jan, "Peterhead"] == 0.4).all()          # no outage hours added
    assert (n.generators_t.p_min_pu.loc[jan, "Peterhead"] <= n.generators_t.p_max_pu.loc[jan, "Peterhead"]).all()
