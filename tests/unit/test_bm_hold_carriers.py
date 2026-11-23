"""participation.generators.hold_carriers (scripts/market/market_utils._apply_participation_policy): generators of a
held carrier leave the BM and are held at their Stage 1 position, whatever the non-participant behaviour."""

import logging

import pandas as pd
import pypsa

from scripts.market.market_utils import _apply_participation_policy


def _network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-08-01", periods=2, freq="h"))
    n.add("Bus", "b")
    n.add("Generator", "ew", bus="b", carrier="embedded_wind", p_nom=10.0)
    n.add("Generator", "w", bus="b", carrier="wind_onshore", p_nom=10.0)
    return n


def _apply(n, generators):
    gens = n.generators.index
    zero = pd.Series(0.0, index=gens)
    su = pd.Series(dtype=float)
    cfg = {"balancing": {"participation": {"generators": generators}}}
    return _apply_participation_policy(n, cfg, logging.getLogger("t"), zero.copy(), zero.copy(), su, su.copy())


def test_held_carrier_bids_only_at_the_penalty_under_fallback_priced():
    n = _network()
    offer, bid, _, _ = _apply(n, {"behavior": "fallback_priced", "exclude_carriers": ["embedded_wind"],
                                  "hold_carriers": ["embedded_wind"], "penalty_offer_price": 6000.0,
                                  "penalty_bid_price": 6000.0})
    assert offer["ew"] == 6000.0 and bid["ew"] == 6000.0      # kept at Stage 1 unless nothing else can relieve
    assert offer["w"] == 0.0 and bid["w"] == 0.0
    assert "ew" not in set(n._bm_eligible_generators)
    assert len(n._bm_fixed_generators) == 0                    # never fixed outright: 22 May 2020 was infeasible


def test_exclusion_alone_keeps_fallback_prices():
    n = _network()
    _apply(n, {"behavior": "fallback_priced", "exclude_carriers": ["embedded_wind"]})
    assert len(n._bm_fixed_generators) == 0
    assert "ew" in set(n._bm_participating_generators)


def test_historical_hydro_inflow_equals_metered_output_day_by_day(tmp_path):
    """scripts/renewables/large_hydro_inflow.apply_historical_metered_hydro: fleet inflow = ESPENI, daily."""
    from scripts.renewables.large_hydro_inflow import (
        ESPENI_HYDRO_COLUMN, ESPENI_TIME_COLUMN, apply_historical_metered_hydro)
    from scripts.solve.hydro_constraints import large_hydro_inflow_pu

    times = pd.date_range("2020-01-01", "2020-12-31 23:30", freq="30min", tz="UTC")
    mw = pd.Series(500.0, index=times)
    mw[times.normalize() == pd.Timestamp("2020-08-02", tz="UTC")] = 260.0
    mw.iloc[100] = 1100.0                                  # the year's metered peak
    path = tmp_path / "espeni.csv"
    pd.DataFrame({ESPENI_TIME_COLUMN: times.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                  ESPENI_HYDRO_COLUMN: mw.to_numpy()}).to_csv(path, index=False)
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-08-01", periods=48, freq="h"))
    n.add("Bus", "b")
    n.add("Generator", "a", bus="b", carrier="large_hydro", p_nom=1000.0)
    n.add("Generator", "c", bus="b", carrier="large_hydro", p_nom=375.0)
    hydro = {"enabled": True, "large_hydro": {"reservoir_model": True, "inflow_profile_factor": 0.55}}
    cfg = {"modelled_year": 2020, "renewables": {"hydro": hydro}}
    assert apply_historical_metered_hydro(n, cfg, 2020, str(path), logging.getLogger("t"))
    inflow = large_hydro_inflow_pu(n, n.snapshots, hydro, ["a", "c"]).mul(n.generators.p_nom[["a", "c"]]).sum(axis=1)
    assert abs(inflow.iloc[0] - 500.0) < 1e-6 and abs(inflow.iloc[30] - 260.0) < 1e-6
    assert abs(n.generators.at["a", "p_max_pu"] - 1100.0 / 1375.0) < 1e-9
    future = dict(cfg, modelled_year=2030, FES_year=2025, FES_scenario="Holistic Transition")
    assert not apply_historical_metered_hydro(n, future, 2020, str(path), logging.getLogger("t"))


def test_declared_buses_move_stations_and_offshore_landings(tmp_path):
    """scripts/generators/declared_buses.apply_declared_buses: station patterns and REPD landing rows."""
    from scripts.generators.declared_buses import apply_declared_buses

    n = pypsa.Network()
    for b in ["FLEE41", "GRAI41", "TONG1-", "HARK13"]:
        n.add("Bus", b)
    n.add("Generator", "Grain CHP*", bus="FLEE41", carrier="CCGT", p_nom=1517.0)
    n.add("Generator", "Grainger Farm", bus="FLEE41", carrier="solar_pv", p_nom=1.0)
    n.add("Generator", "Robin_Rigg_East", bus="TONG1-", carrier="wind_offshore", p_nom=84.0)
    n.generators["repd_ref_id"] = ["", "", "2496.0"]
    (tmp_path / "s.csv").write_text("generator_pattern,bus,source\n" + r'"Grain( CHP\*| GT)?",GRAI41,x' + "\n")
    (tmp_path / "c.csv").write_text("repd_ref_id,site_name,landing_bus\n2496,Robin Rigg East,HARK13\n")
    moved = apply_declared_buses(n, tmp_path / "s.csv", tmp_path / "c.csv")
    assert n.generators.at["Grain CHP*", "bus"] == "GRAI41"
    assert n.generators.at["Grainger Farm", "bus"] == "FLEE41"
    assert n.generators.at["Robin_Rigg_East", "bus"] == "HARK13"
    assert len(moved) == 2


def test_declared_link_outage_zeroes_the_link_inside_its_window():
    """scripts/solve/link_outages.apply_declared_link_outages: end exclusive, other hours untouched."""
    import pytest
    from scripts.solve.link_outages import apply_declared_link_outages

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-01-09 22:00", periods=6, freq="h"))
    n.add("Bus", "a")
    n.add("Bus", "b")
    n.add("Link", "W", bus0="a", bus1="b", p_nom=2250.0, p_min_pu=-1.0)
    cfg = {"network": {"transmission": {"declared_link_outages": [
        {"link": "W", "start": "2020-01-10", "end": "2020-01-10 02:00", "source": "test"}]}}}
    assert apply_declared_link_outages(n, cfg, logging.getLogger("t")) == 1
    pmax, pmin = n.links_t.p_max_pu["W"], n.links_t.p_min_pu["W"]
    assert list(pmax) == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert list(pmin) == [-1.0, -1.0, 0.0, 0.0, -1.0, -1.0]
    with pytest.raises(ValueError, match="not in the network"):
        apply_declared_link_outages(n, {"network": {"transmission": {"declared_link_outages": [
            {"link": "X", "start": "2020-01-10", "end": "2020-01-11"}]}}}, logging.getLogger("t"))
