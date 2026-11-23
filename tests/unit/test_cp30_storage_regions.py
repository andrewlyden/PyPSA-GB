"""CP30 ES5 regional storage split (9 Oct 2026): new build follows ES5 shares, existing sites and totals hold."""
import pandas as pd
import pytest

from scripts.CP30.storage_regions import apply_es5_storage_regions


def _scaled():
    return pd.DataFrame({
        "technology": ["Battery", "Battery", "Battery", "LAES", "LAES", "LAES"],
        "power_mw": [600.0, 300.0, 100.0, 500.0, 200.0, 100.0],
        "duration_h": [2.0, 2.0, 2.0, 8.0, 8.0, 8.0],
        "energy_mwh": [1200.0, 600.0, 200.0, 4000.0, 1600.0, 800.0],
        "existing_site": [None, None, None, "Dinorwig", None, None],
    })


def _targets():
    return pd.DataFrame({"carrier": ["Battery", "Battery", "Battery", "LAES", "LAES", "LAES"],
                         "cp30_region": ["A", "B", "C", "A", "B", "C"],
                         "capacity_mw": [100.0, 300.0, 600.0, 400.0, 0.0, 400.0]})


def test_new_build_follows_es5_shares_and_totals_hold():
    regions = pd.Series(["A", "B", "C", "A", "B", "C"])
    out = apply_es5_storage_regions(_scaled(), _targets(), regions)
    bat = out[out.technology.eq("Battery")].groupby("cp30_region").power_mw.sum()
    assert bat.to_dict() == pytest.approx({"A": 100.0, "B": 300.0, "C": 600.0})
    laes = out[out.technology.eq("LAES")]
    # Dinorwig (existing, region A) stays at 500; the 300 MW of new build all goes to C (A is already over, B is 0).
    assert laes.loc[laes.existing_site.eq("Dinorwig"), "power_mw"].item() == pytest.approx(500.0)
    assert laes.groupby("cp30_region").power_mw.sum().to_dict() == pytest.approx({"A": 500.0, "C": 300.0})
    assert (out.power_mw > 0).all()                        # emptied new-build rows are dropped
    assert out.power_mw.sum() == pytest.approx(_scaled().power_mw.sum())
    assert (out.energy_mwh == out.power_mw * out.duration_h).all()


def test_a_region_without_proxy_rows_passes_its_share_on():
    regions = pd.Series(["A", "A", "B", "A", "B", "C"])     # no battery row in C
    out = apply_es5_storage_regions(_scaled(), _targets(), regions)
    bat = out[out.technology.eq("Battery")].groupby("cp30_region").power_mw.sum()
    assert bat.sum() == pytest.approx(1000.0)
    assert bat["B"] / bat["A"] == pytest.approx(300.0 / 100.0)


# Reconciled on the placed bus's region (16 Oct 2026): the GSP-node move can cross a region edge.

def _region_files(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    polygons = gpd.GeoDataFrame({"GSPGroup": ["_D", "_G"], "GSPs": ["DINO_1", "KIBY_G|WASF_1"]},
                                geometry=[box(0, 0, 100_000, 100_000), box(100_000, 0, 200_000, 100_000)],
                                crs="EPSG:27700")
    polygon_file = tmp_path / "gsp.geojson"
    polygons.to_file(polygon_file, driver="GeoJSON")
    group_file = tmp_path / "groups.csv"
    pd.DataFrame({"gsp_group": ["_D", "_G"], "cp30_region": ["Wales", "North England"]}).to_csv(group_file, index=False)
    return polygon_file, group_file


def _placed_network():
    from types import SimpleNamespace
    # KIBY11 is the node of the North England (Kirkby/Washway Farm) polygon but sits in the Welsh one.
    return SimpleNamespace(buses=pd.DataFrame({"x": [50_000.0, 150_000.0, 20_000.0], "y": [50_000.0] * 3},
                                              index=["KIBY11", "WASF11", "DINO11"]))


def _labelled():
    return pd.DataFrame({
        "site_name": ["lancs_gsp", "lancs_other", "welsh"], "technology": "Battery",
        "power_mw": [200.0, 100.0, 100.0], "duration_h": 2.0, "energy_mwh": [400.0, 200.0, 200.0],
        "bus": ["KIBY11", "WASF11", "DINO11"], "gsp_ids": ["KIBY_G|WASF_1", "KIBY_G|WASF_1", "DINO_1"],
        "cp30_region": ["North England", "North England", "Wales"],   # by the site's own polygon
    })


def _battery_targets():
    return pd.DataFrame({"carrier": ["Battery", "Battery"], "cp30_region": ["North England", "Wales"],
                         "capacity_mw": [300.0, 100.0]})


def test_reconcile_on_placed_bus_restores_es5_by_bus_region(tmp_path):
    from scripts.CP30.storage_regions import reconcile_storage_regions_to_buses, storage_bus_regions

    polygon_file, group_file = _region_files(tmp_path)
    network = _placed_network()
    out, crossed = reconcile_storage_regions_to_buses(_labelled(), network, _battery_targets(), polygon_file,
                                                      group_file)
    assert crossed["site_name"].tolist() == ["lancs_gsp"]
    assert crossed[["label_region", "bus_region"]].iloc[0].tolist() == ["North England", "Wales"]
    by_bus = out.groupby(storage_bus_regions(out, network.buses, polygon_file, group_file))["power_mw"].sum()
    assert by_bus.to_dict() == pytest.approx({"North England": 300.0, "Wales": 100.0})   # ES5, judged by bus
    assert out["power_mw"].sum() == pytest.approx(400.0)                                  # national total holds
    assert out["bus"].tolist() == _labelled()["bus"].tolist()                             # nothing moves
    assert (out["energy_mwh"] == out["power_mw"] * out["duration_h"]).all()               # durations kept


def test_reconcile_is_a_no_op_when_no_row_crossed(tmp_path):
    from scripts.CP30.storage_regions import reconcile_storage_regions_to_buses

    polygon_file, group_file = _region_files(tmp_path)
    rows = _labelled()
    rows.loc[0, "cp30_region"] = "Wales"
    out, crossed = reconcile_storage_regions_to_buses(rows, _placed_network(), _battery_targets(), polygon_file,
                                                      group_file)
    assert crossed.empty
    pd.testing.assert_frame_equal(out, rows)


def test_reconcile_keeps_transmission_site_overrides(tmp_path):
    from types import SimpleNamespace

    from scripts.CP30.storage_regions import storage_bus_regions

    polygon_file, group_file = _region_files(tmp_path)
    buses = pd.DataFrame({"x": [50_000.0, 150_000.0], "y": [50_000.0, 50_000.0]}, index=["CRUA2Q", "WASF11"])
    rows = pd.DataFrame({"bus": ["CRUA2Q", "WASF11"], "site_code": ["CRUA", ""]})
    region = storage_bus_regions(rows, SimpleNamespace(buses=buses).buses, polygon_file, group_file)
    assert region.tolist() == ["South Scotland", "North England"]   # spatial_allocation.TRANSMISSION_SITE_REGIONS
