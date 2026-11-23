import logging

import pandas as pd
import pypsa
import pytest

import scripts.generators.support_scheme_tranches as tranches
from scripts.generators.support_scheme_tranches import (
    apply_generator_monthly_availability,
    apply_physical_commissioning_gates,
    split_historical_cfd_tranches,
)
from scripts.market.revenue_tracking import compute_roc_income


def _network():
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2020-01-01", periods=4, freq="h"))
    network.add("Bus", "b")
    network.add(
        "Generator", "shared", bus="b", carrier="wind_offshore", p_nom=100.0,
        support_type="CfD", operational_date="02/01/2020",
    )
    network.generators_t.p_max_pu["shared"] = [0.1, 0.2, 0.3, 0.4]
    return network


def test_cfd_split_conserves_capacity_and_profile(monkeypatch):
    network = _network()
    mapping = pd.DataFrame({
        "cfd_id": ["A", "B"], "model_generator": ["shared", "shared"],
        "allocation_share": [0.4, 0.5], "mapping_method": ["test", "test"],
        "notes": ["", ""],
    })
    actual = pd.DataFrame({
        "Settlement_Date": ["2020-01-01", "2020-02-01"], "CfD_ID": ["A", "B"]
    })
    def read_csv(path, **kwargs):
        frame = (mapping if str(path) == "mapping" else actual).copy()
        return frame[kwargs["usecols"]] if kwargs.get("usecols") else frame
    monkeypatch.setattr(tranches.pd, "read_csv", read_csv)

    before = network.generators.p_nom.sum()
    split_historical_cfd_tranches(network, "mapping", "actual")

    assert network.generators.p_nom.sum() == before
    assert network.generators.at["shared", "p_nom"] == pytest.approx(10.0)
    assert network.generators.at["shared", "support_type"] == "merchant"
    assert network.generators.at["shared__CfD__A", "p_nom"] == 40.0
    pd.testing.assert_series_equal(
        network.generators_t.p_max_pu["shared__CfD__A"],
        network.generators_t.p_max_pu["shared"], check_names=False,
    )


def test_commissioning_gate_and_monthly_override_replace_carrier_factor(monkeypatch):
    network = _network()
    network.generators.at["shared", "p_min_pu"] = 0.4
    assert apply_physical_commissioning_gates(network) == 1
    assert network.generators_t.p_max_pu["shared"].eq(0.0).all()
    assert network.generators_t.p_min_pu["shared"].eq(0.0).all()
    assert network.generators.at["shared", "p_min_pu"] == 0.0

    availability = pd.DataFrame({
        "generator": ["shared"], "year": [2020], "month": [1],
        "availability_factor": [0.75],
    })
    monkeypatch.setattr(tranches.pd, "read_csv", lambda *args, **kwargs: availability.copy())
    apply_generator_monthly_availability(network, "availability", 2020)
    assert network.generators_t.p_max_pu["shared"].eq(0.75).all()


def test_roc_income_uses_only_eligible_share():
    network = _network()
    network.generators.at["shared", "support_type"] = "ROC"
    network.generators.at["shared", "ro_banding"] = 1.0
    network.generators.at["shared", "roc_eligible_share"] = 0.25
    network.generators_t.p["shared"] = 100.0
    result = compute_roc_income(network, 50.0, logging.getLogger("test"))
    assert result.iloc[0]["total_dispatch_MWh"] == 400.0
    assert result.iloc[0]["roc_eligible_dispatch_MWh"] == 100.0
    assert result.iloc[0]["roc_income_total"] == 5000.0
