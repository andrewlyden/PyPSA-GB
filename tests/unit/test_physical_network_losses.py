from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.consumer_bill.aggregate_physical_network_losses import (
    loss_representation_status,
    resistance_weighted_i2r_loss_proxy_mwh,
)


def test_lossless_linear_network_is_flagged_not_treated_as_zero_physical_loss():
    assert loss_representation_status(0.0, 20.0) == "lossless_linear_network"
    assert loss_representation_status(0.1, 20.0) == "represented_ac_losses_available"


def test_i2r_proxy_uses_resistance_weighted_flow_squared():
    snapshots = pd.date_range("2020-01-01", periods=1, freq="h")
    network = SimpleNamespace(
        snapshot_weightings=pd.DataFrame({"generators": [1.0]}, index=snapshots),
        lines=pd.DataFrame({"bus0": ["bus"], "r": [4.0]}, index=["line"]),
        buses=pd.DataFrame({"v_nom": [200.0]}, index=["bus"]),
        lines_t=SimpleNamespace(p0=pd.DataFrame({"line": [100.0]}, index=snapshots)),
        transformers=pd.DataFrame(columns=["r", "s_nom"]),
        transformers_t=SimpleNamespace(p0=pd.DataFrame(index=snapshots)),
    )

    total, line, transformer = resistance_weighted_i2r_loss_proxy_mwh(network)

    assert line == pytest.approx(1.0)  # 100² × 4 / 200² MW for one hour
    assert transformer == 0.0
    assert total == pytest.approx(1.0)
