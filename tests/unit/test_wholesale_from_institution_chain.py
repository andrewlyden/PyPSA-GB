"""18 Oct 2026: the institution chain on a wholesale_from market year (cp30_network reads stage1's Stage 1).

The CP30 market months own no Stage 1 files, so every place that read
``{market}_wholesale_*`` must read the month's ``wholesale_from`` parent instead,
and a market year that owns its Stage 1 must read exactly what it did before.
Also: a central dispatch Stage 1 takes the balancing stage's network delivery case.
"""

import pandas as pd
import pytest

import scripts.market.fti_settle_year as fsy
import scripts.market.fti_settlement as fs
import scripts.public_ownership.nationalised_institutions as ni

PFX = "CP30_FFR_2030_"


def _write_dispatch(path, value):
    idx = pd.date_range("2030-01-01", periods=4, freq="h")
    pd.DataFrame({"g1": [value] * 4, "g2": [0.0] * 4}, index=idx).to_csv(path)


class TestNationalisedInstitutions:
    def test_stage_1_months_default_to_the_market_months(self):
        assert ni._paired_stage_1(["a", "b"], None) == ["a", "b"]
        assert ni._paired_stage_1(["a", "b"], ["pa", "pb"]) == ["pa", "pb"]
        with pytest.raises(ValueError, match="pair month by month"):
            ni._paired_stage_1(["a", "b"], ["pa"])

    def test_solve_identity_reads_the_parent_stage_1_schedule(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ni, "MARKET_DIR", tmp_path)
        (tmp_path / "child_balancing.nc").write_bytes(b"x")
        _write_dispatch(tmp_path / "parent_wholesale_dispatch.csv", 1.0)
        own = ni.solve_identity(["child"])
        assert own["files"] == 1  # the child owns no Stage 1 file
        paired = ni.solve_identity(["child"], ["parent"])
        assert paired["files"] == 2
        assert paired["fingerprint"] != own["fingerprint"]

    def test_a_month_that_owns_its_stage_1_is_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ni, "MARKET_DIR", tmp_path)
        (tmp_path / "m_balancing.nc").write_bytes(b"x")
        _write_dispatch(tmp_path / "m_wholesale_dispatch.csv", 1.0)
        assert ni.solve_identity(["m"]) == ni.solve_identity(["m"], ["m"])

    def test_counter_trade_reads_the_parent_schedule(self, tmp_path, monkeypatch):
        # The schedule a balancing month's counter-trade is measured against is its parent's.
        from scripts.interconnectors.market_coupling import COUNTER_TRADE_PREMIUM_COLUMN

        monkeypatch.setattr(ni, "MARKET_DIR", tmp_path)
        idx = pd.date_range("2030-01-01", periods=2, freq="h")
        pd.DataFrame({"ic": [100.0, 100.0]}, index=idx).to_csv(tmp_path / "parent_wholesale_links.csv")
        flow = pd.DataFrame({"ic": [150.0, 100.0]}, index=idx)
        links = pd.DataFrame({COUNTER_TRADE_PREMIUM_COLUMN: [10.0]}, index=["ic"])
        with pytest.raises(FileNotFoundError):
            ni.counter_trade_premium("child", flow, links)
        cost = ni.counter_trade_premium("child", flow, links, schedule_scenario="parent")
        assert float(cost.to_numpy().sum()) == pytest.approx(500.0)


class TestFtiSettleYear:
    def test_default_is_the_market_months_themselves(self):
        stage_1 = fsy.market_stage_1("cp30_network")
        assert stage_1["Jan"] == f"{PFX}Jan_cp30_network"
        assert stage_1["Dec"] == f"{PFX}Dec_cp30_network"

    def test_parents_are_paired_january_to_december(self):
        parents = [f"{PFX}{m}_stage1" for m in fsy.MONTHS]
        stage_1 = fsy.market_stage_1("cp30_network", parents)
        assert stage_1["Jul"] == f"{PFX}Jul_stage1"
        with pytest.raises(ValueError, match="12 names"):
            fsy.market_stage_1("cp30_network", parents[:11])
        with pytest.raises(ValueError, match="January to December"):
            fsy.market_stage_1("cp30_network", list(reversed(parents)))


class TestFtiSettlementWindowShare:
    def test_window_share_reads_the_parent_schedule(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fs, "MARKET_DIR", tmp_path)
        _write_dispatch(tmp_path / "child_balancing_dispatch.csv", 3.0)
        _write_dispatch(tmp_path / "parent_wholesale_dispatch.csv", 1.0)
        idx = pd.date_range("2030-01-01", periods=4, freq="h")
        sched = pd.DataFrame({"g1": [1.0] * 2, "g2": [0.0] * 2}, index=idx[:2])
        phys = pd.DataFrame({"g1": [3.0] * 2, "g2": [0.0] * 2}, index=idx[:2])
        with pytest.raises(FileNotFoundError):
            fs._window_share("child", sched, phys)
        assert fs._window_share("child", sched, phys, "parent") == pytest.approx(0.5)


class TestCentralNetworkDelivery:
    @staticmethod
    def _network():
        import pypsa

        n = pypsa.Network()
        n.add("Bus", ["a", "b"], v_nom=400.0)
        n.add("Line", "ab", bus0="a", bus1="b", x=0.1, s_nom=100.0)
        n.add("Load", "l", bus="b", p_set=10.0)
        return n

    def test_no_case_is_a_no_op(self):
        from scripts.market.solve_wholesale import apply_central_network_delivery
        import logging

        cfg = {"market": {"balancing": {}}, "network": {"transmission": {}}}
        assert apply_central_network_delivery(self._network(), cfg, logging.getLogger("t")) is cfg

    def test_the_case_reaches_stage_1_on_a_copy(self):
        from scripts.market.solve_wholesale import apply_central_network_delivery
        import logging

        derate = {"enabled": True, "mode": "replay", "unmapped": "default",
                  "file": "data/network/neso_boundary_outage_generic_2022_25.csv"}
        cfg = {"market": {"balancing": {"network_delivery": {
            "case": "outage_neso", "secured_capability": {"historic_derate": derate}}}},
            "network": {"transmission": {"outage_schedule": {"secured_capability": {"capability_year": 2030}}}}}
        out = apply_central_network_delivery(self._network(), cfg, logging.getLogger("t"))
        assert out is not cfg
        secured = out["network"]["transmission"]["outage_schedule"]["secured_capability"]
        assert secured["historic_derate"] == derate and secured["capability_year"] == 2030
        assert "historic_derate" not in cfg["network"]["transmission"]["outage_schedule"]["secured_capability"]
