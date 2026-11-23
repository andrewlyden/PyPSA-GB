"""Stage 1 interconnector hurdle override (16 Oct 2026, scripts/market/interconnector_hurdle.py, variant s1_nh)."""
import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.interconnectors.add_to_network import _foreign_bid_ask_prices, add_interconnector_links
from scripts.interconnectors.market_coupling import EXPORT_BID_FLOOR_COLUMN, get_piecewise_market_prices
from scripts.market.interconnector_hurdle import (
    apply_interconnector_hurdle_override,
    recover_reference,
    resolve_hurdle_override,
)

MARKETS = {"continent": {"counterparties": ["France"]}}
FRANCE = [-120.0, -20.0, 0.0, 0.4, 5.0, 60.0, 250.0]
OVERRIDE = {"interconnector_hurdle_override": {"enabled": True, "spread_gbp_per_mwh": 0.0,
                                               "export_bid_floor_gbp_per_mwh": None}}


def _pricing(spread, floor=1.0, **extra):
    return {"source": "cp30", "foreign_price_basis": "single", "transaction_spread_gbp_per_mwh": 1.0,
            "circulation_epsilon_gbp_per_mwh": 0.01, "cp30_import_price_gbp_per_mwh": 89.43,
            "cp30_export_price_gbp_per_mwh": 40.09, "volume_tranches": {"markets": MARKETS},
            "single_price_level_offsets_gbp_per_mwh": {"continent": 0.0},
            "single_price_transaction_spread_gbp_per_mwh": spread,
            "single_price_export_bid_floor_gbp_per_mwh": floor, **extra}


def _network(pricing):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=len(FRANCE), freq="h"))
    for bus in ("GB", "Ext_FR"):
        n.add("Bus", bus)
    cables = pd.DataFrame([{"name": "FR", "from_bus": "GB", "to_bus": "Ext_FR", "capacity_mw": 300.0,
                            "losses_percent": 2.5, "counterparty_country": "France"}])
    entsoe = pd.DataFrame({"France": FRANCE}, index=pd.date_range("2024-01-01", periods=len(FRANCE), freq="h"))
    add_interconnector_links(n, cables, pd.DataFrame(), pd.DataFrame(), modelled_year=2030,
                             entsoe_prices_df=entsoe, pricing_config=pricing)
    return n


def test_off_by_default_and_validated():
    assert resolve_hurdle_override({}) is None
    assert resolve_hurdle_override({"interconnector_hurdle_override": {"enabled": False}}) is None
    with pytest.raises(ValueError):
        resolve_hurdle_override({"interconnector_hurdle_override": {"enabled": True, "spread": 0}})
    with pytest.raises(ValueError):
        resolve_hurdle_override({"interconnector_hurdle_override": {"enabled": True, "spread_gbp_per_mwh": -1}})
    n = _network(_pricing(20.0))
    before = n.links_t.marginal_cost.copy()
    assert apply_interconnector_hurdle_override(n, {}) is None
    pd.testing.assert_frame_equal(n.links_t.marginal_cost, before)


@pytest.mark.parametrize("half", [0.0, 0.5, 10.0, 13.5])
def test_reference_is_recovered_exactly_from_the_builders_ask(half):
    reference = pd.Series(np.linspace(-300.0, 300.0, 601))
    pricing = {"transaction_spread_gbp_per_mwh": 2 * half, "circulation_epsilon_gbp_per_mwh": 0.01}
    ask, _ = _foreign_bid_ask_prices(reference, 0.975, pricing)
    back = recover_reference(ask, 0.975, half, 0.01)
    if half >= 0.01:
        assert back.to_numpy() == pytest.approx(reference.to_numpy(), abs=1e-9)
    else:   # below epsilon the builder's half is epsilon: the recovery is that of piecewise_market_prices
        assert np.abs(back.to_numpy() - reference.to_numpy()).max() <= 0.011


@pytest.mark.parametrize("floor", [None, 1.0])
def test_override_equals_a_network_built_at_that_spread(floor):
    built20 = _network(_pricing(20.0, floor=1.0))
    target = _network(_pricing(0.0, floor=floor))
    cfg = {"interconnector_hurdle_override": {**OVERRIDE["interconnector_hurdle_override"],
                                              "export_bid_floor_gbp_per_mwh": floor}}
    report = apply_interconnector_hurdle_override(built20, cfg)
    assert report["links"] == 1
    ask_o, bid_o = get_piecewise_market_prices(built20, "IC_FR", built20.snapshots)
    ask_t, bid_t = get_piecewise_market_prices(target, "IC_FR", target.snapshots)
    assert ask_o.tolist() == pytest.approx(ask_t.tolist(), abs=1e-9)
    assert bid_o.tolist() == pytest.approx(bid_t.tolist(), abs=1e-9)
    for col in ("marginal_cost", "market_spread_half", "market_export_bid", "market_export_bid_offset"):
        assert built20.links.at["IC_FR", col] == pytest.approx(target.links.at["IC_FR", col], abs=1e-9)
    f_o, f_t = built20.links.at["IC_FR", EXPORT_BID_FLOOR_COLUMN], target.links.at["IC_FR", EXPORT_BID_FLOOR_COLUMN]
    assert (np.isnan(f_o) and np.isnan(f_t)) or f_o == pytest.approx(f_t)


def test_no_hurdle_trades_at_the_foreign_price_and_keeps_the_tranches():
    n = _network(_pricing(20.0))
    # The IC1 tranche attributes as scripts/interconnectors/volume_tranches.py stamps them.
    for col, value in (("market_tranche_group", "continent"), ("market_tranche_width_mw", 100.0),
                       ("market_tranche_count", 3), ("market_tranche_import_step", 1.52),
                       ("market_tranche_export_step", 0.68)):
        n.links.loc["IC_FR", col] = value
    tranche_cols = [c for c in n.links.columns if c.startswith("market_tranche")]
    assert len(tranche_cols) == 5
    tranches = n.links.loc["IC_FR", tranche_cols].copy()
    apply_interconnector_hurdle_override(n, OVERRIDE)
    ask, bid = get_piecewise_market_prices(n, "IC_FR", n.snapshots)
    positive = pd.Series(FRANCE, index=n.snapshots) >= 1.0
    # Within the 0.01 circulation epsilon of the foreign price where it is positive (no 10 GBP half hurdle).
    assert (ask[positive] - pd.Series(FRANCE, index=n.snapshots)[positive]).abs().max() <= 0.011
    assert (bid[positive] - pd.Series(FRANCE, index=n.snapshots)[positive]).abs().max() <= 0.011
    pd.testing.assert_series_equal(n.links.loc["IC_FR", tranche_cols], tranches)


def test_separate_basis_is_refused():
    n = _network(_pricing(20.0, foreign_price_basis="separate"))
    with pytest.raises(ValueError, match="single foreign-price basis"):
        apply_interconnector_hurdle_override(n, OVERRIDE)


# Run B (17 Oct 2026): the level offset and the balancing-stage switch.

@pytest.mark.parametrize("offset,spread", [(6.75, 7.5), (-3.0, 20.0), (0.0, 20.0)])
def test_level_offset_equals_a_network_built_at_that_level(offset, spread):
    built = _network(_pricing(20.0, floor=1.0))
    target = _network(_pricing(spread, floor=1.0, single_price_level_offsets_gbp_per_mwh={"continent": offset}))
    cfg = {"interconnector_hurdle_override": {"enabled": True, "spread_gbp_per_mwh": spread,
                                              "export_bid_floor_gbp_per_mwh": 1.0,
                                              "level_offset_gbp_per_mwh": offset}}
    report = apply_interconnector_hurdle_override(built, cfg)
    assert report["level_offset_gbp_per_mwh"] == offset and report["stage"] == "wholesale"
    ask_o, bid_o = get_piecewise_market_prices(built, "IC_FR", built.snapshots)
    ask_t, bid_t = get_piecewise_market_prices(target, "IC_FR", target.snapshots)
    assert ask_o.tolist() == pytest.approx(ask_t.tolist(), abs=1e-9)
    assert bid_o.tolist() == pytest.approx(bid_t.tolist(), abs=1e-9)
    for col in ("marginal_cost", "market_spread_half", "market_export_bid", "market_export_bid_offset"):
        assert built.links.at["IC_FR", col] == pytest.approx(target.links.at["IC_FR", col], abs=1e-9)


def test_new_keys_default_to_the_old_behaviour():
    cfg = resolve_hurdle_override(OVERRIDE)
    assert cfg["level_offset_gbp_per_mwh"] == 0.0 and cfg["apply_in_balancing"] is False
    with pytest.raises(ValueError):
        resolve_hurdle_override({"interconnector_hurdle_override": {"enabled": True,
                                                                    "level_offset_gbp_per_mwh": float("nan")}})


def test_balancing_stage_applies_only_when_asked():
    base = {"enabled": True, "spread_gbp_per_mwh": 7.5, "export_bid_floor_gbp_per_mwh": 1.0,
            "level_offset_gbp_per_mwh": 6.75}
    n = _network(_pricing(20.0))
    before = n.links_t.marginal_cost.copy()
    assert apply_interconnector_hurdle_override(n, {"interconnector_hurdle_override": base}, stage="balancing") is None
    pd.testing.assert_frame_equal(n.links_t.marginal_cost, before)
    stage1 = _network(_pricing(20.0))
    apply_interconnector_hurdle_override(stage1, {"interconnector_hurdle_override": base})
    bm = _network(_pricing(20.0))
    report = apply_interconnector_hurdle_override(
        bm, {"interconnector_hurdle_override": {**base, "apply_in_balancing": True}}, stage="balancing")
    assert report["stage"] == "balancing"
    pd.testing.assert_frame_equal(bm.links_t.marginal_cost, stage1.links_t.marginal_cost)
    assert bm.links.at["IC_FR", "market_export_bid"] == pytest.approx(stage1.links.at["IC_FR", "market_export_bid"])
    with pytest.raises(ValueError):
        apply_interconnector_hurdle_override(bm, {"interconnector_hurdle_override": base}, stage="bm")
