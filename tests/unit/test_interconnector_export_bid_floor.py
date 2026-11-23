"""Single-basis export-bid floor: no export hurdle when GB is in surplus (27 Sep 2026)."""
import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.interconnectors.add_to_network import add_interconnector_links, single_price_export_bid_floor
from scripts.interconnectors.market_coupling import (
    EXPORT_BID_FLOOR_COLUMN,
    build_piecewise_interconnector_cost_callback,
    export_efficiency,
    floor_export_bid,
    get_piecewise_market_prices,
)

MARKETS = {"continent": {"counterparties": ["France"]}}
FRANCE = [-20.0, 0.4, 5.0, 60.0]          # foreign price (offset 0): negative, below F, below hurdle/2, high


def _pricing(**extra):
    return {"source": "cp30", "foreign_price_basis": "single", "transaction_spread_gbp_per_mwh": 1.0,
            "circulation_epsilon_gbp_per_mwh": 0.01, "cp30_import_price_gbp_per_mwh": 89.43,
            "cp30_export_price_gbp_per_mwh": 40.09, "volume_tranches": {"markets": MARKETS},
            "single_price_level_offsets_gbp_per_mwh": {"continent": 0.0},
            "single_price_transaction_spread_gbp_per_mwh": 27.0, **extra}


def _network(prices=FRANCE, pricing=None):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=len(prices), freq="h"))
    for bus in ("GB", "Ext_FR"):
        n.add("Bus", bus)
    cables = pd.DataFrame([{"name": "FR", "from_bus": "GB", "to_bus": "Ext_FR", "capacity_mw": 300.0,
                            "losses_percent": 2.5, "counterparty_country": "France"}])
    entsoe = pd.DataFrame({"France": list(prices)}, index=pd.date_range("2024-01-01", periods=len(prices), freq="h"))
    add_interconnector_links(n, cables, pd.DataFrame(), pd.DataFrame(), modelled_year=2030,
                             entsoe_prices_df=entsoe, pricing_config=pricing or _pricing())
    return n


def test_floor_is_off_by_default():
    assert single_price_export_bid_floor(_pricing()) is None
    n = _network()
    assert np.isnan(n.links.at["IC_FR", EXPORT_BID_FLOOR_COLUMN])
    _, bid = get_piecewise_market_prices(n, "IC_FR", n.snapshots)
    assert bid.tolist() == pytest.approx([-33.5, -13.1, -8.5, 46.5])   # price - 27/2
    assert floor_export_bid(-8.5, 5.0, None) == -8.5
    assert floor_export_bid(-8.5, 5.0, float("nan")) == -8.5
    with pytest.raises(ValueError):
        single_price_export_bid_floor(_pricing(single_price_export_bid_floor_gbp_per_mwh=-1.0))


def test_floor_lifts_only_the_export_bid_and_never_above_the_foreign_price():
    off = _network()
    on = _network(pricing=_pricing(single_price_export_bid_floor_gbp_per_mwh=1.0))
    assert on.links.at["IC_FR", EXPORT_BID_FLOOR_COLUMN] == pytest.approx(1.0)
    ask_off, _ = get_piecewise_market_prices(off, "IC_FR", off.snapshots)
    ask_on, bid_on = get_piecewise_market_prices(on, "IC_FR", on.snapshots)
    assert ask_on.tolist() == pytest.approx(ask_off.tolist())            # imports keep the full hurdle
    # negative price keeps its bid; 0.4 < F -> the price itself; 5 -> F; 60 is above F already.
    assert bid_on.tolist() == pytest.approx([-33.5, 0.4, 1.0, 46.5])
    assert (bid_on <= pd.Series(FRANCE, index=on.snapshots) + 1e-9).all()
    assert (ask_on >= export_efficiency(0.975) * bid_on - 1e-9).all()   # still convex


def test_floor_is_ignored_on_the_separate_basis():
    n = _network(pricing=_pricing(foreign_price_basis="separate", single_price_export_bid_floor_gbp_per_mwh=1.0))
    assert np.isnan(n.links.at["IC_FR", EXPORT_BID_FLOOR_COLUMN])


def test_floor_helper_on_scalars_and_series():
    assert floor_export_bid(-8.5, 5.0, 1.0) == pytest.approx(1.0)
    assert floor_export_bid(-13.1, 0.4, 1.0) == pytest.approx(0.4)
    assert floor_export_bid(-33.5, -20.0, 1.0) == pytest.approx(-33.5)
    assert floor_export_bid(46.5, 60.0, 1.0) == pytest.approx(46.5)
    bid = pd.Series([-33.5, -13.1, -8.5, 46.5])
    ref = pd.Series(FRANCE)
    assert floor_export_bid(bid, ref, 1.0).tolist() == pytest.approx([-33.5, 0.4, 1.0, 46.5])


@pytest.mark.parametrize("floor, exports", [(None, False), (0.5, False), (1.0, True)])
def test_surplus_exports_against_a_merchant_bid_only_when_the_floor_beats_it_after_the_loss(floor, exports):
    """GB surplus priced at the merchant Stage 1 bid (+0.5), foreign price 5 (below the half hurdle).

    Per MWh GB sends, the export earns eta x bid: 0.975 x 1.0 beats 0.5, 0.975 x 0.5 does not.
    """
    extra = {} if floor is None else {"single_price_export_bid_floor_gbp_per_mwh": floor}
    n = _network(prices=[5.0], pricing=_pricing(**extra))
    n.add("Generator", "merchant_wind", bus="GB", p_nom=1000.0, marginal_cost=0.5)
    n.add("Load", "demand", bus="GB", p_set=0.0)
    status, _ = n.optimize(solver_name="highs", extra_functionality=build_piecewise_interconnector_cost_callback(n))
    assert status == "ok"
    p0 = float(n.links_t.p0.iloc[0]["IC_FR"])
    if exports:
        assert p0 == pytest.approx(-300.0)
        sent = float(n.links_t.p1.iloc[0]["IC_FR"])                     # GB withdrawal = eta x 300
        assert sent == pytest.approx(0.975 * 300.0)
        assert float(n.generators_t.p.iloc[0]["merchant_wind"]) == pytest.approx(sent)
    else:
        assert p0 == pytest.approx(0.0, abs=1e-6)
