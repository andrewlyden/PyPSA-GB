"""Decision 29: solve-time TYNDP price sensitivity on a shared 2050 network."""
import pandas as pd
import pypsa
import pytest

from scripts.interconnectors import foreign_price_override as fpo
from scripts.interconnectors.add_to_network import add_interconnector_links

PRICING = {"source": "derived", "foreign_price_source": "tyndp", "transaction_spread_gbp_per_mwh": 1.0,
           "tyndp": {"scenario": "DE", "route": "tyndp_shape"}}


def _network(level):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2050-01-01", periods=2, freq="h"))
    for bus in ("GB", "Ext_FR"):
        n.add("Bus", bus)
    cables = pd.DataFrame([{"name": "FR", "from_bus": "GB", "to_bus": "Ext_FR", "capacity_mw": 300.0,
                            "losses_percent": 2.5, "counterparty_country": "France"}])
    add_interconnector_links(n, cables, pd.DataFrame(), pd.DataFrame(), modelled_year=2050,
                             entsoe_prices_df=_shape(level), pricing_config=PRICING)
    return n


def _shape(level):
    return pd.DataFrame({"France": [level - 10.0, level + 10.0]},
                        index=pd.date_range("2013-01-01", periods=2, freq="h"))


def _config(override=None):
    cfg = {"modelled_year": 2050, "renewables_year": 2013, "interconnectors": {"pricing": PRICING}, "market": {}}
    if override is not None:
        cfg["market"]["foreign_price_override"] = override
    return cfg


def test_absent_or_disabled_is_a_no_op():
    n = _network(20.0)
    before = n.links_t.marginal_cost.copy()
    assert fpo.apply_foreign_price_override(n, _config(), {}) is None
    assert fpo.apply_foreign_price_override(n, _config({"enabled": False, "tyndp": {"scenario": "GA"}}), {}) is None
    pd.testing.assert_frame_equal(n.links_t.marginal_cost, before)
    assert fpo.shape_path(_config()) is None and fpo.market_demand_path(_config()) is None


def test_shape_path_names_the_override_variant():
    cfg = _config({"enabled": True, "tyndp": {"scenario": "GA"}})
    assert fpo.shape_path(cfg) == "resources/interconnectors/tyndp_GA_2050_shape_wind2013.csv"


def test_only_a_tyndp_built_network_may_be_repriced():
    cfg = _config({"enabled": True, "tyndp": {"scenario": "GA"}})
    cfg["interconnectors"]["pricing"] = {"source": "derived"}
    with pytest.raises(ValueError, match="TYNDP"):
        fpo.effective_pricing(cfg)
    with pytest.raises(ValueError, match="unknown keys"):
        fpo.override_block({"market": {"foreign_price_override": {"enabled": True, "level": 3}}})


def test_reprice_at_the_build_variant_is_exact_and_another_moves_the_links(tmp_path):
    n = _network(20.0)
    diffs = fpo.verify(n, _config(), _shape(20.0), None)
    assert max(v for v in diffs.values()) == 0
    path = tmp_path / "ga.csv"
    _shape(30.0).to_csv(path)
    summary = fpo.apply_foreign_price_override(
        n, _config({"enabled": True, "tyndp": {"scenario": "GA"}}), {"foreign_price_shape": str(path)})
    assert summary["levels"]["France"] == {"before": 20.0, "after": 30.0}
    assert n.links.at["IC_FR", "market_import_level"] == pytest.approx(30.0)
    assert n.links_t.marginal_cost["IC_FR"].tolist() == pytest.approx([20.5, 40.5])


# ── Y8 follow-up: the IC1 tranche refit (market.ic1_tranche_refit, 2050 only) ──────────────

TRANCHES = {"enabled": True, "elasticity": 1.0, "tranches": 3, "eu_scenario": "EU Green",
            "pathway_alignment": "Holistic Transition", "demand_year": 2050,
            "markets": {"continent": {"counterparties": ["France"], "demand_countries": ["France"]}}}


def _tranche_network(tmp_path):
    from scripts.interconnectors.volume_tranches import apply_volume_tranches

    demand_file = tmp_path / "es2.csv"
    pd.DataFrame([{"Variable": "Demand (TWh)", "EU Scenario": "EU Green", "FES Pathway Alignment": "Holistic Transition",
                   "Country": "France", "2050": 876.0}]).to_csv(demand_file, index=False)
    n = _network(20.0)
    apply_volume_tranches(n, TRANCHES, pd.Series({"France": 876.0}))
    pricing = {**PRICING, "volume_tranches": {**TRANCHES, "demand_file": str(demand_file)}}
    return n, pricing, str(demand_file)


def _refit_config(pricing, basis, **solve):
    cfg = {"modelled_year": 2050, "interconnectors": {"pricing": pricing},
           "market": {"ic1_tranche_refit": {"enabled": True, "level_basis": basis}}, **solve}
    return cfg


def test_refit_absent_is_a_no_op_and_annual_reproduces_the_build(tmp_path):
    n, pricing, demand = _tranche_network(tmp_path)
    built = n.links[["market_tranche_import_step", "market_tranche_export_step"]].copy()
    assert fpo.refit_tranche_steps(n, {"interconnectors": {"pricing": pricing}, "market": {}}, demand) is None
    assert fpo.refit_market_demand_path({"market": {}}) is None
    fpo.refit_tranche_steps(n, _refit_config(pricing, "annual"), demand)
    pd.testing.assert_frame_equal(n.links[built.columns], built)
    # step = elasticity x level / demand (GW) x width (GW): 20 / 100 GW x 0.1 GW
    assert built.iloc[0, 0] == pytest.approx(20.0 / 100.0 * 0.1)


def test_refit_traded_bases_use_the_ask_and_the_bid(tmp_path):
    n, pricing, demand = _tranche_network(tmp_path)
    table = fpo.refit_tranche_steps(n, _refit_config(pricing, "traded_annual"), demand)
    # hourly ask 10.5 and 30.5 (shape +/- 10 around 20, plus the 0.5 spread half); bid one spread below
    assert table.at["continent", "import_level"] == pytest.approx(20.5)
    assert table.at["continent", "export_level"] == pytest.approx(19.5)
    assert n.links.at["IC_FR", "market_tranche_import_step"] == pytest.approx(20.5 / 100.0 * 0.1)
    window = _refit_config(pricing, "traded_window",
                           solve_period={"enabled": True, "start": "2050-01-01 01:00", "end": "2050-01-01 01:00"})
    table = fpo.refit_tranche_steps(n, window, demand)
    assert table.at["continent", "import_level"] == pytest.approx(30.5)


def test_refit_rejects_unknown_keys_and_bases():
    with pytest.raises(ValueError, match="unknown keys"):
        fpo.tranche_refit_block({"market": {"ic1_tranche_refit": {"enabled": True, "steps": 2}}})
    with pytest.raises(ValueError, match="level_basis"):
        fpo.tranche_refit_block({"market": {"ic1_tranche_refit": {"enabled": True, "level_basis": "hourly"}}})
