"""Item IC1: foreign prices that respond to how much GB trades.

A flat CP.29 price let every exported MWh earn the same price however much GB
sold. The tranches make the foreign price respond to GB's aggregate trade with
each coupled market: the import price rises and the export price falls by a
constant-elasticity step at each third of GB's capacity into that market,

    step = elasticity x level / market demand (GW) x tranche width (GW),

with the market's 2030 demand from FES 2025 ES2 and the CP.29 level for each
direction. The cost stays convex in the signed flow, so the LP stays an LP.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.interconnectors.add_to_network import add_interconnector_links
from scripts.interconnectors.market_coupling import (
    allocate_tranche_cost,
    build_piecewise_interconnector_cost_callback,
    get_volume_tranche_groups,
    interconnector_trade_cost,
    link_price_tables,
    tranche_cost,
)
from scripts.interconnectors.volume_tranches import (
    apply_volume_tranches,
    derive_market_tranches,
    load_market_demand_twh,
)

MARKETS = {
    "continent": {
        "counterparties": ["France", "Belgium", "Netherlands", "Denmark"],
        "demand_countries": ["France", "Belgium", "Netherlands", "Germany", "Denmark"],
    },
    "sem": {
        "counterparties": ["Ireland", "Northern Ireland", "Isle Of Man"],
        "demand_countries": ["SEM"],
    },
}
DEMAND = pd.Series({
    "France": 546.8, "Belgium": 119.2, "Netherlands": 149.9, "Germany": 671.4,
    "Denmark": 49.7, "SEM": 46.5,
})


def _ic_table(rows):
    return pd.DataFrame(rows).set_index("name")


# ── derivation ───────────────────────────────────────────────────────────────

def test_derivation_reproduces_the_approved_continent_steps():
    ic = _ic_table([
        {"name": "IC_IFA", "p_nom": 4047.8, "counterparty_country": "France"},
        {"name": "IC_Nemo", "p_nom": 1035.3, "counterparty_country": "Belgium"},
        {"name": "IC_Britned", "p_nom": 2436.0, "counterparty_country": "Netherlands"},
        {"name": "IC_Viking", "p_nom": 1421.0, "counterparty_country": "Denmark"},
    ]).assign(market_import_level=89.43, market_export_level=40.09)
    table = derive_market_tranches(ic, MARKETS, DEMAND, elasticity=1.0, n_tranches=3)
    row = table.loc["continent"]
    assert row["capacity_mw"] == pytest.approx(8940.1)
    assert row["width_mw"] == pytest.approx(8940.1 / 3)
    demand_gw = 1537.0 / 8.76
    assert row["demand_gw"] == pytest.approx(demand_gw)
    # The proposal's table: import +0 / +1.52 / +3.04, export -0 / -0.68 / -1.36.
    assert row["import_step"] == pytest.approx(89.43 / demand_gw * 8.9401 / 3, rel=1e-9)
    assert row["import_step"] == pytest.approx(1.52, abs=0.01)
    assert row["export_step"] == pytest.approx(0.68, abs=0.01)
    assert "sem" not in table.index  # no SEM cable in this table: no row, no error


def test_an_unassigned_counterparty_is_refused_not_dropped():
    ic = _ic_table([{"name": "IC_NSL", "p_nom": 1421.0, "counterparty_country": "Norway"}]).assign(
        market_import_level=89.43, market_export_level=40.09)
    with pytest.raises(ValueError, match="Norway"):
        derive_market_tranches(ic, MARKETS, DEMAND, elasticity=1.0, n_tranches=3)


def test_a_market_without_demand_data_is_refused():
    ic = _ic_table([{"name": "IC_EW", "p_nom": 500.0, "counterparty_country": "Ireland"}]).assign(
        market_import_level=89.43, market_export_level=40.09)
    with pytest.raises(ValueError, match="SEM"):
        derive_market_tranches(ic, MARKETS, DEMAND.drop("SEM"), elasticity=1.0, n_tranches=3)


def test_demand_is_read_for_one_eu_scenario_and_pathway(tmp_path):
    frame = pd.DataFrame([
        {"Country": "France", "EU Scenario": "EU Green", "FES Pathway Alignment": "Holistic Transition",
         "Variable": "Demand (TWh)", "2030": 546.8},
        {"Country": "France", "EU Scenario": "EU Green", "FES Pathway Alignment": "Electric Engagement",
         "Variable": "Demand (TWh)", "2030": 547.1},
        {"Country": "France", "EU Scenario": "EU Green", "FES Pathway Alignment": "Holistic Transition",
         "Variable": "Capacity (MW)", "2030": 9.0},
    ])
    path = tmp_path / "es2.csv"
    frame.to_csv(path, index=False)
    demand = load_market_demand_twh(path, "EU Green", "Holistic Transition", 2030)
    assert demand.to_dict() == {"France": 546.8}


# ── the cost curve ───────────────────────────────────────────────────────────

GROUP = {"links": ["a", "b"], "width": 100.0, "count": 3, "import_step": 2.0, "export_step": 1.0}


def test_tranche_cost_is_zero_in_the_first_tranche_and_convex_beyond():
    flow = pd.Series([50.0, 250.0, -150.0, -250.0, 0.0])
    cost = tranche_cost(flow, GROUP)
    # 250 import: 2*(250-100) + 2*(250-200); 150 export: 1*(150-100);
    # 250 export: 1*(250-100) + 1*(250-200).
    assert cost.tolist() == pytest.approx([0.0, 400.0, 50.0, 200.0, 0.0])
    grid = np.linspace(-300, 300, 61)
    values = tranche_cost(pd.Series(grid), GROUP).to_numpy()
    assert np.all(np.diff(values, 2) >= -1e-9)


def test_allocation_splits_the_group_cost_by_share_of_same_direction_flow():
    flows = pd.DataFrame({"a": [150.0, -120.0], "b": [100.0, -30.0]})
    allocated = allocate_tranche_cost(flows, GROUP)
    assert allocated.sum(axis=1).tolist() == pytest.approx(tranche_cost(flows.sum(axis=1), GROUP).tolist())
    assert allocated.loc[0, "a"] == pytest.approx(400.0 * 150.0 / 250.0)
    assert allocated.loc[1, "b"] == pytest.approx(50.0 * 30.0 / 150.0)


def test_inconsistent_parameters_inside_a_group_are_refused():
    links = pd.DataFrame({
        "foreign_market_pricing": ["piecewise_bidirectional"] * 2,
        "interconnector_name": ["a", "b"],
        "market_tranche_group": ["continent", "continent"],
        "market_tranche_width_mw": [100.0, 90.0],
        "market_tranche_count": [3, 3],
        "market_tranche_import_step": [2.0, 2.0],
        "market_tranche_export_step": [1.0, 1.0],
    }, index=["IC_a", "IC_b"])
    with pytest.raises(ValueError, match="continent"):
        get_volume_tranche_groups(links)


# ── in the LP ────────────────────────────────────────────────────────────────

def _network(hours=1, import_step=4.0, export_step=10.0):
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    network.add("Bus", "GB")
    network.add("Bus", "External_France")
    cables = pd.DataFrame([
        {"name": name, "from_bus": "GB", "to_bus": "External_France", "capacity_mw": 100.0,
         "losses_percent": 10.0, "counterparty_country": "France"}
        for name in ("One", "Two")
    ])
    prices = pd.DataFrame([{"country": "france", "year": 2030, "pathway": "Central",
                            "estimated_price_gbp_per_mwh": 20.0}])
    add_interconnector_links(network, cables, pd.DataFrame(), prices, modelled_year=2030,
                             fes_pathway="Central", pricing_config={"transaction_spread_gbp_per_mwh": 1.0})
    for link in ("IC_One", "IC_Two"):
        network.links.loc[link, "market_tranche_group"] = "continent"
        network.links.loc[link, "market_tranche_width_mw"] = 200.0 / 3
        network.links.loc[link, "market_tranche_count"] = 3
        network.links.loc[link, "market_tranche_import_step"] = import_step
        network.links.loc[link, "market_tranche_export_step"] = export_step
    return network


def _solve(network):
    status, condition = network.optimize(
        solver_name="highs", extra_functionality=build_piecewise_interconnector_cost_callback(network))
    assert status == "ok", condition
    return network


def test_the_import_price_rises_with_volume():
    small = _network()
    small.add("Load", "demand", bus="GB", p_set=40.0)
    small.add("Generator", "domestic", bus="GB", p_nom=500.0, marginal_cost=1000.0)
    large = _network()
    large.add("Load", "demand", bus="GB", p_set=150.0)
    large.add("Generator", "domestic", bus="GB", p_nom=500.0, marginal_cost=1000.0)
    price_small = float(_solve(small).buses_t.marginal_price["GB"].iloc[0])
    price_large = float(_solve(large).buses_t.marginal_price["GB"].iloc[0])
    # 40 MW delivered is 44.4 MW sent, inside the first tranche: the ask alone.
    assert price_small == pytest.approx(20.5 / 0.9, rel=1e-6)
    # 150 MW delivered is 166.7 MW sent, in the third tranche: two steps on top.
    assert price_large == pytest.approx((20.5 + 2 * 4.0) / 0.9, rel=1e-6)


def test_the_export_price_falls_with_volume_and_stops_the_trade():
    network = _network(export_step=10.0)
    network.add("Generator", "cheap", bus="GB", p_nom=500.0, marginal_cost=12.0)
    _solve(network)
    sent = -network.links_t.p0[["IC_One", "IC_Two"]].sum(axis=1).iloc[0]
    # A flat bid would export the full 200 MW; the second tranche's price is
    # below the unit's cost, so the trade stops at the first boundary.
    assert sent == pytest.approx(200.0 / 3, rel=1e-6)


def test_the_post_solve_valuation_reproduces_the_lp_trade_cost():
    network = _network()
    network.add("Load", "demand", bus="GB", p_set=150.0)
    network.add("Generator", "domestic", bus="GB", p_nom=500.0, marginal_cost=1000.0)
    _solve(network)
    flows = network.links_t.p0[["IC_One", "IC_Two"]]
    links, ask_t = link_price_tables(network)
    valued = interconnector_trade_cost(flows, links, ask_t)
    generator_cost = float((network.generators_t.p * network.generators.marginal_cost).sum().sum())
    assert valued.to_numpy().sum() == pytest.approx(network.objective - generator_cost, rel=1e-6)


def test_links_without_a_group_are_priced_exactly_as_before():
    network = _network()
    network.links.loc[["IC_One", "IC_Two"], "market_tranche_group"] = ""
    assert get_volume_tranche_groups(network.links) == {}
    network.add("Load", "demand", bus="GB", p_set=150.0)
    network.add("Generator", "domestic", bus="GB", p_nom=500.0, marginal_cost=1000.0)
    price = float(_solve(network).buses_t.marginal_price["GB"].iloc[0])
    assert price == pytest.approx(20.5 / 0.9, rel=1e-6)


def test_apply_volume_tranches_writes_persistable_link_attributes():
    network = _network()
    network.links[["market_tranche_group"]] = ""
    network.links["market_import_level"] = 89.43
    network.links["market_export_level"] = 40.09
    table = apply_volume_tranches(
        network, {"elasticity": 1.0, "tranches": 3, "markets": MARKETS}, DEMAND,
        logger=logging.getLogger(__name__))
    groups = get_volume_tranche_groups(network.links)
    assert set(groups) == {"continent"}
    assert sorted(groups["continent"]["links"]) == ["IC_One", "IC_Two"]
    assert groups["continent"]["width"] == pytest.approx(200.0 / 3)
    assert table.loc["continent", "import_step"] == pytest.approx(89.43 / (1537.0 / 8.76) * 0.2 / 3)
    for column in ("market_tranche_width_mw", "market_tranche_import_step", "market_tranche_export_step"):
        assert network.links[column].dtype.kind == "f"
