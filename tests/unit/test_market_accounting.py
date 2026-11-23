import pandas as pd
import pypsa

from scripts.market.market_accounting import create_market_accounting_report


def _network_with_two_hours_of_demand():
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "GB")
    network.add(
        "Load",
        "demand",
        bus="GB",
        p_set=pd.Series([100.0, 100.0], index=snapshots),
    )
    return network


def test_accounting_separates_feasibility_slack_and_normalises_to_demand():
    redispatch = pd.DataFrame(
        {
            "carrier": ["CCGT", "load_shedding"],
            "increase_MWh": [10.0, 2.0],
            "decrease_MWh": [5.0, 0.0],
            "offer_cost": [100.0, 12_000.0],
            "bid_cost": [10.0, 0.0],
        }
    )

    accounting, report = create_market_accounting_report(
        _network_with_two_hours_of_demand(), redispatch
    )

    assert accounting["demand_MWh"] == 200.0
    assert accounting["modelled_redispatch_cost_GBP"] == 12_110.0
    assert accounting["ordinary_redispatch_cost_excluding_feasibility_slack_GBP"] == 110.0
    assert accounting["feasibility_slack_energy_MWh"] == 2.0
    assert accounting["feasibility_slack_cost_GBP"] == 12_000.0
    assert accounting["gross_redispatch_MWh"] == 17.0
    assert accounting["gross_redispatch_pct_demand"] == 8.5
    assert accounting["is_short_run"] is True
    assert accounting["annualisation_allowed"] is False
    assert len(accounting["warnings"]) == 2
    assert set(report["metric"]) >= {
        "demand_MWh",
        "modelled_redispatch_cost_GBP",
        "feasibility_slack_cost_GBP",
    }


def test_forced_turn_down_by_non_participants_is_reported_apart_from_ordinary_cost():
    # Central dispatch, 26 Sep 2026: forecast error forced sub-1 MW plant (left out
    # of the BM at a 6,000 GBP/MWh penalty bid) down; ~82% of the reported cost.
    redispatch = pd.DataFrame(
        {
            "carrier": ["CCGT", "solar_pv", "wind_onshore"],
            "increase_MWh": [10.0, 0.0, 0.0],
            "decrease_MWh": [0.0, 2.0, 5.0],
            "offer_cost": [100.0, 0.0, 0.0],
            "bid_cost": [0.0, 12_000.0, 250.0],   # 6,000 GBP/MWh penalty vs a 50 GBP/MWh bid
        }
    )
    accounting, report = create_market_accounting_report(_network_with_two_hours_of_demand(), redispatch)
    assert accounting["modelled_redispatch_cost_GBP"] == 12_350.0      # the LP's own accounting
    assert accounting["non_participant_turn_down_MWh"] == 2.0
    assert accounting["non_participant_turn_down_cost_at_penalty_GBP"] == 12_000.0
    assert accounting["ordinary_redispatch_cost_excluding_feasibility_slack_GBP"] == 350.0
    assert any("non-participants" in w for w in accounting["warnings"])
    assert "non_participant_turn_down_MWh" in set(report["metric"])


def test_accounting_handles_empty_redispatch_without_dividing_by_zero():
    accounting, report = create_market_accounting_report(
        _network_with_two_hours_of_demand(), pd.DataFrame()
    )

    assert accounting["modelled_redispatch_cost_GBP"] == 0.0
    assert accounting["gross_redispatch_MWh"] == 0.0
    assert accounting["modelled_redispatch_cost_GBP_per_MWh_demand"] == 0.0
    assert report.loc[
        report["metric"] == "modelled_redispatch_cost_GBP", "value"
    ].item() == 0.0



def _network_with_hydrogen_and_a_foreign_market():
    """GB demand 100 MW, a 30 MW_H2 offtake on a hydrogen bus fed by a 50 MW
    electrolyser, and a foreign-market balance generator behind a cable."""
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    network = _network_with_two_hours_of_demand()
    network.add("Bus", "GB_H2", carrier="H2_gas")
    network.add("Bus", "HVDC_External_France", carrier="AC")
    network.add("Load", "h2_offtake", bus="GB_H2", p_set=30.0)
    network.add("Link", "electrolysis_GB", bus0="GB", bus1="GB_H2", p_nom=100.0,
                efficiency=0.7, carrier="electrolysis")
    network.add("Generator", "EU_market_balance_France", bus="HVDC_External_France",
                p_nom=1000.0, carrier="EU_market")
    network.links_t.p0 = pd.DataFrame({"electrolysis_GB": [50.0, 50.0]}, index=snapshots)
    return network


def test_gb_gross_redispatch_leaves_out_the_foreign_side_and_uses_electricity_demand():
    # Audit 26 Sep 2026: foreign-market balance rows were 4.6 of 14.7 TWh gross in
    # CP30 December, and demand_MWh counted the hydrogen offtake (MWh of H2) while
    # the electricity electrolysers drew was left out.
    redispatch = pd.DataFrame(
        {
            "component": ["ccgt", "EU_market_balance_France", "electrolysis_GB"],
            "type": ["generator", "generator", "link"],
            "carrier": ["CCGT", "EU_market", "electrolysis"],
            "increase_MWh": [10.0, 40.0, 3.0],
            "decrease_MWh": [5.0, 0.0, 2.0],
            "offer_cost": [100.0, 0.0, 30.0],
            "bid_cost": [-45.0, 0.0, -10.0],
        }
    )
    accounting, report = create_market_accounting_report(
        _network_with_hydrogen_and_a_foreign_market(), redispatch
    )
    # Legacy metrics unchanged: every row, every Load (the bill checks demand_MWh).
    assert accounting["demand_MWh"] == 260.0
    assert accounting["gross_redispatch_MWh"] == 60.0
    assert accounting["gross_redispatch_including_foreign_side_MWh"] == 60.0
    # GB-only: electrical load 200 MWh plus 100 MWh drawn by electrolysis.
    assert accounting["gb_electricity_demand_MWh"] == 300.0
    assert accounting["gb_gross_redispatch_MWh"] == 20.0
    assert accounting["gb_gross_redispatch_pct_gb_electricity_demand"] == 20.0 / 300.0 * 100.0
    assert {"gb_gross_redispatch_MWh", "gb_electricity_demand_MWh"} <= set(report["metric"])
    # Negative bid costs (NESO receives the saving) reduce the modelled cost.
    assert accounting["modelled_redispatch_cost_GBP"] == 75.0
