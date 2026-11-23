"""Unit tests for contract-value balancing bids (plan item B1).

The defect these guard against is not a crash. It is a bid stack that lets the
balancing LP turn down an expensive contract as readily as a cheap one, which is
why nearly every test here is an *ordering* assertion rather than a value one.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market.cfd_bm_bids import (
    build_contract_value_bid_frames,
    demand_weighted_reference,
    load_scenario_strike_table,
    resolve_reference_price,
)
from scripts.market.market_utils import calculate_bid_offer_prices
from config.config_loader import load_layer


STRIKE_TABLE_CSV = (
    "cfd_round,technology,delivery_year,strike_price_gbp_per_mwh_2012,capacity_mw,"
    "price_basis,source\n"
    "Investment Contract,offshore,2016-17,150.00,1000,2012 prices,test\n"
    "AR4,offshore,2026-27,37.35,5000,2012 prices,test\n"
    "AR6,solar,2026-27,50.07,3000,2012 prices,test\n"
    "HPC,nuclear,2026-27,92.50,3200,2012 prices,test\n"
    "AR8_modelled,offshore,2030-31,65.45,,2012 prices,test\n"
    "AR8_modelled,solar,2027-28,46.82,,2012 prices,test\n"
)

INDEXATION = {
    "from_price_year": 2012,
    "to_price_year": 2024,
    "from_index": 96.1,
    "to_index": 133.9,
    "factor": 133.9 / 96.1,
    "index": "ONS CPI annual average, all items, 2015=100 (CDID D7BT)",
}


@pytest.fixture
def strike_table_file(tmp_path):
    path = tmp_path / "cfd_strike_prices_by_round.csv"
    path.write_text(STRIKE_TABLE_CSV, encoding="utf-8")
    return path


@pytest.fixture
def scenario_config(strike_table_file):
    """The CP30 shape: fixed support pricing plus contract-value bids (``support.bm_bids``)."""
    return {
        "modelled_year": 2030,
        "support": {
            "dispatch_costs": {"enabled": True,
                               "roc_buyout_prices": {"2030": 78.04}},
            "labels": {
                "enabled": True,
                "cfd_strike_table": {
                    "file": str(strike_table_file),
                    "model_price_year": 2024,
                    "indexation": INDEXATION,
                },
            },
            "bm_bids": _bm_bids(),
        },
    }


def _bm_bids():
    return {
        "enabled": True,
        "mode": "fixed",
        "exclude_carriers": ["nuclear"],
        "cfd": {"offer_price": 0.0, "bid_price": 0.0},
        "roc": {"offer_price": 0.0, "bid_multiplier": 1.0},
        "contract_value_bids": {
            "enabled": True,
            # Pinned: these tests are about the contract-value reservation price
            # (the CP30 default is the strike basis, tested separately below).
            "bid_basis": "contract_value",
            "cfd_offer_price": None,
            "missing_strike_policy": "error",
            "merchant": {
                "bid_price": -0.5,
                "carriers": ["wind_offshore", "solar_pv"],
            },
            "reference_price_basis": {"baseload_carriers": []},
        },
    }


@pytest.fixture
def balancing_config():
    """The CP30 market.balancing shape (the support pricing is in ``support.bm_bids``)."""
    return {
        "bid_offer_source": "derived",
        "default_offer_markup": 0.10,
        "default_bid_discount": 0.10,
        "carrier_overrides": {
            "nuclear": {"mode": "absolute", "offer_price": 999.0, "bid_price": 150.0},
            "wind_offshore": {"mode": "absolute", "offer_price": 0.0, "bid_price": 90.0},
            "solar_pv": {"mode": "absolute", "offer_price": 0.0, "bid_price": 55.0},
        },
    }


@pytest.fixture
def network():
    """Four snapshots, one contract per allocation round, one of each label."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=4, freq="h"))
    n.add("Bus", "bus", v_nom=400)
    n.add("Load", "load", bus="bus", p_set=1000.0)
    units = [
        # name,            carrier,        support,    round,                 p_nom
        ("ic_offshore", "wind_offshore", "CfD", "Investment Contract", 500.0),
        ("ar4_offshore", "wind_offshore", "CfD", "AR4", 800.0),
        ("ar6_solar", "solar_pv", "CfD", "AR6", 300.0),
        ("future_solar", "solar_pv", "CfD", "AR8_modelled", 200.0),
        ("hpc", "nuclear", "CfD", "HPC", 3200.0),
        ("merchant_offshore", "wind_offshore", "merchant", "", 400.0),
        ("merchant_solar", "solar_pv", "merchant", "", 250.0),
        ("roc_offshore", "wind_offshore", "ROC", "", 150.0),
    ]
    for name, carrier, support, cfd_round, p_nom in units:
        n.add("Generator", name, bus="bus", carrier=carrier, p_nom=p_nom,
              marginal_cost=0.0, support_type=support, cfd_round=cfd_round,
              ro_banding=1.0 if support == "ROC" else np.nan)
    return n


@pytest.fixture
def reference_price(network):
    # One hour above every strike, one at zero, two in between.
    return pd.Series([0.0, 40.0, 120.0, 400.0], index=network.snapshots)


def _frames(network, scenario_config, balancing_config, reference_price):
    logger = logging.getLogger("test_cfd_bm_bids")
    gen_offer = pd.Series(0.0, index=network.generators.index)
    gen_bid = pd.Series(
        {"ic_offshore": 0.0, "ar4_offshore": 0.0, "ar6_solar": 0.0,
         "future_solar": 0.0, "hpc": 150.0, "merchant_offshore": 90.0,
         "merchant_solar": 55.0, "roc_offshore": 78.04},
    ).reindex(network.generators.index)
    return build_contract_value_bid_frames(
        network, scenario_config, balancing_config, gen_offer, gen_bid,
        reference_price, logger,
    )


@pytest.mark.unit
class TestContractValueBids:

    def test_bid_orders_by_strike(self, network, scenario_config, balancing_config,
                                  reference_price):
        """The whole point: an expensive contract bids above a cheap one, always."""
        bids = _frames(network, scenario_config, balancing_config,
                       reference_price).bid_tv
        for snapshot in network.snapshots:
            assert bids.at[snapshot, "ic_offshore"] >= bids.at[snapshot, "ar4_offshore"]
            assert bids.at[snapshot, "ar6_solar"] >= bids.at[snapshot, "future_solar"]
        # And strictly so while the reference is below both strikes.
        cheap_hour = network.snapshots[0]
        assert bids.at[cheap_hour, "ic_offshore"] > bids.at[cheap_hour, "ar4_offshore"]

    def test_bid_is_strike_minus_reference_floored_at_zero(
            self, network, scenario_config, balancing_config, reference_price):
        table = load_scenario_strike_table(scenario_config)
        strike, _, _ = table.lookup("wind_offshore", "Investment Contract")
        bids = _frames(network, scenario_config, balancing_config,
                       reference_price).bid_tv["ic_offshore"]
        expected = (strike - reference_price).clip(lower=0.0)
        pd.testing.assert_series_equal(bids, expected, check_names=False)
        # The 400 GBP/MWh hour clears above every strike, so nothing is forgone.
        assert bids.iloc[3] == 0.0

    def test_merchant_renewables_drop_to_short_run_cost(
            self, network, scenario_config, balancing_config, reference_price):
        """Merchant units hold no contract, so the blended subsidy must go."""
        bids = _frames(network, scenario_config, balancing_config,
                       reference_price).bid_tv
        # Minus the Stage 1 merchant variable cost (decision D1): NESO receives it.
        assert (bids["merchant_offshore"] == -0.5).all()
        assert (bids["merchant_solar"] == -0.5).all()
        # ...and a merchant unit is now cheaper to turn down than any contract.
        cheap_hour = network.snapshots[0]
        assert bids.at[cheap_hour, "merchant_offshore"] < bids.at[cheap_hour, "ar4_offshore"]

    def test_roc_and_excluded_carriers_keep_their_static_bid(
            self, network, scenario_config, balancing_config, reference_price):
        """ROC banding x buyout is already right; nuclear keeps its restart cost."""
        bids = _frames(network, scenario_config, balancing_config,
                       reference_price).bid_tv
        assert bids["roc_offshore"].to_list() == pytest.approx([78.04] * 4)
        assert bids["hpc"].to_list() == pytest.approx([150.0] * 4)

    def test_offer_prices_are_untouched_by_default(
            self, network, scenario_config, balancing_config, reference_price):
        offers = _frames(network, scenario_config, balancing_config,
                         reference_price).offer_tv
        assert (offers == 0.0).all().all()

    def test_cfd_offer_price_override_applies_to_cfd_only(
            self, network, scenario_config, balancing_config, reference_price):
        scenario_config["support"]["bm_bids"]["contract_value_bids"][
            "cfd_offer_price"] = -5.0
        offers = _frames(network, scenario_config, balancing_config,
                         reference_price).offer_tv
        assert (offers["ar4_offshore"] == -5.0).all()
        assert (offers["merchant_offshore"] == 0.0).all()
        assert (offers["hpc"] == 0.0).all()

    def test_audit_covers_every_repriced_generator(
            self, network, scenario_config, balancing_config, reference_price):
        audit = _frames(network, scenario_config, balancing_config,
                        reference_price).audit
        assert set(audit.generator) == {
            "ic_offshore", "ar4_offshore", "ar6_solar", "future_solar",
            "merchant_offshore", "merchant_solar",
        }
        cfd = audit.set_index("generator").loc["ic_offshore"]
        assert cfd.pricing == "cfd_contract_value"
        assert cfd.strike_key == "Investment Contract_offshore"
        assert cfd.previous_static_bid_gbp_per_mwh == 0.0
        assert cfd.mean_bid_gbp_per_mwh > 0.0
        assert audit.set_index("generator").at[
            "merchant_offshore", "previous_static_bid_gbp_per_mwh"] == 90.0

    def test_a_missing_strike_is_refused_not_silently_zeroed(
            self, network, scenario_config, balancing_config, reference_price):
        network.generators.loc["ar4_offshore", "carrier"] = "tidal_stream"
        with pytest.raises(ValueError, match="No strike-table entry"):
            _frames(network, scenario_config, balancing_config, reference_price)

    def test_retain_static_policy_warns_and_keeps_the_scalar(
            self, network, scenario_config, balancing_config, reference_price, caplog):
        network.generators.loc["ar4_offshore", "carrier"] = "tidal_stream"
        scenario_config["support"]["bm_bids"]["contract_value_bids"][
            "missing_strike_policy"] = "retain_static"
        with caplog.at_level(logging.WARNING):
            frames = _frames(network, scenario_config, balancing_config, reference_price)
        assert (frames.bid_tv["ar4_offshore"] == 0.0).all()
        assert "No strike-table entry" in caplog.text

    def test_a_missing_reference_price_is_refused(
            self, network, scenario_config, balancing_config):
        with pytest.raises(ValueError, match="wholesale clearing price"):
            _frames(network, scenario_config, balancing_config, None)

    def test_a_short_reference_price_is_refused(
            self, network, scenario_config, balancing_config, reference_price):
        with pytest.raises(ValueError, match="missing for 2 of 4"):
            _frames(network, scenario_config, balancing_config,
                    reference_price.iloc[:2])

    def test_baseload_carriers_use_the_window_mean(
            self, network, scenario_config, balancing_config, reference_price):
        """A listed baseload carrier settles flat, not hourly."""
        support = scenario_config["support"]["bm_bids"]
        support["exclude_carriers"] = []
        support["contract_value_bids"]["reference_price_basis"]["baseload_carriers"] = [
            "nuclear"]
        frames = _frames(network, scenario_config, balancing_config, reference_price)
        assert frames.bid_tv["hpc"].nunique() == 1
        assert frames.audit.set_index("generator").at[
            "hpc", "reference_price_basis"] == "window_demand_weighted_mean"
        # The intermittent contracts still settle hourly.
        assert frames.bid_tv["ar4_offshore"].nunique() > 1


@pytest.mark.unit
class TestWiringThroughCalculateBidOfferPrices:

    def test_frames_are_attached_when_time_varying(
            self, network, scenario_config, balancing_config, reference_price):
        network._bm_reference_price = reference_price
        market_config = {"balancing": balancing_config}
        calculate_bid_offer_prices(
            network, market_config, logging.getLogger("test"), time_varying=True,
            scenario_config=scenario_config,
        )
        assert network._bm_bid_tv.shape == (4, len(network.generators))
        assert network._bm_bid_tv.at[
            network.snapshots[0], "ic_offshore"] > network._bm_bid_tv.at[
            network.snapshots[0], "ar4_offshore"]
        assert not network._bm_contract_value_audit.empty

    def test_scalar_path_is_unchanged_when_not_time_varying(
            self, network, scenario_config, balancing_config, reference_price):
        """The wholesale stage and the diagnostics keep the old scalar behaviour."""
        network._bm_reference_price = reference_price
        market_config = {"balancing": balancing_config}
        calculate_bid_offer_prices(
            network, market_config, logging.getLogger("test"), time_varying=False,
            scenario_config=scenario_config,
        )
        assert not hasattr(network, "_bm_bid_tv")

    def test_disabled_leaves_the_scalar_cfd_bid_in_place(
            self, network, scenario_config, balancing_config, reference_price):
        scenario_config["support"]["bm_bids"]["contract_value_bids"][
            "enabled"] = False
        network._bm_reference_price = reference_price
        _, gen_bid, _, _ = calculate_bid_offer_prices(
            network, {"balancing": balancing_config}, logging.getLogger("test"),
            time_varying=True, scenario_config=scenario_config,
        )
        assert gen_bid["ic_offshore"] == 0.0
        assert not hasattr(network, "_bm_bid_tv")


@pytest.mark.unit
class TestHelpers:

    def test_reference_price_accepts_a_frame(self, network, reference_price):
        frame = pd.DataFrame({"wholesale_price": reference_price,
                              "price_spread": 0.0})
        pd.testing.assert_series_equal(
            resolve_reference_price(network, frame), reference_price,
            check_names=False,
        )

    def test_demand_weighted_reference_is_flat_and_within_range(
            self, network, reference_price):
        series = demand_weighted_reference(reference_price, network)
        assert series.nunique() == 1
        assert reference_price.min() <= series.iloc[0] <= reference_price.max()

    def test_strike_table_requires_a_configured_file(self):
        with pytest.raises(ValueError, match="support.labels.cfd_strike_table.file"):
            load_scenario_strike_table({})


@pytest.mark.unit
class TestStrikeBidBasis:
    """``contract_value_bids.bid_basis: strike`` (user decision, 27 Sep 2026)."""

    def _strike_config(self, scenario_config, balancing_config):
        scenario_config["support"]["bm_bids"]["contract_value_bids"]["bid_basis"] = "strike"
        return balancing_config

    def test_every_paying_hour_bids_the_whole_strike(
            self, network, scenario_config, balancing_config, reference_price):
        table = load_scenario_strike_table(scenario_config)
        strike, _, _ = table.lookup("wind_offshore", "Investment Contract")
        bids = _frames(network, scenario_config, self._strike_config(scenario_config, balancing_config),
                       reference_price).bid_tv["ic_offshore"]
        # Even the 400 GBP/MWh hour, where contract value is zero.
        assert np.allclose(bids.to_numpy(), strike)

    def test_the_negative_price_rule_still_zeroes_unpaid_hours(
            self, network, scenario_config, balancing_config):
        cfg = self._strike_config(scenario_config, balancing_config)
        scenario_config["support"]["bm_bids"]["contract_value_bids"]["negative_price_rule"] = "gb_contract_terms"
        negative = pd.Series([-5.0, 40.0, 120.0, 400.0], index=network.snapshots)
        bids = _frames(network, scenario_config, cfg, negative).bid_tv["ar4_offshore"]
        assert bids.iloc[0] == 0.0          # AR4+: no difference payment below zero
        assert (bids.iloc[1:] > 0.0).all()

    def test_ordering_by_strike_is_kept_and_merchant_untouched(
            self, network, scenario_config, balancing_config, reference_price):
        bids = _frames(network, scenario_config, self._strike_config(scenario_config, balancing_config),
                       reference_price).bid_tv
        assert (bids["ic_offshore"] > bids["ar4_offshore"]).all()
        assert (bids["merchant_offshore"] == -0.5).all()   # the fixture's merchant price

    def test_the_audit_records_the_basis(
            self, network, scenario_config, balancing_config, reference_price, tmp_path):
        from scripts.consumer_bill.extract_future_balancing_costs import bid_basis_from_audit

        audit = _frames(network, scenario_config, self._strike_config(scenario_config, balancing_config),
                        reference_price).audit
        cfd = audit.loc[audit["pricing"].eq("cfd_contract_value")]
        assert set(cfd["bid_basis"]) == {"strike"}
        path = tmp_path / "contract_value_bids.csv"
        audit.to_csv(path, index=False)
        assert bid_basis_from_audit(path) == "strike"
        # An audit written before the column existed was contract value.
        audit.drop(columns=["bid_basis"]).to_csv(path, index=False)
        assert bid_basis_from_audit(path) == "contract_value"
        assert bid_basis_from_audit(None) == "contract_value"

    def test_an_unknown_basis_is_refused(self, network, scenario_config, balancing_config, reference_price):
        scenario_config["support"]["bm_bids"]["contract_value_bids"]["bid_basis"] = "empirical"
        with pytest.raises(ValueError, match="bid_basis"):
            _frames(network, scenario_config, balancing_config, reference_price)

    def test_code_defaults_match_defaults_yaml(self):
        from pathlib import Path

        import yaml

        from scripts.market.cfd_bm_bids import DEFAULT_BID_BASIS, DEFAULT_MERCHANT_BID

        root = Path(__file__).resolve().parents[2]
        cvb = load_layer("source", "CP30")["support"]["bm_bids"]["contract_value_bids"]
        assert cvb["bid_basis"] == DEFAULT_BID_BASIS == "strike"
        assert float(cvb["merchant"]["bid_price"]) == DEFAULT_MERCHANT_BID == 20.0
