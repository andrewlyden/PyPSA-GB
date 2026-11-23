"""The F19 valuation decisions (user, 14 Oct 2026) and legacy offshore at deal multiples (user, 24 Oct 2026).

* The purchase price counts the battery top-up sized on the private case's account (owner cash),
  as the valuation file and the streams already do.
* Lever ``legacy_price_path``: the tariff reads each case's own path's streams.
* The legacy corporation tax is profiled on the falling streams in the multi-year view.
* A named P3 case prices legacy offshore wind at observed deal multiples; everything else
  stays at its owners' value.
* The private WACC bases are labelled against CEPA 2025's pre- and post-tax series.

(Drax on its 2027-31 guidance and closing at its CfD's end: test_public_ownership_legacy_valuation_f19.py.)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.public_ownership import offshore_deal_multiple as odm
from scripts.public_ownership import legacy_valuation as lv
from scripts.public_ownership import nationalised_tariff as nt
from scripts.public_ownership.calculate_financing_scenarios import (
    LEVER_PARAMETER_DEFAULTS, calculate_financing_scenarios,
)
from scripts.public_ownership.finance import NATIONALISED_CASE
from scripts.public_ownership.levers import (
    LEVER_LEGACY_OFFSHORE_PRICE, LEVERS, OFFSHORE_DEAL_MULTIPLE, OFFSHORE_OWNERS_VALUE, lever_default,
)
from scripts.public_ownership.reference_runs import ACQUISITION_TERM_LEVERS, expanded_cases, same_fleet_tariff_cases

from tests.unit.test_public_ownership_cfd_repricing import _asset, _costs
from tests.unit.test_public_ownership_legacy_valuation import _stations
from tests.unit.test_public_ownership_nationalised_case import (
    NO_OVERHEAD, _acquisition, _register_inputs, _snapshot_with_public_rate, _terms,
)
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
DEALS = ROOT / "data/public_ownership/financing/legacy_offshore_transaction_multiples.csv"


@pytest.fixture(scope="module")
def po():
    return load_defaults()["public_ownership"]


# ── legacy offshore wind at deal multiples ───────────────────────────────────

class TestDealMultiples:
    def test_the_central_multiple_is_the_median_of_the_central_deals_in_2024_money(self):
        deals = pd.read_csv(DEALS)
        # Each deal's 2024 figure is its nominal per-MW value rebased by ONS CPIH annual averages.
        rebased = deals["gbp_m_per_mw_nominal"] * deals["cpih_2024"] / deals["cpih_deal_year"]
        assert np.allclose(rebased, deals["gbp_m_per_mw_2024"], atol=2e-3)
        assert np.allclose(deals["value_gbp_m"] / deals["stake_mw"], deals["gbp_m_per_mw_nominal"], atol=2e-3)
        m = odm.multiple(None)
        central = deals.loc[deals["in_central"], "gbp_m_per_mw_2024"]
        assert m["gbp_m_per_mw"] == pytest.approx(float(central.median())) == pytest.approx(4.609)
        assert m["low_gbp_m_per_mw"] == pytest.approx(3.970) and m["high_gbp_m_per_mw"] == pytest.approx(7.968)
        assert len(m["deals_used"]) == 5 and len(m["deals_reported_only"]) == 1
        assert deals["source"].str.startswith("https://").all()

    def test_parameters_are_checked_and_the_config_equals_the_code_defaults(self, po):
        assert po["lever_parameters"][odm.PARAMETER_KEY] == odm.DEFAULT_PARAMETERS
        assert LEVER_PARAMETER_DEFAULTS[odm.PARAMETER_KEY] == odm.DEFAULT_PARAMETERS
        with pytest.raises(ValueError, match="statistic"):
            odm.parameters({"statistic": "mode"})
        with pytest.raises(ValueError, match="unknown"):
            odm.parameters({"cohort": "x"})
        assert odm.multiple({"statistic": "high"})["gbp_m_per_mw"] == pytest.approx(7.968)

    def test_the_lever_is_an_acquisition_term_with_owners_value_central(self):
        assert lever_default(LEVER_LEGACY_OFFSHORE_PRICE) == OFFSHORE_OWNERS_VALUE
        assert set(LEVERS[LEVER_LEGACY_OFFSHORE_PRICE].values) == {OFFSHORE_OWNERS_VALUE, OFFSHORE_DEAL_MULTIPLE}
        assert (OFFSHORE_OWNERS_VALUE, OFFSHORE_DEAL_MULTIPLE) == (odm.OWNERS_VALUE, odm.TRANSACTION_MULTIPLE)
        assert LEVER_LEGACY_OFFSHORE_PRICE in ACQUISITION_TERM_LEVERS

    def test_the_named_case_joins_p3s_same_fleet_family(self, po):
        case = "public_ownership@fully_nationalised_market_value_offshore_deal_multiples"
        assert case in expanded_cases(po)
        family = same_fleet_tariff_cases(po, "public_ownership@fully_nationalised_market_value")
        assert case in family

    def test_only_legacy_offshore_moves_to_the_multiple(self):
        def run(**levers):
            return calculate_financing_scenarios(
                _register_inputs(), _costs(), _snapshot_with_public_rate(), cases=("private", NATIONALISED_CASE),
                axes=_terms(purchase_price_basis="market_value", **levers), acquisition=_acquisition(),
                ro_stations=_stations(), state_operating_overhead=NO_OVERHEAD)

        base, _ = run()
        deal, totals = run(legacy_offshore_price="transaction_multiple")
        b = base.loc[base["case_id"].eq(NATIONALISED_CASE)].set_index("asset_id")
        d = deal.loc[deal["case_id"].eq(NATIONALISED_CASE)].set_index("asset_id")
        central = odm.multiple(None)["gbp_m_per_mw"]
        for asset_id, mw in (("old_wind", 600.0), ("old_cfd", 400.0)):
            assert float(d.loc[asset_id, "acquisition_price_gbp"]) == pytest.approx(central * 1e6 * mw)
            assert d.loc[asset_id, "acquisition_basis"] == odm.BASIS
        assert float(d.loc["gas::legacy", "acquisition_price_gbp"]) == pytest.approx(
            float(b.loc["gas::legacy", "acquisition_price_gbp"]))
        summary = totals.attrs["acquisition"][NATIONALISED_CASE]
        assert summary["legacy_offshore_price"] == odm.TRANSACTION_MULTIPLE
        assert summary["offshore_transaction_multiple"]["capacity_mw"] == pytest.approx(1000.0)
        assert summary["offshore_transaction_multiple"]["transaction_multiple_price_gbp"] == pytest.approx(
            central * 1e6 * 1000.0)

    def test_the_multiple_refuses_cancelled_contracts(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            calculate_financing_scenarios(
                _register_inputs(), _costs(), _snapshot_with_public_rate(), cases=("private", NATIONALISED_CASE),
                axes=_terms(purchase_price_basis="market_value", legacy_offshore_price="transaction_multiple",
                            legacy_contracts_under_acquisition="cancelled_residual_on_bills"),
                acquisition=_acquisition(), ro_stations=_stations(), state_operating_overhead=NO_OVERHEAD)


# ── the battery top-up in the purchase price ────────────────────────────────

def _with_legacy_battery():
    assets = _register_inputs()
    battery = _asset("old_battery", "Battery", "Lithium batteries", "legacy", "", "", "not_applicable", np.nan, 100.0,
                     5.0e4, component_type="StorageUnit", energy_capacity_mwh=400.0, commissioning_year=2024)
    battery.update({"market_revenue_gbp": 2.0e6, "support_revenue_gbp": 0.0, "cfd_support_gbp": 0.0,
                    "cfd_curtailment_compensation_gbp": 0.0, "roc_support_gbp": 0.0, "fuel_carbon_gbp": 0.0})
    return pd.concat([assets, pd.DataFrame([battery])], ignore_index=True)


def _run_battery(support_lines):
    return calculate_financing_scenarios(
        _with_legacy_battery(), _costs(), _snapshot_with_public_rate(), cases=("private", NATIONALISED_CASE),
        axes=_terms(purchase_price_basis="market_value"), acquisition=_acquisition(), ro_stations=_stations(),
        state_operating_overhead=NO_OVERHEAD, support_lines=support_lines)


def test_the_purchase_counts_the_battery_top_up_as_owner_cash():
    without, _ = _run_battery(None)
    with_top_up, totals = _run_battery({"battery_top_up": {"carriers": ["Battery"], "basis": "owner_cost"}})
    private = with_top_up.loc[with_top_up["case_id"].eq("private")].set_index("asset_id")
    top_up = float(private.loc["old_battery", "battery_revenue_top_up_gbp"])
    assert top_up > 0.0  # the battery earns below its cost, so the private case tops it up
    price = lambda reg: float(reg.loc[reg["case_id"].eq(NATIONALISED_CASE)].set_index("asset_id")  # noqa: E731
                              .loc["old_battery", "acquisition_price_gbp"])
    # The private top-up makes the battery's own cash zero rent at its WACC, so the owners now value it.
    assert price(with_top_up) > price(without) >= 0.0
    summary = totals.attrs["acquisition"][NATIONALISED_CASE]
    assert summary["purchase_top_ups_gbp_model_year"]["battery_revenue_top_up_gbp"] == pytest.approx(top_up)
    # Nothing else is priced differently.
    a = without.loc[without["case_id"].eq(NATIONALISED_CASE)].set_index("asset_id")
    b = with_top_up.loc[with_top_up["case_id"].eq(NATIONALISED_CASE)].set_index("asset_id")
    for asset_id in ("old_wind", "old_cfd", "gas::legacy"):
        assert float(b.loc[asset_id, "acquisition_price_gbp"]) == pytest.approx(float(a.loc[asset_id, "acquisition_price_gbp"]))


# ── each case's price path, and the corporation tax profile ─────────────────

def _profile(level):
    rows = []
    for year in range(2030, 2061):
        rows.append({"scope_group": "valued", "carrier": "wind_offshore", "support_type": "CfD",
                     "stream": "net_operating_cash", "year": year, "gbp": level * (1.0 if year < 2045 else 0.5)})
        rows.append({"scope_group": "thermal", "carrier": "CCGT", "support_type": "", "stream": "net_operating_cash",
                     "year": year, "gbp": 50.0 if year < 2035 else -10.0})
    return pd.DataFrame(rows)


def test_the_profile_by_price_path_gives_each_case_its_own_streams():
    cp30, model = _profile(200.0), _profile(100.0)
    both = lv.profile_by_price_path({lv.PRICE_PATH_CP30: cp30, lv.PRICE_PATH_MODEL: model}, central=cp30)
    assert set(both["price_path"]) == set(lv.PRICE_PATHS)
    pd.testing.assert_frame_equal(lv.case_profile(both, lv.PRICE_PATH_MODEL, cp30), model)
    pd.testing.assert_frame_equal(lv.case_profile(both, lv.PRICE_PATH_CP30, model), cp30)
    assert lv.case_profile(None, lv.PRICE_PATH_MODEL, cp30) is cp30
    # Without a CP30-level settlement both values read the central profile (the plan's path then too).
    only = lv.profile_by_price_path({lv.PRICE_PATH_CP30: model}, central=model)
    pd.testing.assert_frame_equal(lv.case_profile(only, lv.PRICE_PATH_MODEL, cp30), model)
    with pytest.raises(ValueError):
        lv.case_profile(both, "flat", cp30)
    # The lifetime break-even now moves with the path (it was identical before).
    a = nt.lifetime_break_even_price(lv.case_profile(both, lv.PRICE_PATH_CP30, cp30), public_rate=0.0275,
                                     term_years=30, model_year=2030)
    b = nt.lifetime_break_even_price(lv.case_profile(both, lv.PRICE_PATH_MODEL, cp30), public_rate=0.0275,
                                     term_years=30, model_year=2030)
    assert b["break_even_purchase_price_lifetime_gbp"] < a["break_even_purchase_price_lifetime_gbp"]


# ── each case's asset lives and capacity procurement (27 Oct 2026) ──────────

CENTRAL_KEY = (lv.PRICE_PATH_MODEL, lv.LIVES_CENTRAL, "auction")


def test_each_case_reads_the_streams_of_all_its_valuation_settings():
    central, cp30, longer, cost_based = _profile(100.0), _profile(200.0), _profile(300.0), _profile(50.0)
    frame = lv.profile_by_settings({
        CENTRAL_KEY: central, (lv.PRICE_PATH_CP30, lv.LIVES_CENTRAL, "auction"): cp30,
        (lv.PRICE_PATH_MODEL, lv.LIVES_EXTENDED, "auction"): longer,
        (lv.PRICE_PATH_MODEL, lv.LIVES_CENTRAL, "cost_based_existing"): cost_based})
    assert list(frame.columns[:3]) == list(lv.PROFILE_SETTING_COLUMNS)

    def levers(path=lv.PRICE_PATH_MODEL, lives=lv.LIVES_CENTRAL, capacity="auction"):
        return {"legacy_price_path": path, "legacy_asset_lives": lives, "capacity_procurement": capacity}

    pd.testing.assert_frame_equal(lv.case_profile(frame, levers(), cp30), central)
    pd.testing.assert_frame_equal(lv.case_profile(frame, levers(lives=lv.LIVES_EXTENDED), central), longer)
    pd.testing.assert_frame_equal(lv.case_profile(frame, levers(capacity="cost_based_existing"), central), cost_based)
    pd.testing.assert_frame_equal(lv.case_profile(frame, levers(path=lv.PRICE_PATH_CP30), central), cp30)
    # A lever the case does not name is at its central value; a price path alone still works.
    pd.testing.assert_frame_equal(lv.case_profile(frame, {"legacy_price_path": lv.PRICE_PATH_MODEL}, cp30), central)
    pd.testing.assert_frame_equal(lv.case_profile(frame, lv.PRICE_PATH_CP30, central), cp30)
    # A point the valuation did not write is refused, never given the central streams.
    with pytest.raises(ValueError, match="no rows"):
        lv.case_profile(frame, levers(path=lv.PRICE_PATH_CP30, lives=lv.LIVES_EXTENDED), central)
    # The frame of 14 Oct 2026 (price path only) holds central lives and the auction only.
    old = lv.profile_by_price_path({lv.PRICE_PATH_MODEL: central, lv.PRICE_PATH_CP30: cp30}, central=central)
    pd.testing.assert_frame_equal(lv.case_profile(old, levers(), cp30), central)
    with pytest.raises(ValueError, match="asset_lives"):
        lv.case_profile(old, levers(lives=lv.LIVES_EXTENDED), central)
    # The tariff's summary names the point.
    assert nt.legacy_streams_settings(levers(lives=lv.LIVES_EXTENDED), frame) == {
        "price_path": lv.PRICE_PATH_MODEL, "asset_lives": lv.LIVES_EXTENDED, "capacity_procurement": "auction"}
    assert nt.legacy_streams_settings(levers(), None) == "the valuation's central streams"


def test_the_valuation_writes_every_point_a_case_sits_at_central_first():
    # 10-11 Nov 2026 (user): the central price path is the anchored one (lever legacy_price_path default
    # "anchored", acquisition.valuation anchor_streams: all, horizon_year: life), and the valuation runs on four
    # paths (lv.PRICE_PATHS: cp30_level, model_2030 = prices held at 2030, anchored, today_level), so every path
    # is written at the central lives and auction before the points the cases sit at.
    from scripts.public_ownership import levers as lev

    assert lev.lever_default(lev.LEVER_LEGACY_PRICE_PATH) == lv.PRICE_PATH_ANCHORED
    assert lv.PRICE_PATHS == (lv.PRICE_PATH_CP30, lv.PRICE_PATH_MODEL, lv.PRICE_PATH_ANCHORED, lv.PRICE_PATH_TODAY)
    central = (lv.PRICE_PATH_ANCHORED, "central", "auction")
    paths = [(lv.PRICE_PATH_CP30, "central", "auction"), (lv.PRICE_PATH_MODEL, "central", "auction"),
             (lv.PRICE_PATH_TODAY, "central", "auction")]
    settings = pd.DataFrame({
        "case_id": ["private", "p3", "p3_longer", "p3_cm", "p2_cm"],
        "legacy_price_path": [lv.PRICE_PATH_ANCHORED] * 5,
        "legacy_asset_lives": ["central", "central", "extended", "central", "central"],
        "capacity_procurement": ["auction", "auction", "auction", "cost_based_existing", "cost_based_existing"],
    })
    keys = lv.valuation_setting_keys(settings, central_path=lv.PRICE_PATH_ANCHORED)
    assert keys == [central, *paths, (lv.PRICE_PATH_ANCHORED, "extended", "auction"),
                    (lv.PRICE_PATH_ANCHORED, "central", "cost_based_existing")]
    # Without a settings table (or a lever column): the price paths at the central lives and auction, the
    # named central path first, the others in lv.PRICE_PATHS order.
    assert lv.valuation_setting_keys(None, central_path=lv.PRICE_PATH_CP30) == [
        (lv.PRICE_PATH_CP30, "central", "auction"), CENTRAL_KEY, central, (lv.PRICE_PATH_TODAY, "central", "auction")]
    assert lv.valuation_setting_keys(settings[["case_id"]], central_path=lv.PRICE_PATH_ANCHORED) == [central, *paths]
    with pytest.raises(ValueError, match="no legacy streams"):
        lv.valuation_setting_keys(settings.assign(legacy_asset_lives="longest"), central_path=lv.PRICE_PATH_ANCHORED)
    # The local constants are the lever catalogue's.
    from scripts.public_ownership import levers as lev

    assert (lv.LIVES_CENTRAL, lv.LIVES_EXTENDED) == (lev.LIVES_CENTRAL, lev.LIVES_EXTENDED)
    assert [lever for _, lever, _ in lv.PROFILE_SETTINGS] == [
        lev.LEVER_LEGACY_PRICE_PATH, lev.LEVER_ASSET_LIVES, lev.LEVER_CAPACITY_PROCUREMENT]
    assert lev.lever_default(lev.LEVER_ASSET_LIVES) == lv.LIVES_CENTRAL
    assert lev.lever_default(lev.LEVER_CAPACITY_PROCUREMENT) == "auction"


def test_extended_lives_and_cost_based_capacity_move_the_streams():
    from tests.unit.test_public_ownership_legacy_valuation import _register as small_register

    register = small_register()
    # A legacy CCGT holding a Capacity Market receipt above its missing money, and wind with a life to extend.
    register.loc[register["asset_id"].eq("ccgt::legacy"), ["market_revenue_gbp", "fuel_carbon_gbp"]] = [200.0, 150.0]
    register["capacity_market_revenue_gbp"] = np.where(register["asset_id"].eq("ccgt::legacy"), 40.0, 0.0)
    block = {"horizon_year": 2070, "retirement_life_years_by_carrier": {"wind_onshore": 25, "wind_offshore": 30}}
    extension = {"extra_years": 5, "carriers": ["wind_offshore", "solar_pv"],
                 "retirement_year_overrides": {"Sizewell B": 2055}}
    extended = lv.extended_lives_block(block, extension)
    assert extended["retirement_life_years_by_carrier"] == {"wind_onshore": 25, "wind_offshore": 35.0}
    assert extended["retirement_year_overrides"] == {"Sizewell B": 2055}
    assert block["retirement_life_years_by_carrier"]["wind_offshore"] == 30      # the block is not mutated
    stations = _stations()

    def streams(rows, valuation_block):
        _, profile, _ = lv.value_legacy_fleet(rows, 0.0275, lv.settings_from_config(valuation_block), stations)
        return profile.groupby(["carrier", "stream"])["gbp"].sum()

    base = streams(register, block)
    longer = streams(register, extended)
    assert longer[("wind_offshore", "net_operating_cash")] > base[("wind_offshore", "net_operating_cash")]
    assert longer[("wind_onshore", "net_operating_cash")] == pytest.approx(base[("wind_onshore", "net_operating_cash")])
    # Cost-based capacity: the CCGT's 40 of receipt becomes max(0, 30 fixed - 50 margin) = 0.
    cost_based = streams(lv.capacity_procurement.valuation_rows(register), block)
    assert cost_based[("CCGT", "net_operating_cash")] < base[("CCGT", "net_operating_cash")]
    assert cost_based[("wind_offshore", "net_operating_cash")] == pytest.approx(base[("wind_offshore", "net_operating_cash")])


def test_the_legacy_corporation_tax_falls_with_each_carriers_streams():
    profile = _profile(200.0)
    tax = {"wind_offshore": 40.0, "CCGT": 10.0}
    path = nt.legacy_corporation_tax_path(tax, profile, [2030, 2031, 2040, 2045, 2050], 2030)
    assert path[2030] == pytest.approx(50.0)
    assert path[2031] == pytest.approx(40.0 + 10.0)        # both carriers' streams as in the model year
    assert path[2040] == pytest.approx(40.0)               # CCGT's streams turn negative: no tax
    assert path[2045] == pytest.approx(20.0)               # offshore streams halve
    assert nt.legacy_corporation_tax_path({}, profile, [2030], 2030) == {2030: 0.0}
    rows = pd.DataFrame({"case_id": "p3", "finance_vintage": ["legacy", "legacy", "new_build"],
                         "carrier": ["wind_offshore", "CCGT", "wind_offshore"],
                         "corporation_tax_forgone_gbp": [40.0, 10.0, 99.0]})
    assert nt.legacy_corporation_tax_by_carrier(rows) == {"wind_offshore": 40.0, "CCGT": 10.0}


def test_the_multi_year_view_reports_the_profiled_tax_beside_the_bill():
    profile = _profile(200.0)
    lines = {nt.LINE_LEGACY_RENEWABLES: 100.0, nt.LINE_LEGACY_THERMAL: 0.0}
    conv = {"factor": 1.038374717833, "final_demand_mwh": 318.29e6, "mwh": 3.1, "vat_rate": 0.05}
    frame = nt.multi_year(profile, lines, conv, term_years=30, model_year=2030,
                          corporation_tax_by_carrier={"wind_offshore": 40.0})
    by_year = frame.set_index(frame["year"].astype(str))["legacy_corporation_tax_forgone_gbp"]
    assert by_year["2030"] == pytest.approx(40.0) and by_year["2045"] == pytest.approx(20.0)
    assert by_year["15-year average"] == pytest.approx(40.0)
    assert by_year["30-year average"] == pytest.approx(40.0 * 0.5 + 20.0 * 0.5)
    without = nt.multi_year(profile, lines, conv, term_years=30, model_year=2030)
    assert "legacy_corporation_tax_forgone_gbp" not in without.columns


# ── the private WACC basis labels (CEPA 2025) ───────────────────────────────

def test_every_private_wacc_basis_is_labelled_against_cepa_2025():
    table = pd.read_csv(ROOT / "data/public_ownership/financing/private_wacc_cepa_2025_basis.csv")
    assert set(table["wacc_basis"]) == {"observed_gearing_shielded", "cepa_vanilla", "uniform_gearing_shielded"}
    assert table["tax_basis_label"].notna().all() and table["position_against_cepa"].notna().all()
    vanilla = table.loc[table["wacc_basis"].eq("cepa_vanilla")]
    # Column D is CEPA's pre-tax real (CPI) hurdle rate at 50% gearing, exactly.
    assert np.allclose(vanilla["model_private_rate"], vanilla["cepa_2025_pre_tax_real"])
    operative = table.loc[table["wacc_basis"].eq("observed_gearing_shielded")]
    between = operative["position_against_cepa"].eq("between_pre_and_post_tax")
    assert (operative.loc[between, "model_private_rate"] < operative.loc[between, "cepa_2025_pre_tax_real"]).all()
    assert (operative.loc[between, "model_private_rate"] > operative.loc[between, "cepa_2025_post_tax_real"]).all()
