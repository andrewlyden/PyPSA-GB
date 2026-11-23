"""F19 (user, 6 Oct 2026): the owners' operating-asset rate, energy from waste, and the P3 tariff fixes.

* The owners' rate: since F29 (26 Oct 2026) workbook v2.0's post-tax operating rate with the
  snapshot, each row's hurdle without it; the listed-fund lever is retired.
* Energy-from-waste gate fees are capped at the UK's throughput and credited in the tariff.
* The NZT DPA is paid once (the kept remaining_ccs line), and the ledgers still close.
* Under I1 the imbalance allowance stays as I1's forecast-error cost; I3 replaces it.
* The break-even price is the lifetime one; the 2030 snapshot is kept, renamed.
* P3's bill change splits into new-build ownership, the legacy purchase and the rest, exactly.
* The multi-year view runs the GB halves' annuity off at its term.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.public_ownership import legacy_valuation as lv
from scripts.public_ownership import nationalised_tariff as nt
from scripts.public_ownership.finance import NATIONALISED_CASE, capital_recovery_factor
from scripts.public_ownership.levers import LEVER_OWNER_DISCOUNT_RATE, LEVERS
from scripts.public_ownership.nationalised_institutions import compare

from tests.unit.test_public_ownership_legacy_valuation import PUBLIC_RATE, _row as _vrow, _stations
from tests.unit.test_public_ownership_nationalised_institutions import CENTRAL, I1_SUMMARY, MARKET
from tests.unit.test_public_ownership_nationalised_tariff import CONV, _private_bill, _register, _row
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]


def _value(rows, **overrides):
    settings = lv.settings_from_config({"horizon_year": 2070, **overrides})
    return lv.value_legacy_fleet(pd.DataFrame(rows), PUBLIC_RATE, settings, _stations())


# ── the owners' rate ─────────────────────────────────────────────────────────

class TestOwnerRate:
    """F29 (26 Oct 2026) retired the operating-asset lever: with the workbook snapshot every row is valued at
    workbook v2.0's post-tax operating rate (``test_public_ownership_workbook_v5.py``); without one, at its hurdle."""
    ROWS = [_vrow("farm", "wind_offshore", "merchant", market=100.0, commissioning=2025, wacc=0.0827),
            _vrow("pv", "solar_pv", "merchant", market=50.0, commissioning=2025, wacc=0.07),
            _vrow("gas", "CCGT", "merchant", market=100.0, fuel=20.0, commissioning=2015, wacc=0.09)]

    def test_without_the_workbook_snapshot_each_row_keeps_its_hurdle(self):
        valuation, _, _ = _value(self.ROWS)
        v = valuation.set_index("asset_id")
        assert v.loc["farm", "owner_discount_rate"] == pytest.approx(0.0827)
        assert v.loc["pv", "owner_discount_rate"] == pytest.approx(0.07)
        assert v.loc["gas", "owner_discount_rate"] == pytest.approx(0.09)
        assert set(v["owner_discount_rate_source"]) == {"new_build_hurdle_row_wacc"}

    def test_the_retired_bases_and_rates_are_refused(self):
        with pytest.raises(ValueError, match="owner_discount_rate_basis"):
            lv.settings_from_config({"owner_discount_rate_basis": "operating_asset"})
        with pytest.raises(ValueError, match="operating_asset_real_rates"):
            lv.settings_from_config({"operating_asset_real_rates": {"solar_pv": 0.06}})
        assert LEVER_OWNER_DISCOUNT_RATE not in LEVERS

    def test_the_gate_fee_cap_in_config_matches_the_code(self):
        config = load_defaults()
        block = config["public_ownership"]["acquisition"]["valuation"]
        assert "operating_asset_real_rates" not in block
        assert float(block["gate_fee_income"]["max_tonnes_per_year"]) ==             lv.DEFAULT_SETTINGS["gate_fee_income"]["max_tonnes_per_year"]
        named = config["public_ownership"]["sensitivities"]["named"]
        assert "fully_nationalised_market_value_owner_rate_new_build_hurdle" not in named


# ── energy from waste ────────────────────────────────────────────────────────

class TestGateFeeCap:
    def test_tonnes_above_the_uk_throughput_are_scaled_down_alike(self):
        per_mwh_tonnes = 3.6 / (9.62 * 0.25)
        rows = [{**_vrow(f"efw{i}", "waste_to_energy", "merchant", market=10.0), "dispatch_mwh": mwh}
                for i, mwh in enumerate((6.0e6, 4.0e6))]
        cap = 0.5 * 10.0e6 * per_mwh_tonnes
        valuation, _, summary = _value(rows, gate_fee_income={"max_tonnes_per_year": cap})
        uncapped, _, _ = _value(rows, gate_fee_income={"max_tonnes_per_year": None})
        fees = valuation["gate_fee_model_year_gbp"].to_numpy()
        assert fees == pytest.approx(0.5 * uncapped["gate_fee_model_year_gbp"].to_numpy())
        assert summary["gate_fee_income"]["tonnes_scale"] == pytest.approx(0.5)
        assert summary["gate_fee_income"]["income_model_year_gbp"] == pytest.approx(fees.sum())

    def test_under_the_cap_nothing_moves(self):
        rows = [{**_vrow("efw", "waste_to_energy", "merchant", market=10.0), "dispatch_mwh": 1000.0}]
        capped, _, _ = _value(rows)
        uncapped, _, _ = _value(rows, gate_fee_income={"max_tonnes_per_year": None})
        assert capped["gate_fee_model_year_gbp"].iloc[0] == pytest.approx(uncapped["gate_fee_model_year_gbp"].iloc[0])


# ── the tariff ───────────────────────────────────────────────────────────────

def _register_with_dpa_and_efw():
    frame = _register()
    extra = [
        _row("private", "nzt", "CCS Gas", "new_build", owned=False, market=30.0e6, support=40.0e6, resolved=False),
        _row(NATIONALISED_CASE, "nzt", "CCS Gas", "new_build", owned=False, market=30.0e6, support=40.0e6,
             resolved=False),
        _row("private", "efw", "waste_to_energy", "legacy", owned=False, capital=5.0e6, fixed=6.0e6, market=4.0e6),
        _row(NATIONALISED_CASE, "efw", "waste_to_energy", "legacy", owned=True, scope="thermal",
             capital=9.0e6 * 0.05, fixed=6.0e6, market=4.0e6, price=9.0e6),
    ]
    frame = pd.concat([frame, pd.DataFrame(extra)], ignore_index=True)
    frame["dpa_support_gbp"] = np.where(frame["asset_id"].eq("nzt"), 40.0e6, 0.0)
    return frame


def _bill_with_kept_lines():
    bill = _private_bill()
    extra = bill.iloc[:2].copy()
    extra["component_id"] = ["remaining_ccs", "procurement_imbalance"]
    extra[nt.VALUE] = [1.5, 2.0]
    return pd.concat([bill, extra], ignore_index=True)


def _build(**revenue_kw):
    register, private = _register_with_dpa_and_efw(), _bill_with_kept_lines()
    lines, diag = nt.revenue_requirement(register, **revenue_kw)
    kept = nt.kept_replaced_lines("I1")
    allocation = nt.recovery_rates(private, CONV, {}, kept_replaced=kept)
    components, summary = nt.build_bill(private, lines, CONV, allocation=allocation, kept_replaced=kept)
    detail = {"cfd_economic_curtailment_payment_gbp": 1.0e6, "interconnector_retrading_cost_gbp": 0.0}
    recon, recon_summary = nt.reconcile(private, components, lines, diag, CONV, detail, allocation=allocation,
                                        kept_replaced=kept)
    return register, private, lines, diag, allocation, components, summary, recon, recon_summary


def _closes(summary, recon):
    change = summary["delta_ex_vat_gbp"]
    for ledger in ("bill", "register"):
        assert recon.loc[recon["ledger"].eq(ledger), "gbp_per_3100kwh_ex_vat"].sum() == pytest.approx(change)


class TestDpaPaidOnce:
    def test_the_plant_paid_as_today_is_not_paid_its_dpa_again_and_the_ledgers_close(self):
        _, _, twice, _, _, _, _, _, _ = _build()
        _, _, lines, diag, _, _, summary, recon, recon_summary = _build(dpa_paid_by_kept_line=True)
        assert twice[nt.LINE_UNACQUIRED] - lines[nt.LINE_UNACQUIRED] == pytest.approx(40.0e6)
        assert diag["dpa_paid_through_kept_line_gbp"] == pytest.approx(40.0e6)
        assert any("remaining_ccs" in part for part in recon_summary["residual_parts_gbp"])
        _closes(summary, recon)


class TestGateFeeCredit:
    def test_the_gate_fees_the_price_counts_are_credited_on_owned_legacy_plant(self):
        fees = pd.Series({"efw": 7.0e6, "old_gas": 0.0, "new_wind": 99.0e6})   # new build is never credited
        _, _, lines, diag, _, components, summary, recon, _ = _build(legacy_gate_fee_gbp=fees)
        assert lines[nt.LINE_GATE_FEE_INCOME] == pytest.approx(-7.0e6)
        assert diag["efw_gate_fee_legacy_gbp"] == pytest.approx(7.0e6)
        assert nt.LINE_GATE_FEE_INCOME in set(components["component_id"])
        _closes(summary, recon)


class TestImbalanceUnderI1:
    def test_the_imbalance_allowance_is_kept_outside_the_pool_and_the_ledgers_close(self):
        _, private, _, _, allocation, components, summary, recon, _ = _build()
        ids = set(components["component_id"])
        assert "procurement_imbalance" in ids
        assert "procurement_imbalance" not in summary["replaced_lines_gbp_per_3100kwh_ex_vat"]
        assert "procurement_imbalance" not in allocation["replaced_rates"]
        assert "I1's forecast-error cost" in summary["kept_lines_declared"]["procurement_imbalance"]
        assert nt.kept_replaced_lines("I3") == {}
        _closes(summary, recon)

    def test_i3_replaces_it_and_the_institution_change_counts_it(self):
        _, base = compare(I1_SUMMARY, MARKET, CENTRAL, realtime_correction_gbp=0.0, forecast_error_charge_gbp=1.0e9)
        kept = {**I1_SUMMARY, "i1_forecast_error_cost": {"component_id": "procurement_imbalance",
                                                         "gbp_per_3100kwh_ex_vat": 2.0, "system_gbp": 186.0e6}}
        frame, with_kept = compare(kept, MARKET, CENTRAL, realtime_correction_gbp=0.0, forecast_error_charge_gbp=1.0e9)
        row = frame.set_index("component_id").loc["procurement_imbalance"]
        assert row["i1_system_gbp"] == pytest.approx(186.0e6) and row["i3_system_gbp"] == 0.0
        assert with_kept["institution_change_system_gbp"] - base["institution_change_system_gbp"] == pytest.approx(-186.0e6)
        assert with_kept["institution_change_ex_vat_gbp"] - base["institution_change_ex_vat_gbp"] == pytest.approx(-2.0)
        audit = with_kept["audit"]
        assert audit["realtime_correction_replaced_by_charge_gbp"] == pytest.approx(186.0e6)
        assert audit["i1_imbalance_allowance_removed_gbp"] == pytest.approx(186.0e6)


# ── break-even, three parts and the multi-year view ──────────────────────────

def _profile(streams_by_year):
    return pd.DataFrame([{"scope_group": "valued", "carrier": "wind_offshore", "support_type": "CfD",
                          "stream": "cfd_top_up", "year": y, "gbp": v} for y, v in streams_by_year.items()])


def test_the_lifetime_break_even_makes_the_purchase_worth_nothing_at_the_public_rate():
    profile = _profile({y: (100.0e6 if y < 2040 else 20.0e6) for y in range(2030, 2071)})
    out = nt.lifetime_break_even_price(profile, public_rate=0.02745, term_years=30, model_year=2030)
    price = out["break_even_purchase_price_lifetime_gbp"]
    annuity = price * capital_recovery_factor(0.02745, 30)
    years = np.arange(2030, 2071)
    flows = np.where(years < 2040, 100.0e6, 20.0e6) - np.where(years < 2060, annuity, 0.0)
    assert float((flows * 1.02745 ** -(years - 2030)).sum()) == pytest.approx(0.0, abs=1.0)


def test_the_snapshot_break_even_is_kept_and_labelled():
    summary = {"total_gbp_inc_vat": 900.0, "private_total_gbp_inc_vat": 1000.0}
    out = nt.break_even_purchase_price(summary, {"acquisition_price_gbp": 100e9}, None, CONV, public_rate=0.0275,
                                       term_years=30.0, headline_private=None, exchequer_funded=False)
    assert out["break_even_purchase_price_2030_snapshot_bill_gbp"] > 100e9
    assert "SNAPSHOT" in out["snapshot_note"]


class TestThreeParts:
    @pytest.fixture(scope="class")
    def split(self):
        _, _, lines, diag, allocation, _, summary, _, recon_summary = _build(
            dpa_paid_by_kept_line=True, legacy_gate_fee_gbp=pd.Series({"efw": 7.0e6}))
        profile = _profile({y: (90.0e6 if y < 2037 else 30.0e6) for y in range(2030, 2071)})
        parts, path = nt.three_part_split(recon_summary, lines, diag, CONV, allocation, profile, public_rate=0.02745,
                                          term_years=30, model_year=2030)
        return parts, path, summary, lines, allocation

    def test_the_parts_add_up_to_the_bill_change_in_both_units(self, split):
        parts, _, summary, _, _ = split
        assert sum(parts["gbp_per_3100kwh_inc_vat"].values()) == pytest.approx(summary["delta_inc_vat_gbp"])
        assert parts["total_gbp_inc_vat"] == pytest.approx(summary["delta_inc_vat_gbp"])
        assert set(parts["system_gbp"]) == {nt.PART_NEW_BUILD, nt.PART_LEGACY, nt.PART_REST}

    def test_each_part_holds_its_own_items(self, split):
        parts, _, _, lines, _ = split
        legacy = parts["items_system_gbp"][nt.PART_LEGACY]
        assert legacy["legacy acquisition annuity, renewables and nuclear"] == pytest.approx(
            lines[nt.LINE_LEGACY_RENEWABLES])
        # The legacy purchase takes the valuation's own 2030 streams (the profile its path runs on) ...
        streams = next(v for k, v in legacy.items() if k.startswith("legacy streams captured"))
        assert streams == pytest.approx(-90.0e6)
        # ... and the rest carries the gap to the register's own payments to the same plant, by name.
        rest = parts["items_system_gbp"][nt.PART_REST]
        assert nt.RECONCILING_ITEM in rest
        assert "new-build capital: private WACC to public rate" in parts["items_system_gbp"][nt.PART_NEW_BUILD]
        assert any("remaining_ccs" in k for k in rest)

    def test_the_legacy_path_moves_with_the_streams_and_its_value_is_reported(self, split):
        parts, path, _, lines, allocation = split
        p = path.set_index("year")
        rate = nt.line_rate(allocation, CONV, nt.LINE_LEGACY_RENEWABLES) * 1.05
        # Streams fall by 60 m from 2037: the legacy part's bill change rises by that at the annuity's rate.
        assert p.loc[2037, "legacy_purchase_gbp_inc_vat"] - p.loc[2036, "legacy_purchase_gbp_inc_vat"] == \
            pytest.approx(60.0e6 * rate)
        assert (p["new_build_ownership_gbp_inc_vat"] == p.loc[2030, "new_build_ownership_gbp_inc_vat"]).all()
        assert parts["legacy_purchase"]["pv_span"] == [2030, 2070]
        assert np.isfinite(parts["legacy_purchase"]["pv_public_rate_system_gbp"])


def test_the_gb_halves_annuity_runs_off_at_its_term():
    lines = {nt.LINE_LEGACY_RENEWABLES: 10.0e6, nt.LINE_LEGACY_THERMAL: 0.0, nt.LINE_IC_ACQUISITION: 5.0e6}
    profile = _profile({y: 10.0e6 for y in range(2030, 2071)})
    years = nt.multi_year(profile, lines, CONV, term_years=15, model_year=2030, bill_delta_model_year_gbp_per_hh=-50.0,
                          years=[2030, 2044, 2045])
    by = years.set_index("year")
    flat = nt.household(1.0, CONV["final_demand_mwh"], CONV["factor"], CONV["mwh"])
    # In 2045 both annuities stop: the bill falls by the legacy annuity and by the GB halves' one.
    assert by.loc[2044, "bill_delta_gbp_per_3100kwh_ex_vat"] == pytest.approx(-50.0)
    assert by.loc[2045, "bill_delta_gbp_per_3100kwh_ex_vat"] == pytest.approx(-50.0 - 15.0e6 * flat)
    assert by.loc[2045, "interconnector_and_fees_annuity_gbp"] == 0.0


class TestSizewellCRabPaidOnce:
    """Money audit (13 Nov 2026): a later year's P3 tariff paid Sizewell C's RAB twice, through the plant paid as today
    (its register support) and through the bill's own sizewell_c_rab line; now once, through the kept line."""

    def _build(self, **revenue_kw):
        register = _register_with_dpa_and_efw()
        szc = [_row("private", "szc", "nuclear", "in_construction", owned=False, market=20.0e6, support=50.0e6,
                    resolved=False),
               _row(NATIONALISED_CASE, "szc", "nuclear", "in_construction", owned=False, market=20.0e6, support=50.0e6,
                    resolved=False)]
        register = pd.concat([register, pd.DataFrame(szc)], ignore_index=True)
        register["rab_support_gbp"] = np.where(register["asset_id"].eq("szc"), 50.0e6, 0.0)
        private = _bill_with_kept_lines()
        extra = private.iloc[:1].copy()
        extra["component_id"] = [nt.RAB_LINE]
        extra[nt.VALUE] = [12.0]
        private = pd.concat([private, extra], ignore_index=True)
        lines, diag = nt.revenue_requirement(register, dpa_paid_by_kept_line=True, **revenue_kw)
        kept = nt.kept_replaced_lines("I1")
        allocation = nt.recovery_rates(private, CONV, {}, kept_replaced=kept)
        components, summary = nt.build_bill(private, lines, CONV, allocation=allocation, kept_replaced=kept)
        detail = {"cfd_economic_curtailment_payment_gbp": 1.0e6, "interconnector_retrading_cost_gbp": 0.0}
        recon, recon_summary = nt.reconcile(private, components, lines, diag, CONV, detail, allocation=allocation,
                                            kept_replaced=kept)
        return lines, diag, summary, recon, recon_summary

    def test_the_rab_is_not_paid_again_as_today_and_the_ledgers_close(self):
        twice, _, _, _, _ = self._build()
        lines, diag, summary, recon, recon_summary = self._build(rab_paid_by_kept_line=True)
        assert twice[nt.LINE_UNACQUIRED] - lines[nt.LINE_UNACQUIRED] == pytest.approx(50.0e6)
        assert diag["rab_paid_through_kept_line_gbp"] == pytest.approx(50.0e6)
        assert any("sizewell_c_rab" in part for part in recon_summary["residual_parts_gbp"])
        _closes(summary, recon)
