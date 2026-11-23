"""Item N6: the network finance account.

A RAV-regulated network makes public ownership arithmetic, so the tests are about
getting the arithmetic on one basis and never letting an assumption pass as a
published figure.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.public_ownership.network_finance_account import (
    PRICE_BASIS, ROLL_FORWARD_RULES, build_account, distribution_rav_2030, price_factor,
    sector_account, settings,
)
from config.config_loader import load_defaults

CONV = {"factor": 1.038374717833, "final_demand_mwh": 318.29e6, "mwh": 3.1}
PUBLIC_RATE = 0.02745
BLOCKS = {
    "distribution": {
        "price_basis_key": "2020/21", "closing_rav_gbp_m_2027_28": 69027.0,
        "net_additions_gbp_m_2027_28": 6076.7, "depreciation_gbp_m_2027_28": 4000.4,
        "tax_allowance_gbp_m_period_average": 735.5, "vanilla_wacc_cpih_real": 0.0393,
    },
    "transmission": {
        "price_basis_key": "2023/24", "total_rav_gbp_m": 54210.2,
        "wacc_semi_nominal": 0.0564, "notional_gearing": 0.55,
    },
}


def _cfg(**overrides):
    return settings({"distribution_roll_forward": "ed2_net_growth", "roll_forward_years": 2.5,
                     "acquisition_premium_over_rav": 0.0, "semi_nominal_inflation": 0.02,
                     "fixed_rate_debt_share": 0.75, "include_tax_allowance": True, **overrides})


class TestPriceBasis:
    def test_financial_year_factors_come_from_the_declared_quarters(self):
        # The factor is the model year index over the mean of the four quarters.
        quarters = PRICE_BASIS["financial_year_quarters"]["2020/21"]
        assert price_factor("2020/21") == pytest.approx(132.9 / (sum(quarters) / 4))
        # 2020/21 money buys less in 2024 than 2023/24 money does, so it is uplifted more.
        assert price_factor("2020/21") > price_factor("2023/24") > 1.0

    def test_an_undeclared_basis_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="no CPIH quarters declared"):
            price_factor("2019/20")


class TestSettings:
    def test_supported_rules_only(self):
        for rule in ROLL_FORWARD_RULES:
            kwargs = {"distribution_rav_2030_gbp": 9.0e10} if rule == "explicit" else {}
            assert settings({"distribution_roll_forward": rule, **kwargs})["distribution_roll_forward"] == rule
        with pytest.raises(ValueError, match="distribution_roll_forward"):
            settings({"distribution_roll_forward": "guess"})

    def test_explicit_needs_a_value_and_shares_are_checked(self):
        with pytest.raises(ValueError, match="distribution_rav_2030_gbp"):
            settings({"distribution_roll_forward": "explicit"})
        with pytest.raises(ValueError, match="roll_forward_years"):
            settings({"roll_forward_years": -1})

    def test_repository_defaults_parse(self):
        block = load_defaults()["public_ownership"]
        # Item K1: the wires follow the networks lever of each case, not a global key.
        assert "network_ownership" not in block
        # 30 Sep 2026: no base holds the wires; the acquired treatment is P3's grid axis
        # and P2b, and the account is still built because those cases touch the wires.
        assert "networks" not in block["sensitivities"]["named"]["fully_nationalised"]
        assert block["sensitivities"]["named"]["fully_nationalised_market_value_rav_at_statute"]["networks"] == "acquired"
        settings(block.get("network_account"))


class TestRollForward:
    def test_each_rule_says_what_it_did(self):
        flat, flat_rule = distribution_rav_2030(BLOCKS, _cfg(distribution_roll_forward="hold_flat"))
        grown, grown_rule = distribution_rav_2030(BLOCKS, _cfg())
        assert flat == pytest.approx(69027.0e6 * price_factor("2020/21"))
        assert "held flat" in flat_rule
        # Growth is the final ED2 year's additions less depreciation, for the configured years.
        step = (6076.7 - 4000.4) * 1e6 * price_factor("2020/21")
        assert grown == pytest.approx(flat + step * 2.5)
        assert "conservative" in grown_rule and "ED3" in grown_rule

    def test_explicit_overrides_and_is_labelled(self):
        rav, rule = distribution_rav_2030(BLOCKS, _cfg(distribution_roll_forward="explicit",
                                                       distribution_rav_2030_gbp=1.0e11))
        assert rav == 1.0e11 and "explicit" in rule


class TestSectorAccount:
    def test_saving_is_the_wedge_on_the_rav_plus_the_tax_allowance(self):
        a = sector_account(rav_gbp=100e9, allowed_wacc_cpih_real=0.04, public_rate=0.0275,
                           premium=0.0, tax_allowance_gbp=1e9)
        assert a["financing_saving_gbp"] == pytest.approx(100e9 * (0.04 - 0.0275))
        assert a["total_saving_gbp"] == pytest.approx(100e9 * 0.0125 + 1e9)

    def test_a_premium_is_an_annuity_at_the_public_rate_over_thirty_years(self):
        """User decision 7: the premium is charged on the purchase's own basis, not as perpetual interest."""
        from scripts.public_ownership.finance import capital_recovery_factor
        a = sector_account(rav_gbp=100e9, allowed_wacc_cpih_real=0.04, public_rate=0.0275,
                           premium=0.2, tax_allowance_gbp=0.0)
        annuity = 0.2 * 100e9 * capital_recovery_factor(0.0275, 30.0)
        assert a["premium_annual_cost_gbp"] == pytest.approx(annuity)
        assert a["financing_saving_gbp"] == pytest.approx(100e9 * (0.04 - 0.0275) - annuity)
        # The annuity is dearer than perpetual interest (0.0275 x 20 bn = 0.55 bn), so the
        # break-even premium on this basis is lower than the perpetual one.
        assert annuity > 0.0275 * 0.2 * 100e9
        assert a["break_even_premium_over_rav"] < a["break_even_premium_over_rav_perpetual_basis"]

    def test_break_even_premium_zeroes_the_financing_saving(self):
        a = sector_account(rav_gbp=100e9, allowed_wacc_cpih_real=0.04, public_rate=0.0275,
                           premium=0.0, tax_allowance_gbp=0.0)
        at_break_even = sector_account(rav_gbp=100e9, allowed_wacc_cpih_real=0.04, public_rate=0.0275,
                                       premium=a["break_even_premium_over_rav"], tax_allowance_gbp=0.0)
        assert at_break_even["financing_saving_gbp"] == pytest.approx(0.0, abs=1.0)


class TestAccount:
    @pytest.fixture(scope="class")
    def built(self):
        return build_account(BLOCKS, PUBLIC_RATE, _cfg(), CONV)

    def test_transmission_wacc_is_converted_but_distribution_is_not(self, built):
        _, summary = built
        # Semi-nominal 5.64% loses inflation x gearing x fixed-rate share.
        assert summary["transmission"]["allowed_wacc_cpih_real"] == pytest.approx(0.0564 - 0.02 * 0.55 * 0.75)
        assert summary["distribution"]["allowed_wacc_cpih_real"] == 0.0393
        assert "semi-nominal" in summary["transmission"]["wacc_basis"]
        assert "already CPIH-real" in summary["distribution"]["wacc_basis"]

    def test_only_distribution_carries_a_tax_allowance_and_it_can_be_switched_off(self, built):
        _, summary = built
        assert summary["distribution"]["tax_allowance_saving_gbp"] > 0
        assert summary["transmission"]["tax_allowance_saving_gbp"] == 0.0
        assert summary["transmission"]["tax_basis"].startswith("nil:")   # F27 item 7: nil in the RIIO-3 FD
        _, without = build_account(BLOCKS, PUBLIC_RATE, _cfg(include_tax_allowance=False), CONV)
        assert without["distribution"]["tax_allowance_saving_gbp"] == 0.0

    def test_rows_are_signed_as_bill_reductions_and_carry_their_basis(self, built):
        frame, summary = built
        assert set(frame["bill_line_adjusted"]) == {"tnuos", "duos"}
        # A saving lowers the bill, so the per-household figure is negative.
        assert (frame["saving_gbp_per_3100kwh_ex_vat"] < 0).all()
        assert frame["rav_basis"].str.len().gt(20).all()
        assert summary["total_saving_gbp"] == pytest.approx(frame["total_saving_gbp"].sum())

    def test_the_distribution_assumption_is_never_presented_as_published(self, built):
        _, summary = built
        assert "not a published figure" in " ".join(summary["notes"])
        assert "RIIO-3 runs to 31 March 2031" in " ".join(summary["notes"])


class TestF27Item7:
    """F27 item 7 (user, 20 Oct 2026): calendar-year transmission RAV, ED2 final-year tax, excess returns."""

    EXCESS = {"central": "riio2", "high": "riio1", "equity_share": {"transmission": 0.45, "distribution": 0.40},
              "rates": {"riio2": {"transmission": 0.0105, "distribution": 0.0016},
                        "riio1": {"transmission": 0.0242, "distribution": 0.0227}}}

    def test_the_calendar_rav_tax_basis_and_excess_return_enter_the_saving_and_close(self):
        from scripts.public_ownership.network_finance_account import check_account_closes, incidence_amounts

        cfg = _cfg(transmission_rav_2030_gbp=74.0e9, transmission_rav_2030_gbp_source="calendar 2030 RAV",
                   distribution_tax_allowance_basis="final_year", distribution_tax_allowance_final_year_gbp_m=294.7,
                   excess_return=self.EXCESS)
        frame, summary = build_account(BLOCKS, PUBLIC_RATE, cfg, CONV)
        t, d = summary["transmission"], summary["distribution"]
        assert t["rav_gbp"] == 74.0e9 and t["rav_basis"] == "calendar 2030 RAV"
        assert t["excess_return_saving_gbp"] == pytest.approx(74.0e9 * 0.45 * 0.0105)
        assert d["excess_return_saving_gbp"] == pytest.approx(d["rav_gbp"] * 0.40 * 0.0016)
        assert d["tax_allowance_saving_gbp"] == pytest.approx(294.7e6 * price_factor(BLOCKS["distribution"]["price_basis_key"]))
        assert "ED2 final-year" in d["tax_basis"]
        for account in (t, d):
            assert account["total_saving_gbp"] == pytest.approx(
                account["financing_saving_gbp"] + account["tax_allowance_saving_gbp"] + account["excess_return_saving_gbp"])
        row = frame.set_index("sector").loc["transmission"]
        assert row["excess_return_high_gbp"] == pytest.approx(74.0e9 * 0.45 * 0.0242)
        check_account_closes(frame.assign(treatment="acquired"), {})
        # The excess goes where the wedge goes: kept by the Exchequer under return_retained.
        kept = incidence_amounts(frame, "acquired", {"network_incidence": "return_retained"})
        assert sum(v["retained_by_exchequer_gbp"] for v in kept.values()) == pytest.approx(frame["total_saving_gbp"].sum())

    def test_without_the_keys_the_account_is_unchanged(self):
        _, before = build_account(BLOCKS, PUBLIC_RATE, _cfg(), CONV)
        assert before["transmission"]["excess_return_saving_gbp"] == 0.0
        with pytest.raises(ValueError, match="final_year needs"):
            _cfg(distribution_tax_allowance_basis="final_year")
        with pytest.raises(ValueError, match="must name one of rates"):
            _cfg(excess_return={**self.EXCESS, "high": "riio3"})
