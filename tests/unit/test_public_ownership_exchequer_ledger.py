"""Item D2: the Exchequer ledger beside the bill headline.

Small hand-checkable inputs: corporation tax as the wedge inside a capital charge,
the public account's deficits (refused without the finance layer's field unless
the fallback is asked for), domestic VAT on the bill, and the net-of-Exchequer
figure per case, the bill headline itself untouched.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership.exchequer_ledger import (
    UPSTREAM_CT_COLUMN, UPSTREAM_DEFICIT_FIELD, build, corporation_tax_forgone, domestic_vat, ledger_row,
    per_household, state_account, tax_wedges,
)
from scripts.public_ownership.finance import capital_recovery_factor as crf

CONV = {"factor": 1.04, "final_demand_mwh": 300.0e6, "mwh": 3.1, "vat_rate": 0.05, "domestic_mwh": 100.0e6}
WEDGES = {"offshorewind": {"private": 0.02, "coordination": 0.015, "tau": 0.25},
          "largescalenuclear": {"private": 0.025, "coordination": 0.02, "tau": 0.25}}


def _tax(capital, rate, life, wedge):
    return capital - capital / crf(rate, life) * crf(rate - wedge, life)


def _row(case, asset, vintage, *, share, capital, wacc, life=25.0, price=np.nan, tech="Offshore wind",
         public_return="public_rate"):
    return {"case_id": case, "asset_id": asset, "finance_vintage": vintage, "finance_technology": tech,
            "ownership_share": share, "state_owned": share > 0, "annual_capital_recovery_gbp": capital,
            "wacc": wacc, "remaining_life_years": life, "acquisition_price_gbp": price, "public_return": public_return}


def _register():
    rows = [
        _row("private", "new_wind", "new_build", share=0.0, capital=100.0e6, wacc=0.08),
        _row("private", "coord_wind", "new_build", share=0.0, capital=50.0e6, wacc=0.08),
        _row("private", "old_nuke", "legacy", share=0.0, capital=300.0e6, wacc=0.09, life=20.0,
             tech="Large-scale nuclear"),
        # P2: the state owns new_wind outright; coord_wind stays private.
        _row("p2", "new_wind", "new_build", share=1.0, capital=70.0e6, wacc=0.03),
        _row("p2", "coord_wind", "new_build", share=0.0, capital=50.0e6, wacc=0.08),
        _row("p2", "old_nuke", "legacy", share=0.0, capital=300.0e6, wacc=0.09, life=20.0, tech="Large-scale nuclear"),
        # P1: coord_wind financed at the coordination rate, still a taxpayer.
        _row("p1", "new_wind", "new_build", share=0.0, capital=100.0e6, wacc=0.08, public_return="cepa_decomposition"),
        _row("p1", "coord_wind", "new_build", share=0.3, capital=45.0e6, wacc=0.07, public_return="cepa_decomposition"),
        _row("p1", "old_nuke", "legacy", share=0.0, capital=300.0e6, wacc=0.09, life=20.0, tech="Large-scale nuclear",
             public_return="cepa_decomposition"),
        # Nationalised: everything owned; the legacy plant bought for 1 bn.
        _row("nat", "new_wind", "new_build", share=1.0, capital=70.0e6, wacc=0.03),
        _row("nat", "coord_wind", "new_build", share=1.0, capital=35.0e6, wacc=0.03),
        _row("nat", "old_nuke", "legacy", share=1.0, capital=50.0e6, wacc=0.0275, life=20.0, price=1.0e9,
             tech="Large-scale nuclear"),
    ]
    return pd.DataFrame(rows)


class TestCorporationTax:
    def test_state_owned_new_build_loses_the_owner_tax(self):
        out = corporation_tax_forgone(_register(), "p2", WEDGES)
        assert out["corporation_tax_forgone_gbp"] == pytest.approx(_tax(100.0e6, 0.08, 25.0, 0.02))
        assert out["corporation_tax_forgone_legacy_gbp"] == 0.0

    def test_a_p1_joint_venture_still_pays_tax_at_the_scenario_2_rate_and_wedge(self):
        out = corporation_tax_forgone(_register(), "p1", WEDGES)
        capex = 50.0e6 / crf(0.08, 25.0)
        expected = _tax(50.0e6, 0.08, 25.0, 0.02) - capex * (crf(0.07, 25.0) - crf(0.07 - 0.015, 25.0))
        assert out["corporation_tax_forgone_gbp"] == pytest.approx(expected)
        assert out["corporation_tax_rows_still_taxed"] == 1

    def test_acquired_legacy_is_on_the_same_as_new_capital_base_as_the_register(self):
        # The finance layer's method (calculate_financing_scenarios._corporation_tax_forgone).
        out = corporation_tax_forgone(_register(), "nat", WEDGES)
        assert out["corporation_tax_forgone_legacy_gbp"] == pytest.approx(_tax(300.0e6, 0.09, 20.0, 0.025))
        assert out["corporation_tax_forgone_new_build_gbp"] == pytest.approx(
            _tax(100.0e6, 0.08, 25.0, 0.02) + _tax(50.0e6, 0.08, 25.0, 0.02))

    def test_workbook_wedges_are_the_finance_layer_wedges(self):
        import yaml

        from scripts.public_ownership.finance import corporation_tax_wedges
        # F29: the workbook v2.0 snapshot (rule extract_public_ownership_workbook_assumptions).
        path = Path("resources/public_ownership/workbook_v5_assumptions.yaml")
        if not path.exists():
            pytest.skip("the workbook v2.0 snapshot is a workflow output")
        snapshot = yaml.safe_load(path.read_text(encoding="utf-8"))
        wedges = tax_wedges(snapshot)
        wind = wedges["offshorewind"]
        assert wind["tau"] == pytest.approx(0.25)
        assert 0.0 < wind["coordination"] < wind["private"] < 0.05
        assert wind["private"] == pytest.approx(corporation_tax_wedges(snapshot)["Offshore wind"]["private"])


class TestStateAccount:
    ACCOUNT = {"retained_in_public_account_gbp": -172.0e6, "storage_surplus_not_recycled_gbp": -1000.0e6,
               "capacity_market_route_surplus_not_recycled_gbp": 20.0e6,
               "dispatchable_cfd_route_surplus_not_recycled_gbp": 0.0, "shortfall_on_exchequer_gbp": 5.0e6}

    def test_the_finance_layer_field_is_required(self):
        with pytest.raises(ValueError, match=UPSTREAM_DEFICIT_FIELD):
            state_account(self.ACCOUNT, fallback=False, case_id="p1")
        with pytest.raises(ValueError, match="no case"):
            state_account(None, fallback=True, case_id="p1")

    def test_upstream_wins_and_the_fallback_is_labelled(self):
        up = state_account({**self.ACCOUNT, UPSTREAM_DEFICIT_FIELD: 400.0e6}, fallback=False, case_id="p1")
        assert up["state_deficits_funded_gbp"] == 400.0e6 and up["state_account_provenance"].startswith("finance layer")
        fb = state_account(self.ACCOUNT, fallback=True, case_id="p1")
        assert fb["state_deficits_funded_gbp"] == pytest.approx(5.0e6 + 172.0e6 + 1000.0e6)
        assert fb["state_surplus_retained_gbp"] == pytest.approx(20.0e6)
        assert fb["state_account_provenance"].startswith("computed_here")


def _entry(**kw):
    base = {"bill_ex_vat": 900.0, "bill_inc_vat": 945.0, "carbon_receipts_gbp": 1.5e9, "corporation_tax_forgone_gbp": 0.0,
            "network_tax_forgone_gbp": 0.0, "state_deficits_funded_gbp": 0.0, "state_surplus_retained_gbp": 0.0,
            "exchequer_residual_gbp": 0.0, "financing_saving_gbp": 0.0, "financing_saving_tax_transfer_gbp": 0.0}
    return {**base, **kw}


def test_domestic_vat_is_the_bill_rate_on_domestic_supply_only():
    # 900 GBP per 3.1 MWh in report money, 5% on 100 TWh, back to model money.
    assert domestic_vat(900.0, CONV) == pytest.approx(0.05 * 900.0 / 3.1 * 100.0e6 / 1.04)


def test_ledger_row_nets_the_exchequer_off_the_headline_and_splits_the_saving():
    private = _entry()
    case = _entry(bill_ex_vat=880.0, bill_inc_vat=924.0, corporation_tax_forgone_gbp=300.0e6,
                  network_tax_forgone_gbp=895.5e6, state_deficits_funded_gbp=100.0e6,
                  financing_saving_gbp=1000.0e6, financing_saving_tax_transfer_gbp=300.0e6)
    row = ledger_row(case, private, CONV)
    vat_change = domestic_vat(880.0, CONV) - domestic_vat(900.0, CONV)
    net = vat_change - 300.0e6 - 895.5e6 - 100.0e6
    assert row["exchequer_net_vs_private_gbp"] == pytest.approx(net)
    assert row["exchequer_net_vs_private_gbp_per_household"] == pytest.approx(net * 1.04 / 300.0e6 * 3.1)
    # The headline is untouched; the net-of-Exchequer figure sits beside it.
    assert row["bill_inc_vat"] == 924.0
    assert row["net_of_exchequer_gbp_inc_vat"] == pytest.approx(924.0 - per_household(net, CONV))
    assert row["financing_saving_resource_gbp"] == pytest.approx(700.0e6)
    assert sum(row["exchequer_net_vs_private_parts_gbp"].values()) == pytest.approx(net)
    same = ledger_row(private, private, CONV)
    assert same["exchequer_net_vs_private_gbp"] == 0.0 and same["net_of_exchequer_gbp_inc_vat"] == 945.0


def _summaries():
    bill = {"cases": {
        "private": {"ex_vat_gbp": 900.0, "total_gbp_inc_vat": 945.0, "financing_saving_gbp_2024": 0.0,
                    "levers": {"networks": "private"}},
        "p2": {"ex_vat_gbp": 880.0, "total_gbp_inc_vat": 924.0, "financing_saving_gbp_2024": 400.0e6,
               "levers": {"networks": "acquired"}},
    }}
    account = {"cases": {c: {UPSTREAM_DEFICIT_FIELD: d, "retained_in_public_account_gbp": 0.0}
                         for c, d in (("private", 0.0), ("p2", 50.0e6))}}
    tariff = {"cases": {"nat": {
        "levers": {"networks": "acquired"},
        "bill": {"ex_vat_gbp": 700.0, "total_gbp_inc_vat": 735.0},
        "diagnostics": {"cancelled_contract_residual_exchequer_transfer_gbp": 10.0e6},
        "reconciliation": {"attribution_gbp": {"new-build capital: private WACC to public rate": -60.0e6}}}}}
    institution = {"institution_case_id": "nat_central", "i1_case_id": "nat", "i3_ex_vat_gbp": 695.0,
                   "i3_inc_vat_gbp": 729.75}
    network = pd.DataFrame({"treatment": ["acquired", "acquired"], "sector": ["transmission", "distribution"],
                            "tax_allowance_saving_gbp": [0.0, 895.5e6]})
    carbon = {k: {"carbon_receipts_gbp": v, "carbon_uk_ets_gbp": v * 0.9, "carbon_price_support_gbp": v * 0.1,
                  "chp_heat_credit_carbon_gbp": 0.0} for k, v in (("market", 1.5e9), ("central", 1.6e9))}
    return bill, account, tariff, institution, network, carbon


class TestBuild:
    def test_every_case_gets_a_row_and_the_institution_case_its_own_carbon(self):
        bill, account, tariff, institution, network, carbon = _summaries()
        register = _register().replace({"case_id": {"p2": "p2"}})
        frame, summary = build(register=register, bill=bill, account=account, network_lines=network, conv=CONV,
                               wedges=WEDGES, carbon=carbon, tariff=tariff, institution=institution, fallback=True)
        by_case = frame.set_index("case_id")
        assert set(by_case.index) == {"private", "p2", "nat", "nat_central"}
        assert by_case.loc["p2", "network_tax_forgone_gbp"] == pytest.approx(895.5e6)
        assert by_case.loc["p2", "state_deficits_funded_gbp"] == pytest.approx(50.0e6)
        assert by_case.loc["nat", "state_deficits_funded_gbp"] == 0.0  # the tariff recovers every cost
        assert by_case.loc["nat", "exchequer_residual_gbp"] == pytest.approx(10.0e6)
        assert by_case.loc["nat", "financing_saving_gbp"] == pytest.approx(60.0e6)
        # I3 takes the central carbon and its I1 twin's ownership and tax.
        assert by_case.loc["nat_central", "carbon_receipts_gbp"] == pytest.approx(1.6e9)
        assert by_case.loc["nat_central", "carbon_change_vs_private_gbp"] == pytest.approx(0.1e9)
        assert by_case.loc["nat_central", "corporation_tax_forgone_gbp"] == pytest.approx(
            by_case.loc["nat", "corporation_tax_forgone_gbp"])
        assert by_case.loc["private", "exchequer_net_vs_private_gbp"] == 0.0
        assert summary["coverage"] and "transmission" in summary["coverage"][4]
        assert (frame["transmission_tax_allowance"] == "not tabulated in the RIIO-3 annex; not counted").all()

    def test_a_register_without_the_tax_column_is_refused_without_the_fallback(self):
        bill, account, tariff, institution, network, carbon = _summaries()
        with pytest.raises(ValueError, match=UPSTREAM_CT_COLUMN):
            build(register=_register(), bill=bill, account=account, network_lines=network, conv=CONV, wedges=WEDGES,
                  carbon=carbon)
        register = _register().assign(**{UPSTREAM_CT_COLUMN: 1.0e6})
        frame, _ = build(register=register, bill=bill, account=account, network_lines=network, conv=CONV,
                         wedges=WEDGES, carbon=carbon)
        # The finance layer's per-row figure is summed as it stands (3 rows in p2).
        assert frame.set_index("case_id").loc["p2", "corporation_tax_forgone_gbp"] == pytest.approx(3.0e6)
        assert frame.set_index("case_id").loc["p2", "corporation_tax_provenance"].startswith("finance layer")

    def test_the_financing_summary_split_is_used_when_given(self):
        bill, account, tariff, institution, network, carbon = _summaries()
        financing = pd.DataFrame({"case_id": ["private", "p2"], "annual_financing_saving_vs_private_gbp": [0.0, 420.0e6],
                                  "annual_financing_saving_tax_transfer_gbp": [0.0, 150.0e6]})
        frame, _ = build(register=_register(), bill=bill, account=account, network_lines=network, conv=CONV,
                         wedges=WEDGES, carbon=carbon, fallback=True, financing=financing)
        p2 = frame.set_index("case_id").loc["p2"]
        assert p2["financing_saving_gbp"] == pytest.approx(420.0e6)
        assert p2["financing_saving_resource_gbp"] == pytest.approx(270.0e6)
        assert p2["financing_split_provenance"] == "financing summary"

    def test_the_institution_case_needs_its_own_dispatch(self):
        bill, account, tariff, institution, network, carbon = _summaries()
        with pytest.raises(ValueError, match="central-dispatch months"):
            build(register=_register(), bill=bill, account=account, network_lines=network, conv=CONV, wedges=WEDGES,
                  carbon={"market": carbon["market"]}, tariff=tariff, institution=institution, fallback=True)
