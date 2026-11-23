"""F19 decision 3 (user, 6 Oct 2026): the size of the corporation tax the state's plant does not pay.

The base keeps state plant untaxed (bill-payers first).  What changes is the size:

* new build forgoes the tax embedded in the private capital charge only where the
  comparator's plant would pay it, so none on a row whose comparator rent is below
  its required return;
* acquired legacy plant forgoes 25% of its 2030 operating margin after capital
  allowances on its tax written-down value and interest, not the as-new embedded tax.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.public_ownership import calculate_financing_scenarios as cfs
from scripts.public_ownership.finance import capital_recovery_factor
from scripts.public_ownership.levers import DEFAULT_SETTINGS
from tests.unit.test_public_ownership_levers import ROW
from config.config_loader import load_defaults

BASIS = {**cfs.CORPORATION_TAX_BASIS_DEFAULTS, "model_year": 2030}
WEDGE = {"private": 0.02, "public_coordination": 0.015}


def test_the_code_defaults_match_the_repository_config():
    block = load_defaults()["public_ownership"]
    assert cfs.corporation_tax_basis(block["corporation_tax_basis"]) == cfs.CORPORATION_TAX_BASIS_DEFAULTS
    with pytest.raises(ValueError, match="unknown keys"):
        cfs.corporation_tax_basis({"rate_of_tax": 0.25})
    with pytest.raises(ValueError, match="share"):
        cfs.corporation_tax_basis({"gearing": 1.5})


def test_legacy_tax_is_on_the_margin_after_allowances_and_interest():
    capex = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    built = np.array([2010.0, 2025.0, 2010.0, np.nan])
    margin = np.array([100.0, 100.0, -50.0, 100.0])
    tax = cfs.legacy_corporation_tax(margin, capex, built, BASIS)
    # 2010: the special-rate pool at 6% from the first year, 20 years on: WDV = 1000 x 0.94^20.
    wdv_2010 = 1000.0 * 0.94 * 0.94 ** 19
    expected_2010 = 0.25 * (100.0 - 0.06 * wdv_2010 - 0.7 * wdv_2010 * 0.07)
    assert tax[0] == pytest.approx(expected_2010)
    # 2025: the 50% special-rate first-year allowance, then 6% for four years.
    wdv_2025 = 1000.0 * 0.5 * 0.94 ** 4
    assert tax[1] == pytest.approx(max(0.0, 0.25 * (100.0 - 0.06 * wdv_2025 - 0.7 * wdv_2025 * 0.07)))
    # A loss pays none; an undated row is taken as built the year before (2029: the 50% FYA taken).
    assert tax[2] == 0.0
    wdv_undated = 1000.0 * 0.5
    assert tax[3] == pytest.approx(max(0.0, 0.25 * (100.0 - 0.06 * wdv_undated - 0.049 * wdv_undated)))


def _rows(vintage, commissioning):
    n = len(vintage)
    return SimpleNamespace(technology=np.array(["Offshore wind"] * n), capex_total=np.full(n, 1.0e6),
                           commissioning_year=np.asarray(commissioning, dtype=float), capacity=np.ones(n))


def _terms(comparator=None):
    terms = SimpleNamespace(settings=dict(DEFAULT_SETTINGS, new_build_generation="public"),
                            wedges={"Offshore wind": WEDGE}, rates={"Offshore wind": ROW}, rate_ramp=False,
                            comparator=comparator, tax_basis=BASIS if comparator is not None else None)
    return terms


def test_new_build_forgoes_no_tax_where_the_comparator_is_loss_making():
    vintage = np.array(["new_build", "new_build", "legacy"])
    rows = _rows(vintage, [2028, 2028, 2012])
    idx, owned = np.arange(3), np.ones(3, dtype=bool)
    wacc, remaining = np.full(3, 0.035), np.full(3, 25.0)
    old = cfs._corporation_tax_forgone_vector(rows, _terms(), idx, owned, vintage, wacc, remaining)
    private = ROW["wacc"]["private"]  # workbook v2.0's Sc1 offshore rate (F29)
    embedded = 1.0e6 * (capital_recovery_factor(private, 25) - capital_recovery_factor(private - 0.02, 25))
    assert old == pytest.approx([embedded] * 3)
    comparator = {"rent": np.array([5.0e4, -10.0, 2.0e4]), "capital": np.array([0.0, 0.0, 1.0e5])}
    new = cfs._corporation_tax_forgone_vector(rows, _terms(comparator), idx, owned, vintage, wacc, remaining)
    assert new[0] == pytest.approx(embedded)        # the comparator earns its return: the tax is forgone
    assert new[1] == 0.0                           # below its required return: no tax to forgo
    # The legacy row: 25% of (rent + capital charge) less allowances and interest on its WDV.
    expected = cfs.legacy_corporation_tax(np.array([1.2e5]), np.array([1.0e6]), np.array([2012.0]), BASIS)[0]
    assert new[2] == pytest.approx(expected) and expected > 0.0
    # Judged on the cohort where the comparator says so: a cohort that earns its return keeps the tax.
    cohort = dict(comparator, loss_making=np.array([False, False, False]))
    kept = cfs._corporation_tax_forgone_vector(rows, _terms(cohort), idx, owned, vintage, wacc, remaining)
    assert kept[1] == pytest.approx(embedded)
    # Unowned rows still forgo nothing.
    none = cfs._corporation_tax_forgone_vector(rows, _terms(comparator), idx, np.zeros(3, dtype=bool), vintage,
                                               wacc, remaining)
    assert none == pytest.approx([0.0] * 3)


def test_a_bid_stack_group_judges_its_comparator_on_the_market_group():
    """F27 item 8 (23 Oct 2026): private on the state's bid stack is not a feasible case.

    With a market-group reference, the comparator's rent, capital charge and cohort loss come from
    the reference's rows (re-ordered to this register's assets), never from building it here.
    """
    import pandas as pd

    rows = SimpleNamespace(n=3, asset_ids=np.array(["a", "b", "c"]))
    reference = pd.DataFrame({
        "case_id": ["private"] * 3, "asset_id": ["c", "a", "b"],
        "producer_rent_gbp": [7.0, 5.0, -3.0], "annual_capital_recovery_gbp": [70.0, 50.0, 30.0],
        "carrier": ["wind_offshore", "wind_offshore", "solar_pv"], "support_type": ["CfD", "CfD", "merchant"],
        "cfd_round": ["AR6", "AR6", ""], "finance_vintage": ["new_build"] * 3})
    keyed = [([SimpleNamespace(case_id="public_ownership")], SimpleNamespace())]

    def build(*_):
        raise AssertionError("a referenced comparator is not rebuilt on the bid stack")

    out = cfs._comparator_rows(keyed, {}, {}, None, build, build, rows,
                               reference=lambda case_id: reference.assign(case_id=case_id))
    (arrays,) = out.values()
    assert arrays["rent"].tolist() == [5.0, -3.0, 7.0]
    assert arrays["capital"].tolist() == [50.0, 30.0, 70.0]
    assert arrays["loss_making"].tolist() == [False, True, False]
    with pytest.raises(ValueError, match="not in the market group's register"):
        cfs._comparator_rows(keyed, {}, {}, None, build, build, rows, reference=lambda case_id: reference.iloc[:2])
