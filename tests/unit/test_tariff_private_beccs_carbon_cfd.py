"""BECCS the purchase leaves private, in the tariff cases' three-party identity (fix 8 Nov 2026).

Under F28 Drax BECCS is legacy (committed before the 2028 legislation year), and its carrier
(CCS Biomass) is in no acquisition group, so every tariff case leaves it private and the
Exchequer keeps paying its carbon-removal CfD. The ledger counted only owned plant's CfD, which
booked a GBP 225 m Exchequer saving that does not happen; and where a background lever (the
worst case's private WACC basis) moves that CfD, the owners' change had no investor item.
"""
import pandas as pd
import pytest

from scripts.public_ownership import investor_ledger as inv
from scripts.public_ownership.exchequer_ledger import _private_plant_carbon_cfd_request

from tests.unit.test_public_ownership_investor_ledger import M, TestTariffCase, _tariff_case


def _rows():
    def row(case, asset, owned, cfd, rr=1.0):
        return {"case_id": case, "asset_id": asset, "state_owned": owned, "annual_required_revenue_gbp": rr,
                "carbon_removal_cfd_revenue_gbp": cfd}
    return pd.DataFrame([
        row("private", "beccs_new", False, 50 * M), row("private", "drax", False, 225 * M),
        row("private", "wind", False, 0.0),
        # The case owns the new-build BECCS (the tariff books its CfD) and leaves Drax private,
        # whose CfD a background lever has moved.
        row("nat", "beccs_new", True, 48 * M), row("nat", "drax", False, 215 * M), row("nat", "wind", True, 0.0),
    ])


def test_only_the_rows_the_tariff_does_not_cost_count_against_private():
    _, cases, compute = _private_plant_carbon_cfd_request("nat")
    assert cases == ("nat", "private")
    assert compute(_rows()) == pytest.approx(-10 * M)


def test_an_owned_row_without_a_cost_counts_as_left_private():
    rows = _rows()
    rows.loc[(rows.case_id == "nat") & (rows.asset_id == "beccs_new"), "annual_required_revenue_gbp"] = None
    _, _, compute = _private_plant_carbon_cfd_request("nat")
    assert compute(rows) == pytest.approx(-10 * M - 2 * M)


def test_the_owners_change_closes_the_identity_against_the_exchequer_saving():
    base = dict(corporation_tax_legacy_gbp=8 * M, corporation_tax_new_build_gbp=9 * M, network=TestTariffCase.NETWORK)
    parties = inv.tariff_case_parties(_tariff_case(), private_plant_carbon_cfd_change_gbp=-10 * M, **base)
    assert parties["investor_parts_gbp"][inv.PRIVATE_PLANT_CARBON_CFD] == pytest.approx(-10 * M)
    # The Exchequer pays the private owners 10 m less: its change is 10 m better than without the lever.
    row = inv.three_party_row(parties, vat_change_gbp=0.0, exchequer_change_gbp=-(8 + 9 + 5) * M + 10 * M,
                              case_id="nat")
    assert row["three_party_identity_residual_gbp"] == pytest.approx(0.0, abs=1.0)
    with pytest.raises(ValueError, match="a transfer is booked to one party and not the other"):
        inv.three_party_row(inv.tariff_case_parties(_tariff_case(), **base), vat_change_gbp=0.0,
                            exchequer_change_gbp=-(8 + 9 + 5) * M + 10 * M, case_id="nat")


def test_the_default_is_no_change():
    parties = inv.tariff_case_parties(_tariff_case(), corporation_tax_legacy_gbp=8 * M,
                                      corporation_tax_new_build_gbp=9 * M, network=TestTariffCase.NETWORK)
    assert parties["investor_parts_gbp"][inv.PRIVATE_PLANT_CARBON_CFD] == 0.0


def test_acquired_legacy_beccs_mirrors_new_build():
    """User, 8 Nov 2026: P3 buys Drax BECCS. Its former owners lose the carbon CfD (the legacy margin leaves it
    out) and the Exchequer no longer pays them; the pair nets to nothing on the bill."""
    case = _tariff_case()
    case["diagnostics"]["private_legacy_carbon_removal_cfd_gbp"] = 225 * M
    base = dict(corporation_tax_legacy_gbp=8 * M, corporation_tax_new_build_gbp=9 * M, network=TestTariffCase.NETWORK)
    parties = inv.tariff_case_parties(case, **base)
    without = inv.tariff_case_parties(_tariff_case(), **base)
    lost = "carbon-removal CfD the former owners of acquired BECCS no longer receive"
    saved = "carbon-removal CfD the Exchequer no longer pays the former owners of acquired BECCS"
    assert parties["investor_parts_gbp"][lost] == pytest.approx(-225 * M)
    assert parties["exchequer_parts_gbp"][saved] == pytest.approx(225 * M)
    assert sum(parties["investor_parts_gbp"].values()) == pytest.approx(sum(without["investor_parts_gbp"].values())
                                                                        - 225 * M)


def test_a_bid_stack_that_moves_the_cap_and_floor_pass_through_pays_the_gb_halves():
    """9 Nov 2026: on re-solved months a bid stack moved the pass-through line, which the ledger refused."""
    from scripts.public_ownership.nationalised_tariff import INTERCONNECTOR_PASS_THROUGH_LINE

    zero = {c: 0.0 for c in ("producer_rent_gbp", "unresolved_net_cash_gbp", "revenue_gbp", "fuel_carbon_gbp",
                             "required_revenue_gbp", *(inst for _, inst in inv.DISPATCH_INSTRUMENTS.values()),
                             *inv.RENT_ONLY_INSTRUMENTS)}
    parties = inv.dispatch_step_parties(line_changes={INTERCONNECTOR_PASS_THROUGH_LINE: 2 * M}, position_group=zero,
                                        position_market=zero, carbon_change_gbp=0.0, retrading_change_gbp=0.0)
    assert parties["investor_parts_gbp"][inv.DISPATCH_PASS_THROUGH] == pytest.approx(2 * M)
    row = inv.three_party_row(parties, vat_change_gbp=0.0, exchequer_change_gbp=0.0, case_id="stack")
    assert row["three_party_identity_residual_gbp"] == pytest.approx(0.0, abs=1.0)
