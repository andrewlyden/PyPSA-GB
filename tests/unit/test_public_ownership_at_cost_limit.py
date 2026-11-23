"""A contract that cannot deliver its rent through its strike keeps the ordinary basis.

Handing a contract's whole rent to the bill cuts its strike by
``rent / energy the contract settles``. A mostly curtailed contract settles
little energy, so that cut can exceed the strike and imply a negative strike,
which a difference contract cannot express. Such a contract stays on the
financing-saving basis instead (decided 19 Sep 2026; on the CP30 2030 year it
affects AR3 remote island wind in the bounding-pair case alone).
"""

import pandas as pd
import pytest

from scripts.market.cfd_strike_table import StrikeTable
from scripts.public_ownership.cfd_repricing import (
    ROUTE_CFD_BILL, limit_at_cost_to_deliverable,
)


STRIKE = 55.74
SETTLED = 100_000.0


def _table() -> StrikeTable:
    return StrikeTable(
        strikes={("AR3", "remote_island_wind"): STRIKE, ("AR6", "offshore"): 82.03},
        price_year=2024, indexation={}, sources={},
    )


def _savings(at_cost_gbp, financing_gbp=1_000_000.0, carrier="remote_island_wind", cfd_round="AR3"):
    """One private row (the denominator) and one re-priced row for the case."""
    rows = []
    for case_id in ("private", "public_ownership@all_alternatives"):
        applied = case_id != "private"
        rows.append({
            "case_id": case_id, "asset_id": "island_wind_1", "carrier": carrier,
            "cfd_round": cfd_round, "support_type": "CfD", "dispatch_mwh": SETTLED,
            "route": ROUTE_CFD_BILL, "r1_eligible": True,
            "saving_bill_term_gbp": at_cost_gbp if applied else 0.0,
            "saving_financing_term_gbp": financing_gbp if applied else 0.0,
            "saving_economic_life_gbp": financing_gbp if applied else 0.0,
            "at_cost_applied": applied,
        })
    return pd.DataFrame(rows)


class TestNotDeliverable:
    def test_the_contract_reverts_to_the_financing_saving(self):
        # Rent of 102.67 GBP/MWh against a 55.74 GBP/MWh strike.
        savings, limited = limit_at_cost_to_deliverable(_savings(10_267_000.0), _table())
        case = savings["case_id"].eq("public_ownership@all_alternatives")
        assert savings.loc[case, "saving_bill_term_gbp"].iloc[0] == pytest.approx(1_000_000.0)
        assert not savings.loc[case, "at_cost_applied"].iloc[0]
        assert len(limited) == 1
        row = limited.iloc[0]
        assert row["strike_key"] == "AR3_remote_island_wind"
        assert row["at_cost_gbp_per_mwh_2024"] == pytest.approx(102.67)
        assert row["strike_gbp_per_mwh_2024"] == pytest.approx(STRIKE)
        assert row["financing_saving_gbp_2024"] == pytest.approx(1_000_000.0)

    def test_the_private_row_is_never_touched(self):
        savings, _ = limit_at_cost_to_deliverable(_savings(10_267_000.0), _table())
        private = savings[savings["case_id"].eq("private")].iloc[0]
        assert private["saving_bill_term_gbp"] == pytest.approx(0.0)

    def test_an_undeliverable_financing_saving_is_capped_at_the_strike(self):
        # Review 26 Sep 2026: under gb_contract_terms the paid energy shrinks, so even
        # the ordinary basis can exceed the strike on real inputs (P1 all-repriced,
        # AR3 remote island wind). The cut stops at a zero strike rather than
        # stopping the bill layer; the rest stays with the public owner.
        savings, limited = limit_at_cost_to_deliverable(
            _savings(10_267_000.0, financing_gbp=9_000_000.0), _table()
        )
        case = savings["case_id"].eq("public_ownership@all_alternatives")
        assert savings.loc[case, "saving_bill_term_gbp"].iloc[0] == pytest.approx(STRIKE * SETTLED)
        assert not savings.loc[case, "at_cost_applied"].iloc[0]
        assert limited.iloc[0]["reason"].startswith("even the financing cut exceeds the strike")


class TestDeliverable:
    def test_a_contract_that_can_deliver_keeps_the_at_cost_basis(self):
        # 40 GBP/MWh of rent sits below the 55.74 GBP/MWh strike.
        savings, limited = limit_at_cost_to_deliverable(_savings(4_000_000.0), _table())
        case = savings["case_id"].eq("public_ownership@all_alternatives")
        assert savings.loc[case, "saving_bill_term_gbp"].iloc[0] == pytest.approx(4_000_000.0)
        assert savings.loc[case, "at_cost_applied"].iloc[0]
        assert limited.empty

    def test_exactly_at_the_strike_is_still_deliverable(self):
        savings, limited = limit_at_cost_to_deliverable(_savings(STRIKE * SETTLED), _table())
        case = savings["case_id"].eq("public_ownership@all_alternatives")
        assert savings.loc[case, "saving_bill_term_gbp"].iloc[0] == pytest.approx(STRIKE * SETTLED)
        assert limited.empty

    def test_a_case_that_never_applied_at_cost_is_returned_unchanged(self):
        frame = _savings(4_000_000.0)
        frame["at_cost_applied"] = False
        savings, limited = limit_at_cost_to_deliverable(frame, _table())
        assert savings["saving_bill_term_gbp"].tolist() == frame["saving_bill_term_gbp"].tolist()
        assert limited.empty
