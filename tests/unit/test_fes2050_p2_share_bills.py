"""FES 2050 P2 deployment-share bill groups (decision 16, 3 Oct 2026): the fixes they needed.

* A public-ownership bid audit where the state holds every CfD unit (P2 at 100%) has no
  private CfD weight left, but the solve still priced CfD on the BM1 rule.
* The public account carries a year-gated lever the case sets (state_deployment_share), so
  the bill summaries and the balancing merge can tell the P2 shares apart.
* A bill line booked from a public account (cost_from) declares that file as a rule input.
"""

import pandas as pd
import pytest

from scripts.consumer_bill.extract_future_balancing_costs import LABEL_PRICING, audit_declares_contract_value_bids


def _audit(tmp_path, cfd_weight, state_share, pricing=None):
    path = tmp_path / "audit.csv"
    pd.DataFrame({"generator": ["g1", "g2"], "pricing": [pricing or LABEL_PRICING] * 2,
                  "cfd_weight": cfd_weight, "state_share": state_share}).to_csv(path, index=False)
    return path


def test_all_state_cfd_audit_counts_as_contract_value_priced(tmp_path):
    assert audit_declares_contract_value_bids(_audit(tmp_path, [0.0, 0.0], [1.0, 0.8]))


def test_private_label_audit_without_cfd_weight_is_not_contract_value_priced(tmp_path):
    # No state share and no CfD weight: the solve priced no CfD on contract value.
    assert not audit_declares_contract_value_bids(_audit(tmp_path, [0.0, 0.0], [0.0, 0.0]))


def test_private_cfd_weight_alone_still_counts(tmp_path):
    assert audit_declares_contract_value_bids(_audit(tmp_path, [0.4, 0.0], [0.0, 0.0]))


def test_public_account_carries_the_gated_share_lever():
    import inspect

    from scripts.public_ownership import public_account

    source = inspect.getsource(public_account.build_public_account)
    assert "GATED_LEVERS" in source and "settings_by_case.get(case_id)" in source


def test_bill_rules_declare_a_cost_from_public_account():
    import re
    from pathlib import Path

    text = Path("rules/consumer_bill.smk").read_text(encoding="utf-8")
    body = text[text.index("def _consumer_bill_support_inputs"):text.index("rule extract_future_balancing_costs")]
    assert re.search(r"cost_from.*public_account", body, re.S)
