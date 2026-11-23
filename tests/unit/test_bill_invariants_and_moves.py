"""The workflow's money invariants and headline-move check (user, 15 Nov 2026)."""

import pandas as pd

from scripts.public_ownership import headline_moves as hm
from scripts.public_ownership import verify_bill_invariants as bi

S3 = "public_ownership@fully_nationalised_market_value"


def _components(credit):
    rows = [("private", "remaining_ccs", 4.88), (S3, "remaining_ccs", 4.88)]
    if credit is not None:
        rows.append((S3, bi.DPA_LINE, credit))
    return pd.DataFrame(rows, columns=["case_id", "component_id", bi.VALUE])


def test_a_buying_case_owning_dpa_plant_must_return_the_dpa():
    summary = {"cases": {S3: {"diagnostics": {"dpa_on_owned_costed_rows_gbp": 4.03e8}}}}
    assert bi.dpa_failures(_components(-4.88), summary) == []
    assert bi.dpa_failures(_components(None), summary)            # the 15 Nov bug: paid twice
    assert bi.dpa_failures(_components(-2.0), summary)
    not_owned = {"cases": {S3: {"diagnostics": {"dpa_on_owned_costed_rows_gbp": 0.0}}}}
    assert bi.dpa_failures(_components(None), not_owned) == []


def test_s2_must_carry_sizewell_cs_public_rate_return_where_its_rab_is_private():
    by_case = pd.DataFrame([("private", "sizewell_c_rab", 12.4), ("public_ownership", "sizewell_c_rab", 12.4)],
                           columns=["case_id", "component_id", bi.VALUE])
    account = {"settings": {"sizewell_c_rab_public_rate": {"enabled": True, "existing_state_capital_share": 0.807}},
               "cases": {"public_ownership": {"state_new_build_share": 1.0, "rab_support_gbp": 2e9,
                                              "rab_support_state_owned_gbp": 0.0}}}
    from scripts.public_ownership import sizewell_c_rab

    account["settings"] = {sizewell_c_rab.CONFIG_KEY: sizewell_c_rab.settings_from_config(
        {sizewell_c_rab.CONFIG_KEY: {"enabled": True}})}
    assert bi.sizewell_failures(by_case, account)                  # the 15 Nov bug: skipped in 2040/2050
    fixed = pd.concat([by_case, pd.DataFrame([("public_ownership", bi.SIZEWELL_ADJUSTMENT, -1.9)],
                                             columns=by_case.columns)])
    assert bi.sizewell_failures(fixed, account) == []
    account["cases"]["public_ownership"]["rab_support_state_owned_gbp"] = 2e9   # the case holds the RAB itself
    assert bi.sizewell_failures(by_case, account) == []


def test_every_headline_move_needs_a_declared_reason():
    reference = pd.DataFrame({"case_id": ["private", S3], "bill_inc_vat": [1000.0, 900.0],
                              "household_total_delta_vs_private_gbp_inc_vat": [0.0, -100.0]}).set_index("case_id")
    now = reference.copy()
    now.loc[S3, "bill_inc_vat"] = 895.0
    now.loc[S3, "household_total_delta_vs_private_gbp_inc_vat"] = -105.0
    none = pd.DataFrame(columns=["group_pattern", "case_pattern", "reason", "decided"])
    report = hm.compare(now, reference, none, "cp30_2030_no_acceleration")
    assert len(report) == 2 and not report["explained"].any()
    declared = pd.DataFrame([{"group_pattern": "cp30_2030_*", "case_pattern": S3 + "*", "reason": "DPA credit",
                              "decided": "2026-11-15"}])
    report = hm.compare(now, reference, declared, "cp30_2030_no_acceleration")
    assert report["explained"].all()
    assert hm.compare(reference, reference, none, "g").empty      # nothing moved: nothing to explain
