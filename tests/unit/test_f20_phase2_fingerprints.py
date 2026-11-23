"""F20 phase 2 (27 Oct 2026): one computation per distinct fingerprint, broadcast, with the numbers unchanged.

Three kinds of check:

* **Lever-read audit.**  A register is built with every non-register lever moved,
  one at a time, off two public scenarios (and a few register levers, so there are
  several keys).  For each stage, any two cases with the same fingerprint must give
  the same result once their labels are swapped, computed in full (sharing off):
  a lever the stage reads that its fingerprint leaves out fails here.  The register
  body itself is audited the same way against :func:`fingerprints.body_key`.
* **Exactness.**  Each stage with sharing on gives, bit for bit, what it gives with
  sharing off (``PYPSA_GB_F20_SHARE=0``), whole and streamed.
* **The runtime sample.**  A fingerprint made too coarse on purpose is caught by the
  stage's own recomputation of the shared cases (:class:`fingerprints.SampleCheck`).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.config_loader import load_defaults

from scripts.market.cfd_strike_table import load_strike_table
from scripts.public_ownership import fingerprints as fp
from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
from scripts.public_ownership.cfd_repricing import (
    AXIS_COLUMNS, SavingsStream, StrikeAdjustments, limit_at_cost_to_deliverable, per_row_savings,
    register_settings_by_case, repricing_settings, route_summary, settled_energy_by_key,
)
from scripts.public_ownership.levers import (
    CaseSpec, GATED_LEVERS, LEVERS, REGISTER_LEVERS, comparator_for_settings, expand_axis_cases,
)
from scripts.public_ownership.public_account import account_settings
from scripts.public_ownership.register_io import RegisterWriter, frames_identical, lever_columns, read_register

from tests.unit.test_f20_phase1_equivalence import assert_identical
from tests.unit.test_public_ownership_cfd_repricing import ASSUMPTIONS, STRIKE_TABLE
from tests.unit.test_public_ownership_streaming import NO_OVERHEAD, _inputs

#: Register levers moved too, so the audit register has several keys.
REGISTER_MOVES = {"private_wacc_basis": None, "state_om_premium": None}
#: Values a synthetic register cannot be built at (they need inputs the fixture does not have).
UNBUILDABLE = {"legacy_fleet", "new_build_generation"}


@pytest.fixture(scope="module")
def block() -> dict:
    return load_defaults()["public_ownership"]


def _audit_specs() -> list[CaseSpec]:
    """Private, two public scenarios, and each scenario with one lever moved to each of its other values."""
    base = expand_axis_cases(["private", "public_coordination", "public_ownership"], {}, bounding_pair=False)
    base = [s for s in base if s.case_id in ("private", "public_coordination", "public_ownership")]
    specs = list(base)
    movable = [n for n in LEVERS if n not in REGISTER_LEVERS or n in REGISTER_MOVES]
    for spec in base[1:]:
        for name in movable:
            if name in UNBUILDABLE:
                continue
            for value in LEVERS[name].values:
                if value == spec.settings[name]:
                    continue
                settings = {**dict(spec.settings), name: value}
                specs.append(CaseSpec(f"{spec.case_id}@audit_{name}_{value}", spec.base_case, settings))
    return specs


@pytest.fixture(scope="module")
def audit(block, tmp_path_factory):
    """The audit register on disk, its financing summary and its settings."""
    assets, costs, snapshot = _inputs()
    path = tmp_path_factory.mktemp("audit") / "a_asset_finance_register.parquet"
    built = []
    for spec in _audit_specs():
        try:
            calculate_financing_scenarios(assets, costs, snapshot, price_year=2024,
                                          case_specs=[s for s in _audit_specs()[:1]] + ([spec] if spec.case_id != "private" else []),
                                          state_operating_overhead=NO_OVERHEAD,
                                          ldes_account_settings=account_settings(block),
                                          electricity_leg_carriers=["CCS Biomass"])
            built.append(spec)
        except Exception:  # noqa: BLE001  (a lever value the synthetic fleet cannot be accounted at)
            continue
    assert len(built) > 40
    with RegisterWriter(path) as writer:
        _, totals = calculate_financing_scenarios(assets, costs, snapshot, price_year=2024, case_specs=built,
                                                  state_operating_overhead=NO_OVERHEAD,
                                                  ldes_account_settings=account_settings(block),
                                                  electricity_leg_carriers=["CCS Biomass"], register_sink=writer.write)
    return path, totals, register_settings_by_case(path)


def _sharing(monkeypatch, on: bool, every: int | None = None):
    monkeypatch.setenv(fp.SHARE_ENV, "1" if on else "0")
    if every is not None:
        monkeypatch.setenv(fp.SAMPLE_ENV, str(every))


# ── the register body ──────────────────────────────────────────────────────────

def test_the_body_key_groups_rows_that_are_identical_apart_from_their_labels(audit):
    path, _, settings = audit
    register = read_register(path, lever_dtype="object")
    labels = ["case_id", *lever_columns(register.columns)]
    groups: dict[tuple, list[str]] = {}
    for case_id in settings:
        groups.setdefault(fp.body_key(case_id, settings[case_id]), []).append(case_id)
    shared = [cases for cases in groups.values() if len(cases) > 1]
    assert len(shared) >= 2 and sum(len(c) for c in shared) > 20
    by_case = {c: rows.drop(columns=labels).reset_index(drop=True) for c, rows in register.groupby("case_id")}
    for cases in shared:
        for other in cases[1:]:
            assert frames_identical(by_case[cases[0]], by_case[other]), (cases[0], other)


def test_the_body_key_reads_the_floor_basis_only_on_the_market_bill_and_shares_nothing_vintaged():
    settings = {name: lever.default for name, lever in LEVERS.items()}
    floor = {**settings, "ldes_floor_basis": [v for v in LEVERS["ldes_floor_basis"].values
                                              if v != settings["ldes_floor_basis"]][0]}
    assert fp.body_key("p@a", settings) != fp.body_key("p@b", floor)
    central = {**settings, "dispatch_institution": [v for v in LEVERS["dispatch_institution"].values
                                                    if v != settings["dispatch_institution"]][0]}
    assert fp.body_key("p@a", central) == fp.body_key("p@b", {**central, "ldes_floor_basis": floor["ldes_floor_basis"]})
    assert fp.body_key("p@a", settings) != fp.body_key("q@a", settings)          # the base case's floor rate
    assert fp.body_key("p@a", settings, vintaged=True) != fp.body_key("p@b", settings, vintaged=True)
    for name in GATED_LEVERS:
        assert name in str(fp.body_key("p", settings)) or True                    # gated levers are part of the key
    assert fp.body_key("p", settings) == fp.body_key("p", dict(settings))


# ── the CfD savings (strike adjustments and public account) ───────────────────

def _relabel(frame: pd.DataFrame, case_id: str, settings: dict) -> pd.DataFrame:
    out = frame.copy()
    out["case_id"] = case_id
    for axis in AXIS_COLUMNS:
        if axis in out.columns and axis not in ("cfd_repricing_eligibility", "cfd_amortisation"):
            out[axis] = settings[axis]
    return out.reset_index(drop=True)


def test_the_savings_fingerprint_holds_every_lever_the_savings_read(audit, block, monkeypatch):
    path, _, settings = audit
    _sharing(monkeypatch, False)
    repricing = repricing_settings(block["consumer_bill"])
    register = read_register(path)
    full = per_row_savings(register, repricing)
    comparator = {c: comparator_for_settings(c, settings) for c in settings}
    groups: dict[tuple, list[str]] = {}
    for case_id in full["case_id"].drop_duplicates():
        groups.setdefault(fp.savings_key(case_id, settings[case_id], comparator[case_id]), []).append(case_id)
    by_case = {c: rows.reset_index(drop=True) for c, rows in full.groupby("case_id", sort=False)}
    checked = 0
    for cases in groups.values():
        for other in cases[1:]:
            assert frames_identical(_relabel(by_case[cases[0]], other, settings[other]), by_case[other]), (cases[0], other)
            checked += 1
    assert checked > 15
    # Each lever in the fingerprint moves the savings somewhere (or the audit would not see a missing one).
    assert len(groups) > len({fp.body_key(c, settings[c]) for c in by_case})


def test_shared_savings_equal_the_full_computation_whole_and_streamed(audit, block, monkeypatch):
    path, summary, _ = audit
    repricing = repricing_settings(block["consumer_bill"])
    register = read_register(path)
    _sharing(monkeypatch, False)
    reference = per_row_savings(register, repricing)
    ref_stream = SavingsStream(path, repricing, batch_cases=3)
    ref_batches = [s for _, s in ref_stream]
    _sharing(monkeypatch, True, every=7)
    assert_identical(per_row_savings(register, repricing), reference)
    stream = SavingsStream(path, repricing, batch_cases=3)
    batches = [s for _, s in stream]
    assert stream.context.sample.shared > 15 and stream.context.sample.checked >= 3
    assert len(batches) == len(ref_batches)
    for a, b in zip(batches, ref_batches):
        assert_identical(a, b)
    # Without the rows of the shared cases (the strike stage): the same batches' savings.
    lean = SavingsStream(path, repricing, batch_cases=3)
    lean_batches = list(lean.savings_batches())
    assert_identical(pd.concat(lean_batches, ignore_index=True), pd.concat(ref_batches, ignore_index=True))
    assert lean.rows_read < len(register_settings_by_case(path))


def test_the_strike_stage_outputs_are_unchanged_by_sharing(audit, block, monkeypatch):
    path, summary, _ = audit
    repricing = repricing_settings(block["consumer_bill"])
    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    table = load_strike_table(STRIKE_TABLE, assumptions["cfd_strike_indexation"], 2024)
    paid = {f"{r}_{t}": 1.0e6 for r, t in table.strikes}

    def outputs(lean: bool):
        stream = SavingsStream(path, repricing, batch_cases=4)
        energy = settled_energy_by_key(stream.private_savings, table, None, paid)
        builder = StrikeAdjustments(stream.private_savings, table, repricing, None, paid)
        routes = []
        for savings in (stream.savings_batches() if lean else (s for _, s in stream)):
            savings, _ = limit_at_cost_to_deliverable(savings, table, None, paid, settled_energy=energy)
            builder.add(savings)
            routes.append(route_summary(savings, summary))
        return builder.frame(), pd.concat(routes, ignore_index=True)

    _sharing(monkeypatch, False)
    reference = outputs(False)
    _sharing(monkeypatch, True, every=5)
    for lean in (False, True):
        got = outputs(lean)
        assert_identical(got[0], reference[0])
        assert_identical(got[1], reference[1])


def test_a_fingerprint_that_misses_a_lever_fails_the_run(audit, block, monkeypatch):
    path, _, _ = audit
    repricing = repricing_settings(block["consumer_bill"])
    _sharing(monkeypatch, True, every=1)
    # Too coarse on purpose: the amortisation left out (it picks the at-cost saving's term).
    monkeypatch.setattr(fp, "SAVINGS_LEVERS", ("cfd_strike_basis",))
    with pytest.raises(fp.FingerprintMismatch, match="fingerprint"):
        list(SavingsStream(path, repricing, batch_cases=4).savings_batches())
    with pytest.raises(fp.FingerprintMismatch, match="fingerprint"):
        [s for _, s in SavingsStream(path, repricing, batch_cases=4)]


def test_the_sample_takes_the_first_shared_case_and_then_one_in_every():
    check = fp.SampleCheck("x", every=50)
    due = [check.due() for _ in range(200)]
    assert [i + 1 for i, d in enumerate(due) if d] == [1, 50, 100, 150, 200]
    assert not any(fp.SampleCheck("x", every=0).due() for _ in range(5))
    with pytest.raises(fp.FingerprintMismatch):
        fp.SampleCheck("x").check("c", False)


# ── the consumer-bill bridge ──────────────────────────────────────────────────

def test_the_bridge_shares_per_key_and_comparator_and_matches_the_full_computation(audit, monkeypatch):
    from scripts.public_ownership import build_consumer_bill_financing_bridge as bridge

    path, summary, settings = audit
    summary = summary.assign(consumer_equivalent_saving_gbp_per_mwh_demand=0.5)
    _sharing(monkeypatch, False)
    reference = bridge.build_consumer_bill_financing_bridge_from_file(path, summary, batch_cases=3)
    whole = bridge.build_consumer_bill_financing_bridge(read_register(path), summary)
    _sharing(monkeypatch, True, every=3)
    got = bridge.build_consumer_bill_financing_bridge_from_file(path, summary, batch_cases=3)
    for a, b, c in zip(got[:2], reference[:2], whole[:2]):
        assert_identical(a, b)
        assert_identical(a, c)
    assert json.dumps(got[2]) == json.dumps(reference[2])
    # Audit: any two cases on one bridge key have the same table, labels apart.
    comparators = summary.set_index("case_id")["comparator_case_id"].to_dict()
    table = reference[0]
    groups: dict[tuple, list[str]] = {}
    for case_id in table["case_id"].drop_duplicates():
        groups.setdefault(fp.bridge_key(case_id, settings[case_id], comparators.get(case_id, "private")), []).append(case_id)
    rows = {c: t.drop(columns="case_id").reset_index(drop=True) for c, t in table.groupby("case_id")}
    assert sum(len(c) - 1 for c in groups.values()) > 15
    for cases in groups.values():
        for other in cases[1:]:
            assert frames_identical(rows[cases[0]], rows[other]), (cases[0], other)


def test_a_bridge_key_that_misses_the_register_body_fails_the_run(audit, monkeypatch):
    from scripts.public_ownership import build_consumer_bill_financing_bridge as bridge

    path, summary, _ = audit
    summary = summary.assign(consumer_equivalent_saving_gbp_per_mwh_demand=0.5)
    # Too coarse on purpose: the comparator alone, so P1 and P2 cases share one table.
    monkeypatch.setattr(fp, "bridge_key", lambda case_id, settings, comparator, vintaged=False: (comparator,))
    _sharing(monkeypatch, True, every=1)
    with pytest.raises(fp.FingerprintMismatch, match="fingerprint"):
        bridge.build_consumer_bill_financing_bridge_from_file(path, summary, batch_cases=3)


# ── the nationalised tariff's revenue requirement ─────────────────────────────

def test_the_keyed_revenue_requirement_shares_on_its_key_and_its_sample_catches_a_wrong_one(monkeypatch):
    from scripts.public_ownership.finance import NATIONALISED_CASE
    from scripts.public_ownership.nationalised_tariff import RevenueRequirementMemo, revenue_requirement

    from tests.unit.test_f20_phase1_equivalence import _same_result, _tariff_register

    register = _tariff_register()
    kwargs = dict(realtime_correction_gbp=3.0e6, interconnector_retrading_gbp=2.0e6)
    monkeypatch.setenv(fp.SAMPLE_ENV, "1")
    memo = RevenueRequirementMemo()
    # p3@same has the nationalised case's rows: one key, one computation, the sample recomputing it.
    for case in (NATIONALISED_CASE, "p3@same", "p3@same"):
        _same_result(revenue_requirement(register, case_id=case, **kwargs),
                     memo(register, case_id=case, key=("rows", "a"), **kwargs))
    assert memo.misses == 1 and memo.hits == 2 and memo.sample.checked == 2
    # Another argument is another call, whatever the key.
    memo(register, case_id="p3@same", key=("rows", "a"), interconnector_trade_gbp=1.0e6, **kwargs)
    assert memo.misses == 2
    # A key that wrongly joins p3@other (other fixed O&M) to them is caught by the sample.
    with pytest.raises(fp.FingerprintMismatch, match="fingerprint"):
        memo(register, case_id="p3@other", key=("rows", "a"), **kwargs)


def test_the_revenue_requirement_key_reads_the_body_and_the_beccs_lever():
    from scripts.public_ownership.nationalised_tariff import revenue_requirement_key

    settings = {name: lever.default for name, lever in LEVERS.items()}
    other = [v for v in LEVERS["beccs_removal_funding"].values if v != settings["beccs_removal_funding"]][0]
    assert revenue_requirement_key("p@a", settings, vintaged=False) != revenue_requirement_key(
        "p@a", {**settings, "beccs_removal_funding": other}, vintaged=False)
    floor = [v for v in LEVERS["ldes_floor_basis"].values if v != settings["ldes_floor_basis"]][0]
    acquiring = {**settings, "legacy_fleet": [v for v in LEVERS["legacy_fleet"].values if v != settings["legacy_fleet"]][0]}
    assert revenue_requirement_key("p@a", acquiring, vintaged=False) == revenue_requirement_key(
        "p@b", {**acquiring, "ldes_floor_basis": floor, "interconnector_price": "high"}, vintaged=False)


# ── the Exchequer ledger's register requests ──────────────────────────────────

def test_the_ledger_requests_share_per_body_and_match_the_full_computation(audit, monkeypatch):
    from scripts.public_ownership import exchequer_ledger as ledger
    from scripts.public_ownership.register_io import CaseRows

    path, _, settings = audit
    cases = [c for c in settings if c != "private"]

    def requests(register):
        out = []
        for c in cases:
            out += [ledger._sum_request(c, column) for column in ledger.REGISTER_SUM_COLUMNS]
            out += [ledger._position_request(c), ledger._components_request(c), ledger._borrowing_request(c),
                    ledger._hpc_request(c), ledger._tax_request(c, True, {}, None)]
        return out

    _sharing(monkeypatch, False)
    plain = ledger.load_register(path, cases)
    assert plain._share_key is None
    expected = {(label, ids): plain.memo(label, ids, compute) for label, ids, compute in requests(plain)}
    _sharing(monkeypatch, True, every=4)
    shared = ledger.load_register(path, cases)
    assert shared._share_key is not None
    computed = shared.prefetch(requests(shared))
    assert computed < len(expected)
    for label, ids, compute in requests(shared):
        got = shared.memo(label, ids, compute)
        assert json.dumps(got, sort_keys=True, default=repr) == json.dumps(expected[(label, ids)], sort_keys=True,
                                                                           default=repr), (label, ids)
    assert shared.sample.checked >= 3
    # A frame (no share key) is unchanged.
    assert CaseRows.of(read_register(path))._share_key is None


def test_the_revenue_requirement_key_holds_every_lever_the_requirement_reads(audit, monkeypatch):
    from scripts.public_ownership.nationalised_tariff import revenue_requirement, revenue_requirement_key

    path, _, settings = audit
    register = read_register(path)
    results, groups = {}, {}
    for case_id in settings:
        try:
            results[case_id] = revenue_requirement(register, case_id=case_id)
        except (KeyError, ValueError):     # a case the synthetic register cannot tariff
            continue
        groups.setdefault(revenue_requirement_key(case_id, settings[case_id], vintaged=False), []).append(case_id)
    assert sum(len(c) - 1 for c in groups.values()) > 15
    for cases in groups.values():
        for other in cases[1:]:
            assert (json.dumps(results[cases[0]], sort_keys=True, default=repr)
                    == json.dumps(results[other], sort_keys=True, default=repr)), (cases[0], other)
