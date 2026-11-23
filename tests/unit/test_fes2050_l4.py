"""FES 2050 plan, task Y9: lever L4 state_deployment_share, and its per-share BM1 tables."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
from scripts.public_ownership.finance import deployment_selection, deployment_share
from scripts.public_ownership.levers import (
    ALL_LEVERS, DEFAULT_SETTINGS, GATED_LABELS, GATED_LEVERS, LABELS, LEVER_STATE_DEPLOYMENT_SHARE, LEVERS,
    LeverConfigError, expand_cases, lever_default, parse_scenarios, parse_sensitivities,
)
from config.config_loader import load_defaults

SHARES = ["0.25", "0.50", "0.75", "1.00"]
SCENARIOS = {
    "private": {},
    "public_control": {"new_build_generation": "state_stake", "public_return": "cepa_decomposition"},
    "public_ownership": {"new_build_generation": "public", "public_return": "public_rate",
                         "public_rate_basis": "real_cpi"},
}
NO_OVERHEAD = {"enabled": False}


def _specs(extra=None):
    bundles = parse_scenarios(SCENARIOS)
    sens = {"bounding_pair": False, "one_at_a_time": [{LEVER_STATE_DEPLOYMENT_SHARE: SHARES}], **(extra or {})}
    return expand_cases(bundles, parse_sensitivities(sens, bundles))


# ── the catalogue ────────────────────────────────────────────────────────────


def test_l4_is_catalogued_and_gated_out_of_the_2030_levers():
    lever = GATED_LEVERS[LEVER_STATE_DEPLOYMENT_SHARE]
    assert lever.values == ("all", "0.25", "0.50", "0.75", "1.00")
    assert lever.default == "all" == lever_default(LEVER_STATE_DEPLOYMENT_SHARE)
    assert LEVER_STATE_DEPLOYMENT_SHARE not in LEVERS and LEVER_STATE_DEPLOYMENT_SHARE not in DEFAULT_SETTINGS
    assert LEVER_STATE_DEPLOYMENT_SHARE in ALL_LEVERS and lever.register
    assert not set(GATED_LABELS) & set(LABELS)
    assert [deployment_share({LEVER_STATE_DEPLOYMENT_SHARE: v}) for v in lever.values] == [None, 0.25, 0.5, 0.75, 1.0]


def test_l4_expands_under_p2_only_and_refuses_values_it_does_not_implement():
    ids = [spec.case_id for spec in _specs()]
    assert [c for c in ids if "deployment_share" in c] == [
        "public_ownership@deployment_share_25", "public_ownership@deployment_share_50",
        "public_ownership@deployment_share_75", "public_ownership@deployment_share_100"]
    # a case that does not name it carries no key for it (so a 2030 case is unchanged)
    by = {spec.case_id: spec for spec in _specs()}
    assert LEVER_STATE_DEPLOYMENT_SHARE not in by["public_ownership"].settings
    assert by["public_ownership@deployment_share_50"].register_key() != by["public_ownership"].register_key()
    bundles = parse_scenarios(SCENARIOS)
    with pytest.raises(LeverConfigError, match="not implemented"):
        parse_sensitivities({"one_at_a_time": [{LEVER_STATE_DEPLOYMENT_SHARE: ["0.30"]}]}, bundles)
    with pytest.raises(LeverConfigError, match="not implemented"):
        parse_scenarios({**SCENARIOS, "public_ownership": {**SCENARIOS["public_ownership"],
                                                           LEVER_STATE_DEPLOYMENT_SHARE: "half"}})


def test_the_repository_config_takes_no_gated_lever_so_2030_cases_are_unchanged():
    import yaml

    from scripts.public_ownership.reference_runs import configured_case_specs

    block = load_defaults()["public_ownership"]
    specs = configured_case_specs(block)
    assert all(set(spec.settings) == set(LEVERS) for spec in specs)


# ── the selection (decision 17) ──────────────────────────────────────────────


def test_the_selection_takes_the_same_share_of_every_cell_in_whole_rows():
    rng = np.random.default_rng(7)
    n = 400
    capacity = rng.uniform(1, 20, n)
    cells = [("Solar PV" if i % 2 else "Onshore wind", 2030 + i % 5) for i in range(n)]
    order = [f"{i:04d}" for i in rng.permutation(n)]
    eligible = rng.uniform(size=n) > 0.1
    frame = pd.DataFrame({"capacity": capacity, "cell": cells, "eligible": eligible})
    for share in (0.25, 0.5, 0.75):
        keep = deployment_selection(capacity, cells, order, eligible, share)
        assert not (keep & ~eligible).any()
        frame["keep"] = keep
        for cell, rows in frame.loc[frame["eligible"]].groupby("cell"):
            achieved = rows.loc[rows["keep"], "capacity"].sum()
            assert abs(achieved - share * rows["capacity"].sum()) <= rows["capacity"].max() / 2 + 1e-9, cell
        assert (deployment_selection(capacity, cells, order, eligible, share) == keep).all()   # deterministic
    assert (deployment_selection(capacity, cells, order, eligible, 1.0) == eligible).all()
    assert not deployment_selection(capacity, cells, order, eligible, 0.0).any()
    # nested: a larger share keeps every plant a smaller one keeps
    assert not (deployment_selection(capacity, cells, order, eligible, 0.25)
                & ~deployment_selection(capacity, cells, order, eligible, 0.5)).any()


def _snapshot():
    return {"rate_basis": "real_CPI",
            "assumptions": {"public_coordination_state_equity_share": {"value": 0.30},
                            "public_borrowing_real": {"value": 0.02745}},
            "technologies": [{"technology": t, "risk_rating": "Medium",
                              "wacc": {"private": 0.08, "public_coordination": 0.07,
                                       "public_ownership_new_build": 0.0325, "public_ownership_legacy": 0.02745}}
                             for t in ("Solar PV", "Onshore wind")]}


def _costs():
    return pd.DataFrame([{"technology": t, "capex_power_gbp_per_kw": c, "capex_energy_gbp_per_kwh": 0.0,
                          "fixed_om_gbp_per_kw_year": 10.0, "variable_om_gbp_per_mwh": 0.0,
                          "economic_life_years": 30.0, "price_year": 2024, "real_or_nominal": "real",
                          "status": "reviewed"} for t, c in (("Solar PV", 600.0), ("Onshore wind", 1400.0))])


def _assets():
    rng = np.random.default_rng(11)
    rows = []
    for i in range(120):
        technology = "Solar PV" if i % 3 else "Onshore wind"
        year = 2020 + (i % 30)
        rows.append({"model_year": 2050, "component_type": "Generator", "asset_id": f"G{i:03d}::b{year}",
                     "carrier": "solar_pv" if technology == "Solar PV" else "wind_onshore",
                     "finance_technology": technology, "finance_vintage": "legacy" if year < 2025 else "new_build",
                     "build_year": year, "commissioning_year": float(year),
                     "capacity_mw": float(rng.uniform(5, 60)), "dispatch_mwh": 1e5, "evidence_status": "resolved"})
    return pd.DataFrame(rows)


def test_l4_owns_its_share_of_each_cell_and_at_one_reproduces_p2():
    """Decision 59 (3 Oct 2026): every row inside P2's boundary is held at exactly the share."""
    from scripts.public_ownership.finance import capital_recovery_factor, co_ownership_rate

    register, totals = calculate_financing_scenarios(_assets(), _costs(), _snapshot(), case_specs=_specs(),
                                                     price_year=2024, state_operating_overhead=NO_OVERHEAD)
    rows = register.loc[register["finance_vintage"].eq("new_build")]
    p2 = rows.loc[rows["case_id"].eq("public_ownership")]
    for share in SHARES:
        case = rows.loc[rows["case_id"].eq(f"public_ownership@deployment_share_{int(float(share) * 100)}")]
        assert case["state_owned"].astype(bool).all()           # every boundary row is co-owned
        state_mw = case["capacity_mw"] * case["ownership_share"].astype(float)
        cells = case.groupby(["finance_technology", "build_year"])["capacity_mw"].sum()
        got = state_mw.groupby([case["finance_technology"], case["build_year"]]).sum()
        assert np.allclose(got.reindex(cells.index).to_numpy(), float(share) * cells.to_numpy(), rtol=0, atol=1e-9)
        # the row's annuity is its owners' share-weighted annuities (state 0.0325, private 0.08)
        n = case["remaining_life_years"].astype(float)
        expect = [co_ownership_rate(0.0325, 0.08, float(share), y) for y in n]
        assert np.allclose(case["wacc"].astype(float), expect, atol=1e-10)
        crf = [capital_recovery_factor(r, y) for r, y in zip(case["wacc"].astype(float), n)]
        mix = [float(share) * capital_recovery_factor(0.0325, y) + (1 - float(share)) * capital_recovery_factor(0.08, y)
               for y in n]
        assert np.allclose(crf, mix, rtol=1e-9)
    full = rows.loc[rows["case_id"].eq("public_ownership@deployment_share_100")]
    drop = ["case_id", LEVER_STATE_DEPLOYMENT_SHARE]
    pd.testing.assert_frame_equal(p2.drop(columns=drop).reset_index(drop=True).astype(object),
                                  full.drop(columns=drop).reset_index(drop=True).astype(object), check_exact=True)
    by = totals.set_index("case_id")["annual_required_revenue_gbp"]
    assert by["public_ownership@deployment_share_100"] == by["public_ownership"]
    assert by["public_ownership"] < by["public_ownership@deployment_share_75"] < by["public_ownership@deployment_share_25"]
    assert by["public_ownership@deployment_share_25"] < by["private"]


RES = Path("resources/public_ownership")


@pytest.mark.skipif(not (RES / "cp30_2030_cp30_network_asset_inputs.csv").exists(), reason="the CP30 2030 register inputs are absent")
def test_l4_at_one_reproduces_the_2030_p2_register_on_real_inputs():
    """Targeted parity on the real CP30 2030 inputs (read-only): P2 and L4 at 1.00, row for row."""
    import yaml

    from scripts.public_ownership.calculate_financing_scenarios import register_options_from_config
    from scripts.public_ownership.levers import CaseSpec
    from scripts.public_ownership.reference_runs import configured_case_specs
    from scripts.public_ownership.settlement import attach_settlement

    config = load_defaults()
    block = config["public_ownership"]
    assets = attach_settlement(pd.read_csv(RES / "cp30_2030_cp30_network_asset_inputs.csv", low_memory=False),
                               pd.read_csv(RES / "cp30_2030_cp30_network_asset_settlement.csv"))
    snapshot = yaml.safe_load(Path(block["assumptions_snapshot"]).read_text(encoding="utf-8"))
    by = {spec.case_id: spec for spec in configured_case_specs(block)}
    p2 = by["public_ownership"]
    l4 = CaseSpec("public_ownership@deployment_share_100", "public_ownership",
                  dict(p2.settings, **{LEVER_STATE_DEPLOYMENT_SHARE: "1.00"}), "deployment_share_100")
    register, totals = calculate_financing_scenarios(
        assets, pd.read_csv(block["technology_costs"]), snapshot, case_specs=[by["private"], p2, l4],
        price_year=int(block["price_year"]), state_operating_overhead=block.get("state_operating_overhead"),
        state_equity_share_value=block.get("state_stake_share"), **register_options_from_config(block))
    drop = ["case_id", LEVER_STATE_DEPLOYMENT_SHARE]
    a = register.loc[register["case_id"].eq("public_ownership")].drop(columns=drop).reset_index(drop=True)
    b = register.loc[register["case_id"].eq(l4.case_id)].drop(columns=drop).reset_index(drop=True)
    assert len(a) > 10_000
    pd.testing.assert_frame_equal(a.astype(object), b.astype(object), check_exact=True)
    t = totals.set_index("case_id").drop(columns=[LEVER_STATE_DEPLOYMENT_SHARE])
    pd.testing.assert_series_equal(t.loc["public_ownership"].astype(object), t.loc[l4.case_id].astype(object),
                                   check_names=False)


# ── the per-share BM1 tables ────────────────────────────────────────────────


def test_one_share_table_per_deployment_share_in_the_bm1_format(tmp_path):
    from scripts.public_ownership.ownership_shares import (
        deployment_share_case_id, load_shares, write_deployment_share_tables,
    )
    from scripts.public_ownership.register_io import write_register

    register, _ = calculate_financing_scenarios(_assets(), _costs(), _snapshot(), case_specs=_specs(),
                                                price_year=2024, state_operating_overhead=NO_OVERHEAD)
    path = tmp_path / "fes_ht_2050_asset_finance_register.parquet"
    write_register(register, path)
    written = write_deployment_share_tables(path, tmp_path / "fes_ht_2050_generator_state_shares")
    assert sorted(written) == [25, 50, 75, 100]
    assert written[50].name == "fes_ht_2050_generator_state_shares_deployment_share_50.csv"
    new = register["finance_vintage"].eq("new_build")
    for percent, file in written.items():
        case = deployment_share_case_id(percent)
        shares = load_shares(file, case)
        table = pd.read_csv(file)
        assert set(table["case_id"]) == {case}
        state = (table["state_capacity_mw"].sum()
                 / register.loc[new & register["case_id"].eq(case), "capacity_mw"].sum())
        assert state == pytest.approx(percent / 100, abs=0.08)
        assert shares.between(0.0, 1.0).all()
    with pytest.raises(ValueError, match="no state_deployment_share value"):
        deployment_share_case_id(30)
