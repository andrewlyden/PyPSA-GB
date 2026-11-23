"""GB Energy dated from the GB Energy Act 2025 (user decision, option E, 28 Oct 2026).

GB Energy's local part and its 30% offshore stake are judged at its own year (the Act, Royal Assent
15 May 2025), by the legislation-year rules, while every other case keeps the group's year (2028).
Rows split at 2028 are not split again: each carries its share committed on or after 2025
(``legislation_year.eligible_share_at``), and GB Energy holds that share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import calculate_financing_scenarios as cfs
from scripts.public_ownership.legislation_year import (
    STATUS_POST, apply_legislation_year, eligible_share_at, load_lead_times,
)
from scripts.public_ownership.levers import CaseSpec, expand_axis_cases

LEADS = load_lead_times()


def _row(asset_id, carrier, *, vintage="new_build", capacity=100.0, **extra):
    row = {"asset_id": asset_id, "carrier": carrier, "component_type": "Generator", "finance_vintage": vintage,
           "finance_eligible": True, "capacity_mw": capacity, "energy_capacity_mwh": 0.0, "dispatch_mwh": capacity * 1000,
           "support_type": None, "cfd_round": None, "cfd_award_year": np.nan, "contract_vintage": "not_applicable",
           "commissioning_year": np.nan, "development_status": None, "cohort_legacy_share": np.nan,
           "vintage_method": "asset_date"}
    row.update(extra)
    return row


ROWS = [
    # Offshore: AR7 awarded 2026 (GBE's, not the 2028 group's), AR6 awarded 2023 (nobody's), a modelled
    # round on the even profile (part GBE's), a fleet row commissioned before 2025.
    _row("ar7", "wind_offshore", capacity=1000.0, support_type="CfD", cfd_round="AR7", cfd_award_year=2026.0,
         contract_vintage="post_policy", commissioning_year=2029.0),
    _row("ar6", "wind_offshore", capacity=800.0, support_type="CfD", cfd_round="AR6", cfd_award_year=2023.0,
         contract_vintage="post_policy", commissioning_year=2027.0),
    _row("future_off", "wind_offshore", capacity=600.0, support_type="CfD", cfd_round="AR8_modelled",
         contract_vintage="post_policy"),
    _row("old_off", "wind_offshore", vintage="legacy", capacity=500.0, commissioning_year=2018.0),
    # Local: undated solar on the 2025-2030 profile (split at 2028), dated onshore committed 2026 and 2023,
    # a pipeline project (profile from the snapshot year + 1) and one committed in the snapshot.
    _row("solar", "solar_pv", capacity=600.0, support_type="merchant"),
    _row("on26", "wind_onshore", capacity=300.0, support_type="merchant", commissioning_year=2028.0),
    _row("on23", "wind_onshore", capacity=200.0, support_type="merchant", commissioning_year=2025.0),
    _row("pipe", "solar_pv", capacity=250.0, support_type="merchant", development_status="Awaiting Construction"),
    _row("built", "solar_pv", capacity=150.0, support_type="merchant", development_status="Under Construction"),
]


def _apply(year):
    return apply_legislation_year(pd.DataFrame(ROWS), legislated_year=year, policy_start_year=2025, model_year=2030,
                                  transfer_year=2030, leads=LEADS)


def _base(asset_id):
    return str(asset_id).split("::", 1)[0]


def test_the_share_at_2025_is_what_a_2025_legislation_year_would_make_eligible():
    at_2028 = _apply(2028)
    share = eligible_share_at(at_2028, 2025, policy_start_year=2025, model_year=2030, leads=LEADS)
    gbe = (at_2028["capacity_mw"] * share).groupby(at_2028["asset_id"].map(_base)).sum()
    at_2025 = _apply(2025)
    direct = at_2025["capacity_mw"].where(at_2025["legislation_status"].eq(STATUS_POST), 0.0)
    direct = direct.groupby(at_2025["asset_id"].map(_base)).sum()
    pd.testing.assert_series_equal(gbe.sort_index(), direct.reindex(gbe.index).sort_index(), check_names=False)
    by_id = dict(zip(at_2028["asset_id"], share))
    assert by_id["ar7"] == 1.0 and by_id["ar6"] == 0.0 and by_id["old_off"] == 0.0 and by_id["built"] == 0.0
    # The undated solar split at 2028: its part committed before 2028 is 3/4 GBE's (profile years 2025-2028
    # less a one-year lead: 2024 is before the Act).
    assert by_id["solar::pre_legislation"] == pytest.approx(0.75) and by_id["solar::new_build"] == 1.0
    # A share at a year later than the register's own is refused (rows would have to split again).
    with pytest.raises(ValueError, match="later year"):
        eligible_share_at(_apply(2025), 2028, policy_start_year=2025, model_year=2030, leads=LEADS)


def _selection_rows(frame, share):
    from types import SimpleNamespace

    n = len(frame)
    rows = SimpleNamespace(
        carrier=frame["carrier"].to_numpy(dtype=object), vintage=frame["finance_vintage"].to_numpy(dtype=object),
        capacity=frame["capacity_mw"].to_numpy(dtype=float), resolved=np.ones(n, dtype=bool),
        excluded=np.zeros(n, dtype=bool), pre_policy_cfd=frame["contract_vintage"].eq("pre_policy").to_numpy(),
        pre_2025_dpa=np.zeros(n, dtype=bool), gbe_eligible_share=share)
    rows.base_vintage = rows.vintage
    rows.cells = lambda: [(c, 2028) for c in rows.carrier]
    rows.order_keys = lambda: [f"{i:03d}" for i in range(n)]
    return rows


def test_gb_energy_counts_its_share_committed_from_2025_in_the_stake_and_the_local_part():
    at_2028 = _apply(2028)
    share = eligible_share_at(at_2028, 2025, policy_start_year=2025, model_year=2030, leads=LEADS)
    gbe = dict(cfs.LEVER_PARAMETER_DEFAULTS["gbe_announced"], local_capacity_mw=500.0)
    out = cfs.gbe_selection(_selection_rows(at_2028, share), gbe)
    by_id = dict(zip(at_2028["asset_id"], out["offshore"]))
    # The stake: AR7 whole and the modelled round's share, not AR6 nor the old fleet.
    assert by_id["ar7"] and not by_id["ar6"] and not by_id["old_off"]
    future = at_2028.loc[at_2028["asset_id"].map(_base).eq("future_off")]
    future_mw = float((future["capacity_mw"] * share[future.index]).sum())
    assert out["offshore_mw"] == pytest.approx(1000.0 + future_mw)
    # The local cohort in MW committed from 2025: solar 600 x 5/6, onshore 300, pipeline solar whole.
    assert out["local_cohort_mw"] == pytest.approx(600.0 * 5 / 6 + 300.0 + 250.0)
    # At the group's year (no GBE year) the same rows give the old, smaller cohort.
    old = cfs.gbe_selection(_selection_rows(at_2028, None), dict(gbe, local_capacity_mw=100.0))
    assert old["offshore_mw"] == 0.0 and old["local_cohort_mw"] < out["local_cohort_mw"]


@pytest.fixture(scope="module")
def registers():
    """The streaming fixture's fleet re-classed at 2028, with P2, GBE and its partnership twin, at both GBE years."""
    from config.config_loader import load_defaults
    from scripts.public_ownership.public_account import account_settings
    from tests.unit.test_public_ownership_streaming import NO_OVERHEAD, _inputs

    block = load_defaults()["public_ownership"]
    assets, costs, snapshot = _inputs()
    assets = assets.assign(cfd_award_year=np.where(assets["asset_id"].eq("ar7"), 2026.0, assets["cfd_award_year"]),
                           commissioning_year=np.where(assets["asset_id"].eq("solar"), np.nan, 2029.0),
                           development_status=None, vintage_method="asset_date",
                           finance_eligible=~assets["carrier"].eq("mystery"))
    assets["energy_capacity_mwh"] = assets.get("energy_capacity_mwh", 0.0)
    assets = apply_legislation_year(assets, legislated_year=2028, policy_start_year=2025, model_year=2030,
                                    transfer_year=2030, leads=LEADS)
    base = expand_axis_cases(["private", "public_ownership"], {}, bounding_pair=False)
    base = [s for s in base if s.case_id in ("private", "public_ownership")]
    p2 = next(s for s in base if s.case_id == "public_ownership")
    specs = base + [CaseSpec(f"public_ownership@{v}", "public_ownership", {**dict(p2.settings), "new_build_generation": v})
                    for v in ("gbe_announced", "gbe_announced_partnership_half")]
    out = {}
    for year in (None, 2025):
        params = {"gbe_announced": dict(cfs.LEVER_PARAMETER_DEFAULTS["gbe_announced"], local_capacity_mw=10.0,
                                        legislated_year=year)}
        out[year] = cfs.calculate_financing_scenarios(
            assets, costs, snapshot, price_year=2024, case_specs=specs, state_operating_overhead=NO_OVERHEAD,
            ldes_account_settings=account_settings(block), electricity_leg_carriers=["CCS Biomass"],
            lever_parameter_block=params)
    return out


def test_non_gbe_cases_are_unchanged_and_gbe_owns_its_pre_legislation_rows(registers):
    before, after = registers[None][0], registers[2025][0]
    common = [c for c in before.columns if c in after.columns]
    for case in ("private", "public_ownership"):
        a = before.loc[before["case_id"].eq(case), common].reset_index(drop=True)
        b = after.loc[after["case_id"].eq(case), common].reset_index(drop=True)
        pd.testing.assert_frame_equal(a.astype(object), b.astype(object), check_exact=True)
    assert "gbe_eligible_share" in after.columns and "gbe_eligible_share" not in before.columns
    gbe = after.loc[after["case_id"].eq("public_ownership@gbe_announced")].set_index("asset_id")
    # AR7, awarded 2026, is pre-legislation at 2028 and GB Energy's stake from 2025: new build, 30% state.
    ar7 = gbe.loc["ar7"]
    assert ar7["gbe_part"] == cfs.GBE_PART_STAKE and ar7["finance_vintage"] == "new_build"
    assert float(ar7["ownership_share"]) == pytest.approx(0.30) and bool(ar7["state_owned"])
    assert float(ar7["gbe_eligible_share"]) == 1.0
    # The private case keeps it legacy and private; no row is both GBE-owned and bought.
    private = after.loc[after["case_id"].eq("private")].set_index("asset_id")
    assert private.loc["ar7", "finance_vintage"] == "legacy" and float(private.loc["ar7", "ownership_share"]) == 0.0
    assert gbe["acquisition_price_gbp"].isna().all()
    owned = gbe.loc[gbe["gbe_part"].notna()]
    assert (owned["ownership_share"].astype(float) > 0).all() and owned["state_owned"].astype(bool).all()
    # The partnership twin holds its local rows at half the GBE share and the stake as GBE does.
    half = after.loc[after["case_id"].eq("public_ownership@gbe_announced_partnership_half")].set_index("asset_id")
    assert float(half.loc["ar7", "ownership_share"]) == pytest.approx(0.30)
    local = owned.index[owned["gbe_part"].eq(cfs.GBE_PART_LOCAL)]
    for asset in local:
        assert float(half.loc[asset, "ownership_share"]) == pytest.approx(
            0.5 * float(gbe.loc[asset, "ownership_share"]))


def test_gbe_owned_contracts_are_re_priced_and_the_routes_close(registers):
    from config.config_loader import load_defaults
    from scripts.public_ownership.cfd_repricing import (
        ROUTE_CFD_BILL, per_row_savings, repricing_settings, route_summary,
    )

    register, totals = registers[2025]
    settings = repricing_settings(load_defaults()["public_ownership"]["consumer_bill"])
    savings = per_row_savings(register, settings)
    ar7 = savings.loc[savings["asset_id"].eq("ar7")].set_index("case_id")
    assert ar7.loc["public_ownership@gbe_announced", "route"] == ROUTE_CFD_BILL
    assert ar7.loc["public_ownership", "route"] != ROUTE_CFD_BILL      # P2 leaves the pre-legislation AR7 private
    routes = route_summary(savings, totals)
    gbe = routes.loc[routes["case_id"].str.contains("gbe_announced")]
    assert gbe["identity_within_tolerance"].all()
