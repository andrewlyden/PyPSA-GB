"""Fleet continuity between the acquisition year and a later year (user, 14 Nov 2026)."""

import pandas as pd

from scripts.public_ownership import fleet_continuity as fc

CASE = fc.BUYING_CASE


def _register(rows):
    out = []
    for row in rows:
        a, c, v, o, mw = row[:5]
        year = row[5] if len(row) > 5 else 2020
        contract = row[6] if len(row) > 6 else None
        out.append({"case_id": CASE, "asset_id": a, "carrier": c, "finance_vintage": v, "state_owned": o,
                    "capacity_mw": mw, "ownership_exclusion": "", "build_year": year, "cfd_round": contract})
    return pd.DataFrame(out)


def test_owning_more_than_was_bought_or_leaving_bought_plant_private_fails():
    reg = _register([("a", "wind_offshore", "legacy", True, 100.0), ("b", "wind_offshore", "legacy", True, 100.0),
                     ("c", "wind_offshore", "legacy", False, 100.0),          # built 2020: bought, left private
                     ("d", "wind_offshore", "in_construction", False, 300.0, 2033, "PAR7_2033"),   # never bought
                     ("g", "CCS Gas", "legacy", True, 300.0), ("e", "waste_to_energy", "legacy", False, 50.0),
                     ("s", "nuclear", "in_construction", True, 281.0, 2038, "SZC_RAB")])
    caps = {"offshore_wind": {"none": 300.0}, "gas_ccs": {"none": 100.0}, "nuclear": {"HPC": 3260.0}}
    table = fc.continuity_table(reg, {"offshore_wind": 300.0, "gas_ccs": 100.0, "energy_from_waste": 50.0,
                                      "nuclear": 3260.0}, round_caps=caps)
    problems = fc.failures(table)
    assert any("offshore_wind: leaves 100" in p for p in problems)        # bought 300, eligible 300, owns 200
    assert any("gas_ccs: owns 300" in p for p in problems)                # owns 300 against 100 bought
    assert any("nuclear: owns 281 MW the purchase did not buy" in p for p in problems)   # Sizewell C
    assert not any("PAR7" in p or "energy_from_waste" in p for p in problems)


def test_a_consistent_year_passes_and_reports_bought_plant_its_fleet_no_longer_runs():
    reg = _register([("a", "Battery", "legacy", True, 100.0)])
    valuation = pd.DataFrame({"carrier": ["Battery", "Battery"], "capacity_mw": [100.0, 300.0],
                              "retirement_year": [2045, 2041]})
    table = fc.continuity_table(reg, {"battery": 400.0}, valuation=valuation, model_year=2040).set_index("group")
    assert fc.failures(table.reset_index()) == []
    assert table.loc["battery", "bought_not_running_mw"] == 300.0
    assert table.loc["battery", "repaid_not_running_mw"] == 300.0         # still repaid in 2040, not running


def test_owning_plant_on_a_contract_the_purchase_did_not_buy_fails():
    """15 Nov 2026: Sizewell C and post-AR7 offshore were owned in 2040/2050 though never bought; Hinkley above 2,317."""
    import pandas as pd
    from scripts.public_ownership import fleet_continuity as fc

    register = pd.DataFrame({
        "case_id": fc.BUYING_CASE, "asset_id": ["hpc1", "hpc2", "szc", "off"],
        "carrier": ["nuclear", "nuclear", "nuclear", "wind_offshore"],
        "finance_vintage": ["in_construction"] * 4, "state_owned": [True, True, True, False],
        "capacity_mw": [1630.0, 1630.0, 281.0, 300.0], "ownership_exclusion": None,
        "cfd_round": ["HPC", "HPC", "SZC_RAB", "PAR7_2031"]})
    caps = {"nuclear": {"HPC": 2317.0, "none": 1198.0}, "offshore_wind": {"AR7": 8445.0}}
    problems = fc.contract_failures(fc.contract_ownership(register, caps))
    assert any("SZC_RAB" in p and "did not buy" in p for p in problems)
    assert any("HPC" in p and "2,317" in p for p in problems)
    assert not any("PAR7" in p for p in problems)                      # unowned: fine


def test_a_contract_at_its_cap_does_not_fail_a_correct_year():
    """Independent review (15 Nov 2026): a private row on a contract already owned up to what the purchase bought of
    it is not 'bought plant left private', even while the group as a whole has headroom."""
    reg = _register([("a", "wind_offshore", "legacy", True, 400.0, 2025, "AR4"),
                     ("b", "wind_offshore", "legacy", False, 300.0, 2026, "AR4"),     # AR4 bought 400: at its cap
                     ("c", "wind_offshore", "legacy", True, 200.0, 2015, None)])
    caps = {"offshore_wind": {"AR4": 400.0, "none": 600.0}}
    table = fc.continuity_table(reg, {"offshore_wind": 1000.0}, round_caps=caps)
    assert fc.failures(table) == []
    # Without the contract caps the same row would be counted as ownable.
    assert fc.failures(fc.continuity_table(reg, {"offshore_wind": 1000.0})) != []
