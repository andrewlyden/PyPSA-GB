"""F28 (d): the refurbishment rule and its per-technology windows (sources of 27 Oct 2026)."""

import pytest

from scripts.public_ownership.fleet_vintages import (
    parse_refurbishment, refurbishment_flag, refurbishment_years,
)

HYDRO = "Hydropower"
PHES = "Pumped hydro energy storage (PHES)"


def test_config_and_command_line_forms_round_trip_with_a_window():
    rule = parse_refurbishment({"technologies": [HYDRO, PHES], "first_year": 2026, "last_year": 2045,
                                "windows": {PHES: [2026, 2036]}})
    assert rule.window(HYDRO) == (2026, 2045) and rule.window(PHES) == (2026, 2036)
    flag = refurbishment_flag(rule)
    assert flag == f' --refurbishment "{HYDRO};{PHES}=2026:2045|{PHES}=2026:2036"'
    assert parse_refurbishment(flag.split('"')[1]) == rule


def test_without_windows_the_flag_is_unchanged():
    # Keeps the rule's params byte-identical for groups that set no window.
    rule = parse_refurbishment({"technologies": [HYDRO, PHES], "first_year": 2026, "last_year": 2045})
    assert refurbishment_flag(rule) == f' --refurbishment "{HYDRO};{PHES}=2026:2045"'
    assert rule.windows == ()


def test_schedule_follows_the_technology_window():
    rule = parse_refurbishment({"technologies": [HYDRO, PHES], "first_year": 2026, "last_year": 2045,
                                "windows": {PHES: [2026, 2036]}})
    cohorts = [(1963, 300.0), (1974, 440.0), (1983, 1728.0)]
    phes = refurbishment_years(cohorts, rule, PHES)
    hydro = refurbishment_years(cohorts, rule, HYDRO)
    assert max(phes.values()) <= 2036 and min(phes.values()) >= 2026
    assert max(hydro.values()) > 2036
    # Oldest first in both.
    assert list(phes.values()) == sorted(phes.values()) and list(hydro.values()) == sorted(hydro.values())
    assert refurbishment_years(cohorts, rule) == hydro


def test_a_window_must_name_a_refurbished_technology_and_run_forwards():
    with pytest.raises(ValueError, match="not a refurbished technology"):
        parse_refurbishment({"technologies": [HYDRO], "first_year": 2026, "last_year": 2045,
                             "windows": {PHES: [2026, 2036]}})
    with pytest.raises(ValueError, match="last year precedes"):
        parse_refurbishment(f"{HYDRO}=2026:2045|{HYDRO}=2040:2030")
