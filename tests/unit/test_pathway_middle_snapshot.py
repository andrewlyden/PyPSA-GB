"""The 2030-2050 pathway with a solved year between the ends (2040 when it is run; 10 Nov 2026).

A middle snapshot is one more anchor: every interpolated line passes through it, and each
anchor year's pathway reproduces that year's ledger by mechanism and party.
"""
import pandas as pd
import pytest

from scripts.public_ownership import pathway_2030_2050 as pw


def _mechanisms(financing: float, legacy: float, market: float) -> dict:
    """A snapshot's three mechanisms, each balanced: consumers gain what the resource saving is."""
    def parties(value):
        return {"consumers": value, "exchequer": 0.0, "investors": 0.0, "resource": value}
    return {"financing": parties(financing), "legacy": parties(legacy), "market_and_other": parties(market)}


def _financing(years) -> pd.DataFrame:
    return pd.DataFrame({"year": list(years), "tech_group": "solar",
                         "capital_saving_gbp": [100.0 + 10.0 * (y - 2030) for y in years],
                         "tax_forgone_gbp": 0.0, "overhead_gbp": 0.0})


def test_parse_snapshots_takes_middle_years_only():
    assert pw.parse_snapshots(["2040=ledger_2040.json"]) == {2040: "ledger_2040.json"}
    with pytest.raises(pw.PathwayError):
        pw.parse_snapshots(["2050=x.json"])
    with pytest.raises(pw.PathwayError):
        pw.parse_snapshots(["2040"])


def test_a_middle_snapshot_is_an_anchor_the_lines_pass_through():
    snapshots = {2030: _mechanisms(100.0, 0.0, 10.0), 2040: _mechanisms(250.0, 0.0, 400.0),
                 2050: _mechanisms(300.0, 0.0, 50.0)}
    frame = pw.assemble_case("case", snapshots=snapshots, financing=_financing(pw.YEARS))
    checks = pw.anchor_check(frame, {"case": snapshots}, tolerance=1e-6)
    assert checks["passes"].all()
    market = frame.loc[frame["line"].eq("market_and_other_interpolated") & frame["party"].eq("consumers")]
    by_year = market.set_index("year")["gbp"]
    # Piecewise linear through 2040: up to 400 by 2040, then down to 50 by 2050.
    assert by_year[2035] == pytest.approx(205.0)
    assert by_year[2040] == pytest.approx(400.0)
    assert by_year[2045] == pytest.approx(225.0)
    # Without the middle year the line would be straight from 10 to 50.
    two = pw.assemble_case("case", snapshots={2030: snapshots[2030], 2050: snapshots[2050]},
                           financing=_financing(pw.YEARS))
    straight = two.loc[two["line"].eq("market_and_other_interpolated") & two["party"].eq("consumers")]
    assert straight.set_index("year")["gbp"][2040] == pytest.approx(30.0)
