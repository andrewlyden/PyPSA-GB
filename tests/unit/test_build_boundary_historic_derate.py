"""Boundary outage tables from NESO day-ahead limits (scripts/solve/build_boundary_historic_derate.py)."""
import pandas as pd
import pytest

import scripts.solve.build_boundary_historic_derate as b


def _inputs(tmp_path):
    t = pd.date_range("2024-10-01", "2025-09-30 23:30", freq="30min")
    limit = pd.Series(1000.0, index=t)
    limit[(t.month == 7) & (t.day <= 10)] = 500.0          # ten July outage days
    src = pd.DataFrame({"Constraint Group": "SCOTEX", "Date (GMT/BST)": t.strftime("%Y-%m-%dT%H:%M:%S"),
                        "Limit (MW)": limit.to_numpy()})
    src.to_csv(tmp_path / "limits.csv", index=False)
    pd.DataFrame({"source_sheet": ["NA1"], "Equivalent ETYS name": ["B6"],
                  "Capacity Used 2024/25 (outage days)": [200.0], "Capacity Used 2029/30 (outage days)": [100.0],
                  "Capacity Used 2030/31(outage days)": [100.0]}).to_csv(tmp_path / "na1.csv", index=False)
    pd.DataFrame({"boundary": ["B6"], "fes_pathway": ["HT"], "year": [2030], "capability_mw": [1.0],
                  "limit_type": ["thermal"]}).to_csv(tmp_path / "cap.csv", index=False)
    return tmp_path


def test_generic_outages_spread_each_month_and_feed_the_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "NA1_GROUPS", {"B6": ["SCOTEX"]})
    d = _inputs(tmp_path)
    t = b.build(d / "limits.csv", "2024-10-01", "2025-09-30", 0.9, 0.25, d / "na1.csv", 2030,
                method="outage_generic", capability_file=d / "cap.csv")
    b6 = t[t.boundary == "B6"].groupby("month").factor
    assert b6.nunique().eq(1).all()                                    # one factor per month
    assert b6.first()[1] == pytest.approx(1.0)
    assert b6.first()[7] == pytest.approx(0.80 * (1 - 10 / 31 * (1 - 0.70 / 0.80)), abs=1e-4)
    thermal = t[t.boundary == "_default_thermal"].groupby("month").factor.first()
    assert thermal[7] == pytest.approx(b6.first()[7], abs=1e-4)        # the only thermal boundary
    assert t.na1_ratio[t.boundary == "B6"].iloc[0] == pytest.approx(0.5)


def test_ratio_method_is_the_limit_over_its_winter_level(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "NA1_GROUPS", {"B6": ["SCOTEX"]})
    d = _inputs(tmp_path)
    t = b.build(d / "limits.csv", "2024-10-01", "2025-09-30", 0.9, 0.25, d / "na1.csv", 2030)
    july = t[(t.month == 7)]
    assert july[july.day <= 10].factor.eq(0.5).all() and july[july.day > 10].factor.eq(1.0).all()
