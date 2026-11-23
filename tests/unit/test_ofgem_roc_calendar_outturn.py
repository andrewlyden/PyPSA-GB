import pandas as pd

from scripts.consumer_bill.extract_ofgem_roc_calendar_outturn import extract_calendar_outturn


def test_calendar_outturn_filters_output_period_and_excludes_ni(monkeypatch):
    source = pd.DataFrame({
        "Accreditation reference": ["R1", "R1", "N1"], "Station name": ["Station", "Station", "NI"],
        "Scheme": ["RO", "RO", "NIRO"], "Country": ["England", "England", "Northern Ireland"],
        "Technology": ["Onshore wind", "Onshore wind", "Onshore wind"],
        "Output period": ["Jan 2020", "Jan 2021", "Jan 2020"], "Number of certificates": [20, 10, 20],
        "MWh per certificate": [0.5, 0.5, 0.5], "Start certificate number": ["a", "b", "c"], "End certificate number": ["z", "y", "x"],
    })
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: iter([source.copy()]) if kwargs.get("chunksize") else source.copy())

    stations, technology, summary = extract_calendar_outturn("source.csv", 2020)

    assert stations["ofgem_roc_certificates"].item() == 20
    assert technology["ofgem_eligible_generation_mwh"].item() == 10
    assert summary["calendar_year"] == 2020


def test_calendar_outturn_can_select_one_model_month(monkeypatch):
    source = pd.DataFrame({
        "Accreditation reference": ["R1", "R1"], "Station name": ["Station", "Station"],
        "Scheme": ["RO", "RO"], "Country": ["England", "England"],
        "Technology": ["Onshore wind", "Onshore wind"],
        "Output period": ["Jan 2020", "Feb 2020"], "Number of certificates": [10, 20],
        "MWh per certificate": [1.0, 1.0], "Start certificate number": ["a", "b"],
        "End certificate number": ["c", "d"],
    })
    monkeypatch.setattr(
        pd, "read_csv",
        lambda *args, **kwargs: iter([source.copy()]) if kwargs.get("chunksize") else source.copy(),
    )
    stations, _, summary = extract_calendar_outturn(
        "source.csv", 2020, "2020-01-01", "2020-01-31 23:00"
    )
    assert stations["ofgem_roc_certificates"].sum() == 10
    assert summary["period_start_month"] == "2020-01"
    assert summary["period_end_month"] == "2020-01"
