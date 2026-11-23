import pandas as pd

from scripts.consumer_bill.build_repd_roc_station_mapping import build_mapping


def test_mapping_accepts_unique_exact_name_and_audits_capacity_separately():
    repd = pd.DataFrame({
        "Ref ID": ["A", "B"], "Site Name": ["Example Wind Farm", "Different Wind Farm"],
        "Technology Type": ["Onshore Wind", "Onshore Wind"],
        "Installed Capacity (MWelec)": [10.0, 100.0], "Country": ["England", "England"],
        "Operational": ["01/01/2015", "01/01/2015"],
    })
    stations = pd.DataFrame({
        "Accreditation reference": ["R1"], "Station name": ["Example Wind Farm"],
        "Station DNC in kW": [10000], "Technology": ["Onshore wind"],
        "Commissioning date": ["01/01/2015"], "Effective date": ["01/01/2015"], "Country": ["England"],
    })

    result, summary = build_mapping(repd, stations, 2020)

    assert result.loc[result["repd_ref_id"].eq("A"), "mapping_status"].item() == "high_confidence_exact_name"
    assert result.loc[result["repd_ref_id"].eq("B"), "mapping_status"].item() == "review_required"
    assert summary["high_confidence_matches"] == 1


def test_mapping_includes_energy_from_waste_as_fuelled():
    repd = pd.DataFrame({
        "Ref ID": ["E1"], "Site Name": ["Example Energy Centre"],
        "Technology Type": ["EfW Incineration"],
        "Installed Capacity (MWelec)": [25.0], "Country": ["England"],
        "Operational": ["01/01/2015"],
    })
    stations = pd.DataFrame({
        "Accreditation reference": ["R1"], "Station name": ["Example Energy Centre"],
        "Station DNC in kW": [10000], "Technology": ["Fuelled"],
        "Commissioning date": ["01/01/2015"], "Effective date": ["01/01/2015"],
        "Country": ["England"],
    })

    result, _ = build_mapping(repd, stations, 2020)

    assert result["mapping_status"].item() == "high_confidence_exact_name"
    assert not result["capacity_agrees"].item()
