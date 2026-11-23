import pandas as pd

import scripts.consumer_bill.build_lccc_cfd_mapping as lccc_mapping


def test_lccc_mapping_retains_unrepresented_contracts(monkeypatch):
    actuals = pd.DataFrame({
        "Settlement_Date": ["2020-01-01", "2020-01-01"],
        "CfD_ID": ["C1", "C2"],
        "Name_of_CfD_Unit": ["Mapped", "Gap"],
        "Technology": ["Offshore Wind", "Onshore Wind"],
        "Allocation_round": ["AR1", "AR1"],
        "CFD_Generation_MWh": [10.0, 5.0],
        "CFD_Payments_GBP": [100.0, 25.0],
    })
    portfolio = pd.DataFrame({
        "CFD_ID": ["C1", "C2"], "Status": ["Live", "Live"],
        "Maximum_Contract_Capacity_MW": [1.0, 1.0],
    })
    generators = pd.DataFrame({"name": ["asset"], "carrier": ["wind_offshore"], "p_nom": [1.0]})
    overrides = pd.DataFrame({
        "cfd_id": ["C1"], "model_generator": ["asset"], "allocation_share": [1.0],
        "mapping_method": ["test"], "notes": [""],
    })
    inputs = {"actuals": actuals, "portfolio": portfolio, "generators": generators, "overrides": overrides}

    def read_csv(name, *args, **kwargs):
        return inputs[str(name)].copy()

    monkeypatch.setattr(lccc_mapping.pd, "read_csv", read_csv)

    mapping, summary = lccc_mapping.build_mapping("actuals", "portfolio", "generators", "overrides", 2020)

    assert set(mapping["CfD_ID"]) == {"C1", "C2"}
    assert mapping.loc[mapping["CfD_ID"].eq("C2"), "mapping_status"].item() == "not_represented"
    assert summary["contract_register_complete"] is True
    assert summary["mapped_payment_coverage"] == 0.8


def test_lccc_mapping_tracks_a_source_candidate_pending_rebuild(monkeypatch):
    actuals = pd.DataFrame({
        "Settlement_Date": ["2020-01-01"], "CfD_ID": ["C1"],
        "Name_of_CfD_Unit": ["Future in-year asset"], "Technology": ["Offshore Wind"],
        "Allocation_round": ["AR1"], "CFD_Generation_MWh": [10.0], "CFD_Payments_GBP": [100.0],
    })
    portfolio = pd.DataFrame({"CFD_ID": ["C1"], "Status": ["Live"], "Maximum_Contract_Capacity_MW": [1.0]})
    generators = pd.DataFrame({"name": ["unrelated"], "carrier": ["wind_offshore"], "p_nom": [1.0]})
    overrides = pd.DataFrame(columns=["cfd_id", "model_generator", "allocation_share", "mapping_method", "notes"])
    candidates = pd.DataFrame({
        "cfd_id": ["C1"], "source_site_name": ["Future in-year asset"],
        "expected_model_generator": ["future_asset"], "readiness": ["profile_gate_required"],
    })
    inputs = {"actuals": actuals, "portfolio": portfolio, "generators": generators, "overrides": overrides, "candidates": candidates}

    def read_csv(name, *args, **kwargs):
        return inputs[str(name)].copy()

    monkeypatch.setattr(lccc_mapping.pd, "read_csv", read_csv)
    mapping, summary = lccc_mapping.build_mapping(
        "actuals", "portfolio", "generators", "overrides", 2020, source_candidates_path="candidates"
    )

    assert mapping["mapping_status"].item() == "source_mapped_pending_rebuild"
    assert summary["source_mapping_complete"] is True
    assert summary["model_mapping_complete"] is False


def test_lccc_mapping_auto_maps_an_unambiguous_active_bmu(monkeypatch):
    actuals = pd.DataFrame({
        "Settlement_Date": ["2024-01-01"], "CfD_ID": ["C1"],
        "Name_of_CfD_Unit": ["One"], "Technology": ["Offshore Wind"],
        "Allocation_round": ["AR1"], "CFD_Generation_MWh": [10.0], "CFD_Payments_GBP": [100.0],
    })
    portfolio = pd.DataFrame({"CFD_ID": ["C1"], "Status": ["Live"], "Maximum_Contract_Capacity_MW": [5.0]})
    generators = pd.DataFrame({"name": ["asset"], "carrier": ["wind_offshore"], "p_nom": [5.0]})
    overrides = pd.DataFrame(columns=["cfd_id", "model_generator", "allocation_share", "mapping_method", "notes"])
    cfd_bmus = pd.DataFrame({
        "CFD_Id": ["C1"], "BMU_Id": ["B1"], "Effective_From": ["2020-01-01"],
        "Effective_date_to": [None],
    })
    model_bmus = pd.DataFrame({"generator_name": ["asset"], "bmu_id": ["B1"]})
    inputs = {
        "actuals": actuals, "portfolio": portfolio, "generators": generators,
        "overrides": overrides, "cfd_bmus": cfd_bmus, "model_bmus": model_bmus,
    }
    monkeypatch.setattr(
        lccc_mapping.pd, "read_csv", lambda name, *args, **kwargs: inputs[str(name)].copy()
    )

    mapping, summary = lccc_mapping.build_mapping(
        "actuals", "portfolio", "generators", "overrides", 2024,
        cfd_bmu_path="cfd_bmus", model_bmu_path="model_bmus",
    )

    assert mapping["mapping_method"].item() == "automatic_active_lccc_bmu_to_unique_model_bmu"
    assert mapping["model_generator"].item() == "asset"
    assert summary["automatic_bmu_mapped_contracts"] == 1


def test_lccc_mapping_selects_period_and_prefers_explicit_tranche(monkeypatch):
    actuals = pd.DataFrame({
        "Settlement_Date": ["2020-01-10", "2020-02-10"],
        "CfD_ID": ["C1", "C2"], "Name_of_CfD_Unit": ["One", "Two"],
        "Technology": ["Offshore Wind", "Offshore Wind"],
        "Allocation_round": ["AR1", "AR1"],
        "CFD_Generation_MWh": [10.0, 20.0], "CFD_Payments_GBP": [100.0, 200.0],
    })
    portfolio = pd.DataFrame({
        "CFD_ID": ["C1", "C2"], "Status": ["Live", "Live"],
        "Maximum_Contract_Capacity_MW": [40.0, 60.0],
    })
    generators = pd.DataFrame({
        "name": ["asset", "asset__CfD__C1"], "carrier": ["wind_offshore"] * 2,
        "p_nom": [60.0, 40.0],
    })
    overrides = pd.DataFrame({
        "cfd_id": ["C1", "C2"], "model_generator": ["asset", "missing_feb_asset"],
        "allocation_share": [0.4, 1.0], "mapping_method": ["test", "test"],
        "notes": ["", ""],
    })
    inputs = {"actuals": actuals, "portfolio": portfolio, "generators": generators, "overrides": overrides}
    monkeypatch.setattr(
        lccc_mapping.pd, "read_csv", lambda name, *args, **kwargs: inputs[str(name)].copy()
    )
    mapping, summary = lccc_mapping.build_mapping(
        "actuals", "portfolio", "generators", "overrides", 2020,
        period_start="2020-01-01", period_end="2020-01-31 23:00",
    )
    assert mapping["CfD_ID"].tolist() == ["C1"]
    assert mapping["model_generator"].item() == "asset__CfD__C1"
    assert mapping["allocation_share"].item() == 1.0
    assert mapping["mapped_capacity_mw"].item() == 40.0
    assert summary["period_start"] == "2020-01-01"


def test_lccc_mapping_keeps_absent_override_as_unresolved_evidence(monkeypatch):
    actuals = pd.DataFrame({
        "Settlement_Date": ["2024-01-01"], "CfD_ID": ["C1"],
        "Name_of_CfD_Unit": ["New asset"], "Technology": ["Offshore Wind"],
        "Allocation_round": ["AR1"], "CFD_Generation_MWh": [10.0],
        "CFD_Payments_GBP": [100.0],
    })
    portfolio = pd.DataFrame({
        "CFD_ID": ["C1"], "Status": ["Live"], "Maximum_Contract_Capacity_MW": [5.0],
    })
    generators = pd.DataFrame({"name": ["old_asset"], "carrier": ["wind_offshore"], "p_nom": [5.0]})
    overrides = pd.DataFrame({
        "cfd_id": ["C1"], "model_generator": ["new_asset"], "allocation_share": [1.0],
        "mapping_method": ["reviewed_source"], "notes": ["requires model rebuild"],
    })
    inputs = {"actuals": actuals, "portfolio": portfolio, "generators": generators, "overrides": overrides}
    monkeypatch.setattr(
        lccc_mapping.pd, "read_csv", lambda name, *args, **kwargs: inputs[str(name)].copy()
    )

    mapping, summary = lccc_mapping.build_mapping(
        "actuals", "portfolio", "generators", "overrides", 2024,
    )

    assert mapping["mapping_status"].item() == "override_generator_absent"
    assert mapping["allocation_share"].item() == 0.0
    assert mapping["mapped_capacity_mw"].item() == 0.0
    assert summary["model_mapping_complete"] is False
    assert summary["absent_override_generators"] == ["new_asset"]
