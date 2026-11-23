"""OW11: the lever spreadsheet and the case-settings manifest, generated from the case catalogue."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.public_ownership import lever_workbook as lw
from scripts.public_ownership.levers import LEVERS, RETIRED_VALUES


def _root(tmp_path):
    root = tmp_path / "repo"
    (root / "scripts" / "public_ownership").mkdir(parents=True)
    (root / "data" / "public_ownership" / "financing").mkdir(parents=True)
    (root / "scripts" / "public_ownership" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "data" / "public_ownership" / "financing" / "t.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return root


def _config(tmp_path, block):
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps({"public_ownership": block, "consumer_bill": {"forecasts": {"cases": {}}}}), encoding="utf-8")
    return path


class TestManifest:
    def test_it_records_the_settings_the_config_and_the_sources_and_goes_stale_on_change(self, tmp_path):
        root = _root(tmp_path)
        settings = tmp_path / "g_case_settings.csv"
        pd.DataFrame({"case_id": ["private", "p2"], "networks": ["private", "private"]}).to_csv(settings, index=False)
        config_path = _config(tmp_path, {"price_year": 2024})
        manifest = lw.build_manifest(settings, config_path, "g", root)
        assert manifest["case_settings"]["cases"] == 2
        assert set(manifest["sources"]) == {"scripts/public_ownership/a.py", "data/public_ownership/financing/t.csv"}
        config = lw.load_config(config_path)
        assert lw.manifest_status(manifest, case_settings=settings, config=config, root=root, run_group="g") == {
            "current": True, "reasons": []}
        # A source edit, a config edit and new case settings each make it stale, and say why.
        (root / "scripts" / "public_ownership" / "a.py").write_text("x = 2\n", encoding="utf-8")
        status = lw.manifest_status(manifest, case_settings=settings, config={"public_ownership": {"price_year": 2025}},
                                    root=root, run_group="g")
        assert not status["current"]
        assert any("a.py" in r for r in status["reasons"]) and any("public_ownership" in r for r in status["reasons"])
        assert lw.manifest_status(None, case_settings=None, config=config, root=root, run_group="g")["current"] is False


class TestWorkbook:
    @pytest.fixture(scope="class")
    def built(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("levers")
        config = Path("config")
        po = lw.load_config(config)["public_ownership"]
        # The CENTRAL network group carries every case (cp30_2030_cp30_network is headline-only since 18-19 Oct 2026).
        specs = lw.configured_case_specs(po, "cp30_2030_no_acceleration")
        # Saved numbers for two cases: private closes the identity, P2 is off by GBP 5.
        totals = pd.DataFrame({"case_id": ["private", "public_ownership"], "household_total_gbp_inc_vat": [1000.0, 990.0],
                               "consumer_change_vs_private_gbp": [0.0, 10.0], "exchequer_net_vs_private_gbp": [0.0, 0.0],
                               "investor_change_vs_private_gbp": [0.0, 0.0], "resource_saving_vs_private_gbp": [0.0, 5.0]})
        totals_path = tmp / "totals.csv"
        totals.to_csv(totals_path, index=False)
        settings = pd.DataFrame([{"case_id": s.case_id, **{k: s.settings[k] for k in LEVERS}} for s in specs[:3]])
        settings_path = tmp / "cp30_2030_no_acceleration_case_settings.csv"
        settings.to_csv(settings_path, index=False)
        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(json.dumps(lw.build_manifest(settings_path, config, "cp30_2030_no_acceleration", Path("."))),
                                 encoding="utf-8")
        out = tmp / "levers.xlsx"
        counts = lw.build_workbook(config_path=config, run_group="cp30_2030_no_acceleration", case_settings=settings_path,
                                   manifest_path=manifest_path, household_totals=totals_path, root=Path("."), out=out)
        return counts, pd.read_excel(out, sheet_name=None), len(specs)

    def test_one_row_per_value_with_the_five_status_columns(self, built):
        counts, sheets, n_cases = built
        levers = sheets["Levers"]
        expected = sum(len(lever.values) for lever in LEVERS.values()) + len(RETIRED_VALUES)
        assert len(levers) == expected == counts["lever_values"]
        for column in ("Implemented", "Configured (cases)", "Saved (cases)", "Current (cases)", "Reconciled (cases)"):
            assert column in levers.columns
        retired = levers.loc[levers["Implemented"].astype(str).str.startswith("retired")]
        assert set(zip(retired["Lever (config)"], retired["Value (config)"])) == set(RETIRED_VALUES)
        # OW9: every implemented value is taken by at least one configured case, except the premium value the meeting
        # notes (user, 10 Nov 2026) leave unrun: social_beta_20 (v2.0's central). The 1% flat premium (HM Treasury's
        # adopted rate) has its own named cases since 13 Nov 2026 (the risk carried by the Treasury).
        assert counts["values_not_configured"] == 1
        not_configured = levers.loc[levers["Status"].eq("not configured"), ["Lever (config)", "Value (config)"]]
        assert set(map(tuple, not_configured.to_numpy())) == {("systematic_premium", "social_beta_20")}
        assert len(sheets["Cases"]) == n_cases == counts["cases"]

    def test_saved_current_and_reconciled_are_counted_per_case(self, built):
        counts, sheets, _ = built
        cases = sheets["Cases"].set_index("case_id")
        assert counts["saved"] == 2 and bool(cases.loc["private", "saved"]) and bool(cases.loc["public_ownership", "saved"])
        # The two saved cases are among the three whose settings were written: both current.
        assert counts["current"] == 2
        # P2's columns miss the identity by GBP 5: saved, not reconciled.
        assert counts["reconciled"] == 1 and not bool(cases.loc["public_ownership", "reconciled"])
        p2 = sheets["Levers"].query("`Lever (config)` == 'new_build_generation' and `Value (config)` == 'public'").iloc[0]
        assert p2["Saved (cases)"] == 1 and p2["Reconciled (cases)"] == 0 and p2["Status"] == "current, identity not closed"
