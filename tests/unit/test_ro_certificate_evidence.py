"""RO line: observed bandings and the fuelled-station activity test (item MR8)."""

import pandas as pd
import pytest

from scripts.consumer_bill.build_ro_eligible_fleet import read_certificate_evidence

COLUMNS = ["Accreditation reference", "Scheme", "Compliance Period", "MWh per certificate"]


def _certificates(tmp_path, rows):
    path = tmp_path / "certificates.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def test_banding_is_the_reciprocal_in_the_latest_issued_year(tmp_path):
    path = _certificates(tmp_path, [
        ["A", "RO", "2019/2020", 1.0],        # older banding, superseded
        ["A", "RO", "2024/2025", 1 / 1.5],
        ["A", "RO", "2024/2025", 1 / 1.5],
        ["B", "ROS", "2021/2022", 0.5],       # Scotland issues under ROS
        ["C", "NIRO", "2025/2026", 1.0],      # Northern Ireland is excluded
    ])
    evidence = read_certificate_evidence(path)
    assert evidence.loc["A", "observed_banding"] == pytest.approx(1.5)
    assert evidence.loc["A", "last_certificate_year"] == 2024
    assert evidence.loc["B", "observed_banding"] == pytest.approx(2.0)
    assert evidence.loc["B", "last_certificate_year"] == 2021
    assert "C" not in evidence.index


@pytest.mark.skipif(
    not pd.io.common.file_exists("data/market/ofgem_renewables_obligation/certificates_ro_current.csv"),
    reason="Ofgem certificate register not present",
)
def test_fawley_has_no_certificates_and_fails_the_activity_test():
    register = pd.read_csv("data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv")
    fawley = register.loc[register["Station name"].eq("Fawley Power Station"), "Accreditation reference"]
    assert len(fawley) == 1
    evidence = read_certificate_evidence("data/market/ofgem_renewables_obligation/certificates_ro_current.csv")
    assert fawley.iloc[0] not in evidence.index
