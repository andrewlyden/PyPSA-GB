"""The post-solve steps of a long-range year (user, 11 Nov 2026): the labels must carry the installed calibration."""
import pandas as pd

from scripts.public_ownership import post_solve_year as ps


def _labels(strike):
    return pd.DataFrame({"support_type": ["CfD", "cfd", "merchant", "CfD"],
                         "strike_key": ["PAR7_2031_onshore", "PAR7_2031_onshore", None, "AR4_OFFSHORE_2026_27#1"],
                         "strike_gbp_per_mwh_2024": [strike, strike, None, 50.0]})


CALIBRATION = pd.DataFrame({"scheme": ["CfD", "DPA"], "key": ["PAR7_2031_onshore", "DPA_2035_CCS Gas_"],
                            "calibrated_value": [103.1, 200.0]})


def test_labels_on_the_installed_calibration_pass():
    assert ps.labels_match_calibration(_labels(103.1), CALIBRATION).empty


def test_labels_on_a_stale_calibration_are_caught():
    bad = ps.labels_match_calibration(_labels(90.6), CALIBRATION)
    assert list(bad.index) == ["PAR7_2031_onshore"]
    assert bad["gap"].iloc[0] > 12


def test_paths_follow_the_year():
    p = ps.paths(2045)
    assert p["calibration"].endswith("strike_calibration_2045.csv") and p["group"] == "fes_ht_2045"
    assert p["labels"] == "resources/market/FES25_HT_2045_support_labels.csv"
