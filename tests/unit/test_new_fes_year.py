"""The new-year tool writes a study that matches the 2040 study made by hand (comments aside) (11 Nov 2026)."""

import shutil
from pathlib import Path

import pytest
import yaml

from scripts.sources import new_fes_year as n

ROOT = Path(__file__).resolve().parents[2]
NEEDS = [ROOT / f"resources/FES/fes2025_holistic_transition_2040_{k}" for k in ("targets.json", "es1_lines.csv")]


def _settings(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # A free-text source string differs in wording; everything the model reads must match.
    hp = (((data.get("ht40_foreign_and_hydrogen_prices") or {}).get("market") or {}).get("hydrogen_supply_price") or {})
    hp.pop("source", None)
    return data


@pytest.mark.skipif(not all(p.exists() for p in NEEDS), reason="needs the 2040 FES targets")
def test_the_generated_2040_study_matches_the_one_in_use(tmp_path, monkeypatch):
    (tmp_path / "resources/FES").mkdir(parents=True)
    for p in NEEDS:
        shutil.copy(p, tmp_path / "resources/FES" / p.name)
    monkeypatch.setattr(n, "ROOT", tmp_path)
    out = n.write(2040)
    for name in n.FILES:
        assert _settings(out / name) == _settings(ROOT / "config/studies/fes2040" / name), name


def test_convert_keeps_the_2050_references():
    text = "fes_ht_2050: support_regime_2050.json FES 2050 decision 65 FES25_HT_2050_{month}_market modelled_year: 2050"
    assert n.convert(text, 2045) == ("fes_ht_2045: support_regime_2050.json FES 2050 decision 65 "
                                     "FES25_HT_2045_{month}_market modelled_year: 2045")
