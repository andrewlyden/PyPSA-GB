"""The explorer labels every lever, value and named case it can be handed.

The data builder indexes its label tables directly, so a lever or named
sensitivity added to the config without its labels fails only at regeneration
time, after the solves. Item G3's lever L19 would have.
"""

from pathlib import Path

import yaml

import scripts.analysis.build_ownership_explorer_data as explorer
from scripts.public_ownership.levers import LEVERS
from config.config_loader import load_defaults


def _named():
    config = load_defaults()
    return config["public_ownership"]["sensitivities"]["named"]


def test_every_lever_and_value_has_a_label_and_a_guide():
    for name, lever in LEVERS.items():
        assert name in explorer.LEVER_LABELS, name
        assert name in explorer.LEVER_GUIDES, name
        for value in lever.values:
            assert value in explorer.VALUE_LABELS.get(name, {}), (name, value)


def test_every_named_sensitivity_has_a_label_and_a_guide():
    for name in _named():
        assert name in explorer.NAMED_LABELS, name
        assert name in explorer.NAMED_GUIDES, name


def _all_text():
    import json
    return json.dumps([explorer.SCENARIOS, explorer.LEVER_GUIDES, explorer.NAMED_GUIDES, explorer.NAMED_LABELS,
                       explorer.VALUE_LABELS, explorer.PROPOSITION_LABELS], ensure_ascii=False)


def test_no_stale_presentation_text():
    """Audit 26 Sep 2026: stale numbers and claims the later work overturned."""
    text = _all_text()
    assert "959" not in text                        # the pre-L2 private bill
    assert "can only understate" not in text        # BM1 measured the balancing channel
    assert "CP.29" not in text                      # the flat foreign price, replaced by IC1
    # CfD terms come from config (the acquisition term sensitivity is a genuine 15 years).
    assert "15-year contract" not in text and "15-year CfD" not in text
    template = Path("scripts/analysis/ownership_explorer_template.html").read_text(encoding="utf-8")
    assert "15-year contract term" not in template and "year 15" not in template


def test_the_bm1_figure_is_filled_from_the_merge_never_typed():
    from copy import deepcopy
    saved = deepcopy(explorer.SCENARIOS)
    try:
        assert "__BM1_BID_STACK__" in explorer.SCENARIOS["public_control"]["guide"]["held_fixed"]
        explorer._fill_bm1("-0.47 GBP inc VAT")
        assert "__BM1_BID_STACK__" not in _all_text()
        assert "-0.47 GBP inc VAT" in explorer.SCENARIOS["public_control"]["guide"]["held_fixed"]
    finally:
        explorer.SCENARIOS.clear()
        explorer.SCENARIOS.update(saved)
