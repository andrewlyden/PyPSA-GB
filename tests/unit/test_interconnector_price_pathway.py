"""Foreign price rows by FES pathway (scripts/interconnectors/add_to_network._filter_price_differentials).

NESO's current ES2 table lists Holistic Transition and Electric Engagement separately; an earlier layout grouped
them in one label. The pathway's own rows win; the grouped label is used only where the table has no own rows.
"""

import pandas as pd
import pytest

from scripts.interconnectors.add_to_network import _filter_price_differentials


def _table(labels):
    return pd.DataFrame({"country": ["France"] * len(labels), "pathway": labels, "year": [2035] * len(labels),
                         "price_differential_gbp_per_mwh": range(len(labels))})


def test_own_pathway_rows_are_taken_when_the_table_lists_them():
    table = _table(["Holistic Transition", "Electric Engagement", "Counterfactual"])
    assert _filter_price_differentials(table, 2035, "Holistic Transition")["pathway"].tolist() == ["Holistic Transition"]


def test_grouped_label_is_used_only_without_own_rows():
    table = _table(["Holistic Transition, Electric Engagement", "Counterfactual"])
    picked = _filter_price_differentials(table, 2035, "Electric Engagement")
    assert picked["pathway"].tolist() == ["Holistic Transition, Electric Engagement"]


def test_unknown_pathway_is_refused():
    with pytest.raises(ValueError, match="No foreign price data"):
        _filter_price_differentials(_table(["Counterfactual"]), 2035, "Holistic Transition")
