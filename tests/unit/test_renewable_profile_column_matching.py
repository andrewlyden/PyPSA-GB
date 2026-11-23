"""Profile-column selection must not be decided by column order.

Regression for the 588 MW Beatrice farm silently receiving the 10 MW Beatrice
Demonstrator profile because the first substring hit won.
"""

from scripts.generators.integrate_renewable_generators import select_profile_column


def test_canonical_source_suffix_beats_a_longer_prefix_sibling():
    columns = ["Beatrice Demonstrator (REPD)", "Beatrice (REPD)"]
    assert select_profile_column(columns, "Beatrice") == ("Beatrice (REPD)", "canonical")
    assert select_profile_column(columns, "Beatrice Demonstrator") == (
        "Beatrice Demonstrator (REPD)", "canonical",
    )


def test_exact_column_name_wins_over_a_longer_containing_name():
    columns = ["Whitelee Windfarm Extension phase 2", "Whitelee"]
    assert select_profile_column(columns, "Whitelee") == ("Whitelee", "exact")


def test_substring_only_matches_are_still_reported_as_such():
    columns = ["Redcourt Farm"]
    column, kind = select_profile_column(columns, "Court Farm")
    assert (column, kind) == ("Redcourt Farm", "substring")


def test_ambiguous_matches_are_flagged_not_silently_ordered():
    columns = ["Manor Farm North", "Manor Farm South"]
    column, kind = select_profile_column(columns, "Manor Farm")
    assert kind == "substring_ambiguous"
    assert column in columns


def test_non_breaking_space_and_case_are_normalised():
    columns = ["Triton Knoll\xa0 (REPD)"]
    assert select_profile_column(columns, "triton  knoll")[0] == "Triton Knoll\xa0 (REPD)"


def test_missing_site_or_column_returns_no_match():
    assert select_profile_column(["Anything (REPD)"], "Nowhere") == (None, "no_match")
    assert select_profile_column(["Anything (REPD)"], "  ") == (None, "no_site_name")
