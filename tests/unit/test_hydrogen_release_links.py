"""hydrogen_links.release_links (19 Oct 2026): held-carrier links the balancing stage may move.

no_upgrades reverts the 203 MVA Skye line to today's 99 MVA one, and the 94 MW Ardmore electrolyser, held at its
Stage 1 schedule, then forced load shedding on Skye. A released link is priced like the other hydrogen links.
"""
from scripts.market.solve_balancing import resolve_held_hydrogen_carriers, resolve_released_hydrogen_links


def test_default_releases_nothing():
    assert resolve_released_hydrogen_links({}) == ()
    assert resolve_released_hydrogen_links({"hydrogen_links": {"hold_carriers": ["electrolysis"]}}) == ()


def test_release_list_is_read_as_link_names():
    cfg = {"hydrogen_links": {"hold_carriers": ["electrolysis", "H2_turbine"],
                              "release_links": ["electrolysis_ARMO3-"]}}
    assert resolve_released_hydrogen_links(cfg) == ("electrolysis_ARMO3-",)
    # The hold itself is unchanged: release works link by link, not by carrier.
    assert resolve_held_hydrogen_carriers(cfg) == ("electrolysis", "H2_turbine")
