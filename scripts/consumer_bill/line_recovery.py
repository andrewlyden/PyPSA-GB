"""Per-line recovery: how one household line converts to and from its system cost (F18, 6 Oct 2026).

Every bill line is a system cost spread over the volume it is recovered on. Since F18
the bill has two recovery volumes (metered supply and the GSP charging volume, each
with its eligible share) and Ofgem's household loss factor on the lines charged at a
GSP rate, so household and system pounds can no longer be converted with one
denominator. The bill summary therefore carries one entry per line
(``line_recovery``), and every conversion goes through it:

    household = system_gbp x price_year_factor x household_loss_factor
                x household_allocation_factor / volume_mwh x consumption_kwh / 1000

``system_gbp`` is in ``system_price_year`` money and ``price_year_factor`` carries it
to the bill's report year. The projector writes every entry in the model price year
(2024), the money the public-ownership ledgers keep, so the system pounds of any two
lines can be added. ``volume_key`` names the recovery volume of the bill's
``recovery_denominators`` the line is spread over; it is an open name (the P3 tariff
adds pool-specific volumes), checked against the volumes the recovery output carries.

Lines the ownership projector adds (``project_public_ownership_consumer_bill.py``)
carry a copy of the entry of the line they adjust, declared once in
:data:`INHERITED_LINE_RECOVERY`; the summaries carry those copies explicitly, so a
reader never needs the inheritance.
"""

from __future__ import annotations

import math

#: The fields of one entry, in the order the summary writes them.
ENTRY_FIELDS = ("volume_key", "volume_mwh", "household_loss_factor", "household_allocation_factor",
                "price_year_factor", "system_price_year", "basis")

#: The volume keys the bill's own lines use (F18). A recovery output may carry more.
BILL_VOLUME_KEYS = ("metered_supply_mwh", "gsp_charging_volume_mwh", "cfd_eligible_gsp_volume_mwh",
                    "ro_eligible_metered_supply_mwh")

#: Lines written only by the ownership projector -> the bill line whose entry they copy
#: (agreed with the public-ownership session, 6 Oct 2026). The surplus, levy and contract
#: routes are recovered like the CfD (eligible GSP volume at the household loss factor);
#: an adjustment to an existing line is recovered like that line.
INHERITED_LINE_RECOVERY = {
    "sizewell_c_rab_public_rate_adjustment": "sizewell_c_rab",
    "capacity_market_public_financing_adjustment": "capacity_market",
    "allocated_ccs_public_financing_adjustment": "allocated_ccs",
    # F27 item 1 (24 Oct 2026): Net Zero Teesside's DPA re-priced to the state's cost.
    "remaining_ccs_public_financing_adjustment": "remaining_ccs",
    "tnuos_public_financing_adjustment": "tnuos",
    "duos_public_financing_adjustment": "duos",
    "new_build_contracts_at_cost": "cfd",
    # F19 (7 Oct 2026): both are built over metered supply and move the wholesale line (the
    # household's energy cost), so they take its entry, not the CfD's.
    "pay_as_bid_ceiling": "direct_energy",
    "public_cfd_at_cost": "cfd",
    "public_cost_recovery_contracting": "direct_energy",
    "public_surplus_rebate": "cfd",
    "public_surplus_offset_domestic_levies": "cfd",
    "public_portfolio_shortfall_levy": "cfd",
    "public_state_deficit_levy": "cfd",
    "beccs_carbon_removal_levy": "cfd",
    "legacy_levy_ro_to_taxation": "renewables_obligation",
    "legacy_levy_fit_to_taxation": "feed_in_tariffs",
    "legacy_levy_whd_to_taxation": "warm_home_discount",
    # F19 (7 Oct 2026): the battery top-up at each owner's cost, and the Capacity Market moved
    # by the re-cleared 2030/31 T-4 (a share of the CM line).
    "battery_revenue_top_up_public_financing_adjustment": "battery_revenue_top_up",
    "capacity_market_state_bids_at_cost": "capacity_market",
    # F27 item 6 (24 Oct 2026): existing capacity on cost-based contracts, a cut to the CM line.
    "capacity_market_cost_based_existing": "capacity_market",
    # FES 2050 decision 73: the Capacity Market design under ownership, a move on the CM line (2050 only).
    "capacity_market_design_adjustment": "capacity_market",
}


def _positive(value, name) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"line recovery {name} must be finite and positive, got {value!r}")
    return value


def make_entry(volume_key, volume_mwh, *, household_loss_factor=1.0, household_allocation_factor=1.0,
               price_year_factor=1.0, system_price_year, basis) -> dict:
    """One line's recovery entry, validated."""
    if not volume_key or not isinstance(volume_key, str):
        raise ValueError(f"line recovery needs a volume_key, got {volume_key!r}")
    return {
        "volume_key": volume_key,
        "volume_mwh": _positive(volume_mwh, f"{volume_key} volume"),
        "household_loss_factor": _positive(household_loss_factor, "household_loss_factor"),
        "household_allocation_factor": _positive(household_allocation_factor, "household_allocation_factor"),
        "price_year_factor": _positive(price_year_factor, "price_year_factor"),
        "system_price_year": int(system_price_year),
        "basis": str(basis),
    }


def _gbp_per_system_gbp(entry, consumption_kwh) -> float:
    missing = [field for field in ENTRY_FIELDS if field not in entry]
    if missing:
        raise ValueError(f"line recovery entry lacks {missing}")
    return (float(entry["price_year_factor"]) * float(entry["household_loss_factor"])
            * float(entry["household_allocation_factor"]) / _positive(entry["volume_mwh"], "volume_mwh")
            * float(consumption_kwh) / 1000.0)


def system_to_household_gbp(entry, system_gbp, consumption_kwh=3100.0) -> float:
    """A system cost (``entry['system_price_year']`` money) as one household's bill line (report money)."""
    return float(system_gbp) * _gbp_per_system_gbp(entry, consumption_kwh)


def household_to_system_gbp(entry, gbp_per_household, consumption_kwh=3100.0) -> float:
    """One household's bill line (report money) as the system cost it recovers (``system_price_year`` money)."""
    return float(gbp_per_household) / _gbp_per_system_gbp(entry, consumption_kwh)


def entry_for(summary_or_line_recovery, component_id) -> dict:
    """The entry of ``component_id`` from a bill summary or its ``line_recovery`` block.

    A plain lookup: the summaries carry an explicit entry for every line, inherited
    ones included. Raises ``KeyError`` naming the line when it has none.
    """
    block = summary_or_line_recovery
    if isinstance(block, dict) and "line_recovery" in block:
        block = block["line_recovery"]
    if not isinstance(block, dict) or component_id not in block:
        raise KeyError(f"no line_recovery entry for bill line {component_id!r}")
    return block[component_id]


def inherited_entry(line_recovery, component_id) -> dict:
    """A copy of the entry of the line ``component_id`` adjusts (:data:`INHERITED_LINE_RECOVERY`)."""
    parent = INHERITED_LINE_RECOVERY.get(component_id)
    if parent is None:
        raise KeyError(f"bill line {component_id!r} has no recovery of its own and is not declared in "
                       "INHERITED_LINE_RECOVERY (scripts/consumer_bill/line_recovery.py)")
    if parent not in line_recovery:
        raise KeyError(f"bill line {component_id!r} inherits the recovery of {parent!r}, which the bill does not carry")
    entry = dict(line_recovery[parent])
    entry["basis"] = f"inherited from {parent} (INHERITED_LINE_RECOVERY): {line_recovery[parent]['basis']}"
    return entry


def check_volumes(line_recovery, volumes, rtol=1e-9) -> None:
    """Refuse an entry whose volume is not the recovery output's volume of the same name."""
    for component_id, entry in line_recovery.items():
        key = entry["volume_key"]
        if key not in volumes or volumes[key] is None:
            raise ValueError(f"{component_id}: line recovery names volume {key!r}, which the recovery "
                             f"denominators do not carry")
        if not math.isclose(float(entry["volume_mwh"]), float(volumes[key]), rel_tol=rtol):
            raise ValueError(f"{component_id}: line recovery volume {entry['volume_mwh']:,.0f} MWh is not the "
                             f"recovery output's {key} ({float(volumes[key]):,.0f} MWh)")
