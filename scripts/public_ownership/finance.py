"""Pure finance functions used by the public-ownership accounting layer."""

from __future__ import annotations

from typing import Mapping, Sequence


from scripts.public_ownership.levers import (
    DEFAULT_SETTINGS, LEGACY_PRIVATE, LEVER_LEGACY_FLEET, LEVER_NEW_BUILD_GENERATION, LEVER_PRIVATE_WACC_BASIS,
    LEVER_PUBLIC_RATE_BASIS, LEVER_PUBLIC_RETURN, LEVER_SOVEREIGN_YIELD, LEVER_STATE_OVERHEAD,
    LEVER_SYSTEMATIC_PREMIUM, NEW_BUILD_PUBLIC, NEW_BUILD_STATE_STAKE, OVERHEAD_CENTRAL, OVERHEAD_HIGH, OVERHEAD_LOW,
    RATE_BASIS_MARGIN_KEYS, RATE_BASIS_REAL, RETIRED_VALUES, RETURN_CEPA_DECOMPOSITION, RETURN_PUBLIC_RATE,
    SYSTEMATIC_NONE, WACC_FIXED_TMR, WACC_NOTIONAL_GEARING, YIELD_A,
)

# The item-N2 case name and the item-D1 case sets, kept for callers that still
# speak that vocabulary.  Since item K1 every treatment is read from lever
# settings (``rate_for_settings``, ``ownership_share_for_settings``), never from
# a case name.
NATIONALISED_CASE = "public_nationalised"
MARKET_CASES = frozenset({"private", "public_coordination", "public_control", "public_ownership"})
CASES = MARKET_CASES | {NATIONALISED_CASE}
VINTAGES = frozenset({"legacy", "new_build", "in_construction"})
#: The workbook scenario whose gilt path a public case follows under yield assumption B (F29):
#: a stake (P1) Sc2, the state developer (P2) Sc3, a case that also buys the fleet (P3) Sc4.
YIELD_ANALOGUE_SC2, YIELD_ANALOGUE_SC3, YIELD_ANALOGUE_SC4 = "Sc2", "Sc3", "Sc4"
#: F29 (26 Oct 2026): the snapshot schema the finance layer reads (extract_workbook_v5.py, workbook v2.0).
SNAPSHOT_SCHEMA_VERSION = 5


def fisher_real_rate(nominal_rate: float, inflation_rate: float) -> float:
    """Convert a nominal rate to a real rate using the Fisher relationship."""
    if inflation_rate <= -1.0:
        raise ValueError("inflation_rate must be greater than -1")
    if nominal_rate <= -1.0:
        raise ValueError("nominal_rate must be greater than -1")
    return (1.0 + float(nominal_rate)) / (1.0 + float(inflation_rate)) - 1.0


def capital_recovery_factor(rate: float, years: float) -> float:
    """Return the annuity factor for a positive remaining economic life."""
    if years <= 0:
        raise ValueError("years must be positive")
    rate = float(rate)
    if rate <= -1.0:
        raise ValueError("rate must be greater than -1")
    if abs(rate) < 1e-12:
        return 1.0 / float(years)
    growth = (1.0 + rate) ** float(years)
    return rate * growth / (growth - 1.0)


def annual_capital_recovery(
    capacity_mw: float, capex_power_gbp_per_kw: float, wacc: float, remaining_life_years: float,
    *, energy_capacity_mwh: float = 0.0, capex_energy_gbp_per_kwh: float = 0.0,
) -> float:
    """Calculate the annual capital charge, including a separate storage energy term."""
    power_capex = float(capacity_mw) * 1_000.0 * float(capex_power_gbp_per_kw)
    energy_capex = float(energy_capacity_mwh) * 1_000.0 * float(capex_energy_gbp_per_kwh)
    return (power_capex + energy_capex) * capital_recovery_factor(wacc, remaining_life_years)


def annual_required_revenue(
    *, capacity_mw: float, dispatch_mwh: float, capex_power_gbp_per_kw: float,
    fixed_om_gbp_per_kw_year: float, variable_om_gbp_per_mwh: float,
    wacc: float, remaining_life_years: float, energy_capacity_mwh: float = 0.0,
    capex_energy_gbp_per_kwh: float = 0.0,
) -> dict[str, float]:
    """Return a component-level annual resource-cost account.

    Market and support revenues intentionally do not enter this calculation:
    they are transfers assessed in the producer/public account.
    """
    capital = annual_capital_recovery(
        capacity_mw, capex_power_gbp_per_kw, wacc, remaining_life_years,
        energy_capacity_mwh=energy_capacity_mwh,
        capex_energy_gbp_per_kwh=capex_energy_gbp_per_kwh,
    )
    fixed_om = float(capacity_mw) * 1_000.0 * float(fixed_om_gbp_per_kw_year)
    variable_om = float(dispatch_mwh) * float(variable_om_gbp_per_mwh)
    return {
        "annual_capital_recovery_gbp": capital,
        "annual_fixed_om_gbp": fixed_om,
        "annual_variable_om_gbp": variable_om,
        "annual_required_revenue_gbp": capital + fixed_om + variable_om,
    }


def wacc_for_case(
    finance_assumption: Mapping[str, object], case_id: str, finance_vintage: str,
    *, acquisition_in_construction_treatment: str | None = None,
) -> float:
    """Select a reviewed workbook WACC without applying policy timing implicitly."""
    if case_id not in CASES:
        raise ValueError(f"Unknown ownership case: {case_id}")
    if finance_vintage not in VINTAGES:
        raise ValueError(f"Unknown finance vintage: {finance_vintage}")

    rates = finance_assumption.get("wacc", finance_assumption)
    if not isinstance(rates, Mapping):
        raise ValueError("finance assumption lacks a wacc mapping")

    if case_id == "private":
        key = "private"
    elif case_id == NATIONALISED_CASE:
        # Every GB asset is state-financed: legacy at the real public borrowing
        # rate (workbook Assumptions!C82, the Sc.3 rate), new build at the
        # public new-build rate.
        key = "public_ownership_legacy" if finance_vintage == "legacy" else "public_ownership_new_build"
    elif case_id == "public_coordination":
        key = "private" if finance_vintage == "legacy" else "public_coordination"
    elif finance_vintage == "legacy":
        # The initial CP30 policy boundary excludes acquisition/refinancing of
        # the historic fleet: legacy assets remain privately financed.
        key = "private"
    elif finance_vintage == "in_construction":
        if acquisition_in_construction_treatment is None:
            raise ValueError(
                "public ownership of in-construction assets requires an explicit treatment"
            )
        key = acquisition_in_construction_treatment
    else:
        key = "public_ownership_new_build"

    try:
        return float(rates[key])
    except KeyError as exc:
        raise ValueError(f"finance assumption lacks WACC {key!r}") from exc


def state_equity_share(case_id: str, finance_vintage: str, configured_share: float = 0.30) -> float:
    """Return ownership share; the 30% workbook value applies to eligible new build only.

    Under ``public_nationalised`` the state owns every vintage outright; the
    caller zeroes the share for assets it does not acquire (foreign plant).
    """
    if case_id not in CASES or finance_vintage not in VINTAGES:
        raise ValueError("Unknown ownership case or finance vintage")
    if not 0.0 <= configured_share <= 1.0:
        raise ValueError("configured_share must be between zero and one")
    if case_id == "public_coordination" and finance_vintage == "new_build":
        return float(configured_share)
    if case_id == "public_ownership" and finance_vintage == "new_build":
        return 1.0
    if case_id == NATIONALISED_CASE:
        return 1.0
    return 0.0


def levelised_cost_gbp_per_mwh(
    *, capex_power_gbp_per_kw: float, fixed_om_gbp_per_kw_year: float, load_factor: float,
    economic_life_years: float, wacc: float, fuel_cost_gbp_per_mwh: float = 0.0,
) -> float:
    """Reproduce the workbook ``LCOE`` sheet formula for one technology and rate.

    LCOE = (capex x CRF + fixed O&M) x 1000 / (8760 x load factor) + fuel.
    Used for workbook parity tests; production accounting works in annual
    required revenue, not levelised cost.
    """
    if not 0.0 < float(load_factor) <= 1.0:
        raise ValueError("load_factor must be within (0, 1]")
    annual_gbp_per_kw = (
        float(capex_power_gbp_per_kw) * capital_recovery_factor(wacc, economic_life_years)
        + float(fixed_om_gbp_per_kw_year)
    )
    return annual_gbp_per_kw * 1_000.0 / (8_760.0 * float(load_factor)) + float(fuel_cost_gbp_per_mwh)

# ── item K1: treatments read from lever settings ─────────────────────────────

def _setting(settings: Mapping[str, str], name: str) -> str:
    return str(settings.get(name, DEFAULT_SETTINGS[name]))


def _value(cell) -> float:
    return float(cell["value"] if isinstance(cell, Mapping) else cell)


def _snapshot_cell(assumptions: Mapping, key: str, cell: str) -> float:
    value = assumptions.get(key)
    if value is None:
        raise ValueError(f"snapshot lacks assumptions.{key} ({cell}); re-run extract_workbook_v5.py")
    return _value(value)


def require_snapshot_schema(snapshot: Mapping) -> None:
    """Refuse a snapshot of another workbook (schema 3 v11-2, schema 4 v2.0): its rates are not v6.2's."""
    version = snapshot.get("schema_version")
    if version is not None and int(version) != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"the finance snapshot is schema {version}; since 30 Oct 2026 the model reads schema "
                         f"{SNAPSHOT_SCHEMA_VERSION} (workbook v6.2, scripts/public_ownership/extract_workbook_v5.py)")


def _retired_or_unknown(lever: str, value: str, supported) -> ValueError:
    note = RETIRED_VALUES.get((lever, value))
    return ValueError(f"{lever} {value!r}: " + (note if note else f"unknown; supported: {list(supported)}"))


def public_borrowing_rate(snapshot: Mapping, settings: Mapping[str, str]) -> float:
    """The state's cash rate: the real gilt plus the public lending margin ``public_rate_basis`` selects.

    Workbook v2.0 (F29): ``Sc4 Operative!F11`` = rf_cpi (CEPA's index-linked gilt method at the
    workbook's valuation date) + the margin of ``Sc4 Operative!B5``.  ``real_cpi`` is the
    central, the National Loans Fund's 11 bp (``assumptions.public_borrowing_real``); the PWLB
    margins and the flat gilt are the workbook's own alternatives, recalculated in Excel
    (``public_rate_by_lending_margin``).  The state charges no construction premium: what bills
    pay is the cash rate, and the risk the state holds is priced off the bill
    (:func:`systematic_premium`).
    """
    basis = _setting(settings, LEVER_PUBLIC_RATE_BASIS)
    key = RATE_BASIS_MARGIN_KEYS.get(basis)
    if key is None:
        raise _retired_or_unknown(LEVER_PUBLIC_RATE_BASIS, basis, RATE_BASIS_MARGIN_KEYS)
    if basis == RATE_BASIS_REAL:
        return _snapshot_cell(snapshot.get("assumptions") or {}, "public_borrowing_real", "Sc4 Operative!F11")
    block = (snapshot.get("public_rate_by_lending_margin") or {}).get(key)
    if block is None:
        raise ValueError(f"snapshot lacks public_rate_by_lending_margin.{key}; re-run extract_workbook_v5.py")
    return _value(block)


def systematic_premium(rate_row: Mapping, settings: Mapping[str, str], *, operating: bool = False) -> float:
    """The workbook's economic rate less its cash rate for one technology: the systematic premium (F29; v6.2).

    ``Sc4 Operative`` D (new build) or E (operating assets, one CEPA band down) less the cash
    rate, at the ``systematic_premium`` lever's value (``economic_rate_by_premium``: the base case,
    a flat 1.9% in v6.2, and the workbook's alternatives, recalculated in Excel).  It is the social cost of the
    systematic risk the state takes on when it owns the plant: never on the bill, a cost to
    taxpayers in the Exchequer ledger and out of the resource saving.  Zero for a row without
    the workbook's economic rates (a unit-test fixture).
    """
    value = _setting(settings, LEVER_SYSTEMATIC_PREMIUM)
    by_premium = rate_row.get("operating_economic_rate_by_premium" if operating else "economic_rate_by_premium")
    if not by_premium:
        return 0.0
    if value not in by_premium:
        raise _retired_or_unknown(LEVER_SYSTEMATIC_PREMIUM, value, by_premium)
    return float(by_premium[value]) - float(by_premium[SYSTEMATIC_NONE])


def yield_analogue(settings: Mapping[str, str]) -> str:
    """The workbook scenario whose gilt path a case follows (F29): Sc2 for a stake, Sc4 with the fleet bought, else Sc3."""
    if _setting(settings, LEVER_PUBLIC_RETURN) == RETURN_CEPA_DECOMPOSITION:
        return YIELD_ANALOGUE_SC2
    if str(settings.get(LEVER_LEGACY_FLEET, LEGACY_PRIVATE)) != LEGACY_PRIVATE:
        return YIELD_ANALOGUE_SC4
    return YIELD_ANALOGUE_SC3


def yield_shift(snapshot: Mapping, settings: Mapping[str, str]) -> float:
    """Additive shift on every public and P1 rate under yield assumption B.

    Workbook v2.0's ``Gilt-Adjusted WACCs``: the gilt yield rises with the scenario's own debt and
    contingent exposure over GDP (2 bp per pp), and an asset financed in a year carries that
    year's shift for its life; the model year's shift applies.  The private baseline never moves;
    assumption A is zero by construction.
    """
    assumption = _setting(settings, LEVER_SOVEREIGN_YIELD)
    if assumption == YIELD_A:
        return 0.0
    return _value(yield_shift_cell(snapshot, yield_analogue(settings), assumption, snapshot_model_year(snapshot)))


#: Item Y4: the key :func:`with_model_year` stamps on a finance snapshot view.
SNAPSHOT_MODEL_YEAR_KEY = "_model_year"


def with_model_year(snapshot: Mapping, model_year: int) -> dict:
    """A shallow copy of the snapshot that knows the model year it is read for (item Y4).

    The sovereign-yield shift is the workbook gilt path in the model year less
    the baseline gilt, so the year decides which column of the path applies.

    Args:
        snapshot: the workbook assumptions snapshot.
        model_year: the run group's model year.

    Returns:
        The snapshot with ``_model_year`` set; every other key is shared.
    """
    view = dict(snapshot)
    view[SNAPSHOT_MODEL_YEAR_KEY] = int(model_year)
    return view


def snapshot_model_year(snapshot: Mapping) -> int:
    """The model year a snapshot view is read for (:func:`with_model_year`), else the default year."""
    from scripts.public_ownership.model_year import DEFAULT_MODEL_YEAR

    return int(snapshot.get(SNAPSHOT_MODEL_YEAR_KEY, DEFAULT_MODEL_YEAR))


def yield_shift_cell(snapshot: Mapping, analogue: str, assumption: str, model_year: int):
    """The workbook's gilt shift for one scenario analogue and assumption in ``model_year``.

    Schema 4 writes a year-named block (``sovereign_yield_shift_<year>``) for every year of
    the workbook's ``Gilt-Adjusted WACCs`` (2025 to 2050), each with the Sc2, Sc3 and Sc4
    paths of assumption B (the Sc3 path from the ``fiscal_sc3`` Excel variant).

    Args:
        snapshot: the workbook assumptions snapshot.
        analogue: ``Sc2``, ``Sc3`` or ``Sc4``.
        assumption: ``B``.
        model_year: the model year.

    Returns:
        The snapshot cell (a mapping with ``value``).
    """
    named = f"sovereign_yield_shift_{int(model_year)}"
    shifts = (snapshot.get(named) or {}).get(analogue) or {}
    if assumption in shifts:
        return shifts[assumption]
    raise ValueError(f"snapshot lacks {named}.{analogue}.{assumption}; workbook v2.0's gilt paths run 2025 to 2050 "
                     "(assumption C is blank in the workbook and retired)")


def _basis_rate(rate_row: Mapping, basis: str, column: str) -> float:
    rates = rate_row.get("wacc", rate_row)
    if basis == WACC_NOTIONAL_GEARING:
        try:
            return float(rates[column])
        except KeyError as exc:
            raise ValueError(f"finance assumption lacks WACC {column!r}") from exc
    by_basis = rate_row.get("wacc_by_basis") or {}
    try:
        return float(by_basis[basis][column])
    except KeyError as exc:
        if (LEVER_PRIVATE_WACC_BASIS, basis) in RETIRED_VALUES:
            raise _retired_or_unknown(LEVER_PRIVATE_WACC_BASIS, basis, ()) from exc
        raise ValueError(
            f"snapshot lacks the {basis!r} WACC block for {column!r}; re-run extract_workbook_v5.py"
        ) from exc


def private_rate(rate_row: Mapping, settings: Mapping[str, str]) -> float:
    """Scenario 1 lead rate (tax-shielded) on the ERP convention ``private_wacc_basis`` selects."""
    return _basis_rate(rate_row, _setting(settings, LEVER_PRIVATE_WACC_BASIS), "private")


def coordination_rate(rate_row: Mapping, settings: Mapping[str, str]) -> float:
    """Scenario 2 lead rate (P1) on the same convention."""
    return _basis_rate(rate_row, _setting(settings, LEVER_PRIVATE_WACC_BASIS), "public_coordination")


def public_new_build_rate(
    rate_row: Mapping, snapshot: Mapping, settings: Mapping[str, str], *, ownership_share: float = 1.0,
) -> float:
    """What state-financed new build earns, per the ``public_return`` lever (workbook v2.0, F29).

    ``public_rate``: the cash rate (:func:`public_borrowing_rate`), what bills pay;
    ``cepa_decomposition``: the Scenario 2 lead rate (P1: a 30% state stake at the risk-free
    rate and a guarantee lifting the debt two notches).  The sovereign-yield shift is added to
    both.  ``ownership_share`` is kept for the callers: since F29 no lever moves a stake's rate
    by its share (the workbook's Sc2 rate does not use the lending margin).  Retired values
    (``commercial_benchmark`` with F29, ``explicit`` with OW9) fail with a pointer.
    """
    kind = _setting(settings, LEVER_PUBLIC_RETURN)
    if kind == RETURN_PUBLIC_RATE:
        rates = rate_row.get("wacc", rate_row)
        if _setting(settings, LEVER_PUBLIC_RATE_BASIS) == RATE_BASIS_REAL and "public_ownership_new_build" in rates:
            # The row's own workbook column (WACC Scenarios H, the cash rate at the NLF margin).
            return float(rates["public_ownership_new_build"]) + yield_shift(snapshot, settings)
        return public_borrowing_rate(snapshot, settings) + yield_shift(snapshot, settings)
    if kind == RETURN_CEPA_DECOMPOSITION:
        return coordination_rate(rate_row, settings) + yield_shift(snapshot, settings)
    raise _retired_or_unknown(LEVER_PUBLIC_RETURN, kind, (RETURN_PUBLIC_RATE, RETURN_CEPA_DECOMPOSITION))


def public_legacy_rate(snapshot: Mapping, settings: Mapping[str, str]) -> float:
    """The rate an acquired legacy asset is refinanced at: the cash rate plus the yield shift."""
    return public_borrowing_rate(snapshot, settings) + yield_shift(snapshot, settings)


def coordination_at_stake(snapshot: Mapping, stake: float) -> dict:
    """The snapshot with P1's Scenario 2 rates and tax wedges recomputed at another state stake (F29).

    Workbook v2.0's ``Sc2 Operative``: the state's share of the equity requires the risk-free
    rate plus phi of the premium, so the blended post-tax cost of equity is
    ``rf + (1 - sigma (1 - phi)) beta_e ERP``, grossed up by ``1 - tau`` and weighted by
    ``1 - g``; the debt leg (the guarantee's notch uplift) does not depend on the stake.  So
    each technology's lead rate moves by ``(1 - g)(1 - phi) beta_e ERP (sigma_0 - sigma) / (1 - tau)``
    and its tax wedge by ``tau`` times that, on each ERP convention (Fixed TMR's premium is
    TMR less rf).  A snapshot without the workbook's betas is returned with only the stake moved.
    """
    import copy

    out = copy.deepcopy(dict(snapshot))
    assumptions = out.setdefault("assumptions", {})
    cell = assumptions.get("public_coordination_state_equity_share")
    sigma0 = _value(cell) if cell is not None else 0.30
    assumptions["public_coordination_state_equity_share"] = (
        dict(cell, value=float(stake)) if isinstance(cell, Mapping) else {"value": float(stake)})
    needed = ("equity_risk_premium", "corporation_tax_rate", "public_coordination_retained_premium", "risk_free_real")
    if any(assumptions.get(k) is None for k in needed):
        return out
    tau = _value(assumptions["corporation_tax_rate"])
    phi = _value(assumptions["public_coordination_retained_premium"])
    erp = {WACC_NOTIONAL_GEARING: _value(assumptions["equity_risk_premium"])}
    if assumptions.get("total_market_return") is not None:
        erp[WACC_FIXED_TMR] = _value(assumptions["total_market_return"]) - _value(assumptions["risk_free_real"])
    move = float(sigma0) - float(stake)
    for row in out.get("technologies", []):
        if row.get("equity_beta") is None or row.get("gearing") is None:
            continue
        g, beta_e = _value(row["gearing"]), float(row["equity_beta"])
        for basis, premium in erp.items():
            block = (row.get("wacc_by_basis") or {}).get(basis)
            if block is None:
                continue
            post_tax = (1.0 - g) * (1.0 - phi) * beta_e * premium * move
            block["public_coordination"] = float(block["public_coordination"]) + post_tax / (1.0 - tau)
            if basis == WACC_NOTIONAL_GEARING:
                row["wacc"]["public_coordination"] = block["public_coordination"]
                wedge = row.get("corporation_tax_wedge")
                if wedge is not None:
                    wedge["public_coordination"] = float(wedge["public_coordination"]) + post_tax * tau / (1.0 - tau)
        row["wacc_basis_note"] = f"Scenario 2 recomputed at a {float(stake):.0%} state stake (workbook v2.0 formula)"
    return out


def rate_for_settings(
    rate_row: Mapping, snapshot: Mapping, settings: Mapping[str, str], *, finance_vintage: str, owned: bool,
    ownership_share: float = 1.0,
) -> float:
    """The WACC one register row carries under one lever point.

    A row the state does not own is financed privately whatever the case (the
    legacy fleet in the market cases, capacity outside the ownership boundary).
    An owned legacy row is refinanced at the public rate; an owned new-build or
    in-construction row earns the ``public_return``.  ``ownership_share`` is the
    state's stake, which scales a public-rate stress under a joint venture.
    """
    if finance_vintage not in VINTAGES:
        raise ValueError(f"Unknown finance vintage: {finance_vintage}")
    if not owned:
        return private_rate(rate_row, settings)
    if finance_vintage == "legacy":
        return public_legacy_rate(snapshot, settings)
    return public_new_build_rate(rate_row, snapshot, settings, ownership_share=ownership_share)


def ownership_share_for_settings(
    settings: Mapping[str, str], *, finance_vintage: str, owned: bool, configured_share: float = 0.30,
) -> float:
    """The state share of an owned row: the stake on new build, all of an acquired or public asset."""
    if finance_vintage not in VINTAGES:
        raise ValueError(f"Unknown finance vintage: {finance_vintage}")
    if not 0.0 <= configured_share <= 1.0:
        raise ValueError("configured_share must be between zero and one")
    if not owned:
        return 0.0
    if finance_vintage == "legacy":
        return 1.0
    generation = _setting(settings, LEVER_NEW_BUILD_GENERATION)
    if generation == NEW_BUILD_STATE_STAKE:
        return float(configured_share)
    if generation == NEW_BUILD_PUBLIC:
        return 1.0
    from scripts.public_ownership.levers import GBE_VALUES

    if generation in GBE_VALUES:
        # F10: the row's part is set by its GB Energy selection (the register's gbe_part), 1 here.
        return 1.0
    raise ValueError("an owned new-build row under private new build is a contradiction")


# ── FES 2050 plan, task Y9: lever L4 state_deployment_share ─────────────────

def deployment_share(settings: Mapping[str, str]) -> float | None:
    """The state's share of P2's post-2025 capacity the ``state_deployment_share`` lever selects.

    ``None`` for ``all`` (today's P2, no selection); otherwise 0.25, 0.50, 0.75 or 1.00.
    """
    from scripts.public_ownership.levers import (
        DEPLOYMENT_ALL, DEPLOYMENT_SHARES, LEVER_STATE_DEPLOYMENT_SHARE, lever_default,
    )

    value = str(settings.get(LEVER_STATE_DEPLOYMENT_SHARE, lever_default(LEVER_STATE_DEPLOYMENT_SHARE)))
    if value == DEPLOYMENT_ALL:
        return None
    if value not in DEPLOYMENT_SHARES:
        raise ValueError(f"unknown state_deployment_share {value!r}; supported: {[DEPLOYMENT_ALL, *DEPLOYMENT_SHARES]}")
    return float(DEPLOYMENT_SHARES[value])


def deployment_selection(capacity, cells, order_keys, eligible, share: float):
    """Which eligible rows the state keeps at a deployment share (decision 17: the same share of every cell).

    Within each cell (a technology and build year) the eligible rows are taken in
    ``order_keys`` order (a fixed md5 of the asset id, so no location or name bias)
    and kept while the capacity midpoint of the row lies within ``share`` of the
    cell's eligible capacity: whole rows, deterministic, the same share always
    picking the same plant, and at ``share = 1`` every eligible row (the midpoint
    never passes the total).  The achieved share of a cell is within half its
    largest row of the target.

    Args:
        capacity: Row capacity (MW), array-like.
        cells: Cell key per row (hashable), array-like.
        order_keys: Sort key per row within a cell, array-like.
        eligible: Rows the state would own at 100% (bool), array-like.
        share: The deployment share in [0, 1].

    Returns:
        A bool array: eligible rows the state keeps.
    """
    import numpy as np
    import pandas as pd

    if not 0.0 <= float(share) <= 1.0:
        raise ValueError(f"deployment share {share} is not in [0, 1]")
    eligible = np.asarray(eligible, dtype=bool)
    keep = np.zeros(len(eligible), dtype=bool)
    if not eligible.any():
        return keep
    frame = pd.DataFrame({"capacity": np.asarray(capacity, dtype=float), "cell": list(cells),
                          "order": list(order_keys)}).loc[eligible]
    frame = frame.sort_values(["cell", "order"], kind="mergesort")
    cumulative = frame.groupby("cell", sort=False)["capacity"].cumsum()
    total = frame.groupby("cell", sort=False)["capacity"].transform("sum")
    midpoint = cumulative - frame["capacity"] / 2.0
    kept = frame.index[(midpoint <= float(share) * total).to_numpy()]
    keep[np.asarray(kept, dtype=int)] = True
    return keep


def co_ownership_rate(public_rate: float, private_rate: float, state_share: float, years: float) -> float:
    """FES 2050 decision 59: the single rate whose annuity is the share-weighted annuity of a row's two owners.

    A row the state holds at ``state_share`` (lever L4, fractional) is financed ``state_share``
    at ``public_rate`` and the rest at ``private_rate``; its capital charge is
    ``capex x (s CRF(public, n) + (1 - s) CRF(private, n))``, which is ``capex x CRF(r, n)``
    at the returned ``r`` (found by bisection; CRF rises with the rate). At a share of 1 or 0
    it is the owner's own rate.
    """
    s = float(state_share)
    if s >= 1.0:
        return float(public_rate)
    if s <= 0.0:
        return float(private_rate)
    target = s * capital_recovery_factor(public_rate, years) + (1.0 - s) * capital_recovery_factor(private_rate, years)
    lo, hi = min(public_rate, private_rate), max(public_rate, private_rate)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if capital_recovery_factor(mid, years) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── FES 2050 plan decision 19: the private rate by build year ───────────────
#: The wind technologies NESO's FES 2025 hurdle rates step down over 2031-2035.
WIND_RAMP_TECHNOLOGIES = ("Offshore wind", "Onshore wind", "Floating offshore wind")
#: The ramp is the change on this build year's FES rate (the flat central).
WIND_RAMP_BASE_YEAR = 2030


def neso_wind_ramp_deltas(hurdle_table=None) -> dict[str, dict[int, float]]:
    """``{technology: {build year: FES hurdle(year) - FES hurdle(2030)}}`` for the wind ramp technologies.

    From ``data/public_ownership/financing/fes2025_hurdle_rates.csv`` (Y2), FES
    source, central case; years 2025-2050, a later year holding 2050.
    """
    from scripts.public_ownership.vintage_costs import FIRST_BUILD_YEAR, LAST_BUILD_YEAR, hurdle_rate, load_hurdle_rates

    table = load_hurdle_rates() if hurdle_table is None else hurdle_table
    out = {}
    for technology in WIND_RAMP_TECHNOLOGIES:
        base = hurdle_rate(technology, WIND_RAMP_BASE_YEAR, source="FES", table=table)
        out[technology] = {year: hurdle_rate(technology, year, source="FES", table=table) - base
                           for year in range(FIRST_BUILD_YEAR, LAST_BUILD_YEAR + 1)}
    return out


def private_rate_path_delta(settings: Mapping[str, str], technology: str, build_year, deltas) -> float:
    """What the ``private_rate_path`` lever adds to a privately financed row's rate.

    ``flat``: nothing.  ``neso_wind_ramp``: the NESO ramp's change on 2030 for a
    wind row built after 2030 (``deltas`` from :func:`neso_wind_ramp_deltas`); zero
    for any other technology, an undated row or a build year up to 2030.
    """
    from scripts.public_ownership.levers import LEVER_PRIVATE_RATE_PATH, RATE_PATH_NESO_WIND_RAMP, lever_default

    if str(settings.get(LEVER_PRIVATE_RATE_PATH, lever_default(LEVER_PRIVATE_RATE_PATH))) != RATE_PATH_NESO_WIND_RAMP:
        return 0.0
    try:
        year = float(build_year)
    except (TypeError, ValueError):
        return 0.0
    if not year == year or year <= WIND_RAMP_BASE_YEAR:
        return 0.0
    by_year = (deltas or {}).get(str(technology))
    if not by_year:
        return 0.0
    return float(by_year[min(int(year), max(by_year))])



# ── Item OH1: what it costs the state to run the thing ───────────────────────
# DESNZ is explicit that its fixed operating costs are the ones "a developer is
# liable for ... associated with the generator and site itself", covering
# "staffing costs and routine maintenance and any equipment contracts", with
# insurance and connection as separate LCOE inputs.  So the
# ``fixed_om_gbp_per_kw_year`` column every case already carries is SITE-level.
# The increment a state owner adds is the holding company: board, group finance
# and treasury, legal, HR, IT, risk, regulatory and policy.  Charging the whole
# of an offshore OPEX line here would double-count the site block, which is why
# only the admin sliver is taken (plan Group L, item OH1).
#: Lever value -> the configured rate that value selects.
STATE_OVERHEAD_RATE_KEYS: Mapping[str, str] = {
    OVERHEAD_CENTRAL: "central_gbp_per_kw_year",
    OVERHEAD_LOW: "low_gbp_per_kw_year",
    OVERHEAD_HIGH: "high_gbp_per_kw_year",
}
#: Used only when ``public_ownership.state_operating_overhead`` is absent, and
#: kept equal to the repository config so a missing block cannot silently move a
#: number (the plan's central and its two twins).
STATE_OVERHEAD_DEFAULT_RATES: Mapping[str, float] = {
    OVERHEAD_CENTRAL: 3.0, OVERHEAD_LOW: 1.0, OVERHEAD_HIGH: 8.0,
}


def state_overhead_rate(overhead_config: Mapping | None, settings: Mapping[str, str]) -> float:
    """The holding-company rate in GBP per kW per year this case's lever selects.

    A disabled block is zero: the overhead is then not modelled at all, which is
    the pre-OH1 behaviour and is what the ``enabled: false`` switch is for.
    """
    config = dict(overhead_config or {})
    if not config.get("enabled", True):
        return 0.0
    value = _setting(settings, LEVER_STATE_OVERHEAD)
    key = STATE_OVERHEAD_RATE_KEYS[value]
    rate = config.get(key, STATE_OVERHEAD_DEFAULT_RATES[value])
    rate = float(rate)
    if rate < 0.0:
        raise ValueError(f"state_operating_overhead.{key} must not be negative, got {rate}")
    return rate


#: F19 decision 6 (user, 6 Oct 2026): unstaffed, remotely operated storage carries the
#: OH1 note's own bottom-up 1.5 GBP/kW/yr, its twins scaled in proportion (x 1/3 and x 8/3
#: of the central, as the fleet rate's 1.0 and 8.0 are of its 3.0).  Kept equal to config
#: ``state_operating_overhead.unstaffed_storage``; vintaged (2040 on) registers take it only
#: when ``in_vintaged_registers`` is true (2050 has not decided).
STATE_OVERHEAD_STORAGE_DEFAULTS: Mapping = {
    "carriers": ("Battery", "LAES", "pumped_hydro"),
    "central_gbp_per_kw_year": 1.5, "low_gbp_per_kw_year": 0.5, "high_gbp_per_kw_year": 4.0,
    "in_vintaged_registers": False,
}


def state_overhead_carrier_rates(overhead_config: Mapping | None, settings: Mapping[str, str], *,
                                 vintaged: bool = False) -> dict[str, float]:
    """F19 decision 6: the carriers whose rate differs from :func:`state_overhead_rate`, with that rate.

    Empty when the block is disabled, or in a vintaged register unless the block's
    ``in_vintaged_registers`` says otherwise (the 2050 register is unchanged).
    """
    config = dict(overhead_config or {})
    if not config.get("enabled", True):
        return {}
    block = {**STATE_OVERHEAD_STORAGE_DEFAULTS, **dict(config.get("unstaffed_storage") or {})}
    if vintaged and not bool(block["in_vintaged_registers"]):
        return {}
    key = STATE_OVERHEAD_RATE_KEYS[_setting(settings, LEVER_STATE_OVERHEAD)]
    rate = float(block[key])
    if rate < 0.0:
        raise ValueError(f"state_operating_overhead.unstaffed_storage.{key} must not be negative, got {rate}")
    return {str(carrier): rate for carrier in block["carriers"]}


def annual_state_overhead(
    *, capacity_mw: float, ownership_share: float, rate_gbp_per_kw_year: float,
    component_type: str | None = None, applies_to: Sequence[str] | None = None,
) -> float:
    """The state's holding-company cost carried by one row.

    It follows the state's SHARE of the capacity, so a 30% stake carries 30% of
    the centre: the holding company a minority co-investor needs is smaller than
    the one a full owner needs, and the register already knows the share.  Rows
    outside ``applies_to`` carry none -- acquired networks keep their RIIO
    allowed opex in the network account, so an overhead there would double-count.
    """
    if applies_to is not None and component_type is not None and str(component_type) not in set(applies_to):
        return 0.0
    share = float(ownership_share)
    if share <= 0.0:
        return 0.0
    return float(capacity_mw) * 1_000.0 * share * float(rate_gbp_per_kw_year)


# ── Item D4(a): generator TNUoS and insurance, costs of the plant itself ─────
# DESNZ's generation-cost framework carries insurance and "connection and use of
# system charges" as their own LCOE inputs, outside fixed O&M (Electricity
# Generation Costs 2025, Section 1), so ``technology_costs.csv`` carries neither.
# Both are paid by whoever owns the plant, so they are added to the required
# revenue of EVERY row (private and public alike): the financing saving is
# unchanged, but producer rent falls, the at-cost strike rises and the
# acquisition tariff carries them, each exactly once.  The register folds them
# into ``annual_fixed_om_gbp`` (with the parts reported beside it) because the
# tariff, the LDES floor and the legacy valuation all read that column.
TNUOS_CONVENTIONAL_CARBON = "conventional_carbon"
TNUOS_CONVENTIONAL_LOW_CARBON = "conventional_low_carbon"
TNUOS_INTERMITTENT = "intermittent"
TNUOS_CLASSES = (TNUOS_CONVENTIONAL_CARBON, TNUOS_CONVENTIONAL_LOW_CARBON, TNUOS_INTERMITTENT)


def wider_tnuos_tariff(*, tnuos_class: str, annual_load_factor: float, system_peak: float,
                       shared_year_round: float, not_shared_year_round: float, adjustment: float) -> float:
    """One generator's wider TNUoS tariff in GBP/kW (CUSC 14.15, NESO Five-Year View Appendix A).

    * conventional carbon: peak + ALF x (shared + not shared) + adjustment;
    * conventional low carbon: peak + ALF x shared + not shared + adjustment;
    * intermittent: ALF x shared + not shared + adjustment (no peak element).

    The NESO example tariffs (zone 1, 2030/31: 41.42, 80.67 and 51.47 GBP/kW at
    ALFs of 40%, 75% and 45%) are reproduced by the unit tests.
    """
    alf = min(max(float(annual_load_factor), 0.0), 1.0)
    if tnuos_class == TNUOS_CONVENTIONAL_CARBON:
        return system_peak + alf * (shared_year_round + not_shared_year_round) + adjustment
    if tnuos_class == TNUOS_CONVENTIONAL_LOW_CARBON:
        return system_peak + alf * shared_year_round + not_shared_year_round + adjustment
    if tnuos_class == TNUOS_INTERMITTENT:
        return alf * shared_year_round + not_shared_year_round + adjustment
    raise ValueError(f"unknown TNUoS generator class {tnuos_class!r}; supported: {TNUOS_CLASSES}")


def nearest_zone(lat, lon, zones) -> list:
    """The generation charging zone whose reference point is nearest each (lat, lon).

    ``zones`` carries ``zone``, ``reference_lat`` and ``reference_lon``.  A
    declared approximation: NESO defines zones by transport-model node lists,
    not by polygons, so each zone is represented by one point placed from its
    name.  Distance is on a local flat projection (longitude scaled by cos lat).
    Missing coordinates give ``None``.
    """
    import numpy as np

    zlat = np.asarray(zones["reference_lat"], dtype=float)
    zlon = np.asarray(zones["reference_lon"], dtype=float)
    ids = list(zones["zone"])
    out = []
    for a, b in zip(lat, lon):
        if a is None or b is None or not np.isfinite(float(a)) or not np.isfinite(float(b)):
            out.append(None)
            continue
        d = (zlat - float(a)) ** 2 + ((zlon - float(b)) * np.cos(np.radians(float(a)))) ** 2
        out.append(ids[int(np.argmin(d))])
    return out


def annual_insurance(*, capex_total_gbp: float, capacity_mw: float, share_of_capex: float | None,
                     gbp_per_kw_year: float | None) -> float:
    """Annual insurance: a GBP/kW/yr rate where one is sourced, else a share of as-new capex."""
    if gbp_per_kw_year is not None:
        return float(capacity_mw) * 1_000.0 * float(gbp_per_kw_year)
    if share_of_capex is None:
        return 0.0
    return float(capex_total_gbp) * float(share_of_capex)


# ── Exchequer ledger (decision D2): corporation tax inside the financing saving ──

def corporation_tax_wedges(snapshot: Mapping) -> dict[str, dict[str, float]]:
    """Per workbook technology, the corporation tax inside the lead (tax-shielded) rate (F29).

    Workbook v2.0's ``Sc4 Operative!H``: ``(1 - g)(r_e pre-tax - r_e post-tax) - g tau r_d``,
    the equity's tax less the interest shield at the technology's observed gearing, so that
    Sc1 lead = Sc1 resource rate + wedge.  The P1 wedge is the same construction on the Sc2
    tab (the extractor writes both as ``corporation_tax_wedge``; :func:`coordination_at_stake`
    moves P1's with the stake).  Returns ``{}`` for a snapshot without them (unit-test fixtures).
    """
    assumptions = snapshot.get("assumptions") or {}
    if "corporation_tax_rate" not in assumptions:
        return {}
    tau = _value(assumptions["corporation_tax_rate"])
    out = {}
    for row in snapshot.get("technologies", []):
        wedge = row.get("corporation_tax_wedge")
        if not wedge:
            continue
        out[row["technology"]] = {"private": float(wedge["private"]),
                                  "public_coordination": float(wedge["public_coordination"]), "tax_rate": tau}
    return out


def embedded_corporation_tax(capex_total_gbp: float, rate: float, wedge: float, years: float) -> float:
    """Corporation tax carried in an annuitised capital charge: capex x [CRF(r) - CRF(r - wedge)]."""
    if years <= 0 or capex_total_gbp == 0.0:
        return 0.0
    return float(capex_total_gbp) * (capital_recovery_factor(rate, years) - capital_recovery_factor(rate - wedge, years))
