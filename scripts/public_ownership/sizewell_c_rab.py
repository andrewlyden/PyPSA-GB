"""OW10 (was backlog G4): Sizewell C's RAB levy with its return at the public rate, in the public cases.

The 2030 bill carries Sizewell C's nuclear RAB levy as the forecast bill's
``sizewell_c_rab`` line: Ofgem's nRAB allowance (Annex 4 v1.25, Oct-Dec 2026),
paid while the plant is under construction and outside the register.  During
construction that allowance is almost all **return on capital** on the regulated
asset base at the construction-period WACC DESNZ agreed with investors, 6.73% real
(CPIH-real; NAO, *Sizewell C*, HC 33, 20 May 2026, para 2.14 and Appendix One para
17); its other blocks (corporation tax, Civil Nuclear Constabulary costs, business
rates, the cost-of-debt adjustment; NAO Figure 8) are small and not tabulated.

In a public case the state finances that regulated asset base at its own rate, as
it finances the wires it buys at RAV (item N6).  The comparator already finances
most of it: since the final investment decision DESNZ holds 44.9% of the equity
and the National Wealth Fund lends up to GBP 36.6 bn of the GBP 41.6 bn of debt
(NAO, HC 33, summary and paras 2.7-2.8), so about 81% of the capital is
state-funded before any public case.  Only the privately financed remainder moves,
and what moves is its RETURN, not its share of the capital (F19 decision 1, refined
by the user on 6 Oct 2026): the saving is what private capital earns, less its cost
at the public cash rate the case's own new build carries.
The bill moves by

    - line x return_share x f x (R_private - K_private x rate) / (K x rab_real_rate)

per household, recovered over final demand like the line it adjusts, with

* ``K``: Sizewell C's capital, equity plus all the debt (NAO HC 33: GBP 50.1 bn);
* ``K_private``: the privately financed capital, the private equity
  (1 - DESNZ's share) x equity plus the non-NWF debt (GBP 4.68 bn + 5.0 bn);
* ``R_private``: what that capital earns, private equity at its required return and
  private debt at its rate (:func:`capital_rates`).  The equity return is the NAO's
  post-tax nominal equity IRR at the lower regulatory threshold (12.0-13.0%, para
  2.23; the company's own cost estimate is below that threshold), deflated to
  CPIH-real; the debt rate is whatever makes the capital structure in config earn the
  agreed 6.73% real WACC on K, so both are consistent with the WACC by construction
  (10.29% and 6.00% real at the config figures);
* ``f``: the fraction of the private capital the case takes on,
  max(0, state_share - existing_share) / (1 - existing_share), pro rata across the
  private equity and the private debt (1 under P2 and P3, 0 under P1);
* ``rate``: the public cash rate (F29: workbook v2.0 charges the state no construction premium);
* ``return_share``: the return element's share of the line (1.0, the whole line,
  an upper bound on the saving: the other blocks are not tabulated);
* ``state_share``: the share of a project's **capital** the case's state finances,
  its ``ownership_share`` of its own new build (a state owner finances its share of
  the capital, equity and debt alike: 30% under P1, all of it under P2 and P3);
* ``existing_share``: the share of Sizewell C's **capital** (equity plus all the
  debt, not the equity alone) the state already funds in the comparator,
  :func:`existing_state_capital_share` of the NAO figures in config.  Both shares
  are shares of capital, so P1's 30% is below it and moves nothing;
* ``public_rate``: the network finance account's public real rate at the case's
  ``public_rate_basis`` (the rate the wires are refinanced at; the sovereign-yield
  lever moves neither).

It is an ownership effect, never a comparator change: the private case and any
case with no state-owned new build keep the line as it is.  It applies in every
model year (D4, user 14 Nov 2026): a later year's register books Sizewell C's RAB
allowance (FES 2050 decision 41 (c), ``rab_support_gbp``) on private rows in S1 and
S2, which keep the levy; only a case whose state holds those rows (S3, which replaces
the levy) is priced by the register instead.  For the three-party
ledger it is a cost-of-capital saving: the RAB's investors are refinanced at the
regulated value (a wash for them, as the wires at RAV) and nothing passes the
Exchequer.

Configured in ``public_ownership.consumer_bill.sizewell_c_rab_public_rate``; off
(``enabled: false``, today's behaviour) unless the config turns it on.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

#: The forecast bill's Sizewell C line, and the row (market bill) the adjustment is booked on.
LINE = "sizewell_c_rab"
ADJUSTMENT_LINE = "sizewell_c_rab_public_rate_adjustment"
CONFIG_KEY = "sizewell_c_rab_public_rate"
#: Code defaults: the switch off (today's bill); the rate, shares, financing and premia are config/defaults/base/public_ownership/consumer_bill.yaml's.
DEFAULT_SETTINGS: dict = {
    "enabled": False, "rab_real_rate": 0.0673, "return_share": 1.0,
    # F19 decision 1: the comparator's existing state finance (NAO, HC 33, 20 May 2026), GBP bn nominal.
    "state_equity_share": 0.449, "equity_gbp_bn": 8.5, "nwf_debt_gbp_bn": 36.6, "total_debt_gbp_bn": 41.6,
    # F19 decision 1, refined: private equity's required return, the NAO's post-tax nominal equity IRR at the
    # lower regulatory threshold (12.0-13.0%, HC 33 para 2.23), deflated on the bill's CPIH path (2% a year).
    "equity_return_nominal": 0.125, "cpih_inflation": 0.02,
}
#: Keys retired by F29 (26 Oct 2026), and why.
RETIRED_KEYS = {"construction_premium": ("retired 26 Oct 2026 (F29): workbook v2.0 charges the state its cash rate "
                                         "with no construction premium (lever construction_premium retired)")}
_SHARE_KEYS = ("return_share", "state_equity_share", "equity_return_nominal", "cpih_inflation")
_AMOUNT_KEYS = ("equity_gbp_bn", "nwf_debt_gbp_bn", "total_debt_gbp_bn")


def settings_from_config(consumer_bill: Mapping | None) -> dict:
    """``public_ownership.consumer_bill.sizewell_c_rab_public_rate`` over the defaults, checked."""
    block = dict(((consumer_bill or {}).get(CONFIG_KEY)) or {})
    for key in sorted(set(block) & set(RETIRED_KEYS)):
        raise ValueError(f"consumer_bill.{CONFIG_KEY}.{key}: {RETIRED_KEYS[key]}")
    unknown = sorted(set(block) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(f"consumer_bill.{CONFIG_KEY} has unknown keys {unknown}; supported: {sorted(DEFAULT_SETTINGS)}")
    merged = {**DEFAULT_SETTINGS, **block}
    if not isinstance(merged["enabled"], bool):
        raise ValueError(f"consumer_bill.{CONFIG_KEY}.enabled must be true or false")
    if not 0.0 < float(merged["rab_real_rate"]) < 1.0:
        raise ValueError(f"consumer_bill.{CONFIG_KEY}.rab_real_rate must be a rate in (0, 1)")
    for key in _SHARE_KEYS:
        if not 0.0 <= float(merged[key]) <= 1.0:
            raise ValueError(f"consumer_bill.{CONFIG_KEY}.{key} must be a share in [0, 1]")
    for key in _AMOUNT_KEYS:
        if float(merged[key]) < 0.0:
            raise ValueError(f"consumer_bill.{CONFIG_KEY}.{key} must be an amount >= 0")
    if float(merged["nwf_debt_gbp_bn"]) > float(merged["total_debt_gbp_bn"]):
        raise ValueError(f"consumer_bill.{CONFIG_KEY}: nwf_debt_gbp_bn exceeds total_debt_gbp_bn")
    if float(merged["equity_gbp_bn"]) + float(merged["total_debt_gbp_bn"]) <= 0.0:
        raise ValueError(f"consumer_bill.{CONFIG_KEY}: Sizewell C's capital must be positive")
    debt_rate = capital_rates(merged)[1]
    if not 0.0 <= debt_rate < 1.0:
        raise ValueError(f"consumer_bill.{CONFIG_KEY}: equity_return_nominal and rab_real_rate imply a debt rate of "
                         f"{debt_rate:.2%} on this capital structure; they are inconsistent")
    return {"enabled": bool(merged["enabled"]), "rab_real_rate": float(merged["rab_real_rate"]),
            **{key: float(merged[key]) for key in _SHARE_KEYS + _AMOUNT_KEYS}}


def existing_state_capital_share(settings: Mapping) -> float:
    """The share of Sizewell C's capital the state already funds in the comparator (F19 decision 1).

    DESNZ's share of the equity plus the National Wealth Fund's debt, over the equity
    plus all the debt: (0.449 x 8.5 + 36.6) / (8.5 + 41.6) = 0.807 at the NAO figures.
    """
    merged = {**DEFAULT_SETTINGS, **dict(settings or {})}
    capital = float(merged["equity_gbp_bn"]) + float(merged["total_debt_gbp_bn"])
    state = float(merged["state_equity_share"]) * float(merged["equity_gbp_bn"]) + float(merged["nwf_debt_gbp_bn"])
    return min(max(state / capital, 0.0), 1.0) if capital > 0.0 else 0.0


def refinanced_share(settings: Mapping, state_share: float) -> float:
    """The privately financed share of the capital a case takes on: max(0, case share - existing share)."""
    return max(0.0, float(state_share) - existing_state_capital_share(settings))


def capital_rates(settings: Mapping) -> tuple[float, float]:
    """``(equity, debt)`` real rates consistent with the construction-period WACC (F19 decision 1, refined).

    Equity earns its required return, the NAO's post-tax nominal IRR deflated to
    CPIH-real (a vanilla WACC: post-tax equity, the tax allowance is its own block);
    debt earns what makes the whole capital earn ``rab_real_rate``:
    (K x WACC - E x r_e) / D.  At the config figures 12.5% nominal is 10.29% real and
    the debt rate 6.00% real.
    """
    merged = {**DEFAULT_SETTINGS, **dict(settings or {})}
    equity, debt = float(merged["equity_gbp_bn"]), float(merged["total_debt_gbp_bn"])
    r_e = (1.0 + float(merged["equity_return_nominal"])) / (1.0 + float(merged["cpih_inflation"])) - 1.0
    if debt <= 0.0:
        return r_e, 0.0
    return r_e, ((equity + debt) * float(merged["rab_real_rate"]) - equity * r_e) / debt


def private_capital(settings: Mapping) -> tuple[float, float]:
    """``(private equity, private debt)``, GBP bn: the equity DESNZ does not hold and the debt the NWF does not lend."""
    merged = {**DEFAULT_SETTINGS, **dict(settings or {})}
    return ((1.0 - float(merged["state_equity_share"])) * float(merged["equity_gbp_bn"]),
            float(merged["total_debt_gbp_bn"]) - float(merged["nwf_debt_gbp_bn"]))


def return_saving_share(settings: Mapping, state_share: float, rate: float) -> float:
    """The share of the RAB's whole return a case saves: f x (R_private - K_private x rate) / (K x WACC)."""
    merged = {**DEFAULT_SETTINGS, **dict(settings or {})}
    existing = existing_state_capital_share(merged)
    fraction = refinanced_share(merged, state_share) / (1.0 - existing) if existing < 1.0 else 0.0
    if fraction <= 0.0:
        return 0.0
    r_e, r_d = capital_rates(merged)
    equity_p, debt_p = private_capital(merged)
    capital = float(merged["equity_gbp_bn"]) + float(merged["total_debt_gbp_bn"])
    earned = equity_p * r_e + debt_p * r_d
    return fraction * (earned - (equity_p + debt_p) * float(rate)) / (capital * float(merged["rab_real_rate"]))


def premium(settings: Mapping, levers: Mapping | None) -> float:
    """The premium on the refinanced capital: none since F29 (workbook v2.0 charges the state its cash rate)."""
    return 0.0


def state_new_build_share(rows: pd.DataFrame) -> float:
    """The state's share of the case's own new build: the largest ``ownership_share`` on a new-build row, else 0.

    F10: GB Energy as announced (``new_build_generation`` gbe_announced, gbe_announced_partnership_half)
    holds a stake in offshore wind and local power only; it takes on none of Sizewell C's private
    capital (GB Energy-Nuclear's holding is the existing 44.9% the comparator already has), so 0.
    """
    if rows is None or rows.empty or "ownership_share" not in rows.columns or "finance_vintage" not in rows.columns:
        return 0.0
    if "new_build_generation" in rows.columns:
        from scripts.public_ownership.levers import GBE_VALUES

        if rows["new_build_generation"].astype(str).isin(GBE_VALUES).any():
            return 0.0
    new_build = rows["finance_vintage"].astype(str).eq("new_build")
    share = pd.to_numeric(rows.loc[new_build, "ownership_share"], errors="coerce").fillna(0.0)
    return float(min(max(share.max(), 0.0), 1.0)) if len(share) else 0.0


def public_rate(network_lines: pd.DataFrame | None, levers: Mapping, fallback: float | None = None) -> float:
    """The public real rate at the case's ``public_rate_basis``, as the network account refinances the wires."""
    basis = str((levers or {}).get("public_rate_basis", "real_cpi"))
    if network_lines is not None and {"treatment", "public_rate_basis", "public_real_rate"} <= set(network_lines.columns):
        rows = network_lines.loc[network_lines["treatment"].astype(str).eq("acquired")
                                 & network_lines["public_rate_basis"].astype(str).eq(basis)]
        rates = pd.to_numeric(rows["public_real_rate"], errors="coerce").dropna().round(12).unique()
        if len(rates) == 1:
            return float(rates[0])
        if len(rates) > 1:
            raise ValueError(f"the network account carries {len(rates)} public rates for public_rate_basis {basis!r}")
    if fallback is None:
        raise ValueError(f"no public real rate for public_rate_basis {basis!r}: the Sizewell C RAB adjustment (OW10) "
                         "needs the network finance account")
    return float(fallback)


def household_adjustment(line_gbp_per_household: float, settings: Mapping, *, state_share: float,
                         rate: float) -> float:
    """GBP per household (negative when the bill falls) the line moves by: see the module.

    ``state_share`` is the case's share (the existing share comes off here); ``rate`` is
    the state's rate on the refinanced capital, the public rate plus :func:`premium`.
    """
    if not settings.get("enabled") or refinanced_share(settings, state_share) <= 0.0 or not line_gbp_per_household:
        return 0.0
    return -(float(line_gbp_per_household) * float(settings["return_share"])
             * return_saving_share(settings, state_share, rate))


def method_note(settings: Mapping, *, state_share: float, rate: float) -> str:
    merged = {**DEFAULT_SETTINGS, **dict(settings or {})}
    r_e, r_d = capital_rates(merged)
    equity_p, debt_p = private_capital(merged)
    return (f"Sizewell C's RAB levy, its return element ({merged['return_share']:.0%} of the line): the return on "
            f"the privately financed capital the case takes on ({refinanced_share(merged, state_share):.1%} of the "
            f"capital: the case's {state_share:.0%} less the {existing_state_capital_share(merged):.1%} the state "
            "already funds through DESNZ's equity and the National Wealth Fund's debt), private equity "
            f"(GBP {equity_p:.2f} bn) at its required return, {r_e:.2%} real (the NAO's "
            f"{merged['equity_return_nominal']:.1%} post-tax nominal equity IRR at {merged['cpih_inflation']:.0%} CPIH), "
            f"and private debt (GBP {debt_p:.2f} bn) at {r_d:.2%} real (the rate consistent with the "
            f"{merged['rab_real_rate']:.2%} construction-period WACC on this capital structure), less its cost at the "
            f"public cash rate, {rate:.4%}: "
            f"{return_saving_share(merged, state_share, rate):.1%} of the whole return (CPIH-real; NAO, Sizewell C, "
            "HC 33, 20 May 2026, paras 2.7-2.8, 2.14, 2.23 and Appendix One para 17); recovered over final demand like "
            "the levy itself (OW10, F19 decision 1 as refined on 6 Oct 2026).")
