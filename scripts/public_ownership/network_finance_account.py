"""The network finance account (item N6, backlog G2).

Networks are GBP 231 of the 2030 household bill, more than the generation lines
nationalisation reaches, and until now the largest block the ownership work could not
touch.  A regulated network is a simpler object than a generation fleet: its allowed
revenue contains an explicit return on an explicit Regulatory Asset Value, so the effect
of public ownership is arithmetic rather than a modelling exercise.

If the state buys the RAV at ``(1 + premium) x RAV`` and finances it at the public real
rate, the annual consumer saving is::

    RAV x (allowed_wacc - public_rate x (1 + premium)) + tax_allowance

The first term is the financing wedge, the second the corporation-tax allowance a public
owner does not pay.  Setting the bracket to zero gives the premium at which acquisition
stops paying: ``allowed_wacc / public_rate - 1``.

**Everything must be on one basis before it is subtracted.**  Two conversions are made
explicitly here rather than hidden:

1. *Price basis.*  ED2 figures are 2020/21 prices and RIIO-3 figures 2023/24 prices; the
   bill works in 2024.  Financial-year CPIH averages are built from the ONS quarterly
   series and declared in :data:`PRICE_BASIS`.
2. *Real against nominal.*  The ED2 WACC is CPIH-real.  The RIIO-3 WACC is semi-nominal,
   because RIIO-3 moved fixed-rate debt to a nominal allowance.  Subtracting a real public
   rate from it would overstate the transmission saving by about a point on a 54 bn asset
   base.  :func:`semi_nominal_to_cpih_real` converts it.

**The distribution asset base in 2030 is an assumption, not a published figure, and the
module refuses to pretend otherwise.**  RIIO-ED2 ends on 31 March 2028 and 2030 falls in
RIIO-ED3, which had only a methodology decision as at May 2026.  The ED2 closing RAV is
therefore rolled forward under a named rule that is written onto every output row.
Transmission needs no such rule: RIIO-3 runs to March 2031, so 2030 is inside a determined
period.

Usage::

    python scripts/public_ownership/network_finance_account.py <building_blocks.json>
        <snapshot.yaml> <assumptions.json> <recovery.json> <out_summary.json> <out_lines.csv>
        [--config config]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.consumer_bill.extract_network_price_control_building_blocks import (  # noqa: E402
    ED3_PERIOD, RIIO3_PERIOD, calendar_year_inside, semi_nominal_to_cpih_real,
)
from scripts.public_ownership.finance import capital_recovery_factor  # noqa: E402
from scripts.public_ownership.model_year import DEFAULT_MODEL_YEAR, recovery_model_year  # noqa: E402
from scripts.public_ownership.nationalised_tariff import household  # noqa: E402
from config.config_loader import read_config

LINE_TNUOS = "tnuos"
LINE_DUOS = "duos"
SECTORS = ("transmission", "distribution")
ROLL_FORWARD_RULES = ("hold_flat", "ed2_net_growth", "explicit")
#: Item Y12: the declared roll-forward past RIIO-ED3 (needs roll_forward_inputs).
DECLARED_ROLL_FORWARD_RULES = ("peak_demand",)
#: Item Y12: keys a ``network_account.model_years.<year>`` block may add for a year after
#: the determined price controls (written onto the settings only when configured).
YEAR_KEYS = ("transmission_roll_forward", "roll_forward_inputs", "acquisition_year", "guarantee_window")
TRANSMISSION_ROLL_FORWARD_RULES = ("declared",)

# ONS CPIH, all items, 2015=100, series L522 (mm23). Financial-year averages are the
# mean of the four quarters April to March; the 2024 calendar-year average of 132.9 is
# the model price year and matches the factor already used by the bill assumptions,
# which is what ties this conversion to the rest of the bill.
PRICE_BASIS = {
    "index": "ONS CPIH annual/quarterly average, all items, 2015=100 (series L522, dataset MM23)",
    "source_url": "https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l522/mm23",
    "model_price_year": 2024, "model_price_year_index": 132.9,
    "financial_year_quarters": {
        "2020/21": [108.7, 109.1, 109.3, 109.5],
        "2023/24": [128.9, 129.5, 130.3, 130.8],
    },
}


def explicit_rav_key(model_year: int | None = None) -> str:
    """The config key of the explicit distribution RAV for a model year (``distribution_rav_2030_gbp`` for 2030)."""
    return f"distribution_rav_{int(DEFAULT_MODEL_YEAR if model_year is None else model_year)}_gbp"


def explicit_transmission_rav_key(model_year: int | None = None) -> str:
    """F27 item 7: the config key of the transmission RAV of the model's calendar year (2024 prices)."""
    return f"transmission_rav_{int(DEFAULT_MODEL_YEAR if model_year is None else model_year)}_gbp"


#: F27 item 7: the distribution tax allowance on the ED2 period average (as before) or its final year.
DISTRIBUTION_TAX_BASES = ("period_average", "final_year")


def excess_return_settings(block: Mapping) -> dict:
    """F27 item 7 (user, 20 Oct 2026): the return on regulated equity above the allowed return, per sector.

    ``rates`` holds named sets of percentage points of RoRE above the allowed return
    (``riio2``, ``riio1``); ``central`` and ``high`` name the set each uses, and
    ``equity_share`` is one less the notional gearing. Bought wires earn it for the state
    (the central set); the high set is reported beside it, not a case.
    """
    block = dict(block)
    rates = {name: {sector: float(v) for sector, v in dict(values).items()}
             for name, values in dict(block.get("rates") or {}).items()}
    for key in ("central", "high"):
        if block.get(key) not in rates:
            raise ValueError(f"network_account.excess_return.{key} must name one of rates {sorted(rates)}")
    shares = {sector: float(v) for sector, v in dict(block.get("equity_share") or {}).items()}
    for name, values in rates.items():
        if set(values) != set(SECTORS):
            raise ValueError(f"network_account.excess_return.rates.{name} needs {list(SECTORS)}")
    if set(shares) != set(SECTORS) or not all(0.0 < v <= 1.0 for v in shares.values()):
        raise ValueError(f"network_account.excess_return.equity_share needs {list(SECTORS)} in (0, 1]")
    return {"central": block["central"], "high": block["high"], "rates": rates, "equity_share": shares,
            "sources": dict(block.get("sources") or {})}


def excess_return_gbp(settings_block: Mapping | None, sector: str, rav_gbp: float, which: str = "central") -> float:
    """The excess return on ``sector``'s regulated equity, RAV x equity share x pp above allowed (0 without it)."""
    if not settings_block:
        return 0.0
    rates = settings_block["rates"][settings_block[which]]
    return float(rav_gbp) * float(settings_block["equity_share"][sector]) * float(rates[sector])


def price_factor(from_basis: str) -> float:
    """Multiplier from a published financial-year price basis to the model's 2024 basis."""
    quarters = PRICE_BASIS["financial_year_quarters"].get(from_basis)
    if not quarters:
        raise ValueError(f"no CPIH quarters declared for price basis {from_basis!r}; "
                         f"declared: {sorted(PRICE_BASIS['financial_year_quarters'])}")
    return float(PRICE_BASIS["model_price_year_index"]) / (sum(quarters) / len(quarters))


def year_override(block: Mapping | None, model_year: int | None) -> dict:
    """The ``network_account.model_years.<year>`` block for a model year, or empty (item Y12)."""
    years = dict((block or {}).get("model_years") or {})
    if model_year is None:
        return {}
    return dict(years.get(int(model_year)) or years.get(str(int(model_year))) or {})


def settings(block: Mapping | None, model_year: int | None = None) -> dict:
    """Read and check ``public_ownership.network_account``.

    Item Y12: ``model_years.<year>`` overrides the block for that model year only (the
    2050 roll-forward inputs, its guarantee table); the CP30 year reads the block as before.
    """
    block = dict(block or {})
    override = year_override(block, model_year)
    block.pop("model_years", None)
    block.update(override)
    rule = str(block.get("distribution_roll_forward", "ed2_net_growth"))
    if rule not in ROLL_FORWARD_RULES + DECLARED_ROLL_FORWARD_RULES:
        raise ValueError(f"network_account.distribution_roll_forward={rule!r}; supported: "
                         f"{list(ROLL_FORWARD_RULES + DECLARED_ROLL_FORWARD_RULES)}")
    years = float(block.get("roll_forward_years", 2.5))
    if years < 0:
        raise ValueError("network_account.roll_forward_years must not be negative")
    if "acquisition_premium_over_rav" in block and "acquisition_premium_table" in block:
        raise ValueError("network_account: give acquisition_premium_table (the evidence, user decision 7) or the "
                         "retired flat acquisition_premium_over_rav, not both")
    premium = float(block.get("acquisition_premium_over_rav", 0.0))
    if premium < -1.0:
        raise ValueError("network_account.acquisition_premium_over_rav must be greater than -1")
    # Item Y4: the explicit RAV is named for the model year it is for.
    rav_key = explicit_rav_key(model_year)
    explicit = block.get(rav_key)
    if rule == "explicit" and not (isinstance(explicit, (int, float)) and explicit > 0):
        raise ValueError(f"network_account.{rav_key} must be positive under distribution_roll_forward: explicit")
    inflation = float(block.get("semi_nominal_inflation", 0.02))
    fixed_share = float(block.get("fixed_rate_debt_share", 0.75))
    include_tax = bool(block.get("include_tax_allowance", True))
    annuity_years = float(block.get("premium_annuity_years", PREMIUM_ANNUITY_YEARS))
    if annuity_years <= 0:
        raise ValueError("network_account.premium_annuity_years must be positive")
    allowed = dict(block.get("allowed_wacc_cpih_real") or {})
    for sector, value in allowed.items():
        if sector not in SECTORS:
            raise ValueError(f"network_account.allowed_wacc_cpih_real.{sector}: sectors are {list(SECTORS)}")
        if not (isinstance(value, (int, float)) and 0.0 < float(value) < 0.2):
            raise ValueError(f"network_account.allowed_wacc_cpih_real.{sector} must be a real rate in (0, 0.2)")
    out = {"distribution_roll_forward": rule, "roll_forward_years": years,
           "acquisition_premium_over_rav": premium, rav_key: explicit,
           "semi_nominal_inflation": inflation, "fixed_rate_debt_share": fixed_share,
           "include_tax_allowance": include_tax, "premium_annuity_years": annuity_years,
           "allowed_wacc_cpih_real": {k: float(v) for k, v in allowed.items()},
           "allowed_wacc_sources": dict(block.get("allowed_wacc_sources") or {}),
           "acquisition_premium_table": block.get("acquisition_premium_table"),
           "guarantee_pass_through_table": block.get("guarantee_pass_through_table")}
    # Item Y12: the roll-forward keys, present only when a model_years block sets them,
    # so the CP30 year's settings (written into its summary) are unchanged.
    for key in YEAR_KEYS:
        if key in block and block[key] is not None:
            out[key] = block[key]
    # F27 item 7 (23 Oct 2026): the transmission RAV of the model's calendar year, the ED2
    # final year's distribution tax allowance and the excess-return stream bought wires earn.
    t_key = explicit_transmission_rav_key(model_year)
    if block.get(t_key) is not None:
        if not (isinstance(block[t_key], (int, float)) and block[t_key] > 0):
            raise ValueError(f"network_account.{t_key} must be positive")
        out[t_key] = float(block[t_key])
        out[f"{t_key}_source"] = str(block.get(f"{t_key}_source", ""))
    tax_basis = str(block.get("distribution_tax_allowance_basis", "period_average"))
    if tax_basis not in DISTRIBUTION_TAX_BASES:
        raise ValueError(f"network_account.distribution_tax_allowance_basis={tax_basis!r}; "
                         f"supported: {list(DISTRIBUTION_TAX_BASES)}")
    if tax_basis == "final_year":
        value = block.get("distribution_tax_allowance_final_year_gbp_m")
        if not (isinstance(value, (int, float)) and value >= 0):
            raise ValueError("network_account.distribution_tax_allowance_basis: final_year needs "
                             "distribution_tax_allowance_final_year_gbp_m (the ED2 final year, the block's price basis)")
        out["distribution_tax_allowance_basis"] = tax_basis
        out["distribution_tax_allowance_final_year_gbp_m"] = float(value)
        out["distribution_tax_allowance_source"] = str(block.get("distribution_tax_allowance_source", ""))
    if block.get("excess_return"):
        out["excess_return"] = excess_return_settings(block["excess_return"])
    if "transmission_roll_forward" in out and out["transmission_roll_forward"] not in TRANSMISSION_ROLL_FORWARD_RULES:
        raise ValueError(f"network_account.transmission_roll_forward={out['transmission_roll_forward']!r}; "
                         f"supported: {list(TRANSMISSION_ROLL_FORWARD_RULES)}")
    if (rule == "peak_demand" or "transmission_roll_forward" in out) and not out.get("roll_forward_inputs"):
        raise ValueError("network_account: a declared roll-forward (distribution_roll_forward: peak_demand or "
                         "transmission_roll_forward) needs roll_forward_inputs "
                         "(data/market/network_cost_sources/rav_roll_forward_2050.yaml)")
    if "acquisition_year" in out and model_year is not None and int(out["acquisition_year"]) > int(model_year):
        raise ValueError("network_account.acquisition_year must not be after the model year")
    return out


#: User decision 7 (29 Sep 2026): a premium over RAV is charged on the same basis as
#: the legacy fleet's purchase, an annuity at the public rate over 30 years.  Until
#: then it was charged as perpetual interest (``public_rate x premium``), which is why
#: the published break-evens (43% distribution, 75% transmission) halve on this basis.
PREMIUM_ANNUITY_YEARS = 30.0


def distribution_rav(blocks: Mapping, cfg: Mapping, model_year: int | None = None) -> tuple[float, str]:
    """Distribution RAV in the model year on the model price basis, and the rule that produced it.

    ED2 ends on 31 March 2028, so this is always an assumption.  ``hold_flat`` keeps the
    ED2 closing balance; ``ed2_net_growth`` continues the final ED2 year's net additions
    less depreciation, which is conservative because ED3 is expected to step investment
    up for electrification; ``explicit`` takes a configured figure
    (``distribution_rav_<year>_gbp``).  A model year after RIIO-ED3 takes only an
    explicit figure (item Y4): a roll-forward of ``roll_forward_years`` from ED2 is a
    CP30-year rule and would silently return a 2030 asset base.
    """
    year = int(DEFAULT_MODEL_YEAR if model_year is None else model_year)
    dist = blocks["distribution"]
    factor = price_factor(dist["price_basis_key"])
    closing = float(dist["closing_rav_gbp_m_2027_28"]) * 1e6 * factor
    rule = cfg["distribution_roll_forward"]
    if rule == "explicit":
        return float(cfg[explicit_rav_key(year)]), "explicit configured value"
    if rule == "peak_demand":
        result = roll_forward_result(blocks, cfg, year)
        return float(result[f"distribution_rav_{year}_gbp"]), result["distribution_basis"]
    if pd.Timestamp(f"{year}-01-01") > pd.Timestamp(ED3_PERIOD[1]):
        raise ValueError(f"calendar {year} is after RIIO-ED3 ({ED3_PERIOD[0]} to {ED3_PERIOD[1]}): the distribution "
                         f"RAV needs distribution_roll_forward: explicit with network_account.{explicit_rav_key(year)} "
                         "(a declared roll-forward; FES 2050 plan task Y12)")
    if rule == "hold_flat":
        return closing, f"ED2 closing RAV at 31 March 2028 held flat to {year}"
    growth = (float(dist["net_additions_gbp_m_2027_28"]) - float(dist["depreciation_gbp_m_2027_28"])) * 1e6 * factor
    years = float(cfg["roll_forward_years"])
    return closing + growth * years, (
        f"ED2 closing RAV at 31 March 2028 plus {years:g} years of the final ED2 year's net additions "
        f"less depreciation ({growth / 1e9:.2f} bn a year); ED3 is expected to step investment up, so this is conservative")


def distribution_rav_2030(blocks: Mapping, cfg: Mapping) -> tuple[float, str]:
    """The CP30-year name of :func:`distribution_rav`, kept for its callers (the default model year)."""
    return distribution_rav(blocks, cfg)


def ed2_anchor_rav(blocks: Mapping, cfg: Mapping) -> float:
    """The CP30-year distribution RAV (``ed2_net_growth``), the anchor a later year rolls forward from (item Y12)."""
    dist = blocks["distribution"]
    factor = price_factor(dist["price_basis_key"])
    closing = float(dist["closing_rav_gbp_m_2027_28"]) * 1e6 * factor
    growth = (float(dist["net_additions_gbp_m_2027_28"]) - float(dist["depreciation_gbp_m_2027_28"])) * 1e6 * factor
    return closing + growth * float(cfg["roll_forward_years"])


def transmission_anchor_rav(blocks: Mapping) -> float:
    """The RIIO-3 period-average RAV on the model basis (the CP30-year transmission RAV)."""
    trans = blocks["transmission"]
    return float(trans["total_rav_gbp_m"]) * 1e6 * price_factor(trans["price_basis_key"])


_ROLL_FORWARD_CACHE: dict = {}


def roll_forward_result(blocks: Mapping, cfg: Mapping, model_year: int) -> dict:
    """Item Y12: both RAVs rolled forward to ``model_year`` from the configured inputs (cached per inputs and anchors)."""
    from scripts.public_ownership.network_rav_roll_forward import load_inputs, roll_forward

    path = cfg.get("roll_forward_inputs")
    if not path:
        raise ValueError("network_account.roll_forward_inputs is not configured")
    dist = blocks["distribution"]
    anchors = {
        "transmission_anchor_rav_gbp": transmission_anchor_rav(blocks),
        "distribution_anchor_rav_gbp": ed2_anchor_rav(blocks, cfg),
        "distribution_anchor_depreciation_gbp": float(dist["depreciation_gbp_m_2027_28"]) * 1e6
                                                * price_factor(dist["price_basis_key"]),
    }
    key = (str(path), int(model_year), tuple(sorted(anchors.items())))
    if key not in _ROLL_FORWARD_CACHE:
        _ROLL_FORWARD_CACHE[key] = roll_forward(anchors, load_inputs(path), int(model_year),
                                                fy_price_factor=price_factor)
    return _ROLL_FORWARD_CACHE[key]


def premium_annual_cost(rav_gbp: float, premium: float, public_rate: float, annuity_years: float) -> float:
    """The annual cost of a premium over RAV: ``premium x RAV x CRF(public rate, years)``."""
    if premium == 0.0:
        return 0.0
    return float(premium) * float(rav_gbp) * capital_recovery_factor(float(public_rate), float(annuity_years))


def sector_account(*, rav_gbp: float, allowed_wacc_cpih_real: float, public_rate: float,
                   premium: float, tax_allowance_gbp: float,
                   annuity_years: float = PREMIUM_ANNUITY_YEARS,
                   premium_rav_gbp: float | None = None, premium_payable: bool = True,
                   excess_return_gbp: float = 0.0) -> dict:
    """The financing wedge, the premium's annuity, the tax allowance and the premiums at which acquisition stops paying.

    ``financing_saving_gbp = RAV x (allowed WACC - public rate) - premium x RAV x
    CRF(public rate, 30)``: consumers stop paying the allowed return on the RAV and
    pay the public financing cost on the same RAV, plus the annuity on the premium,
    the legacy purchase's own basis (user decision 7, 29 Sep 2026).  The break-even
    premium sets it (and, with the tax allowance, the total) to zero.

    Item Y12: when the wires were bought in an earlier year (decision 2: 2030), the
    premium was paid on that year's RAV, ``premium_rav_gbp``, and its annuity runs only
    while ``premium_payable``; the wedge is on the model year's RAV.

    ``excess_return_gbp`` (F27 item 7): the return above the allowed return the vendors
    earned, now the state's; a transfer from them, in the total saving and the break-even
    premium that counts it, never in the financing wedge (the resource part).
    """
    crf = capital_recovery_factor(float(public_rate), float(annuity_years))
    gross = rav_gbp * (allowed_wacc_cpih_real - public_rate)
    base = rav_gbp if premium_rav_gbp is None else float(premium_rav_gbp)
    premium_cost = premium_annual_cost(base, premium, public_rate, annuity_years) if premium_payable else 0.0
    wedge = gross - premium_cost
    extra = {}
    if premium_rav_gbp is not None:
        extra = {
            "premium_rav_gbp": base, "premium_payable": bool(premium_payable),
            "premium_capital_gbp": float(premium) * base,
            "break_even_premium_over_rav": (gross / (base * crf)) if crf and base else None,
            "break_even_premium_over_rav_with_tax_allowance": ((gross + tax_allowance_gbp) / (base * crf))
                                                              if crf and base else None,
        }
    return {
        "rav_gbp": rav_gbp,
        "allowed_wacc_cpih_real": allowed_wacc_cpih_real,
        "public_real_rate": public_rate,
        "premium_over_rav": float(premium),
        "premium_capital_gbp": float(premium) * rav_gbp,
        "premium_annual_cost_gbp": premium_cost,
        "premium_annuity_years": float(annuity_years),
        "financing_wedge_gross_gbp": gross,
        "financing_wedge": (allowed_wacc_cpih_real - public_rate) - (premium_cost / rav_gbp if rav_gbp else 0.0),
        "financing_saving_gbp": wedge,
        "tax_allowance_saving_gbp": tax_allowance_gbp,
        "excess_return_saving_gbp": float(excess_return_gbp),
        "total_saving_gbp": wedge + tax_allowance_gbp + float(excess_return_gbp),
        "break_even_premium_over_rav": ((allowed_wacc_cpih_real - public_rate) / crf) if crf else None,
        "break_even_premium_over_rav_with_tax_allowance": (
            ((allowed_wacc_cpih_real - public_rate) * rav_gbp + tax_allowance_gbp) / (rav_gbp * crf)
            if crf and rav_gbp else None),
        "break_even_premium_over_rav_with_tax_allowance_and_excess_return": (
            ((allowed_wacc_cpih_real - public_rate) * rav_gbp + tax_allowance_gbp + float(excess_return_gbp))
            / (rav_gbp * crf) if crf and rav_gbp else None),
        "break_even_premium_over_rav_perpetual_basis": (allowed_wacc_cpih_real / public_rate) - 1.0 if public_rate else None,
        **extra,
    }


TREATMENT_ACQUIRED = "acquired"
TREATMENT_DEBT_GUARANTEE = "debt_guarantee"
TREATMENTS = (TREATMENT_ACQUIRED, TREATMENT_DEBT_GUARANTEE)
#: The case levers a network row is keyed on, and each one's central value.
VARIANT_LEVERS = {
    "network_acquisition_premium": "central", "public_rate_basis": "real_cpi",
    "guarantee_pass_through": "new_and_refinanced",
}


def guarantee_sector_account(*, rav_gbp: float, allowed_wacc_cpih_real: float, notional_gearing: float,
                             debt_compression: float, pass_through_share: float = 1.0) -> dict:
    """Item J6: the workbook's Scenario 2 network treatment.

    A state guarantee compresses the cost of debt by ``debt_compression`` (75 bp,
    Assumptions!C31) on the geared share of the RAV; equity, tax and ownership are
    unchanged, so there is no tax-allowance saving and no acquisition
    (``Cost of Capital``!D39: ``wacc_riio3 - 0.55 x qp_spread``).  User decision 6
    (29 Sep 2026): the compression reaches the 2030 allowance only through the debt
    issued or refinanced 2026-2030 (``pass_through_share`` of the notional debt,
    Ofgem's RAV-weighted trailing average), and that is the debt guaranteed.
    """
    debt = float(rav_gbp) * float(notional_gearing) * float(pass_through_share)
    wedge = float(notional_gearing) * float(debt_compression) * float(pass_through_share)
    return {
        "rav_gbp": rav_gbp,
        "allowed_wacc_cpih_real": allowed_wacc_cpih_real,
        "financing_rate": allowed_wacc_cpih_real - wedge,
        "financing_wedge": wedge,
        "financing_saving_gbp": debt * float(debt_compression),
        "tax_allowance_saving_gbp": 0.0,
        "total_saving_gbp": debt * float(debt_compression),
        "break_even_premium_over_rav": None,
        "notional_gearing": float(notional_gearing),
        "debt_compression": float(debt_compression),
        "pass_through_share": float(pass_through_share),
        "guaranteed_debt_gbp": debt,
    }


def build_account(blocks: Mapping, public_rate: float, cfg: Mapping, conv: Mapping, *,
                  treatment: str = TREATMENT_ACQUIRED, debt_compression: float | None = None,
                  premium: float | None = None, pass_through: Mapping[str, float] | None = None,
                  variant: Mapping[str, str] | None = None,
                  expected_loss_bps: Mapping[str, float] | None = None,
                  model_year: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Both sectors, their bill lines, and the assumptions each rests on.

    ``model_year`` (item Y4, default the CP30 year) decides whether the RIIO-3
    transmission RAV applies (the calendar year must lie inside the period) and
    which explicit distribution RAV is read.

    ``treatment`` is the networks lever value: ``acquired`` (the RAV bought and
    financed at ``public_rate``, a ``premium`` over RAV annuitised over 30 years)
    or ``debt_guarantee`` (the allowed WACC less notional gearing times
    ``debt_compression`` on the ``pass_through`` share of the debt, item J6).
    ``variant`` is written onto every row (the case levers the row answers to);
    ``expected_loss_bps`` ({guarantee_cost setting: bp}) adds each setting's
    expected loss on the guaranteed debt, which a guarantee fee equals.
    """
    if treatment not in TREATMENTS:
        raise ValueError(f"network treatment {treatment!r}; supported: {list(TREATMENTS)}")
    year = int(DEFAULT_MODEL_YEAR if model_year is None else model_year)
    rolled = not calendar_year_inside(year, RIIO3_PERIOD)
    if rolled and cfg.get("transmission_roll_forward") not in TRANSMISSION_ROLL_FORWARD_RULES:
        raise ValueError(f"calendar {year} is outside RIIO-3 ({RIIO3_PERIOD[0]} to {RIIO3_PERIOD[1]}): the "
                         "transmission RAV needs a declared roll-forward (network_account.model_years."
                         f"{year}.transmission_roll_forward: declared, with roll_forward_inputs; FES 2050 plan "
                         "task Y12)")
    if treatment == TREATMENT_DEBT_GUARANTEE and debt_compression is None:
        raise ValueError("the debt_guarantee treatment needs the guarantee's debt compression (workbook Assumptions!C31)")
    premium = float(cfg["acquisition_premium_over_rav"] if premium is None else premium)
    shares = {"transmission": 1.0, "distribution": 1.0, **dict(pass_through or {})}
    allowed = dict(cfg.get("allowed_wacc_cpih_real") or {})
    trans, dist = blocks["transmission"], blocks["distribution"]
    t_factor = price_factor(trans["price_basis_key"])
    t_rav = float(trans["total_rav_gbp_m"]) * 1e6 * t_factor
    rf = roll_forward_result(blocks, cfg, year) if rolled else None
    if rf is not None:
        t_rav = float(rf[f"transmission_rav_{year}_gbp"])
    # F27 item 7 (user, 20 Oct 2026): the RAV of the calendar year, not the RIIO-3 period average.
    t_explicit = cfg.get(explicit_transmission_rav_key(year)) if rf is None else None
    if t_explicit is not None:
        t_rav = float(t_explicit)
    excess = cfg.get("excess_return") if treatment == TREATMENT_ACQUIRED else None
    # Item Y12 (decision 2): wires bought in acquisition_year pay the premium on that
    # year's RAV, annuitised from then; the CP30 year buys in its own year, as before.
    acquired_in = cfg.get("acquisition_year")
    premium_terms: dict = {}
    if acquired_in is not None and int(acquired_in) != year:
        payable = year < int(acquired_in) + float(cfg.get("premium_annuity_years", PREMIUM_ANNUITY_YEARS))
        premium_terms = {
            "transmission": {"premium_rav_gbp": transmission_anchor_rav(blocks), "premium_payable": payable},
            "distribution": {"premium_rav_gbp": ed2_anchor_rav(blocks, cfg), "premium_payable": payable},
        }
    if "transmission" in allowed:
        t_wacc = float(allowed["transmission"])
        t_basis = ("Ofgem's own CPIH-real RIIO-3 allowance, configured: "
                   + str((cfg.get("allowed_wacc_sources") or {}).get("transmission", "")))
    else:
        t_wacc = semi_nominal_to_cpih_real(
            float(trans["wacc_semi_nominal"]), gearing=float(trans["notional_gearing"]),
            inflation=float(cfg["semi_nominal_inflation"]), fixed_rate_debt_share=float(cfg["fixed_rate_debt_share"]))
        t_basis = (f"published {float(trans['wacc_semi_nominal']):.4f} semi-nominal converted to CPIH-real at "
                   f"{cfg['semi_nominal_inflation']:.1%} inflation and a {cfg['fixed_rate_debt_share']:.0%} fixed-rate debt share")
    if treatment == TREATMENT_DEBT_GUARANTEE:
        transmission = guarantee_sector_account(rav_gbp=t_rav, allowed_wacc_cpih_real=t_wacc,
                                                notional_gearing=float(trans["notional_gearing"]),
                                                debt_compression=float(debt_compression),
                                                pass_through_share=float(shares["transmission"]))
    else:
        transmission = sector_account(rav_gbp=t_rav, allowed_wacc_cpih_real=t_wacc, public_rate=public_rate,
                                      premium=premium, tax_allowance_gbp=0.0,
                                      annuity_years=float(cfg.get("premium_annuity_years", PREMIUM_ANNUITY_YEARS)),
                                      excess_return_gbp=excess_return_gbp(excess, "transmission", t_rav),
                                      **premium_terms.get("transmission", {}))
    if rf is not None:
        transmission["rav_basis"] = rf["transmission_basis"]
    elif t_explicit is not None:
        transmission["rav_basis"] = (cfg.get(f"{explicit_transmission_rav_key(year)}_source")
                                     or f"calendar {year} RAV, configured")
    else:
        transmission["rav_basis"] = f"RIIO-3 period-average RAV; {year} is inside the determined period"
    transmission["wacc_basis"] = t_basis
    if treatment == TREATMENT_ACQUIRED:
        # F27 item 7: nil, not missing (RIIO-3 FD Finance Annex, Appendix 2, Tables 52, 54 and 56, pp.170-173:
        # no tax allowance for NGET, SHET or SPT in 2026/27-2030/31; full expensing leaves no taxable profit).
        transmission["tax_basis"] = ("nil: the RIIO-3 Final Determinations give no tax allowance for NGET, SHET or "
                                     "SPT in any year (Finance Annex, Appendix 2, Tables 52, 54 and 56)")
    else:
        transmission["tax_basis"] = "unchanged: a guarantee leaves the private owner and its tax position in place"

    d_rav, d_rule = distribution_rav(blocks, cfg, year)
    d_factor = price_factor(dist["price_basis_key"])
    final_year_tax = cfg.get("distribution_tax_allowance_basis") == "final_year"
    d_tax_gbp_m = (float(cfg["distribution_tax_allowance_final_year_gbp_m"]) if final_year_tax
                   else float(dist["tax_allowance_gbp_m_period_average"]))
    d_tax = d_tax_gbp_m * 1e6 * d_factor if cfg["include_tax_allowance"] else 0.0
    tax_scaled = cfg["distribution_roll_forward"] == "peak_demand" and bool(cfg["include_tax_allowance"])
    if tax_scaled:
        # Item Y12: the tax allowance follows the equity return, so it is scaled with the RAV
        # from the ED2 period average it was published on (declared).
        d_tax = d_tax * d_rav / (float(dist["opening_rav_gbp_m_period_average"]) * 1e6 * d_factor)
    if "distribution" in allowed:
        d_wacc = float(allowed["distribution"])
        where = ("is inside ED3" if calendar_year_inside(year, ED3_PERIOD)
                 else "is outside ED3; the allowance is held, declared")
        d_basis = (f"Ofgem's own CPIH-real ED3 allowance (calendar {year} {where}), configured: "
                   + str((cfg.get("allowed_wacc_sources") or {}).get("distribution", "")))
    else:
        d_wacc = float(dist["vanilla_wacc_cpih_real"])
        d_basis = "ED2 vanilla WACC, already CPIH-real"
    if treatment == TREATMENT_DEBT_GUARANTEE:
        distribution = guarantee_sector_account(rav_gbp=d_rav, allowed_wacc_cpih_real=d_wacc,
                                                notional_gearing=float(dist.get("notional_gearing", 0.6)),
                                                debt_compression=float(debt_compression),
                                                pass_through_share=float(shares["distribution"]))
        distribution["tax_basis"] = "unchanged: a guarantee leaves the private owner and its tax position in place"
    else:
        distribution = sector_account(rav_gbp=d_rav, allowed_wacc_cpih_real=d_wacc,
                                      public_rate=public_rate, premium=premium, tax_allowance_gbp=d_tax,
                                      annuity_years=float(cfg.get("premium_annuity_years", PREMIUM_ANNUITY_YEARS)),
                                      excess_return_gbp=excess_return_gbp(excess, "distribution", d_rav),
                                      **premium_terms.get("distribution", {}))
        ed2 = "ED2 final-year" if final_year_tax else "ED2 period-average"
        if not cfg["include_tax_allowance"]:
            distribution["tax_basis"] = "excluded by configuration"
        elif tax_scaled:
            distribution["tax_basis"] = (f"{ed2} tax allowance, rebased and scaled by the rolled-forward RAV over "
                                         "the ED2 period-average RAV (declared)")
        elif final_year_tax:
            distribution["tax_basis"] = (f"{ed2} tax allowance, rebased; an upper bound for calendar 2030 (ED3). "
                                         + str(cfg.get("distribution_tax_allowance_source", "")))
        else:
            distribution["tax_basis"] = f"{ed2} tax allowance, rebased"
    distribution["rav_basis"] = d_rule
    distribution["wacc_basis"] = d_basis

    variant = dict(variant or {})
    if premium_terms:
        span = float(cfg.get("premium_annuity_years", PREMIUM_ANNUITY_YEARS))
        for account in (transmission, distribution):
            account["premium_basis"] = (f"bought in {int(acquired_in)} at (1 + premium) x the {int(acquired_in)} RAV, "
                                        f"the premium annuitised over {span:g} years from {int(acquired_in)}; the "
                                        f"wedge is on the {year} RAV")
    rows = []
    for sector, account, line in (("transmission", transmission, LINE_TNUOS), ("distribution", distribution, LINE_DUOS)):
        row = {
            "treatment": treatment, "sector": sector, "bill_line_adjusted": line,
            **{lever: variant.get(lever, "") for lever in VARIANT_LEVERS},
            "rav_gbp": account["rav_gbp"], "allowed_wacc_cpih_real": account["allowed_wacc_cpih_real"],
            "financing_saving_gbp": account["financing_saving_gbp"],
            "tax_allowance_saving_gbp": account["tax_allowance_saving_gbp"],
            "excess_return_saving_gbp": float(account.get("excess_return_saving_gbp", 0.0)),
            "total_saving_gbp": account["total_saving_gbp"],
            "saving_gbp_per_3100kwh_ex_vat": -household(
                account["total_saving_gbp"], conv["final_demand_mwh"], conv["factor"], conv["mwh"]),
            "household_gbp_per_system_gbp": household(1.0, conv["final_demand_mwh"], conv["factor"], conv["mwh"]),
            "rav_basis": account["rav_basis"], "wacc_basis": account["wacc_basis"], "tax_basis": account["tax_basis"],
        }
        for key in ("public_real_rate", "premium_over_rav", "premium_capital_gbp", "premium_annual_cost_gbp",
                    "financing_wedge_gross_gbp", "break_even_premium_over_rav",
                    "break_even_premium_over_rav_with_tax_allowance", "notional_gearing", "debt_compression",
                    "pass_through_share", "guaranteed_debt_gbp"):
            row[key] = account.get(key)
        if excess:
            # F27 item 7: the central set is in the saving; the high set is reported beside it, not a case.
            row["break_even_premium_over_rav_with_tax_allowance_and_excess_return"] = account.get(
                "break_even_premium_over_rav_with_tax_allowance_and_excess_return")
            row["excess_return_high_gbp"] = excess_return_gbp(excess, sector, account["rav_gbp"], "high")
            row["excess_return_basis"] = (
                f"RoRE above the allowed return at {excess['central']} rates "
                f"({excess['rates'][excess['central']][sector]:.2%} of equity RAV, equity share "
                f"{excess['equity_share'][sector]:.0%}); {excess['high']} rates reported as the high case. "
                + str(excess["sources"].get(sector, "")))
        # Item Y12: the acquisition-year premium base, only when the wires were bought earlier.
        for key in ("premium_rav_gbp", "premium_payable"):
            if key in account:
                row[key] = account[key]
        if treatment == TREATMENT_DEBT_GUARANTEE:
            # A guarantee fee equal to the expected loss (lever network_incidence), per setting.
            for setting, bps in dict(expected_loss_bps or {}).items():
                row[f"guarantee_expected_loss_{setting}_gbp"] = float(account["guaranteed_debt_gbp"]) * float(bps) / 1e4
        rows.append(row)
    frame = pd.DataFrame(rows)
    total = float(frame["total_saving_gbp"].sum())
    declared = rf is not None or cfg["distribution_roll_forward"] == "peak_demand"
    if declared:
        result = rf or roll_forward_result(blocks, cfg, year)
        notes = [f"Transmission and distribution in {year} are declared roll-forwards, not published figures. "
                 + result["statement"],
                 f"Transmission: {transmission['rav_basis']}.", f"Distribution: {distribution['rav_basis']}."]
    else:
        notes = [
            f"Distribution in {year} is a roll-forward of the ED2 closing RAV, not a published figure: ED2 ends "
            "31 March 2028 and RIIO-ED3 had only a methodology decision as at May 2026. The rule is on every row.",
            "Transmission needs no roll-forward: RIIO-3 runs to 31 March 2031.",
        ]
    summary = {
        "treatment": treatment, "variant": variant,
        "public_real_rate": public_rate, "settings": {k: v for k, v in cfg.items()}, "price_basis": PRICE_BASIS,
        "transmission": transmission, "distribution": distribution,
        "total_saving_gbp": total,
        "total_saving_gbp_per_3100kwh_ex_vat": -household(total, conv["final_demand_mwh"], conv["factor"], conv["mwh"]),
        "recovery": "network charges are recovered over final demand, like the LDES cap-and-floor line",
        "notes": [
            "Buying a RAV-regulated network at RAV is close to self-financing: consumers stop paying the "
            "allowed return on the RAV and start paying the public financing cost on the same RAV, so the "
            "saving is the wedge between the two rates; a premium over RAV is an annuity at the public rate "
            "over 30 years on top, the legacy purchase's own basis (user decision 7, 29 Sep 2026).",
            "The break-even premium is where the wedge (or the wedge plus the tax allowance) equals the premium's "
            "annuity; the perpetual-interest break-even the account used before 29 Sep 2026 is reported beside it.",
            *notes,
            "Operating costs, depreciation and pass-through are unchanged: ownership changes the cost of "
            "capital and the tax allowance, not the cost of running the wires.",
        ],
    }
    if declared:
        summary["roll_forward"] = dict(result)
        summary[f"transmission_rav_{year}_gbp"] = t_rav
        summary[f"distribution_rav_{year}_gbp"] = d_rav
    return frame, summary


def variant_key(variant: Mapping[str, str]) -> str:
    return "|".join(f"{lever}={variant.get(lever, '')}" for lever in VARIANT_LEVERS)


def public_rates(snapshot: Mapping) -> dict[str, float]:
    """The public cash rate on each ``public_rate_basis`` value (:func:`finance.public_borrowing_rate`; F29:
    workbook v2.0's lending margins)."""
    from scripts.public_ownership.finance import public_borrowing_rate
    from scripts.public_ownership.levers import RATE_BASIS_MARGIN_KEYS

    return {basis: public_borrowing_rate(snapshot, {"public_rate_basis": basis}) for basis in RATE_BASIS_MARGIN_KEYS}


def build_all(blocks: Mapping, snapshot: Mapping, cfg: Mapping, conv: Mapping, *, premiums: Mapping[str, float],
              pass_through: Mapping[str, Mapping[str, float]], debt_compression: float | None,
              expected_loss_bps: Mapping[str, float] | None = None,
              model_year: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Every row the networks lever and its variants can ask for, and the summary with the central treatments.

    Acquired: each premium setting at each public-rate basis.  The rows carry the variant
    levers, so a reader picks the case's own rows (:func:`select_rows`).  The debt guarantee
    (each pass-through setting) is built only when ``debt_compression`` is given: F29 (26 Oct
    2026) retired ``networks: debt_guarantee`` with v11-2's 75 bp, so the command line passes none.
    """
    rates = public_rates(snapshot)
    frames, variants = [], {}
    central_acquired = central_guarantee = None
    for premium_setting, premium in premiums.items():
        for basis, rate in rates.items():
            variant = {"network_acquisition_premium": premium_setting, "public_rate_basis": basis}
            frame, summary = build_account(blocks, rate, cfg, conv, premium=premium, variant=variant,
                                           model_year=model_year)
            frames.append(frame)
            variants[variant_key(variant)] = {k: v for k, v in summary.items() if k != "notes"}
            if premium_setting == VARIANT_LEVERS["network_acquisition_premium"] and basis == VARIANT_LEVERS["public_rate_basis"]:
                central_acquired = summary
    for setting, shares in (pass_through.items() if debt_compression is not None else ()):
        variant = {"guarantee_pass_through": setting}
        frame, summary = build_account(blocks, rates["real_cpi"], cfg, conv, treatment=TREATMENT_DEBT_GUARANTEE,
                                       debt_compression=debt_compression, pass_through=shares, variant=variant,
                                       expected_loss_bps=expected_loss_bps, model_year=model_year)
        frames.append(frame)
        variants[variant_key(variant)] = {k: v for k, v in summary.items() if k != "notes"}
        if setting == VARIANT_LEVERS["guarantee_pass_through"]:
            central_guarantee = summary
    if central_acquired is None or (debt_compression is not None and central_guarantee is None):
        raise ValueError("the network account needs the central premium and the central pass-through settings")
    summary = dict(central_acquired)
    summary["treatments"] = {TREATMENT_ACQUIRED: {k: v for k, v in central_acquired.items() if k != "notes"}}
    if central_guarantee is not None:
        summary["treatments"][TREATMENT_DEBT_GUARANTEE] = {k: v for k, v in central_guarantee.items() if k != "notes"}
    summary["variants"] = variants
    summary["public_real_rates"] = rates
    summary["acquisition_premium_over_rav_by_setting"] = dict(premiums)
    summary["guarantee_pass_through_by_setting"] = {k: dict(v) for k, v in pass_through.items()}
    summary["break_even"] = {
        sector: {"premium_over_rav_financing_only": central_acquired[sector]["break_even_premium_over_rav"],
                 "premium_over_rav_with_tax_allowance": central_acquired[sector]["break_even_premium_over_rav_with_tax_allowance"],
                 "premium_over_rav_perpetual_basis_before_29_sep_2026": central_acquired[sector]["break_even_premium_over_rav_perpetual_basis"]}
        for sector in SECTORS}
    summary["network_ownership"] = "by the networks lever of each case (item K1)"
    return pd.concat(frames, ignore_index=True), summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("building_blocks")
    parser.add_argument("snapshot")
    parser.add_argument("assumptions")
    parser.add_argument("recovery")
    parser.add_argument("out_summary")
    parser.add_argument("out_lines")
    parser.add_argument("--config", default="config")
    parser.add_argument("--guarantee-cost", default=None,
                        help="network_guarantee_expected_loss.csv: a guarantee fee equals the expected loss")
    args = parser.parse_args(argv)

    po = (read_config(args.config) or {}).get("public_ownership") or {}
    # Item Y4: the model year is the recovery-demand file's forecast year.
    model_year = recovery_model_year(json.loads(Path(args.recovery).read_text(encoding="utf-8")))
    cfg = settings(po.get("network_account"), model_year)
    extracted = json.loads(Path(args.building_blocks).read_text(encoding="utf-8"))
    dist_raw, trans_raw = extracted["distribution"], extracted["transmission"]
    blocks = {
        "distribution": {
            "price_basis_key": "2020/21",
            "closing_rav_gbp_m_2027_28": dist_raw["closing_rav_gbp_m_2027_28"],
            "net_additions_gbp_m_2027_28": dist_raw["net_additions_gbp_m_2027_28"],
            "depreciation_gbp_m_2027_28": dist_raw["depreciation_gbp_m_2027_28"],
            "tax_allowance_gbp_m_period_average": dist_raw["tax_allowance_gbp_m_period_average"],
            "vanilla_wacc_cpih_real": dist_raw["vanilla_wacc_cpih_real_infrequent_issuers"],
            # Item Y12: the base a rolled-forward tax allowance is scaled from.
            "opening_rav_gbp_m_period_average": dist_raw["opening_rav_gbp_m_period_average"],
        },
        "transmission": {
            "price_basis_key": "2023/24",
            "total_rav_gbp_m": trans_raw["implied_total_rav_gbp_m_period_average"],
            "wacc_semi_nominal": trans_raw["wacc_semi_nominal_unweighted_average"],
            "notional_gearing": trans_raw["notional_gearing"],
        },
    }
    snapshot = yaml.safe_load(Path(args.snapshot).read_text(encoding="utf-8"))
    from scripts.public_ownership.finance import require_snapshot_schema

    require_snapshot_schema(snapshot)
    assumptions = json.loads(Path(args.assumptions).read_text(encoding="utf-8"))
    recovery = json.loads(Path(args.recovery).read_text(encoding="utf-8"))
    # F19 (7 Oct 2026): TNUoS and DUoS are recovered over metered supply (their F18 entries); the bill
    # module reads it off the recovery file on either basis, never through the deprecated alias.
    from scripts.consumer_bill.project_future_bill import conversion_denominators

    conv = {"factor": float(assumptions["currency_rebasing"]["factor"]),
            "final_demand_mwh": float(conversion_denominators(recovery)["final_demand_mwh"]),
            "mwh": float(assumptions["benchmark"]["annual_consumption_kwh"]) / 1000.0}
    # F29 (26 Oct 2026): no debt guarantee (workbook v2.0 has no network rate); --guarantee-cost is not read.
    premiums = load_premiums(cfg)
    frame, summary = build_all(blocks, snapshot, cfg, conv, premiums=premiums, pass_through={},
                               debt_compression=None, model_year=model_year)
    # Item Y12: every row's identities, for both treatments, before anything is written.
    check_account_closes(frame, summary)
    for path in (args.out_summary, args.out_lines):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_lines, index=False)
    Path(args.out_summary).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({
        "transmission_rav_gbp_bn": round(summary["transmission"]["rav_gbp"] / 1e9, 1),
        "transmission_wacc_cpih_real": round(summary["transmission"]["allowed_wacc_cpih_real"], 4),
        "distribution_rav_gbp_bn": round(summary["distribution"]["rav_gbp"] / 1e9, 1),
        "distribution_wacc_cpih_real": round(summary["distribution"]["allowed_wacc_cpih_real"], 4),
        "central_premium_over_rav": summary["premium_over_rav"] if "premium_over_rav" in summary else None,
        "total_saving_gbp_bn": round(summary["total_saving_gbp"] / 1e9, 2),
        "saving_gbp_per_household": round(summary["total_saving_gbp_per_3100kwh_ex_vat"], 2),
        "break_even": summary["break_even"],
    }, default=float))


def load_premiums(cfg: Mapping) -> dict[str, float]:
    """The premium over RAV at each ``network_acquisition_premium`` setting (user decision 7)."""
    table = cfg.get("acquisition_premium_table")
    if not table:
        # A config without the evidence table keeps its one flat premium for every setting.
        return {setting: float(cfg["acquisition_premium_over_rav"])
                for setting in ("central", "low", "high", "statutory_at_rav", "quarter")}
    frame = pd.read_csv(table)
    return {str(r.setting): float(r.premium_over_rav) for r in frame.itertuples(index=False)}


#: User decision 6 generalised (item Y12): the guarantee reaches the debt issued or
#: refinanced from the window's first year to the model year.  Under Ofgem's trailing
#: averages (ET: RAV-weighted 14 years, RIIO-3 FD Finance Annex para 2.17; the low
#: setting's ED2-style simple 17 years, ED2 FD Finance Annex p.9) the model year's
#: cost-of-debt allowance is wholly inside the window once the trailing years are.
GUARANTEE_WINDOW_START = 2026
TRAILING_AVERAGE_YEARS = {"new_and_refinanced": {"transmission": 14, "distribution": 14},
                          "new_debt_low": {"transmission": 14, "distribution": 17},
                          "full_stock": {"transmission": 0, "distribution": 0}}


def issuance_window_share(model_year: int, *, window_start: int = GUARANTEE_WINDOW_START,
                          trailing_years: int = 14) -> float | None:
    """The share of the model year's cost-of-debt allowance set by debt issued inside the guarantee window.

    Args:
        model_year: The calendar model year.
        window_start: The first year the guarantee covers new or refinanced debt.
        trailing_years: The allowance's trailing-average length (0 = the whole stock repriced).

    Returns:
        1.0 when every year of the trailing average lies inside the window; None when the
        window covers only part of it (the RAV-weighted share then needs Ofgem's cost-of-debt
        model, as the CP30-year table does).
    """
    if int(trailing_years) <= 0:
        return 1.0
    first_yield_year = int(model_year) - int(trailing_years) + 1
    return 1.0 if first_yield_year >= int(window_start) else None


def derived_pass_through(model_year: int, window: Mapping | None = None) -> dict[str, dict[str, float]]:
    """Every ``guarantee_pass_through`` setting's shares from :func:`issuance_window_share`, or raise.

    Args:
        model_year: The calendar model year.
        window: ``{"start": 2026, "trailing_years": {setting: {sector: years}}}``; defaults
            to :data:`GUARANTEE_WINDOW_START` and :data:`TRAILING_AVERAGE_YEARS`.

    Returns:
        ``{setting: {"transmission": share, "distribution": share}}``.

    Raises:
        ValueError: A setting's trailing average straddles the window start.
    """
    window = dict(window or {})
    start = int(window.get("start", GUARANTEE_WINDOW_START))
    trailing = dict(window.get("trailing_years") or TRAILING_AVERAGE_YEARS)
    out = {}
    for setting, sectors in trailing.items():
        shares = {}
        for sector, years in dict(sectors).items():
            share = issuance_window_share(model_year, window_start=start, trailing_years=int(years))
            if share is None:
                raise ValueError(f"guarantee_pass_through {setting}: the {sector} trailing average in {model_year} "
                                 f"straddles the {start} window start; give a guarantee_pass_through_table from "
                                 "Ofgem's cost-of-debt model (as the CP30 year's)")
            shares[sector] = share
        out[str(setting)] = shares
    return out


def load_pass_through(cfg: Mapping, model_year: int | None = None) -> dict[str, dict[str, float]]:
    """The share of the notional debt the guarantee reaches, per ``guarantee_pass_through`` setting (user decision 6).

    Item Y12: a ``guarantee_window`` (``{start, trailing_years}``) with no table derives the
    shares for ``model_year`` from :func:`derived_pass_through`.
    """
    table = cfg.get("guarantee_pass_through_table")
    if not table and cfg.get("guarantee_window") is not None:
        if model_year is None:
            raise ValueError("network_account.guarantee_window needs the model year")
        return derived_pass_through(int(model_year), dict(cfg["guarantee_window"]))
    if not table:
        return {setting: {"transmission": 1.0, "distribution": 1.0}
                for setting in ("new_and_refinanced", "new_debt_low", "full_stock")}
    frame = pd.read_csv(table)
    out = {}
    for r in frame.itertuples(index=False):
        shares = {"transmission": float(r.transmission_share), "distribution": float(r.distribution_share)}
        if not all(0.0 <= v <= 1.0 for v in shares.values()):
            raise ValueError(f"{table}: pass-through shares for {r.setting} must lie in [0, 1]")
        out[str(r.setting)] = shares
    return out


#: The incidence settings an account row is closed under (see :func:`incidence_amounts`).
CLOSURE_INCIDENCES = {
    TREATMENT_ACQUIRED: ({"network_incidence": "bills"}, {"network_incidence": "return_retained"},
                         {"network_incidence": "bills", "tax_equivalent": "on"}),
    TREATMENT_DEBT_GUARANTEE: ({"network_incidence": "bills"}, {"network_incidence": "return_retained"}),
}


def check_account_closes(lines: pd.DataFrame, summary: Mapping, tolerance_gbp: float = 1.0) -> dict:
    """Item Y12: the network account's identities, row by row, for both treatments; raise on a break.

    * acquired: financing saving = RAV x (allowed WACC - public rate) - the premium's
      annuity, and total = financing + tax allowance;
    * debt guarantee: financing saving = guaranteed debt x compression = total;
    * every incidence: bill + fee to the Exchequer + return it keeps = the row's total
      saving (the guarantee fee setting too, when the lines carry it);
    * the summary's total equals the sum of its sectors.

    Args:
        lines: The account lines (every variant row).
        summary: The account summary (its ``treatments``).
        tolerance_gbp: The largest residual allowed, system GBP.

    Returns:
        ``{"rows_checked", "largest_residual_gbp"}``.

    Raises:
        ValueError: An identity does not close.
    """
    worst, checked = 0.0, 0
    num = lambda r, c: float(getattr(r, c)) if pd.notna(getattr(r, c, None)) else 0.0  # noqa: E731
    for r in lines.itertuples(index=False):
        treatment = str(r.treatment)
        if treatment == TREATMENT_ACQUIRED:
            premium = num(r, "premium_annual_cost_gbp")
            financing = num(r, "rav_gbp") * (num(r, "allowed_wacc_cpih_real") - num(r, "public_real_rate")) - premium
        else:
            financing = num(r, "guaranteed_debt_gbp") * num(r, "debt_compression")
        residuals = [financing - num(r, "financing_saving_gbp"),
                     num(r, "financing_saving_gbp") + num(r, "tax_allowance_saving_gbp")
                     + num(r, "excess_return_saving_gbp") - num(r, "total_saving_gbp")]
        row = pd.DataFrame([r._asdict()])
        settings_list = list(CLOSURE_INCIDENCES[treatment])
        if treatment == TREATMENT_DEBT_GUARANTEE:
            settings_list += [{"network_incidence": "guarantee_fee_on_bills", "guarantee_cost": c.replace(
                "guarantee_expected_loss_", "").replace("_gbp", "")} for c in lines.columns
                if c.startswith("guarantee_expected_loss_")]
        for levers in settings_list:
            for part in incidence_amounts(row, treatment, levers).values():
                residuals.append(part["saving_on_bill_gbp"] + part["fee_to_exchequer_gbp"]
                                 + part["retained_by_exchequer_gbp"] - part["total_saving_gbp"])
        checked += 1
        worst = max(worst, max(abs(x) for x in residuals))
    for treatment, entry in dict(summary.get("treatments") or {}).items():
        parts = sum(float(entry[sector]["total_saving_gbp"]) for sector in SECTORS)
        worst = max(worst, abs(parts - float(entry["total_saving_gbp"])))
    if worst > tolerance_gbp:
        raise ValueError(f"the network account does not close: largest residual GBP {worst:,.2f}")
    return {"rows_checked": checked, "largest_residual_gbp": worst}




def select_rows(lines: pd.DataFrame, treatment: str, levers: Mapping[str, str] | None = None) -> pd.DataFrame:
    """The network account rows one case reads: its treatment, at its own variant levers.

    A lines table written before the variants (no variant columns) is read whole
    for the treatment, as it always was.
    """
    if "treatment" in lines.columns:
        rows = lines.loc[lines["treatment"].astype(str).eq(treatment)]
    elif treatment == TREATMENT_ACQUIRED:
        rows = lines
    else:
        raise ValueError(f"network account lines carry no treatment column; cannot serve {treatment!r}")
    levers = dict(levers or {})
    for lever, central in VARIANT_LEVERS.items():
        if lever not in rows.columns:
            continue
        column = rows[lever].fillna("").astype(str)
        applicable = column.ne("")
        if not applicable.any():
            continue
        wanted = str(levers.get(lever, central))
        rows = rows.loc[~applicable | column.eq(wanted)]
    if rows.empty or rows["sector"].duplicated().any():
        raise ValueError(f"network account lines carry no single row per sector for {treatment!r} at {levers}")
    return rows


def adjustments_for_treatment(lines: pd.DataFrame, treatment: str, levers: Mapping[str, str] | None = None,
                              ) -> dict[str, float]:
    """``{bill_line: GBP per household ex VAT}`` (negative = saving) for one case's networks lever and incidence.

    ``levers`` are the case's settings (``network_incidence``, ``tax_equivalent``,
    ``guarantee_cost`` and the variant levers); without them the central rows go
    to the bill whole, as before 29 Sep 2026.  See :func:`incidence_amounts`.
    """
    if treatment == "private":
        return {}
    rows = select_rows(lines, treatment, levers)
    amounts = incidence_amounts(rows, treatment, levers)
    return {str(line): float(value["bill_gbp_per_3100kwh_ex_vat"]) for line, value in amounts.items()}


def break_even_summary(rows: pd.DataFrame) -> dict:
    """The premium over RAV at which acquiring the wires stops paying, for one case's rows (user decision 7).

    Per sector from the account (``break_even_premium_over_rav`` and ``..._with_tax_
    allowance``), and for the wires together: the gross financing wedge (plus the tax
    allowance) over the premium's annual cost per unit of premium, ``RAV x CRF(public
    rate, 30)``, summed over the sectors.  The premium the case pays sits beside it,
    so a reader sees at once whether the case is above or below break-even.
    """
    acquired = rows.loc[rows["treatment"].astype(str).eq(TREATMENT_ACQUIRED)] if "treatment" in rows.columns else rows
    if acquired.empty:
        return {}
    num = lambda c: pd.to_numeric(acquired[c], errors="coerce").fillna(0.0)  # noqa: E731
    per_unit = 0.0
    has_base = "premium_rav_gbp" in acquired.columns
    for r in acquired.itertuples(index=False):
        # Item Y12: a premium paid in an earlier acquisition year was paid on that year's RAV.
        base = float(r.premium_rav_gbp) if has_base and pd.notna(r.premium_rav_gbp) else float(r.rav_gbp)
        per_unit += base * capital_recovery_factor(float(r.public_real_rate), float(PREMIUM_ANNUITY_YEARS))
    gross, tax = float(num("financing_wedge_gross_gbp").sum()), float(num("tax_allowance_saving_gbp").sum())
    premium = float(num("premium_over_rav").iloc[0]) if "premium_over_rav" in acquired.columns else None
    return {
        "premium_over_rav_paid": premium,
        "premium_setting": (str(acquired["network_acquisition_premium"].iloc[0])
                            if "network_acquisition_premium" in acquired.columns else None),
        "break_even_premium_over_rav": gross / per_unit if per_unit else None,
        "break_even_premium_over_rav_with_tax_allowance": (gross + tax) / per_unit if per_unit else None,
        "by_sector": {str(r.sector): {"break_even_premium_over_rav": float(r.break_even_premium_over_rav),
                                      "break_even_premium_over_rav_with_tax_allowance":
                                          float(r.break_even_premium_over_rav_with_tax_allowance)}
                      for r in acquired.itertuples(index=False)},
        "basis": ("the premium, annuitised at the public rate over 30 years, that equals the financing wedge "
                  "(and, with the tax allowance, the whole saving) on the wires' RAV"),
    }


def incidence_amounts(rows: pd.DataFrame, treatment: str, levers: Mapping[str, str] | None = None) -> dict:
    """Where each sector's saving lands under the case's ``network_incidence`` and ``tax_equivalent`` (user decision 5).

    Per bill line: the household bill adjustment and, in system GBP, the amount the
    bill carries, the guarantee fee the Exchequer receives and the return it keeps.

    * ``bills`` (today): the whole saving (wedge, and the tax allowance when the
      wires are acquired) comes off TNUoS and DUoS.
    * ``guarantee_fee_on_bills`` (debt guarantee): the networks pay the Exchequer a
      fee equal to its expected loss (HM Treasury CLAF para 4.4) and pass it through,
      so the bill keeps the compression less the fee.
    * ``return_retained``: allowed revenue unchanged, the bill does not move; the
      Exchequer keeps the wedge and the tax allowance (acquired) or charges a market
      fee equal to the whole compression (guarantee).
    * ``tax_equivalent: on`` with the wires acquired: the DUoS tax allowance stays on
      the bill and is paid to the Exchequer, the wedge alone reaches the bill.
    """
    levers = dict(levers or {})
    incidence = str(levers.get("network_incidence", "bills"))
    tax_kept = str(levers.get("tax_equivalent", "none")) == "on"
    setting = str(levers.get("guarantee_cost", "central"))
    out = {}
    for r in rows.itertuples(index=False):
        saving, tax = float(r.financing_saving_gbp), float(r.tax_allowance_saving_gbp)
        # F27 item 7: the excess return bought wires earn goes where the wedge goes (0 on older tables).
        excess = float(getattr(r, "excess_return_saving_gbp", 0.0) or 0.0)
        total = float(r.total_saving_gbp)
        if "household_gbp_per_system_gbp" in rows.columns:
            rate = float(r.household_gbp_per_system_gbp)
        else:  # a lines table written before 29 Sep 2026: back the rate out of the row itself
            rate = float(r.saving_gbp_per_3100kwh_ex_vat) / -total if total else 0.0
        fee = 0.0
        retained = 0.0
        on_bill = saving + tax + excess
        if incidence == "return_retained":
            retained, on_bill = saving + tax + excess, 0.0
            if treatment == TREATMENT_DEBT_GUARANTEE:
                fee, retained = saving, 0.0
        elif incidence == "guarantee_fee_on_bills":
            if treatment != TREATMENT_DEBT_GUARANTEE:
                raise ValueError("network_incidence guarantee_fee_on_bills needs networks: debt_guarantee")
            column = f"guarantee_expected_loss_{setting}_gbp"
            if column not in rows.columns:
                raise ValueError(f"the network account carries no {column}; build it with --guarantee-cost")
            fee = float(getattr(r, column))
            on_bill = saving - fee
        elif tax_kept and treatment == TREATMENT_ACQUIRED:
            retained, on_bill = tax, saving + excess
        out[str(r.bill_line_adjusted)] = {"sector": r.sector, "saving_on_bill_gbp": on_bill, "fee_to_exchequer_gbp": fee,
                                          "retained_by_exchequer_gbp": retained, "total_saving_gbp": total,
                                          "excess_return_gbp": excess,
                                          "bill_gbp_per_3100kwh_ex_vat": -on_bill * rate}
    return out


if __name__ == "__main__":
    main()
