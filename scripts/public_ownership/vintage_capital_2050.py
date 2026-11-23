"""How a 2050 vintage recovers its capital: the strike basis, the register's capital charge, and the calibration.

FES 2050 plan, decision 41 (user, 2 October 2026), with the strike basis the user set the
same day in place of decision 6's economic-life levelising.

**One place for the basis.** The study config's ``public_ownership.vintage_capital_2050``
block (``amortisation``, ``capture_prices``, ``strike_calibration``) is read by every stage
that needs it through :func:`basis_from_config`: the 2050 strike table
(``contract_book_2050.build_strike_table_file``), the contract book's DPA availability rates
(``contract_book_2050.build_contract_book``) and the vintaged register's capital charge
(``calculate_financing_scenarios``). Without the block every stage behaves as before (the
``economic_life`` basis), so no 2030 output can move.

**Amortisation modes** (:data:`AMORTISATIONS`):

* ``economic_life``: decision 6, kept as a sensitivity. The strike levelises capital over
  the plant's life (``cfd_strikes_2050.levelised_strike``); the register is unchanged.
* ``contract_term``: capital recovered over the contract term with no tail value (the upper
  bound).
* ``contract_term_with_tail`` (the 2050 central): capital recovered over the contract term
  (20 years; the technology's own term where it differs: tidal and BECCS 15, nuclear 35;
  DPAs 15) less a credit for the merchant tail. The credit is the present value, over the
  years from the term's end to the economic life, of

      max((capture price - fuel - variable O&M) x expected output - fixed costs, 0) a year,

  levelised back over the contract term's output. The capture price is the solved 2050
  year's market revenue per MWh by carrier (the private register, :func:`derive_capture_prices`),
  the same for every vintage's tail (declared). Fixed costs are the strike's own: fixed O&M
  without DESNZ insurance, the D4(a) insurance and the use-of-system charge. Expected output
  is the strike's declared load factor (DESNZ Annex A) for CfD technologies and the solved
  year's load factor for DPA plant, which has no declared one.

**The register under a term basis** (decision 41 (a)). In the 2050 snapshot a new-build
row's capital charge is the annuity its contract pays:

* a row still within its contract: ``live share x (capex - PV of the tail) x CRF(r, term)``,
  at the row's own rate (the private rate, or the public rate on the state's rows);
* a row whose contract has ended (``support_basis = contract_expired_merchant``): zero.
  Its capital was recovered over the term, so its 2050 required revenue is its operating
  cost only, and the corporation tax embedded in its capital charge is zero too.

Sizewell C's RAB keeps the register's own charge (its allowance pays it; decision 41 (c)).

**Merchant plant** (decision 66, user, 3 October 2026; ``merchant_capital: level_annuity``, the
2050 central). A row outside any contract, RAB or RO (storage, new pumped storage, hydro, unabated
biomass, energy from waste, a wholly repowered solar build year) pays a level annuity, capex x
CRF(rate, economic life), at its case's own rate, in every year it runs; past its economic life its
capital is sunk (:func:`merchant_level_rows`). Under ``remaining_life`` (the original rule, kept
where the key is absent) it paid the build-year capex over its remaining life. A long-duration store
on the LDES cap and floor (decision 43) is both: it pays the level annuity, and the floor tops its
income up to that same capex re-annuitised over the same life at the floor rate
(``public_account._floor_parts``, ``capital_annuity_years``), so nothing is counted twice.

**Legacy plant** (decision 42, user, 3 October 2026): a pre-2025 row whose RO or CfD has
ended by the model year is on the same rule, capital sunk (``legacy_expired_contracts``:
the rows the register labels ``contract_expired_merchant``, wind and solar, and the
technologies and build years the config names: in 2050 pumped storage, hydro and energy from
waste). Any other legacy merchant row within its economic life takes decision 66's level annuity.

**Calibration** (decision 41 (b)). The post-AR7 strikes (``PAR7_<build year>``) and the DPA
availability rates are then set so that each private cohort (technology x build year)
breaks even on the modelled output: revenue (market, CfD top-up or DPA payments, BM and
curtailment as solved) equals its required revenue. A strike moves the cohort's CfD support
by exactly its paid energy (the settlement's ``cfd_paid_mwh``); an availability rate moves
its DPA support by ``kW x availability x live share``. :func:`derive_calibration` reads the
live register and the strikes in use, so a calibrated cohort re-derives to itself
(``--check``), as the 2030 auction calibration does (``derive_cost_calibration.py``).

**On the central-dispatch year** (decision 70, user, 4 October 2026): ``calibrate --dispatch-basis
central`` re-derives the post-AR7 strikes on the central year's output beside the market year's
(:mod:`central_strike_diagnostic_2050`): only uncompensated curtailment counts on either basis (BM
turn-down is restored by the CfD bid; central turn-down below the market's day-ahead schedule by
FTI's firm access). A diagnostic: it writes its own CSV and JSON, never the calibration in use.

**The auction margin** (decision 71, user, 4 October 2026; 2050 matches 2030's lever ``auction_margin``).
CfD auctions pay as clear, so only the marginal project breaks even: the post-AR7 cohorts keep a margin
``m`` of their contract revenue above full cost. Each calibrated CfD cohort's strike solves
``rent(s) = m x s x paid MWh`` instead of ``rent(s) = 0``; rent moves by the paid energy per GBP of strike,
so ``s* = (s_in_use - rent(s_in_use) / paid) / (1 - m)``, the break-even strike at ``m = 0`` exactly.
``vintage_capital_2050.auction_margin_share`` (default 0, so a study without the key is unchanged; HT50:
0.05) and ``calibrate --auction-margin`` set it, and the calibration records it per row
(``auction_margin_share``). Only schemes an auction prices take it (:data:`AUCTION_SCHEMES`: the post-AR7
CfDs). DPAs are allocated bilaterally (Net Zero Teesside's agreement; hydrogen-to-power's first
allocation), so their availability rates still break even; Sizewell C's RAB is regulated and is not
calibrated at all. ``margin`` re-margins a calibration from its own columns (no register read), which
writes the installed central file and the two bound variants (``strike_calibration_2050_auction_margin_
{none,high}.csv``); ``bound-inputs`` builds a margin bound's own strike table and bill assumptions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

ECONOMIC_LIFE, CONTRACT_TERM, CONTRACT_TERM_WITH_TAIL = "economic_life", "contract_term", "contract_term_with_tail"
AMORTISATIONS = (ECONOMIC_LIFE, CONTRACT_TERM, CONTRACT_TERM_WITH_TAIL)
#: The modes under which the register charges capital over the contract term (decision 41 (a)).
TERM_MODES = (CONTRACT_TERM, CONTRACT_TERM_WITH_TAIL)
CONFIG_KEY = "vintage_capital_2050"
DEFAULT_CAPTURE_PRICES = "data/public_ownership/financing/capture_prices_2050.csv"
DEFAULT_CALIBRATION = "data/public_ownership/financing/strike_calibration_2050.csv"
#: Vintage-table technology -> the network carrier whose capture price its tail earns.
TECHNOLOGY_CARRIER = {
    "Offshore wind": "wind_offshore", "Floating offshore wind": "wind_offshore", "Onshore wind": "wind_onshore",
    "Solar PV": "solar_pv", "Tidal stream": "marine", "Small modular reactors (SMRs)": "nuclear",
    "Large-scale nuclear": "nuclear", "Hydropower": "large_hydro", "Biomass - with CCUS - mature": "CCS Biomass",
    "Gas generation - CCUS - mature": "CCS Gas", "Hydrogen CCHT": "H2_turbine", "Hydrogen OCHT": "H2_turbine",
    # 11 Nov 2026: 2040 carries live energy-from-waste CfDs with a tail (by 2050 they have ended).
    "Energy from waste": "waste_to_energy",
}
DPA_TECHNOLOGY_SCHEME = {"Gas generation - CCUS - mature": "CCS Gas", "Hydrogen CCHT": "Hydrogen",
                         "Hydrogen OCHT": "Hydrogen"}
SUPPORT_BASIS_EXPIRED = "contract_expired_merchant"
#: Register support bases that hold a live contract whose capital the term basis recovers.
TERM_SUPPORT_BASES = ("post_ar7_modelled", "dpa_modelled", "known_contract")
CAPTURE_COLUMNS = ("carrier", "capture_gbp_per_mwh", "fuel_carbon_gbp_per_mwh", "load_factor", "dispatch_mwh",
                   "capacity_mw", "market_revenue_gbp", "source")
CALIBRATION_COLUMNS = ("scheme", "key", "cfd_round", "technology", "dpa_carrier", "dpa_technology", "build_year",
                       "basis", "base_value", "calibrated_value", "unit", "rent_at_value_in_use_gbp",
                       "denominator", "denominator_unit", "value_in_use", "source")
#: Decision 45 (user, 3 Oct 2026): reported beside each CfD cohort's calibrated strike, never read back.
DECLARED_OUTPUT_COLUMNS = ("declared_load_factor", "delivered_load_factor", "declared_output_strike")
#: The strike table's technology -> the vintage-table technology whose declared load factor it takes.
TABLE_TECHNOLOGY = {"offshore": "Offshore wind", "onshore": "Onshore wind", "solar": "Solar PV",
                    "tidal": "Tidal stream", "nuclear": "Small modular reactors (SMRs)"}
#: A calibrated cohort re-derives to itself within this (GBP/MWh for strikes, GBP/kW/yr for DPAs).
CHECK_TOLERANCE = 0.01
#: Strike-table technologies left uncalibrated: Power BECCS's electricity-leg strike is set
#: after its off-bill removal credit, and its carbon CfD pays the capture leg at cost
#: (``calculate_financing_scenarios.book_carbon_removal_cfd``), so the two legs together
#: already settle the plant at cost; calibrating one leg against the other is circular.
UNCALIBRATED_TECHNOLOGIES = ("beccs",)
#: Decision 49 (b): which strikes the solved BM bid on, so the bill values curtailment at them.
#: ``economic_life`` is decision 6's central (``support_labels.strike_variant: central``);
#: ``calibrated`` is the contract table itself (``strike_variant: calibrated``). Must match the
#: months' ``support_labels.strike_variant`` (a test checks HT50).
BM_STRIKE_BASES = ("economic_life", "calibrated")
BM_STRIKE_BASIS_KEY = "bm_strike_basis"
#: FES 2050 decision 66 (user, 3 Oct 2026): how a merchant row's capital is charged. ``remaining_life``
#: (the register's original rule) annuitises the build-year capex over the row's REMAINING life, so a
#: 2036 battery with one year left was charged its whole capex in 2050; ``level_annuity`` charges every
#: merchant row capex x CRF(rate, economic life), the same in each year it runs, and a merchant row
#: past its economic life has recovered its capital (sunk, as decisions 41 (a) and 42 treat expired
#: contracts). Contract rows keep decision 41's term basis; RAB, RO and expired rows are untouched.
MERCHANT_CAPITAL_KEY = "merchant_capital"
MERCHANT_REMAINING_LIFE, MERCHANT_LEVEL_ANNUITY = "remaining_life", "level_annuity"
MERCHANT_CAPITAL_BASES = (MERCHANT_REMAINING_LIFE, MERCHANT_LEVEL_ANNUITY)
#: Support types that are not merchant (their capital follows their own regime).
NON_MERCHANT_SUPPORT = ("CfD", "DPA", "RAB", "ROC")
#: FES 2050 decision 71 (user, 4 Oct 2026): the share of contract revenue (strike x paid MWh) a calibrated
#: post-AR7 CfD cohort keeps as rent (pay-as-clear auctions). The study config's key, the calibration's column.
AUCTION_MARGIN_KEY = "auction_margin_share"
AUCTION_MARGIN_COLUMN = "auction_margin_share"
#: The 2030 lever's values (``levers.LEVER_AUCTION_MARGIN``; ``lever_parameters.auction_margin_share``), so the
#: two years name and size the margin the same way.
AUCTION_MARGIN_LEVELS = {"none": 0.0, "central": 0.05, "high": 0.10}
#: The calibrated schemes an auction prices. DPAs are allocated bilaterally (decision 71: break even); a RAB
#: (Sizewell C) is regulated and never in the calibration.
AUCTION_SCHEMES = ("CfD",)
#: Decision 71: a register bound's group key, ``{level: none | high, strike_calibration: <file>}``.
GROUP_AUCTION_MARGIN_KEY = "auction_margin"
#: The margin a calibration's own rows must agree on (and with the basis), to this tolerance.
MARGIN_TOLERANCE = 1e-9


class VintageCapitalError(ValueError):
    """A basis, capture table or calibration that cannot be used."""


# ── the basis ────────────────────────────────────────────────────────────────

def _read_config(path) -> dict:
    """``config_loader.read_config``, imported on first use: this script puts the repo root on sys.path itself."""
    from config.config_loader import read_config

    return read_config(path)


def basis_from_config(config: Mapping | None, root: Path | str = ROOT) -> dict:
    """The study's capital basis (``public_ownership.vintage_capital_2050``), or the economic-life default.

    Args:
        config: A resolved config (the whole file or its ``public_ownership`` block).
        root: Where relative paths resolve.

    Returns:
        ``amortisation``, ``capture_prices`` (a path or None), ``strike_calibration`` (a path
        or None) and ``enabled`` (a term mode is in force).
    """
    block = dict(config or {})
    block = dict(block.get("public_ownership", block) or {})
    raw = dict(block.get(CONFIG_KEY) or {})
    # Decision 58's support lines live in public_ownership.support_lines (a 2030 study may use them too).
    _refuse_support_lines_here(raw)
    mode = str(raw.get("amortisation", ECONOMIC_LIFE))
    if mode not in AMORTISATIONS:
        raise VintageCapitalError(f"{CONFIG_KEY}.amortisation={mode!r}; one of {AMORTISATIONS}")
    root = Path(root)

    def _path(key):
        value = raw.get(key)
        if not value:
            return None
        p = Path(value)
        return p if p.is_absolute() else root / p

    capture = _path("capture_prices")
    if mode == CONTRACT_TERM_WITH_TAIL and capture is None:
        raise VintageCapitalError(f"{CONFIG_KEY}: contract_term_with_tail needs capture_prices (the tail's price)")
    merchant_capital = str(raw.get(MERCHANT_CAPITAL_KEY, MERCHANT_REMAINING_LIFE))
    if merchant_capital not in MERCHANT_CAPITAL_BASES:
        raise VintageCapitalError(f"{CONFIG_KEY}.{MERCHANT_CAPITAL_KEY}={merchant_capital!r}; one of {MERCHANT_CAPITAL_BASES}")
    if merchant_capital == MERCHANT_LEVEL_ANNUITY and mode not in TERM_MODES:
        raise VintageCapitalError(f"{CONFIG_KEY}.{MERCHANT_CAPITAL_KEY}: level_annuity needs a term amortisation")
    bm_basis = str(raw.get(BM_STRIKE_BASIS_KEY, "economic_life"))
    if bm_basis not in BM_STRIKE_BASES:
        raise VintageCapitalError(f"{CONFIG_KEY}.{BM_STRIKE_BASIS_KEY}={bm_basis!r}; one of {BM_STRIKE_BASES}")
    if bm_basis == "calibrated" and (mode not in TERM_MODES or _path("strike_calibration") is None):
        raise VintageCapitalError(f"{CONFIG_KEY}.{BM_STRIKE_BASIS_KEY}: calibrated needs a term amortisation "
                                  "and a strike_calibration file")
    return {"amortisation": mode, "capture_prices": capture, "strike_calibration": _path("strike_calibration"),
            "enabled": mode in TERM_MODES, "bm_strike_basis": bm_basis,
            "legacy_expired": legacy_expired_rule(raw.get(LEGACY_EXPIRED_KEY), raw.get(LEGACY_SUNK_KEY)),
            "beccs_removal_value_gbp_per_mwh": beccs_removal_value(raw.get(BECCS_REMOVALS_KEY)),
            "beccs_carbon_cfd_term_years": (None if raw.get(BECCS_CARBON_CFD_TERM_KEY) is None
                                            else int(raw[BECCS_CARBON_CFD_TERM_KEY])),
            # Decision 32: Sizewell C's RAB outside P1's stake and P2's boundary (a switch; see the config).
            "rab_outside_propositions": bool(raw.get("rab_outside_propositions", False)),
            # Carriers held private in every case until the full rebuild (a newly costed carrier).
            "held_private_carriers": [str(c) for c in (raw.get("held_private_carriers") or [])],
            # Decision 54: each private rate path's own break-even calibration (the NESO wind ramp).
            "rate_path_calibrations": {str(k): _path_of(v, root) for k, v in
                                       (raw.get(RATE_PATH_CALIBRATIONS_KEY) or {}).items()},
            # Decision 66: merchant rows on a level annuity over their economic life.
            MERCHANT_CAPITAL_KEY: merchant_capital,
            # Decision 71: the share of contract revenue the post-AR7 CfD cohorts keep (0: break even).
            AUCTION_MARGIN_KEY: auction_margin_share(raw.get(AUCTION_MARGIN_KEY, 0.0))}


def auction_margin_share(value) -> float:
    """Decision 71: a margin as a share of contract revenue, from a number or the 2030 lever's level name.

    ``none`` 0, ``central`` 0.05, ``high`` 0.10 (:data:`AUCTION_MARGIN_LEVELS`), or a share in [0, 0.5).
    """
    if value is None:
        return 0.0
    if isinstance(value, str) and value.strip() in AUCTION_MARGIN_LEVELS:
        return AUCTION_MARGIN_LEVELS[value.strip()]
    try:
        share = float(value)
    except (TypeError, ValueError):
        raise VintageCapitalError(f"{CONFIG_KEY}.{AUCTION_MARGIN_KEY}={value!r}: a share or one of "
                                  f"{sorted(AUCTION_MARGIN_LEVELS)}") from None
    if not (np.isfinite(share) and 0.0 <= share < 0.5):
        raise VintageCapitalError(f"{CONFIG_KEY}.{AUCTION_MARGIN_KEY}={share}: must be in [0, 0.5)")
    return share


#: FES 2050 decision 54: ``{rate path: calibration file}``, each derived on ``private@private_rate_<path>``.
RATE_PATH_CALIBRATIONS_KEY = "strike_calibration_by_rate_path"


def _refuse_support_lines_here(raw: Mapping) -> bool:
    """Decision 58's support lines moved to ``public_ownership.support_lines`` (3 Oct 2026): refuse the old key."""
    if raw.get("support_lines"):
        raise VintageCapitalError(f"{CONFIG_KEY}.support_lines moved to public_ownership.support_lines")
    return True


def _path_of(value, root) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(root) / p


#: Decision 44 (user, 3 Oct 2026): Power BECCS's carbon-removal CfD runs on its own term.
BECCS_CARBON_CFD_TERM_KEY = "beccs_carbon_cfd_term_years"


#: Decision 42 (user, 3 Oct 2026): the config key of the legacy (pre-2025) plant whose RO or CfD has ended.
LEGACY_EXPIRED_KEY = "legacy_expired_contracts"
#: User, 3 Oct 2026 (after decision 42): legacy technologies past their economic life whose capital is
#: sunk too (pumped storage, pre-2002 hydro, energy from waste); refurbishment is O&M.
LEGACY_SUNK_KEY = "legacy_sunk_capital"
#: User, 3 Oct 2026: expired Power BECCS earns the market value of its removals (``ets_value``).
BECCS_REMOVALS_KEY = "beccs_removals_after_cfd"


def beccs_removal_value(raw) -> float | None:
    """The value per MWh of an expired BECCS plant's removals, or None (off).

    ``ets_value``: captured tCO2 per MWh x the declared 2050 removal price
    (``support_regime_2050.json`` ``beccs.removal_price_gbp_per_tco2``, the FES 2050 carbon
    price, the price the off-bill removals contract and the BECCS strike already use), through
    ``support_regime_2050.beccs_strike``'s own arithmetic.
    """
    if not raw:
        return None
    if str(raw) != "ets_value":
        raise VintageCapitalError(f"{CONFIG_KEY}.{BECCS_REMOVALS_KEY}={raw!r}; supported: ets_value")
    from scripts.public_ownership import support_regime_2050 as sr

    params = sr.load_parameters()
    thermal, beccs = params["thermal"], params["beccs"]
    eff = float(thermal["efficiency"]["CCS Biomass"])
    captured = sr._value(thermal["biomass_emission_factor_tco2_per_mwh_thermal"]) * sr._value(beccs["capture_rate"]) / eff
    return float(captured * sr._value(beccs["removal_price_gbp_per_tco2"]))


def legacy_expired_rule(raw: Mapping | None, sunk: Mapping | None = None) -> dict | None:
    """Decision 42's rule: which legacy rows held an RO or CfD that has ended by the model year (None: off).

    ``labelled`` (bool): legacy rows the register itself labels ``contract_expired_merchant``
    (wind and solar: a CfD technology built before 2031, ``fleet_vintages.label_support_contracts``).
    ``technologies``: stock-flow technology -> ``[first, last]`` build year of a supported legacy
    build (the RO accredited from April 2002 to March 2017 and ended by March 2037; the AR1-AR6
    CfDs and the Drax bridge end by 2047), for technologies the register leaves unlabelled.
    """
    if not raw and not sunk:
        return None
    raw = dict(raw or {"labelled": False})
    technologies = {}
    # Sunk-capital technologies (every build year) join the contract rule's technologies.
    sunk_technologies = {str(t): [None, None] for t in (dict(sunk or {}).get("technologies") or [])}
    for technology, years in {**dict(raw.get("technologies") or {}), **sunk_technologies}.items():
        first, last = (list(years) + [None, None])[:2] if isinstance(years, (list, tuple)) else (years, None)
        technologies[str(technology)] = (float(first) if first is not None else -np.inf,
                                         float(last) if last is not None else np.inf)
    return {"labelled": bool(raw.get("labelled", True)), "technologies": technologies}


def legacy_expired_rows(merged: pd.DataFrame, rule: Mapping | None) -> np.ndarray:
    """Decision 42: legacy rows whose RO or CfD has ended, so their capital is sunk (operating cost only)."""
    n = len(merged)
    if not rule or "finance_vintage" not in merged.columns:
        return np.zeros(n, dtype=bool)
    legacy = merged["finance_vintage"].astype(str).eq("legacy").to_numpy()
    out = np.zeros(n, dtype=bool)
    if rule.get("labelled") and "support_basis" in merged.columns:
        out |= merged["support_basis"].fillna("").astype(str).eq(SUPPORT_BASIS_EXPIRED).to_numpy()
    if rule.get("technologies"):
        technology = merged["stock_flow_technology"].fillna("").astype(str).to_numpy() \
            if "stock_flow_technology" in merged.columns else np.full(n, "")
        year = pd.to_numeric(merged.get("build_year"), errors="coerce").to_numpy(dtype=float) \
            if "build_year" in merged.columns else np.full(n, np.nan)
        for name, (first, last) in rule["technologies"].items():
            out |= (technology == name) & np.isfinite(year) & (year >= first) & (year <= last)
    return legacy & out


def load_capture_prices(path) -> pd.DataFrame:
    """The capture-price table (:func:`derive_capture_prices`), indexed by carrier."""
    table = pd.read_csv(path)
    if missing := set(CAPTURE_COLUMNS) - set(table.columns):
        raise VintageCapitalError(f"{path} lacks {sorted(missing)}")
    if table["carrier"].duplicated().any():
        raise VintageCapitalError(f"{path} repeats a carrier")
    return table.set_index("carrier")


def _capture_for(technology: str, capture: pd.DataFrame) -> pd.Series:
    carrier = TECHNOLOGY_CARRIER.get(str(technology))
    if carrier is None or carrier not in capture.index:
        raise VintageCapitalError(f"no capture price for {technology!r} (carrier {carrier!r})")
    return capture.loc[carrier]


def annuity_factor(rate: float, years: float) -> float:
    """Present value of 1 a year for ``years`` years (the inverse of the capital recovery factor)."""
    if years <= 0:
        return 0.0
    if abs(rate) < 1e-12:
        return float(years)
    return (1.0 - (1.0 + rate) ** -float(years)) / rate


def tail_inputs(technology: str, build_year: int, *, capture: pd.DataFrame, insurance_config: Mapping | None = None,
                learning: bool = False, vintage_table=None, strike_inputs=None, params: Mapping | None = None) -> dict:
    """What a vintage's merchant tail earns a year per kW, and the years it earns it.

    CfD technologies take the strike's declared load factor, use-of-system charge and term
    (``cfd_strike_inputs_2050.csv``); DPA technologies the scheme's term and use of system
    and the solved year's load factor. Fuel per MWh is the solved year's (the register's
    fuel and carbon over its output, by carrier): zero for wind, solar and tidal.

    Returns:
        ``margin_gbp_per_kw_year`` (floored at zero), ``life_years``, ``term_years``,
        ``tail_years``, ``load_factor``, ``capture_gbp_per_mwh``, ``fuel_gbp_per_mwh``,
        ``variable_om_gbp_per_mwh``, ``fixed_gbp_per_kw_year``, ``capex_gbp_per_kw``.
    """
    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.public_ownership import vintage_costs as vc

    year = min(max(int(build_year), vc.FIRST_BUILD_YEAR), vc.LAST_BUILD_YEAR)
    cost = vc.costs_for(technology, year, learning=learning, table=vintage_table)
    energy_capex = float(np.nan_to_num(float(cost.get("capex_energy_gbp_per_kwh") or 0.0)))
    duration = float(np.nan_to_num(float(cost.get("duration_hours") or 0.0)))
    capex = float(cost["capex_power_gbp_per_kw"]) + energy_capex * duration
    insured = cfd50.insured_fixed_cost(cost, capex, insurance=cfd50.INSURANCE_RULE, insurance_config=insurance_config)
    row = _capture_for(technology, capture)
    if technology in DPA_TECHNOLOGY_SCHEME:
        from scripts.public_ownership import support_regime_2050 as sr

        params = sr.load_parameters() if params is None else params
        term = int(sr._value(params["dpa"]["term_years"]))
        uos = float(params["dpa"]["use_of_system_gbp_per_kw_year"][DPA_TECHNOLOGY_SCHEME[technology]])
        lf = float(row["load_factor"])
    else:
        spec = cfd50.strike_inputs_for(technology, int(build_year), strike_inputs)
        term = int(spec["term_years"])
        uos = float(spec["use_of_system_gbp_per_kw_year"])
        lf = float(spec["expected_load_factor"])
    life = float(cost["economic_life_years"])
    vom = float(cost["variable_om_gbp_per_mwh"])
    fuel = float(row["fuel_carbon_gbp_per_mwh"])
    fixed = float(insured["fixed_total_gbp_per_kw_year"]) + uos
    margin = (float(row["capture_gbp_per_mwh"]) - fuel - vom) * 8.760 * lf - fixed
    return {"margin_gbp_per_kw_year": max(margin, 0.0), "margin_before_floor_gbp_per_kw_year": margin,
            "life_years": life, "term_years": term, "tail_years": max(life - term, 0.0), "load_factor": lf,
            "capture_gbp_per_mwh": float(row["capture_gbp_per_mwh"]), "fuel_gbp_per_mwh": fuel,
            "variable_om_gbp_per_mwh": vom, "fixed_gbp_per_kw_year": fixed, "capex_gbp_per_kw": capex}


def tail_value_per_kw(inputs: Mapping, rate: float) -> float:
    """Present value at the build year of the tail's margins, per kW (years term+1 to life)."""
    life, term = float(inputs["life_years"]), float(inputs["term_years"])
    if life <= term or inputs["margin_gbp_per_kw_year"] <= 0.0:
        return 0.0
    return float(inputs["margin_gbp_per_kw_year"]) * (annuity_factor(rate, life) - annuity_factor(rate, term))


def tail_credit_gbp_per_mwh(inputs: Mapping, rate: float) -> float:
    """The tail's present value levelised back over the contract term's expected output."""
    from scripts.public_ownership.finance import capital_recovery_factor

    pv = tail_value_per_kw(inputs, rate)
    if pv <= 0.0:
        return 0.0
    return pv * capital_recovery_factor(rate, float(inputs["term_years"])) / (8.760 * float(inputs["load_factor"]))


# ── strikes and DPA rates ────────────────────────────────────────────────────

def strike(technology: str, build_year: int, *, amortisation: str, snapshot: Mapping, capture: pd.DataFrame | None,
           insurance_config: Mapping | None = None, rate_path: str = "flat", learning: bool = False,
           vintage_table=None, strike_inputs=None, hurdle_table=None) -> dict:
    """One post-AR7 strike on a basis (``cfd_strikes_2050.levelised_strike``, less the tail credit when it applies)."""
    from scripts.market import cfd_strikes_2050 as cfd50

    if amortisation not in AMORTISATIONS:
        raise VintageCapitalError(f"amortisation {amortisation!r}; one of {AMORTISATIONS}")
    row = cfd50.levelised_strike(technology, int(build_year), snapshot=snapshot, rate_path=rate_path, learning=learning,
                                 amortisation=ECONOMIC_LIFE if amortisation == ECONOMIC_LIFE else CONTRACT_TERM,
                                 vintage_table=vintage_table, inputs=strike_inputs, hurdle_table=hurdle_table,
                                 insurance_config=insurance_config)
    return with_tail(row, technology, build_year, amortisation=amortisation, capture=capture,
                     insurance_config=insurance_config, learning=learning, vintage_table=vintage_table,
                     strike_inputs=strike_inputs)


def with_tail(row: dict, technology: str, build_year: int, *, amortisation: str, capture: pd.DataFrame | None,
              insurance_config: Mapping | None = None, learning: bool = False, vintage_table=None,
              strike_inputs=None) -> dict:
    """A contract-term strike row less its tail credit (``contract_term_with_tail``); other modes unchanged."""
    row = dict(row)
    credit = 0.0
    if amortisation == CONTRACT_TERM_WITH_TAIL:
        if capture is None:
            raise VintageCapitalError("contract_term_with_tail needs the capture prices")
        inputs = tail_inputs(technology, build_year, capture=capture, insurance_config=insurance_config,
                             learning=learning, vintage_table=vintage_table, strike_inputs=strike_inputs)
        credit = tail_credit_gbp_per_mwh(inputs, float(row["private_rate"]))
        row["tail_margin_gbp_per_kw_year"] = inputs["margin_gbp_per_kw_year"]
        row["tail_years"] = inputs["tail_years"]
    row["tail_credit_gbp_per_mwh"] = credit
    row["strike_before_tail_gbp_per_mwh_2024"] = float(row["strike_gbp_per_mwh_2024"])
    row["strike_gbp_per_mwh_2024"] = float(row["strike_gbp_per_mwh_2024"]) - credit
    row["amortisation"] = amortisation
    return row


def dpa_annual_fixed(technology: str, build_year: int, *, amortisation: str, snapshot: Mapping,
                     capture: pd.DataFrame | None, insurance_config: Mapping | None = None, learning: bool = False,
                     vintage_table=None, params: Mapping | None = None) -> dict:
    """A DPA vintage's annual fixed cost per kW on a basis (``support_regime_2050.dpa_contract``'s arithmetic).

    ``economic_life`` reproduces the scheme's own ``capex x CRF(r, life) + fixed + use of
    system``; the term modes take ``CRF(r, term)`` on capex less the tail's present value.
    """
    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.public_ownership import support_regime_2050 as sr
    from scripts.public_ownership import vintage_costs as vc
    from scripts.public_ownership.finance import capital_recovery_factor

    params = sr.load_parameters() if params is None else params
    scheme = DPA_TECHNOLOGY_SCHEME[technology]
    cost_year = max(int(build_year), vc.FIRST_BUILD_YEAR)
    cost = vc.costs_for(technology, cost_year, learning=learning, table=vintage_table)
    rate = cfd50.private_rate(technology, cost_year, snapshot=snapshot, vintage_table=vintage_table)["rate"]
    capex = float(cost["capex_power_gbp_per_kw"])
    insured = cfd50.insured_fixed_cost(cost, capex, insurance=cfd50.INSURANCE_RULE, insurance_config=insurance_config)
    uos = float(params["dpa"]["use_of_system_gbp_per_kw_year"][scheme])
    term = int(sr._value(params["dpa"]["term_years"]))
    life = float(cost["economic_life_years"])
    tail = 0.0
    if amortisation == ECONOMIC_LIFE:
        capital = capex * capital_recovery_factor(rate, life)
    else:
        if amortisation == CONTRACT_TERM_WITH_TAIL:
            tail = tail_value_per_kw(tail_inputs(technology, build_year, capture=capture, insurance_config=insurance_config,
                                                 learning=learning, vintage_table=vintage_table, params=params), rate)
        capital = max(capex - tail, 0.0) * capital_recovery_factor(rate, min(float(term), life))
    return {"annual_fixed_gbp_per_kw_year": capital + insured["fixed_total_gbp_per_kw_year"] + uos,
            "capital_gbp_per_kw_year": capital, "tail_value_gbp_per_kw": tail, "private_rate": rate}


# ── the register ─────────────────────────────────────────────────────────────

def known_contract_terms() -> dict[str, int]:
    """Contract id -> term (years) of Y11's known 2050 contracts (AR7, AR7a, HPC, Sizewell C)."""
    from scripts.market.cfd_strikes_2050 import KNOWN_CONTRACTS

    table = pd.read_csv(KNOWN_CONTRACTS)
    return {str(c): int(t) for c, t in zip(table["contract_id"], table["term_years"])}


def register_capital_terms(merged: pd.DataFrame, basis: Mapping, *, insurance_config: Mapping | None = None,
                           learning: bool = False, vintage_table=None) -> dict[str, np.ndarray]:
    """Per register row, what the term basis needs: the row kind, its term, live share and tail inputs.

    Returns:
        ``term_row`` (a live contract on the term basis), ``expired_row`` (decision 41 (a):
        capital sunk), ``term_years``, ``live_share``, ``tail_margin_gbp_per_kw_year`` and
        ``life_years`` (for the tail's present value), one array each over ``merged``.
    """
    n = len(merged)
    out = {"term_row": np.zeros(n, dtype=bool), "expired_row": np.zeros(n, dtype=bool),
           "term_years": np.full(n, np.nan), "live_share": np.ones(n),
           "tail_margin_gbp_per_kw_year": np.zeros(n), "life_years": np.full(n, np.nan)}
    if not basis.get("enabled") or "support_basis" not in merged.columns:
        return out
    if "repowered_merchant_share" in merged.columns:
        # FES 2050 decision 62: the repowered (merchant, no CfD) share of a solar build year keeps
        # the economic-life charge; its contract share is the live share the labels already cut.
        out["merchant_share"] = pd.to_numeric(merged["repowered_merchant_share"], errors="coerce").fillna(0.0).to_numpy()
    from scripts.public_ownership.fleet_vintages import modelled_contract_term

    support_basis = merged["support_basis"].fillna("").astype(str).to_numpy()
    support = merged["support_type"].fillna("").astype(str).to_numpy()
    new_build = merged["finance_vintage"].astype(str).eq("new_build").to_numpy()
    technology = merged["stock_flow_technology"].fillna("").astype(str).to_numpy()
    year = pd.to_numeric(merged["build_year"], errors="coerce").to_numpy(dtype=float)
    live = pd.to_numeric(merged.get("cfd_live_share"), errors="coerce").to_numpy(dtype=float)
    contract = merged["cfd_contract_id"].fillna("").astype(str).str.split("#", n=1).str[0].to_numpy() \
        if "cfd_contract_id" in merged.columns else np.full(n, "")
    out["expired_row"] = new_build & (support_basis == SUPPORT_BASIS_EXPIRED)
    # Decision 42: legacy plant whose RO or CfD has ended is on the same rule (capital sunk).
    out["expired_row"] |= legacy_expired_rows(merged, basis.get("legacy_expired"))
    term_row =new_build & np.isin(support_basis, TERM_SUPPORT_BASES) & np.isin(support, ("CfD", "DPA")) & np.isfinite(year)
    known = known_contract_terms()
    capture = (load_capture_prices(basis["capture_prices"])
               if basis["amortisation"] == CONTRACT_TERM_WITH_TAIL else None)
    cache: dict = {}
    for i in np.flatnonzero(term_row):
        t, b = technology[i], int(year[i])
        if support_basis[i] == "known_contract":
            if contract[i] not in known:
                raise VintageCapitalError(f"{merged['asset_id'].iloc[i]}: known contract {contract[i]!r} has no term")
            term = known[contract[i]]
        else:
            term = modelled_contract_term(t, b)
        key = (t, b, term)
        if key not in cache:
            if capture is not None:
                inputs = tail_inputs(t, b, capture=capture, insurance_config=insurance_config, learning=learning,
                                     vintage_table=vintage_table)
                cache[key] = (inputs["margin_gbp_per_kw_year"], inputs["life_years"])
            else:
                from scripts.public_ownership import vintage_costs as vc

                cost = vc.costs_for(t, min(max(b, vc.FIRST_BUILD_YEAR), vc.LAST_BUILD_YEAR), learning=learning,
                                    table=vintage_table)
                cache[key] = (0.0, float(cost["economic_life_years"]))
        margin, life = cache[key]
        out["term_years"][i] = min(float(term), life)
        out["tail_margin_gbp_per_kw_year"][i] = margin
        out["life_years"][i] = life
        out["live_share"][i] = live[i] if np.isfinite(live[i]) else 1.0
    out["term_row"] = term_row
    if basis.get(MERCHANT_CAPITAL_KEY) == MERCHANT_LEVEL_ANNUITY:
        out.update(merchant_level_rows(merged, term_row, out["expired_row"]))
    return out


def merchant_level_rows(merged: pd.DataFrame, term_row: np.ndarray, expired_row: np.ndarray) -> dict[str, np.ndarray]:
    """Decision 66: the merchant rows on a level annuity, and those past their economic life (sunk).

    A merchant row is any row that is not on a live contract's term basis, not expired (decision 41
    (a), 42) and whose support type is not a contract, RAB or RO (:data:`NON_MERCHANT_SUPPORT`). Its
    age is the model year less its commissioning year, the age the register's remaining life uses;
    an undated row (no commissioning year) is left as it is (its remaining life is already its life).
    """
    n = len(merged)
    support = merged["support_type"].fillna("").astype(str).to_numpy() if "support_type" in merged.columns \
        else np.full(n, "")
    life = pd.to_numeric(merged["economic_life_years"], errors="coerce").to_numpy(dtype=float)
    commissioning = pd.to_numeric(merged.get("commissioning_year"), errors="coerce").to_numpy(dtype=float) \
        if "commissioning_year" in merged.columns else np.full(n, np.nan)
    age = pd.to_numeric(merged["model_year"], errors="coerce").to_numpy(dtype=float) - commissioning
    merchant = (~term_row & ~expired_row & ~np.isin(support, NON_MERCHANT_SUPPORT)
                & np.isfinite(life) & (life > 0) & np.isfinite(age))
    return {"level_row": merchant & (age < life), "past_life_row": merchant & (age >= life), "level_life_years": life}


def tail_value_vector(margin: np.ndarray, life: np.ndarray, term: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """:func:`tail_value_per_kw` over arrays (zero where there is no tail)."""
    out = np.zeros(len(margin))
    for i in np.flatnonzero((margin > 0) & np.isfinite(life) & np.isfinite(term) & (life > term)):
        out[i] = margin[i] * (annuity_factor(float(rate[i]), float(life[i])) - annuity_factor(float(rate[i]), float(term[i])))
    return out


# ── calibration (decision 41 (b)) ────────────────────────────────────────────

def load_calibration(path, allow_missing: bool = False) -> pd.DataFrame | None:
    """The calibration table, or None when no file is configured (or, with ``allow_missing``, not yet derived)."""
    if path is None or (allow_missing and not Path(path).exists()):
        return None
    table = pd.read_csv(path)
    if missing := set(CALIBRATION_COLUMNS) - set(table.columns):
        raise VintageCapitalError(f"{path} lacks {sorted(missing)}")
    if table["key"].duplicated().any():
        raise VintageCapitalError(f"{path} repeats a cohort")
    if AUCTION_MARGIN_COLUMN not in table.columns:
        # A calibration from before decision 71 broke every cohort even.
        table[AUCTION_MARGIN_COLUMN] = 0.0
    return table


def calibration_margin(table: pd.DataFrame | None) -> float:
    """Decision 71: the auction margin a calibration was derived at (its auction-priced rows; 0 without any).

    Raises:
        VintageCapitalError: The rows disagree, or a non-auction row carries a margin.
    """
    if table is None or table.empty:
        return 0.0
    shares = (pd.to_numeric(table[AUCTION_MARGIN_COLUMN], errors="coerce").fillna(0.0)
              if AUCTION_MARGIN_COLUMN in table.columns else pd.Series(0.0, index=table.index))
    auction = table["scheme"].astype(str).isin(AUCTION_SCHEMES)
    if (shares[~auction].abs() > MARGIN_TOLERANCE).any():
        raise VintageCapitalError("a calibration row outside the auctioned schemes carries an auction margin")
    values = shares[auction]
    if values.empty:
        return 0.0
    if float(values.max() - values.min()) > MARGIN_TOLERANCE:
        raise VintageCapitalError(f"the calibration mixes auction margins {sorted(set(values.round(6)))}")
    return float(values.iloc[0])


def check_calibration_margin(table: pd.DataFrame | None, margin: float, what: str = "the strike calibration") -> None:
    """Refuse a calibration derived at another auction margin than the basis's (decision 71: re-margin it)."""
    if table is None:
        return
    found = calibration_margin(table)
    if abs(found - float(margin)) > MARGIN_TOLERANCE:
        raise VintageCapitalError(f"{what} was derived at auction margin {found:g}, but the basis's "
                                  f"{CONFIG_KEY}.{AUCTION_MARGIN_KEY} is {float(margin):g}: re-margin it "
                                  "(vintage_capital_2050.py margin) or re-derive it (calibrate)")


def margin_strike(break_even: float, margin: float) -> float:
    """Decision 71: the strike whose rent is ``margin`` x contract revenue, from the break-even strike.

    Rent at strike ``s`` is ``rent(s0) + (s - s0) x paid``; at the break-even strike ``s_be`` it is zero,
    so ``rent(s) = (s - s_be) x paid``, and ``rent(s) = m x s x paid`` gives ``s = s_be / (1 - m)``.
    """
    return float(break_even) / (1.0 - float(margin))


def apply_auction_margin(table: pd.DataFrame, margin) -> pd.DataFrame:
    """Decision 71: a calibration re-margined from its own columns (no register read).

    Each auction-priced row's break-even strike is ``value_in_use - rent_at_value_in_use / denominator``
    (the derivation's own arithmetic, :func:`derive_calibration`); its calibrated value becomes that over
    ``1 - margin``. Other rows (DPAs) are unchanged. The decision-45 declared-output strike, where
    present, is scaled the same way. Every other column is kept, so files at two margins differ only
    in ``calibrated_value``, ``declared_output_strike`` and ``auction_margin_share``.
    """
    m = auction_margin_share(margin)
    out = table.copy()
    if AUCTION_MARGIN_COLUMN not in out.columns:
        out[AUCTION_MARGIN_COLUMN] = 0.0
    old = calibration_margin(out)
    auction = out["scheme"].astype(str).isin(AUCTION_SCHEMES).to_numpy()
    in_use = pd.to_numeric(out["value_in_use"], errors="coerce").to_numpy(dtype=float)
    rent = pd.to_numeric(out["rent_at_value_in_use_gbp"], errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(out["denominator"], errors="coerce").to_numpy(dtype=float)
    calibrated = pd.to_numeric(out["calibrated_value"], errors="coerce").to_numpy(dtype=float).copy()
    for i in np.flatnonzero(auction):
        # The derivation's own expression, so a re-margin at the file's margin reproduces it exactly.
        calibrated[i] = (in_use[i] - rent[i] / d[i]) / (1.0 - m) if d[i] > 0 else in_use[i]
    out["calibrated_value"] = calibrated
    out[AUCTION_MARGIN_COLUMN] = np.where(auction, m, 0.0)
    if "declared_output_strike" in out.columns:
        declared = pd.to_numeric(out["declared_output_strike"], errors="coerce").to_numpy(dtype=float)
        out["declared_output_strike"] = np.where(auction, declared * (1.0 - old) / (1.0 - m), declared)
    return out


def calibrated_strikes(table: pd.DataFrame | None, *, amortisation: str) -> dict[tuple[str, str], tuple[float, float]]:
    """``(cfd_round, technology) -> (base strike, calibrated strike)`` on the basis in force."""
    if table is None:
        return {}
    rows = table.loc[table["scheme"].eq("CfD")]
    if len(rows) and not rows["basis"].eq(amortisation).all():
        raise VintageCapitalError(f"the strike calibration was derived on {sorted(set(rows['basis']))}, "
                                  f"but the basis in force is {amortisation!r}: re-derive it")
    return {(str(r.cfd_round), str(r.technology)): (float(r.base_value), float(r.calibrated_value))
            for r in rows.itertuples(index=False)}


def calibrated_dpa_rates(table: pd.DataFrame | None, *, amortisation: str) -> dict[tuple[str, str, int], tuple[float, float]]:
    """``(dpa carrier, dpa technology, build year) -> (base APR, calibrated APR)``."""
    if table is None:
        return {}
    rows = table.loc[table["scheme"].eq("DPA")]
    if len(rows) and not rows["basis"].eq(amortisation).all():
        raise VintageCapitalError(f"the DPA calibration was derived on {sorted(set(rows['basis']))}, "
                                  f"but the basis in force is {amortisation!r}: re-derive it")
    return {(str(r.dpa_carrier), "" if pd.isna(r.dpa_technology) else str(r.dpa_technology), int(r.build_year)):
            (float(r.base_value), float(r.calibrated_value)) for r in rows.itertuples(index=False)}


def check_base(base_in_file: float, base_now: float, what: str) -> None:
    """Refuse a calibration whose base no longer matches the basis (a stale file)."""
    if not np.isclose(base_in_file, base_now, rtol=0.0, atol=1e-6):
        raise VintageCapitalError(f"{what}: the calibration's base {base_in_file:.6f} differs from the basis's "
                                  f"{base_now:.6f}; re-derive the calibration")


def _register_year_hours(frame: pd.DataFrame) -> float:
    """Hours in the register's model year (8,784 in a leap year such as 2040; 8,760 without the column)."""
    from scripts.public_ownership.model_year import is_vintaged_model_year, frame_model_year
    from scripts.utilities.calendar_year import hours_in_year

    year = frame_model_year(frame)
    return float(hours_in_year(year)) if is_vintaged_model_year(year) else 8760.0


def _register_model_year(frame: pd.DataFrame) -> int:
    """The vintaged register's model year (2050 when it carries none, the default every caller had)."""
    from scripts.public_ownership.model_year import is_vintaged_model_year, frame_model_year

    year = frame_model_year(frame)
    return int(year) if is_vintaged_model_year(year) else 2050


def _config_group_model_year(config: Mapping, run_group: str) -> int:
    """A run group's ``model_year`` in the study config (2050 when absent)."""
    groups = ((config or {}).get("public_ownership") or {}).get("reference_run_groups") or {}
    return int((groups.get(run_group) or {}).get("model_year") or 2050)


def _config_case_model_year(config: Mapping, case: str) -> int:
    """A forecast case's ``modelled_year`` in the study config (2050 when absent)."""
    cases = (((config or {}).get("consumer_bill") or {}).get("forecasts") or {}).get("cases") or {}
    return int((cases.get(case) or {}).get("modelled_year") or 2050)


def derive_capture_prices(register: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Capture price, fuel per MWh and load factor by carrier from the private case's register rows.

    The capture price is market revenue (wholesale plus balancing settlement) over the
    output the register books, which no strike moves: the months are solved.
    """
    rows = register.loc[register["case_id"].astype(str).eq("private")]
    num = lambda c: pd.to_numeric(rows[c], errors="coerce").fillna(0.0)  # noqa: E731
    frame = pd.DataFrame({"carrier": rows["carrier"].astype(str), "dispatch_mwh": num("dispatch_mwh"),
                          "capacity_mw": num("capacity_mw"), "market_revenue_gbp": num("market_revenue_gbp"),
                          "fuel_carbon_gbp": num("fuel_carbon_gbp")})
    g = frame.groupby("carrier", sort=True).sum()
    g = g.loc[g["dispatch_mwh"] > 0]
    year_hours = _register_year_hours(register)
    out = pd.DataFrame({
        "carrier": g.index, "capture_gbp_per_mwh": (g["market_revenue_gbp"] / g["dispatch_mwh"]).to_numpy(),
        "fuel_carbon_gbp_per_mwh": (g["fuel_carbon_gbp"] / g["dispatch_mwh"]).to_numpy(),
        "load_factor": (g["dispatch_mwh"] / (g["capacity_mw"] * year_hours)).to_numpy(),
        "dispatch_mwh": g["dispatch_mwh"].to_numpy(), "capacity_mw": g["capacity_mw"].to_numpy(),
        "market_revenue_gbp": g["market_revenue_gbp"].to_numpy(), "source": source})
    return out[list(CAPTURE_COLUMNS)]


def derive_calibration(register: pd.DataFrame, book: pd.DataFrame, strike_table, *, amortisation: str,
                       base_strikes: Mapping[tuple[str, str], float], base_dpa: Mapping[tuple[str, str, int], float],
                       source: str, case_id: str = "private", technologies=None,
                       include_dpa: bool = True, auction_margin: float = 0.0) -> pd.DataFrame:
    """Each private post-AR7 CfD and DPA cohort's break-even strike or availability rate.

    ``value* = value in use - private cohort rent / d(support)/d(value)``: for a CfD the paid
    energy (the settlement's ``cfd_paid_mwh``, the hours the contract pays, after the
    negative-pricing rule), for a DPA ``kW x expected availability x live share``. Rent is
    the register's producer rent (carbon-removal CfD included: it is the plant's income).

    Args:
        register: Private-case rows of the vintaged register (with ``cfd_paid_mwh``).
        book: The contract book the settlement used (the values in use).
        strike_table: The bill's ``StrikeTable`` (the strikes in use).
        amortisation: The basis in force.
        base_strikes: ``(round, technology) -> strike`` on the basis before calibration.
        base_dpa: ``(carrier, technology, build year) -> APR`` before calibration.
        source: Provenance text.
        case_id: The private case the cohorts break even in (decision 54: ``private@private_rate_neso_wind_ramp``
            derives that rate path's strikes, at the strikes in use).
        technologies: Strike-table technologies to calibrate (None: all but :data:`UNCALIBRATED_TECHNOLOGIES`).
        include_dpa: Calibrate the DPA cohorts too (a rate path moves wind only: False).
        auction_margin: Decision 71: the share of contract revenue (strike x paid MWh) each post-AR7 CfD
            cohort keeps as rent, ``value* = (value in use - rent / paid) / (1 - m)``; 0 breaks even
            (exactly the decision-41 (b) value). DPA cohorts (allocated bilaterally) always break even.
    """
    margin = auction_margin_share(auction_margin)
    rows = register.loc[register["case_id"].astype(str).eq(str(case_id))].copy()
    num = lambda c: pd.to_numeric(rows[c], errors="coerce").fillna(0.0)  # noqa: E731
    rows["rent"] = num("producer_rent_gbp")
    if "cfd_paid_mwh" not in rows.columns:
        raise VintageCapitalError("the register carries no cfd_paid_mwh: settle with the 2050 settlement first")
    rows["paid"] = num("cfd_paid_mwh")
    out = []
    cfd = rows.loc[rows["support_type"].astype(str).eq("CfD") & rows["cfd_round"].astype(str).str.startswith("PAR7_")]
    keys = {}
    for carrier, rnd in cfd[["carrier", "cfd_round"]].drop_duplicates().itertuples(index=False):
        key, fallback = strike_table.resolve(carrier, rnd)
        if key is None or fallback:
            raise VintageCapitalError(f"no strike entry for ({carrier}, {rnd})")
        keys[(carrier, rnd)] = key
    cfd = cfd.assign(_key=[keys[(c, r)] for c, r in zip(cfd["carrier"], cfd["cfd_round"])])
    cfd = cfd.loc[[k[1] not in UNCALIBRATED_TECHNOLOGIES and (technologies is None or k[1] in technologies)
                   for k in cfd["_key"]]]
    for key, g in cfd.groupby("_key", sort=True):
        paid = float(g["paid"].sum())
        # Decision 54: a case on another private rate path settles each contract at the table's strike
        # less the register's offset for that path (constant over the entry).
        offset = (float(pd.to_numeric(g["cfd_strike_offset_gbp_per_mwh"], errors="coerce").fillna(0.0).mean())
                  if "cfd_strike_offset_gbp_per_mwh" in g.columns else 0.0)
        in_use = float(strike_table.strikes[key]) - offset
        rent = float(g["rent"].sum())
        # Decision 71: rent(s) = m x s x paid; at m = 0 the decision-41 (b) break-even strike, bit for bit.
        value = (in_use - rent / paid) / (1.0 - margin) if paid > 0 else in_use
        out.append({"scheme": "CfD", "key": f"{key[0]}_{key[1]}", "cfd_round": key[0], "technology": key[1],
                    "dpa_carrier": "", "dpa_technology": "", "build_year": int(key[0].split("_", 1)[1]),
                    "basis": amortisation, "base_value": float(base_strikes[key]), "calibrated_value": value,
                    "unit": "GBP/MWh 2024", "rent_at_value_in_use_gbp": rent, "denominator": paid,
                    "denominator_unit": "MWh paid", "value_in_use": in_use, "source": source,
                    AUCTION_MARGIN_COLUMN: margin,
                    **declared_output_strike(g, key[1], int(key[0].split("_", 1)[1]), auction_margin=margin)})
    dpa_book = book.loc[book["scheme"].astype(str).eq("DPA")].set_index("unit")
    dpa = rows.loc[rows["support_type"].astype(str).eq("DPA")].copy() if include_dpa else rows.iloc[:0]
    if len(dpa):
        units = dpa_book.reindex(dpa["asset_id"].astype(str))
        if units["dpa_carrier"].isna().any():
            raise VintageCapitalError("DPA register rows missing from the contract book")
        dpa["_carrier"] = units["dpa_carrier"].astype(str).to_numpy()
        dpa["_tech"] = units["dpa_technology"].fillna("").astype(str).to_numpy()
        dpa["_year"] = units["build_year"].astype(int).to_numpy()
        dpa["_apr"] = pd.to_numeric(units["dpa_apr_gbp_per_kw_year"]).to_numpy()
        dpa["_d"] = (pd.to_numeric(units["capacity_mw"]).to_numpy() * 1e3
                     * pd.to_numeric(units["dpa_expected_availability"]).to_numpy()
                     * pd.to_numeric(units["live_share"]).to_numpy())
        for (carrier, tech, year), g in dpa.groupby(["_carrier", "_tech", "_year"], sort=True):
            d = float(g["_d"].sum())
            in_use = float(g["_apr"].iloc[0])
            if not np.allclose(g["_apr"], in_use):
                raise VintageCapitalError(f"DPA cohort {carrier} {tech} {year} has several rates in use")
            rent = float(g["rent"].sum())
            out.append({"scheme": "DPA", "key": f"DPA_{year}_{carrier}_{tech}", "cfd_round": f"DPA_{year}",
                        "technology": "", "dpa_carrier": carrier, "dpa_technology": tech, "build_year": int(year),
                        "basis": amortisation, "base_value": float(base_dpa[(carrier, tech, int(year))]),
                        "calibrated_value": in_use - rent / d if d > 0 else in_use, "unit": "GBP/kW/yr",
                        "rent_at_value_in_use_gbp": rent, "denominator": d, "denominator_unit": "kW x availability x live share",
                        "value_in_use": in_use, "source": source, AUCTION_MARGIN_COLUMN: 0.0})
    return pd.DataFrame(out, columns=list(CALIBRATION_COLUMNS) + [AUCTION_MARGIN_COLUMN] + list(DECLARED_OUTPUT_COLUMNS))


def declared_output_strike(rows: pd.DataFrame, table_technology: str, build_year: int,
                           auction_margin: float = 0.0) -> dict:
    """Decision 45: the strike a CfD cohort would need if it delivered its declared load factor.

    The cohort's required revenue (capital on the term basis, fixed and variable O&M, fuel)
    recovered over the output its declared load factor gives (``cfd_strike_inputs_2050.csv``,
    DESNZ Annex A) on its live share, all of it paid, with variable O&M and fuel scaled to that
    output at the modelled rates. The calibrated strike recovers the same cost over the
    output the solved year delivered (curtailment and BM turn-down included, decision 39), so
    the two differ by the delivered-to-declared output ratio. Reported, never read back. With an
    auction margin (decision 71) it carries the same margin: the cost over ``1 - m``.
    """
    technology = TABLE_TECHNOLOGY.get(str(table_technology))
    blank = {"declared_load_factor": np.nan, "delivered_load_factor": np.nan, "declared_output_strike": np.nan}
    if technology is None:
        return blank
    from scripts.market import cfd_strikes_2050 as cfd50

    num = lambda c: pd.to_numeric(rows[c], errors="coerce").fillna(0.0) if c in rows.columns         else pd.Series(0.0, index=rows.index)  # noqa: E731
    lf = float(cfd50.strike_inputs_for(technology, int(build_year))["expected_load_factor"])
    live = (pd.to_numeric(rows["cfd_live_share"], errors="coerce").fillna(1.0) if "cfd_live_share" in rows.columns
            else pd.Series(1.0, index=rows.index))
    hours = num("capacity_mw") * _register_year_hours(rows) * live
    declared = float((hours * lf).sum())
    dispatch = float(num("dispatch_mwh").sum())
    variable = float(num("annual_variable_om_gbp").sum() + num("fuel_carbon_gbp").sum())
    required = float(num("annual_required_revenue_gbp").sum()) - float(num("annual_variable_om_gbp").sum())
    if declared <= 0:
        return blank
    per_mwh = variable / dispatch if dispatch > 0 else 0.0
    return {"declared_load_factor": lf,
            "delivered_load_factor": dispatch / float(hours.sum()) if float(hours.sum()) > 0 else np.nan,
            "declared_output_strike": (required / declared + per_mwh) / (1.0 - float(auction_margin))}


# ── command line ─────────────────────────────────────────────────────────────

#: Decision 54: the strike-table technologies a private rate path moves (the NESO ramp: wind).
RATE_PATH_TECHNOLOGIES = {"neso_wind_ramp": ("offshore", "onshore")}


def rate_path_private_case(rate_path: str) -> str:
    """``neso_wind_ramp`` -> ``private@private_rate_neso_wind_ramp``: the private case on that path."""
    from scripts.public_ownership.levers import GATED_LEVERS, LEVER_PRIVATE_RATE_PATH, VARIANT_SEPARATOR

    return f"private{VARIANT_SEPARATOR}{GATED_LEVERS[LEVER_PRIVATE_RATE_PATH].label(rate_path)}"


#: FES 2050 decision 70: the dispatch year a calibration reads. ``market`` is decision 41 (b)'s
#: calibration (the register); ``central`` is the strike diagnostic on the central-dispatch year.
DISPATCH_BASES = ("market", "central")


def _cli_central(args) -> int:
    """Decision 70: ``calibrate --dispatch-basis central``, the strike diagnostic on the central-dispatch year.

    Reads the market year's calibration (the strikes to compare with), the contract book and the
    solved market and central months; writes the per-cohort CSV and the JSON summary. A diagnostic:
    no file the finance chain reads is written.
    """
    import logging

    import yaml

    from scripts.public_ownership import central_strike_diagnostic_2050 as csd

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.check:
        raise VintageCapitalError("--check is the market basis's; the central basis is a diagnostic")
    calibration = args.market_calibration
    if calibration is None and args.config:
        basis = basis_from_config(_read_config(args.config) or {})
        calibration = basis["strike_calibration"]
    frame, summary = csd.run(
        study=args.study or csd.DEFAULT_STUDY, months=tuple(args.months or csd.MONTHS), book_path=args.book,
        calibration_path=calibration or csd.DEFAULT_CALIBRATION,
        bill_summary_path=args.bill_summary or csd.DEFAULT_BILL_SUMMARY, out_csv=args.out or csd.DEFAULT_OUT,
        register_path=args.register or csd.DEFAULT_REGISTER,
        out_json=args.json_out, central_suffix=args.central_suffix or csd.CENTRAL_SUFFIX,
        central_price_basis=args.central_price)
    csd.print_summary(summary)
    print(f"wrote {args.out or csd.DEFAULT_OUT}: {len(frame)} cohorts")
    return 0


def _cli_margin(args) -> int:
    """Decision 71: ``margin``, a calibration re-margined from its own columns (the installed file or a bound's)."""
    table = load_calibration(args.calibration)
    before = calibration_margin(table)
    out = apply_auction_margin(table, args.auction_margin)
    after = calibration_margin(out)
    auction = out["scheme"].astype(str).isin(AUCTION_SCHEMES)
    moved = (pd.to_numeric(out["calibrated_value"]) - pd.to_numeric(table["calibrated_value"]))[auction]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    span = f"{moved.min():+.4f} to {moved.max():+.4f}" if len(moved) else "none"
    print(f"auction margin {before:g} -> {after:g} on {int(auction.sum())} post-AR7 CfD cohorts "
          f"(strikes moved {span} GBP/MWh 2024); {int((~auction).sum())} DPA rows unchanged; wrote {args.out}")
    return 0


def bound_margin_spec(config: Mapping, run_group: str, root: Path | str = ROOT) -> dict:
    """Decision 71: a register bound's auction margin (``auction_margin: {level, strike_calibration}``).

    Returns ``level``, ``share`` (the 2030 lever's value), ``strike_calibration`` (a path) and ``parent``
    (the group it bounds). Only ``none`` and ``high`` are bounds; ``central`` is the study's own calibration.
    """
    block = dict(config or {})
    block = dict(block.get("public_ownership", block) or {})
    group = (block.get("reference_run_groups") or {}).get(run_group)
    if group is None:
        raise VintageCapitalError(f"no run group {run_group!r} in the config")
    raw = group.get(GROUP_AUCTION_MARGIN_KEY)
    parent = group.get("register_bound_of")
    if not isinstance(raw, Mapping) or parent is None:
        raise VintageCapitalError(f"{run_group}: an auction-margin bound is a register bound with an "
                                  f"{GROUP_AUCTION_MARGIN_KEY} block")
    if unknown := sorted(set(raw) - {"level", "strike_calibration"}):
        raise VintageCapitalError(f"{run_group}.{GROUP_AUCTION_MARGIN_KEY} has unknown keys {unknown}")
    level = str(raw.get("level"))
    if level not in AUCTION_MARGIN_LEVELS or level == "central":
        raise VintageCapitalError(f"{run_group}.{GROUP_AUCTION_MARGIN_KEY}.level={level!r}: none or high (central is "
                                  "the study's own calibration)")
    if not raw.get("strike_calibration"):
        raise VintageCapitalError(f"{run_group}.{GROUP_AUCTION_MARGIN_KEY} names no strike_calibration file")
    return {"level": level, "share": AUCTION_MARGIN_LEVELS[level], "parent": str(parent),
            "strike_calibration": _path_of(raw["strike_calibration"], root)}


#: Decision 71: the bill-assumption entries a margin bound settles itself, as ``(block, key, file suffix)``;
#: its strikes move them. Every other path stays its parent's, as for every register bound.
BOUND_ASSUMPTION_FILES = (("cfd", "contract_book", "_support_contract_book.csv"),
                          ("dpa", "settlement_summary", "_asset_settlement_support.json"))


def bound_assumptions(assumptions: Mapping, *, parent: str, group: str, strike_table: str) -> dict:
    """Decision 71: the forecast case's bill assumptions with a margin bound's own strike table and settlement.

    ``model_linked_rules.cfd.strike_table`` becomes the bound's table, and the contract book and the DPA
    settlement summary become the bound's own (``{group}_support_contract_book.csv``,
    ``{group}_asset_settlement_support.json``). The BM strike table (``cfd.bm_strike_table``) is kept: the
    bound is re-accounted on the central months, whose BM bid on the central labels.
    """
    import copy

    out = copy.deepcopy(dict(assumptions))
    rules = out["model_linked_rules"]
    rules["cfd"]["strike_table"] = str(strike_table)
    for block, key, suffix in BOUND_ASSUMPTION_FILES:
        value = (rules.get(block) or {}).get(key)
        if not value:
            continue
        head, _, name = str(value).replace("\\", "/").rpartition("/")
        if name != f"{parent}{suffix}":
            raise VintageCapitalError(f"model_linked_rules.{block}.{key}={value!r} is not {parent!r}'s {suffix}")
        rules[block][key] = f"{head}/{group}{suffix}" if head else f"{group}{suffix}"
    if rules["cfd"].get("strike_basis"):
        rules["cfd"]["strike_basis"] = (f"{rules['cfd']['strike_basis']} Decision 71 bound {group}: the post-AR7 "
                                        "strikes at its own auction margin; the BM bid on the central labels.")
    return out


def write_bound_inputs(config_path, run_group: str, assumptions_path, *, strike_table_out, assumptions_out,
                       snapshot_path=None, model_year: int | None = None) -> pd.DataFrame:
    """Decision 71: a margin bound's strike table (its calibration on the study's basis) and bill assumptions.

    The table is :func:`contract_book_2050.basis_strike_frame` with the bound's calibration, whose margin
    must be its level's and whose derivation columns must be the installed central calibration's (it is
    that file re-margined); the assumptions are :func:`bound_assumptions` pointing at the table.
    """
    import json

    import yaml

    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.public_ownership import contract_book_2050 as cb

    config = _read_config(config_path) or {}
    if model_year is None:
        model_year = _config_group_model_year(config, run_group)
    spec = bound_margin_spec(config, run_group)
    basis = basis_from_config(config)
    if not basis["enabled"]:
        raise VintageCapitalError(f"{run_group}: an auction-margin bound needs the study's term basis")
    calibration = load_calibration(spec["strike_calibration"])
    found = calibration_margin(calibration)
    if abs(found - spec["share"]) > MARGIN_TOLERANCE:
        raise VintageCapitalError(f"{spec['strike_calibration']} is at margin {found:g}, not {spec['level']} "
                                  f"({spec['share']:g}): write it with `margin --auction-margin {spec['level']}`")
    check_bound_variant(load_calibration(basis["strike_calibration"]), calibration, spec["strike_calibration"])
    snapshot = cfd50.load_snapshot(snapshot_path) if snapshot_path else cfd50.load_snapshot()
    frame = cb.basis_strike_frame({**basis, AUCTION_MARGIN_KEY: found}, snapshot=snapshot,
                                  insurance_config=cb._insurance_from(config_path), model_year=model_year,
                                  calibration=calibration)
    Path(strike_table_out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(strike_table_out, index=False)
    assumptions = json.loads(Path(assumptions_path).read_text(encoding="utf-8"))
    out = bound_assumptions(assumptions, parent=spec["parent"], group=run_group,
                            strike_table=Path(strike_table_out).as_posix())
    Path(assumptions_out).parent.mkdir(parents=True, exist_ok=True)
    Path(assumptions_out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return frame


#: The columns a re-margined calibration shares with the one it came from (decision 71).
DERIVATION_COLUMNS = ("scheme", "basis", "base_value", "value_in_use", "rent_at_value_in_use_gbp", "denominator")


def check_bound_variant(central: pd.DataFrame | None, variant: pd.DataFrame, path) -> None:
    """A bound's calibration is the central one re-margined: the same cohorts, bases and derivation columns."""
    if central is None:
        return
    a = central.set_index("key").sort_index()
    b = variant.set_index("key").sort_index()
    if not a.index.equals(b.index):
        raise VintageCapitalError(f"{path} does not hold the central calibration's cohorts: re-write it with `margin`")
    num = ["base_value", "value_in_use", "rent_at_value_in_use_gbp", "denominator"]
    same_text = a[["scheme", "basis"]].astype(str).equals(b[["scheme", "basis"]].astype(str))
    same_num = np.allclose(a[num].to_numpy(dtype=float), b[num].to_numpy(dtype=float), rtol=1e-12, atol=1e-9)
    if not (same_text and same_num):
        raise VintageCapitalError(f"{path} was not re-margined from the central calibration in use (its derivation "
                                  "columns differ): re-write it with `margin` from the installed file")


# ── decision 70, the full private central-dispatch case (25 Oct 2026) ─────────
#
# The diagnostic was material (post-AR7 strikes 99.97 -> 97.24 GBP/MWh paid-weighted, about -5.57 GBP per
# 3,100 kWh ex VAT), so the case is built: the market bill on the central year (forecast case
# ``institution: central_dispatch``, FTI's settlement) with the post-AR7 CfDs at the diagnostic's central
# strikes. The forecast case's ``central_dispatch_strikes`` block names the inputs; ``central-inputs``
# writes the strike calibration, strike table and bill assumptions it settles on. No solve.

#: The forecast-case block naming a central-dispatch case's strike inputs (decision 70).
CENTRAL_STRIKES_KEY = "central_dispatch_strikes"
#: The diagnostic's columns the central calibration reads.
DIAGNOSTIC_STRIKE_COLUMNS = ("key", "market_strike_gbp_per_mwh", "central_strike_gbp_per_mwh")
#: How far the diagnostic's market strike may sit from the calibration's (GBP/MWh): it read that file.
DIAGNOSTIC_STRIKE_TOLERANCE = 1e-6


def central_dispatch_calibration(calibration: pd.DataFrame, diagnostic: pd.DataFrame,
                                 source: str = "the decision-70 strike diagnostic") -> pd.DataFrame:
    """Decision 70: the installed calibration with every post-AR7 CfD cohort at its central-year strike.

    ``diagnostic`` is ``fes_ht_2050_central_dispatch_strike_diagnostic.csv`` (one row per cohort, keyed
    as the calibration is). The cohorts must be the calibration's auction-priced rows exactly, and each
    diagnostic market strike must equal that row's ``calibrated_value`` (it was derived from this file;
    a stale diagnostic is refused). Only ``calibrated_value`` (and ``source``) move; the DPA rows, the
    margin and the derivation columns are kept, so the strike table built on it differs from the
    installed one only in the post-AR7 CfD strikes.

    Raises:
        VintageCapitalError: Missing columns, a cohort mismatch, a stale or non-finite strike.
    """
    if missing := set(DIAGNOSTIC_STRIKE_COLUMNS) - set(diagnostic.columns):
        raise VintageCapitalError(f"the strike diagnostic lacks {sorted(missing)}")
    if diagnostic["key"].duplicated().any():
        raise VintageCapitalError("the strike diagnostic repeats a cohort")
    out = calibration.copy()
    auction = out["scheme"].astype(str).isin(AUCTION_SCHEMES)
    keys = out.loc[auction, "key"].astype(str)
    diag = diagnostic.set_index(diagnostic["key"].astype(str))
    if set(keys) != set(diag.index):
        only_cal = sorted(set(keys) - set(diag.index))[:3]
        only_diag = sorted(set(diag.index) - set(keys))[:3]
        raise VintageCapitalError(f"the strike diagnostic's cohorts are not the calibration's post-AR7 CfD rows "
                                  f"(calibration only: {only_cal}; diagnostic only: {only_diag}): re-run the diagnostic")
    market = pd.to_numeric(diag.loc[keys, "market_strike_gbp_per_mwh"], errors="coerce").to_numpy(dtype=float)
    central = pd.to_numeric(diag.loc[keys, "central_strike_gbp_per_mwh"], errors="coerce").to_numpy(dtype=float)
    calibrated = pd.to_numeric(out.loc[auction, "calibrated_value"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(central).all() and np.isfinite(market).all()):
        raise VintageCapitalError("the strike diagnostic has a non-finite strike")
    stale = np.abs(market - calibrated) > DIAGNOSTIC_STRIKE_TOLERANCE
    if stale.any():
        first = keys.to_numpy()[np.flatnonzero(stale)[0]]
        raise VintageCapitalError(f"the strike diagnostic's market strikes are not the calibration's (e.g. {first}): "
                                  "it was derived from another calibration; re-run the diagnostic")
    out.loc[auction, "calibrated_value"] = central
    out.loc[auction, "source"] = (out.loc[auction, "source"].astype(str)
                                  + f"; decision 70: central-dispatch year strike from {source}")
    return out


def central_dispatch_assumptions(assumptions: Mapping, *, case: str, strike_table: str, contract_book: str) -> dict:
    """Decision 70: a forecast case's bill assumptions with the central-dispatch case's strike table and contract book.

    ``model_linked_rules.cfd.strike_table`` and ``.contract_book`` become the case's own (the central
    strikes); everything else, the BM strike table and the DPA settlement summary among it, is the
    parent's. The parent must name a contract book (a 2050 bill).
    """
    import copy

    out = copy.deepcopy(dict(assumptions))
    cfd = out["model_linked_rules"]["cfd"]
    if not cfd.get("contract_book"):
        raise VintageCapitalError(f"{case}: the parent's bill assumptions name no model_linked_rules.cfd.contract_book "
                                  "(a central-dispatch strike case is a 2050 bill)")
    cfd["strike_table"] = str(strike_table)
    cfd["contract_book"] = str(contract_book)
    if cfd.get("strike_basis"):
        cfd["strike_basis"] = (f"{cfd['strike_basis']} Decision 70 ({case}): the post-AR7 strikes re-derived on the "
                               "central-dispatch year (uncompensated curtailment only); the BM strike table is the "
                               "market year's.")
    return out


def central_dispatch_files(case: str, root: str = "resources") -> dict:
    """Decision 70: the files a central-dispatch strike case settles on, named by its forecast case."""
    base = f"{root}/public_ownership/{case}"
    return {"calibration": f"{base}_central_strike_calibration.csv", "strike_table": f"{base}_cfd_strike_table.csv",
            "assumptions": f"{base}_bill_assumptions.json", "contract_book": f"{base}_support_contract_book.csv"}


def central_strikes_spec(case_block: Mapping, case: str) -> dict:
    """A forecast case's ``central_dispatch_strikes`` block, checked: ``assumptions_from``, ``market_group``, ``diagnostic``.

    The case must be a central-dispatch case (``institution: central_dispatch``) whose ``assumptions``
    is the generated file :func:`central_dispatch_files` names.
    """
    raw = (case_block or {}).get(CENTRAL_STRIKES_KEY)
    if not isinstance(raw, Mapping):
        raise VintageCapitalError(f"{case}: no {CENTRAL_STRIKES_KEY} block")
    if unknown := sorted(set(raw) - {"assumptions_from", "market_group", "diagnostic"}):
        raise VintageCapitalError(f"{case}.{CENTRAL_STRIKES_KEY} has unknown keys {unknown}")
    if missing := [k for k in ("assumptions_from", "market_group", "diagnostic") if not raw.get(k)]:
        raise VintageCapitalError(f"{case}.{CENTRAL_STRIKES_KEY} lacks {missing}")
    if case_block.get("institution") != "central_dispatch":
        raise VintageCapitalError(f"{case}: {CENTRAL_STRIKES_KEY} is for a central-dispatch case "
                                  "(institution: central_dispatch)")
    return {k: str(raw[k]) for k in ("assumptions_from", "market_group", "diagnostic")}


def write_central_dispatch_inputs(config_path, case: str, assumptions_path, diagnostic_path, *, calibration_out,
                                  strike_table_out, assumptions_out, contract_book, snapshot_path=None,
                                  model_year: int | None = None) -> pd.DataFrame:
    """Decision 70: write a central-dispatch case's calibration, strike table and bill assumptions.

    The calibration is :func:`central_dispatch_calibration` of the study's installed one; the table is
    :func:`contract_book_2050.basis_strike_frame` with it (the study's basis, margin checked); the
    assumptions are :func:`central_dispatch_assumptions` naming that table and ``contract_book``.
    """
    import json

    import yaml

    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.public_ownership import contract_book_2050 as cb

    config = _read_config(config_path) or {}
    if model_year is None:
        model_year = _config_case_model_year(config, case)
    basis = basis_from_config(config)
    if not basis["enabled"]:
        raise VintageCapitalError(f"{case}: a central-dispatch strike case needs the study's term basis")
    calibration = central_dispatch_calibration(load_calibration(basis["strike_calibration"]),
                                               pd.read_csv(diagnostic_path), source=Path(diagnostic_path).name)
    Path(calibration_out).parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(calibration_out, index=False)
    snapshot = cfd50.load_snapshot(snapshot_path) if snapshot_path else cfd50.load_snapshot()
    frame = cb.basis_strike_frame(basis, snapshot=snapshot, insurance_config=cb._insurance_from(config_path),
                                  model_year=model_year, calibration=calibration)
    Path(strike_table_out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(strike_table_out, index=False)
    assumptions = json.loads(Path(assumptions_path).read_text(encoding="utf-8"))
    out = central_dispatch_assumptions(assumptions, case=case, strike_table=Path(strike_table_out).as_posix(),
                                       contract_book=Path(contract_book).as_posix())
    Path(assumptions_out).parent.mkdir(parents=True, exist_ok=True)
    Path(assumptions_out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return frame


def _cli(argv=None) -> int:
    """``capture``: write the capture prices from a register; ``calibrate``: derive (or ``--check``) the calibration."""
    import yaml

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.public_ownership.register_io import read_register

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--register", required=True)
    c.add_argument("--out", required=True)
    k = sub.add_parser("calibrate")
    k.add_argument("--register", default=None,
                   help="the vintaged register (required on the market basis; central basis: variable costs)")
    k.add_argument("--book", required=True)
    k.add_argument("--assumptions", default=None,
                   help="the 2050 bill assumptions (the strike table in use; required on the market basis)")
    k.add_argument("--config", default=None, help="the study's resolved config (required on the market basis)")
    k.add_argument("--snapshot", default=None)
    k.add_argument("--out", default=None)
    k.add_argument("--check", action="store_true")
    k.add_argument("--rate-path", default="flat",
                   help="decision 54: derive this private rate path's calibration (neso_wind_ramp) on its private case")
    # Decision 70 (user, 4 Oct 2026): the strike diagnostic on the central-dispatch year.
    k.add_argument("--dispatch-basis", choices=DISPATCH_BASES, default="market",
                   help="decision 70: 'central' re-derives the post-AR7 strikes on the central-dispatch year's "
                        "output (central_strike_diagnostic_2050; a diagnostic, never read back)")
    k.add_argument("--market-calibration", default=None,
                   help="central basis: the market year's calibration (default: --config's strike_calibration, "
                        "else the installed file)")
    k.add_argument("--study", default=None, help="central basis: the study (FES25_HT_2050)")
    k.add_argument("--months", nargs="+", default=None, help="central basis: months to read (default all twelve)")
    k.add_argument("--central-suffix", default=None, help="central basis: the central year's run-id suffix")
    k.add_argument("--central-price", default="market_copperplate",
                   help="central basis: market_copperplate (FTI national price, default) or central_stage1")
    k.add_argument("--bill-summary", default=None, help="central basis: the 2050 bill summary (per-household conversion)")
    k.add_argument("--json-out", default=None, help="central basis: the JSON summary (default beside --out)")
    # Decision 71 (user, 4 Oct 2026): the auction margin the post-AR7 CfD cohorts keep.
    k.add_argument("--auction-margin", default=None,
                   help="decision 71: share of contract revenue kept as rent (none | central | high or a number; "
                        f"default: the config's {CONFIG_KEY}.{AUCTION_MARGIN_KEY}, else 0)")
    mg = sub.add_parser("margin", help="decision 71: re-margin a calibration from its own columns (no register read)")
    mg.add_argument("--calibration", required=True, help="the calibration to re-margin")
    mg.add_argument("--auction-margin", required=True, help="none | central | high, or a share")
    mg.add_argument("--out", required=True, help="where to write it (may be --calibration itself)")
    bi = sub.add_parser("bound-inputs", help="decision 71: a margin bound's own strike table and bill assumptions")
    bi.add_argument("--config", required=True, help="the study's resolved config")
    bi.add_argument("--run-group", required=True)
    bi.add_argument("--assumptions", required=True, help="the forecast case's bill assumptions (the parent's)")
    bi.add_argument("--snapshot", default=None)
    bi.add_argument("--strike-table-out", required=True)
    bi.add_argument("--assumptions-out", required=True)
    ci = sub.add_parser("central-inputs", help="decision 70: a central-dispatch case's strikes and bill assumptions")
    ci.add_argument("--config", required=True, help="the study's resolved config")
    ci.add_argument("--case", required=True, help="the central-dispatch forecast case (FES25_HT_2050_cd)")
    ci.add_argument("--assumptions", required=True, help="the market case's bill assumptions (assumptions_from)")
    ci.add_argument("--diagnostic", required=True, help="the decision-70 strike diagnostic CSV")
    ci.add_argument("--snapshot", default=None)
    ci.add_argument("--calibration-out", required=True)
    ci.add_argument("--strike-table-out", required=True)
    ci.add_argument("--assumptions-out", required=True)
    ci.add_argument("--contract-book", required=True, help="the case's contract book (named in the assumptions)")
    args = parser.parse_args(argv)
    if args.command == "margin":
        return _cli_margin(args)
    if args.command == "central-inputs":
        write_central_dispatch_inputs(args.config, args.case, args.assumptions, args.diagnostic,
                                      calibration_out=args.calibration_out, strike_table_out=args.strike_table_out,
                                      assumptions_out=args.assumptions_out, contract_book=args.contract_book,
                                      snapshot_path=args.snapshot)
        print(f"wrote {args.calibration_out}, {args.strike_table_out} and {args.assumptions_out}")
        return 0
    if args.command == "bound-inputs":
        write_bound_inputs(args.config, args.run_group, args.assumptions, strike_table_out=args.strike_table_out,
                           assumptions_out=args.assumptions_out, snapshot_path=args.snapshot)
        print(f"wrote {args.strike_table_out} and {args.assumptions_out}")
        return 0
    if args.command == "calibrate" and args.dispatch_basis == "central":
        return _cli_central(args)
    if args.command == "calibrate":
        for name in ("register", "assumptions", "config"):
            if getattr(args, name) is None:
                parser.error(f"calibrate --{name} is required on the market basis")
    if args.command == "capture":
        reg = read_register(args.register, cases=["private"])
        table = derive_capture_prices(reg, source=f"{Path(args.register).name}, private case: market revenue / output")
        table.to_csv(args.out, index=False)
        print(table.to_string(index=False))
        return 0
    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.public_ownership import contract_book_2050 as cb
    from scripts.public_ownership.build_cfd_strike_adjustments import load_bill_strike_table

    config = _read_config(args.config) or {}
    basis = basis_from_config(config)
    rate_path = str(args.rate_path)
    case_id = "private" if rate_path == "flat" else rate_path_private_case(rate_path)
    reg = read_register(args.register, cases=[case_id])
    book = cb.read_contract_book(args.book)
    table = load_bill_strike_table(args.assumptions, 2024)
    snapshot = cfd50.load_snapshot(args.snapshot) if args.snapshot else cfd50.load_snapshot()
    insurance = cfd50.finance_insurance_config(args.config)
    base = cb.basis_strike_frame(basis, snapshot=snapshot, insurance_config=insurance, rate_path=rate_path,
                                 model_year=_register_model_year(reg))
    base_strikes = {(r.cfd_round, r.technology): float(r.strike_price_gbp_per_mwh_2012) * cfd50.CPI_2012_TO_2024
                    for r in base.itertuples(index=False)}
    capture = load_capture_prices(basis["capture_prices"]) if basis["capture_prices"] else None
    base_dpa = {}
    for (carrier, tech, year) in book.loc[book["scheme"].eq("DPA"), ["dpa_carrier", "dpa_technology", "build_year"]] \
            .drop_duplicates().itertuples(index=False):
        tech = "" if pd.isna(tech) else str(tech)
        base_dpa[(str(carrier), tech, int(year))] = cb.basis_dpa_apr(
            str(carrier), tech or None, int(year), basis=basis, snapshot=snapshot, insurance_config=insurance,
            capture=capture)
    if rate_path != "flat" and (basis.get("rate_path_calibrations") or {}).get(rate_path) is None:
        raise VintageCapitalError(f"{RATE_PATH_CALIBRATIONS_KEY} names no file for {rate_path!r}")
    # Decision 71: the margin the cohorts keep (the CLI's, else the study's; 0 breaks even).
    margin = auction_margin_share(args.auction_margin if args.auction_margin is not None else basis[AUCTION_MARGIN_KEY])
    frame = derive_calibration(reg, book, table, amortisation=basis["amortisation"], base_strikes=base_strikes,
                               base_dpa=base_dpa, source=f"derived from {Path(args.register).name} ({case_id})",
                               case_id=case_id, technologies=RATE_PATH_TECHNOLOGIES.get(rate_path),
                               include_dpa=rate_path == "flat", auction_margin=margin)
    drift = (frame["calibrated_value"] - frame["value_in_use"]).abs()
    print(frame.assign(delta=frame["calibrated_value"] - frame["base_value"])[
        ["key", "base_value", "value_in_use", "calibrated_value", "delta", "rent_at_value_in_use_gbp"]].to_string(index=False))
    print(f"auction margin (decision 71): {margin:g} of contract revenue on the post-AR7 CfD cohorts")
    print(f"largest re-derivation drift: {drift.max():.6f}")
    if args.check:
        failed = False
        installed = (basis["strike_calibration"] if rate_path == "flat"
                     else (basis.get("rate_path_calibrations") or {}).get(rate_path))
        if installed is not None and Path(installed).exists():
            installed_margin = calibration_margin(load_calibration(installed))
            print(f"installed calibration {Path(installed).name}: auction margin {installed_margin:g}")
            if abs(installed_margin - margin) > MARGIN_TOLERANCE:
                print(f"FAIL: the installed calibration's margin {installed_margin:g} is not {margin:g}; "
                      "re-margin it (margin) or re-derive it with --out")
                failed = True
        if drift.max() > CHECK_TOLERANCE:
            print("FAIL: the calibration in use does not earn its margin; re-derive with --out")
            failed = True
        if failed:
            return 1
        print(f"OK: every calibrated cohort earns its margin ({margin:g}; 0 = breaks even)")
        return 0
    if args.out:
        frame.to_csv(args.out, index=False)
        print(f"wrote {args.out}: {len(frame)} cohorts")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
