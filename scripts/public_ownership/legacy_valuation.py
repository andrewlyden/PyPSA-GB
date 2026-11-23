"""Value the legacy fleet's cash flows as a declining profile (item N0, backlog G1).

Acquisition buys the legacy fleet's cash flows.  Consumers pay for that fleet
today through the market price on its output, CfD top-ups and the Renewables
Obligation; if the state owned it, it would receive those flows.  Their value is
a present value of a *declining* profile, never a level annuity of the 2030
snapshot: the RO closes on 31 March 2037, legacy CfDs expire eighteen years after
award, and the plant retires.  This module builds that profile per register row
and discounts it at the public rate (what the flows are worth to the state) and
at each row's private WACC (what the owners would sell for), which is the
purchase-price axis item N4 apportions.

Inputs are the private-case rows of the asset finance register (settlement
columns already apportioned by ``cohort_legacy_share``; use them as stored, see
the G1 data trap), the workbook snapshot for the public real rate, the Ofgem RO
accredited-stations register for the accreditation run-off, and the
``acquisition.valuation`` block of ``config/defaults/base/public_ownership/acquisition.yaml``.

Three streams per row and year, each held flat at its model-year level until
its own expiry, volumes always held (declared).  Prices follow a declared path
(F19 P3 decision 2, 6 Oct 2026): when the rows carry ``market_revenue_path_gbp`` and
``cfd_support_path_gbp`` (the asset settlement re-settled at CP30's CP.32 price level,
``price_level_settlement.py``; lever ``legacy_price_path: cp30_level``), every year after
the model year takes them, so a merchant tail earns CP30's capture price and a CfD's
top-up falls as its reference rises; without them (``model_2030``) every year keeps the
model year's price, the behaviour before 6 Oct 2026.  ROC income has no price in it.

* ``net_operating_cash``: market revenue (wholesale plus the balancing
  settlement) plus Capacity Market revenue (item MR4), less fuel, carbon and
  fixed O&M, while the plant is alive; the CfD curtailment compensation inside
  the balancing settlement only while the contract is also alive; gas plant's balancing
  margin (the settlement less the variable cost of the energy redispatched) follows NESO's
  thermal constraint-cost projection relative to the model year (``balancing_income_path``,
  F27 item 13, user 24 Oct 2026; ``flat`` holds it, the behaviour before);
* ``cfd_top_up``: the difference payments, while the contract and the plant are
  alive; with the other contract income outside ``cfd_support_gbp`` (8 Nov 2026, when P3 came to
  buy the CCS plants): a Power BECCS row's carbon-removal CfD on its CfD's schedule (the dual
  CfD's carbon leg) and a DPA row's payments from commissioning for ``dpa_contract_term_years``;
* ``roc_income``: ROC income scaled by the share of the row's technology still
  accredited on 1 April of each year, from the Ofgem register.

Curtailment compensation is never valued as a stream of its own, and the
``cfd_curtailment_compensation_model_year_gbp_not_valued`` column is never added
to the flows.  When the balancing solve priced CfD bids on contract value
(every CP30 run) the compensation *is* the accepted bid, so it sits inside the
balancing settlement and therefore inside ``market_revenue_gbp``
(``settlement.combine_settlement_revenue``).  It exists only while the CfD
runs, so ``net_operating_cash`` carries it only while the contract is alive and
drops it at expiry (decided 26 Sep 2026; before, it was held until retirement:
115 m a year on the valued group of the 25 Sep 2026 CP30 register).  The column
name records only that no separate stream is built.  Rows are grouped into
``valued`` (renewables and nuclear, the headline), ``thermal`` (the
loss-making dispatchable fleet item N3 treats separately), ``storage`` and
``excluded`` (foreign plant, load shedding); every group is profiled and priced
so N3 can read the thermal figure, but only ``valued`` enters the headline.

Asset data (F19 P3 decision 3, user 6 Oct 2026; non-vintaged model years only, so a 2050
valuation is unchanged):

* **CfD terms.** A contract runs from its start for its round's term
  (``cfd_contract_term_years_by_round``: 15 years for FIDeR and AR1-AR6, 20 from AR7),
  not award plus eighteen.  Starts are the LCCC first settlement (or expected start)
  dates of ``cfd_contract_starts`` per phase, else the commissioning year's middle, else
  the award year plus ``cfd_start_years_after_award_undated``; a contract on part of a
  farm (Seagreen: 454 of 1,075 MW) pays its top-up on that share only.  Per-round end
  overrides stay (HPC 2065; Drax's bridging CfD ends March 2031).  Flows are pro rata
  in the year a contract ends.
* **Named plant.** ``capacity_from_year`` steps a plant's flows to a new capacity
  (Hinkley Point C's unit 2 from 2031); ``new_contracts`` give a plant a CfD from a year
  (Sizewell B, GBP 70.50/MWh in 2025 money from 2035, gov.uk 8 Jul 2026), paying
  strike less its capture price on its output; ``declared_margins`` value a plant's
  contract years at a declared margin (Drax: the midpoint of its published April 2027 to
  March 2031 guidance, GBP 100-200 m a year; F19 valuation decision, user 14 Oct 2026), and
  with ``close_at_end`` the plant closes when that margin ends (Drax at its bridging CfD's end,
  March 2031: no merchant months after it).
* **Gate fees.** ``gate_fee_income`` credits energy from waste with its gate fee: the
  WRAP fee per tonne times the tonnes per MWh its efficiency implies.
* **Closure.** With ``close_when_cash_negative`` an owner retires a plant from the first
  year after the model year in which its total flow is negative (the option every
  owner holds); its retirement year says so.

The owners' value (F29, user 26 Oct 2026: workbook v2.0's Approach 2 on each register row):
``pv_market_gbp`` is the row's after-tax value to its owners (floored at zero by the purchase):
``(1 - tau) x PV(flows) + tau x d / (r + d) x book value``, corporation tax at ``tau`` (25%) on
the flows and the capital allowances still to come on the row's tax written-down value at the
workbook's 18% reducing balance, discounted end of year at the workbook's post-tax operating
rate ``r`` (``Legacy Fleet`` column AF: offshore wind 6.1%, calibrated to Brookfield's December
2024 purchase of 12.45% of Hornsea 1/2, Walney Extension and Burbo Bank Extension, the rest by
the CEPA hurdle spread to offshore).  Hinkley Point C, which the workbook does not value, takes
its new-build nuclear post-tax rate (``owner_rate_named``).  The book value is the workbook's
historic capex by cohort written down over its accounting life (``Legacy A1``), the register's
as-new capex where the workbook has no cohort.  ``pv_market_pre_tax_gbp``,
``pv_no_support_gbp`` (without CfD top-ups, ROC income and the CfD compensation in the
balancing settlement: the workbook's no-support value) and ``net_book_value_gbp`` are written
beside it, the purchase bases the acquisition plan reads.  The method needs the snapshot;
without one (unit fixtures) each row's private WACC values the pre-tax flows.  The state's own
value (``pv_state_gbp``) is pre-tax at the public rate, the model year undiscounted, as before.
The gate fees' tonnes are capped at ``gate_fee_income.max_tonnes_per_year`` (the UK's 16.82 Mt
in 2024), every energy-from-waste row scaled alike.

Usage::

    python scripts/public_ownership/legacy_valuation.py <finance_register.csv> <snapshot.yaml>
        <out_assets.csv> <out_profile.csv> <out_summary.json> [--config config]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.finance import capital_recovery_factor  # noqa: E402
from scripts.public_ownership import capacity_procurement  # noqa: E402
from config.config_loader import read_config


STREAMS = ("net_operating_cash", "cfd_top_up", "roc_income")
#: F29 (user, 26 Oct 2026): how the owners' value is computed.  ``workbook_post_tax`` is workbook
#: v2.0's Approach 2 (after-tax flows, capital allowances, post-tax operating rate, end of year);
#: ``pre_tax`` values the flows at each row's private WACC, the year-zero convention (the
#: fallback without a snapshot).
OWNER_VALUE_WORKBOOK, OWNER_VALUE_PRE_TAX = "workbook_post_tax", "pre_tax"
OWNER_VALUE_METHODS = (OWNER_VALUE_WORKBOOK, OWNER_VALUE_PRE_TAX)
#: Valuation keys retired by F29, and why (a config naming them fails with the pointer).
RETIRED_SETTINGS = {
    "owner_discount_rate_basis": ("retired 26 Oct 2026 (F29): every row is discounted at workbook v2.0's post-tax "
                                  "operating rate (lever legacy_owner_discount_rate retired)"),
    "operating_asset_real_rates": ("retired 26 Oct 2026 (F29): the listed funds' rates are replaced by workbook "
                                   "v2.0's Legacy Fleet column AF"),
}
#: F29: the workbook's legacy model technology (``Legacy Fleet`` column C) of each PyPSA carrier,
#: for the cohort a row's historic capex and accounting life come from (``Legacy A1``).
WORKBOOK_LEGACY_TECHNOLOGY = {
    "wind_offshore": "Offshore wind", "wind_onshore": "Onshore wind", "solar_pv": "Solar PV", "nuclear": "Nuclear",
    "CCGT": "Gas CCGT", "OCGT": "Gas CCGT", "gas_engine": "Gas CCGT", "biomass": "Biomass", "large_hydro": "Hydro",
    "pumped_hydro": "Pumped storage",
}
#: Offshore CfD rounds by the workbook cohort's round words (FIDeR / AR1; AR2; AR3 / AR4 operational by end-2025).
OFFSHORE_COHORT_ROUND = {"Investment Contract": "FIDeR", "MGT_IC": "FIDeR", "FIDeR": "FIDeR", "AR1": "FIDeR",
                         "AR2": "AR2"}
#: F27 item 13 (user, 24 Oct 2026): how gas plant's balancing margin runs after the model year.
BALANCING_PATH_NESO, BALANCING_PATH_FLAT = "neso_constraint_cost", "flat"
BALANCING_PATH_METHODS = (BALANCING_PATH_NESO, BALANCING_PATH_FLAT)
#: The multi-year view's span (the last of ``nationalised_tariff.MULTI_YEAR_OFFSETS``):
#: the profile always reaches the model year plus this (item Y12).
MULTI_YEAR_VIEW_SPAN_YEARS = 30
GROUP_VALUED = "valued"
GROUP_THERMAL = "thermal"
GROUP_STORAGE = "storage"
GROUP_EXCLUDED = "excluded"
GROUPS = (GROUP_VALUED, GROUP_THERMAL, GROUP_STORAGE, GROUP_EXCLUDED)

# Ofgem register technology label -> PyPSA carrier, for the accreditation run-off.
RO_TECHNOLOGY_TO_CARRIER = {
    "onshore wind": "wind_onshore",
    "wind with a dnc of 50kw or less": "wind_onshore",
    "offshore wind": "wind_offshore",
    "offshore wind - floating turbines": "wind_offshore",
    "offshore wind - demonstration turbines": "wind_offshore",
    "photovoltaic": "solar_pv",
    "photovoltaic with a dnc of 50kw or less": "solar_pv",
    "hydro with a dnc of 20mw or less": "large_hydro",
    "hydro with a dnc greater than 20 mw": "large_hydro",
    "hydro with a dnc of 50kw or less": "large_hydro",
    "micro hydro": "large_hydro",
    "tidal stream": "marine",
    "wave power": "marine",
    "fuelled": "biomass",
    "fuelled with a dnc of 50kw or less": "biomass",
    "landfill gas": "biomass",
    "sewage gas": "biomass",
}
RO_SHARE_ALL = "__all__"

DEFAULT_SETTINGS: dict = {
    "horizon_year": 2070,
    # User, 11 Nov 2026: what follows the anchored path. "price": the wholesale price only (10 Nov). "all": every
    # stream per MW by technology (output, wholesale, balancing, fuel and carbon, Capacity Market;
    # price_anchors.STREAMS), so CfD and RO income follow output and price too; every technology the anchors cover.
    "anchor_streams": "price",
    "ro_register": "data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv",
    "ro_closure_date": "2037-03-31",
    "ro_accreditation_years": 20,
    # Vintaged (2050) registers only; a 2030 register dates contracts from their start (F19).
    "cfd_contract_years_from_award": 18,
    # F19 P3 decision 3 (user, 6 Oct 2026): Drax's bridging CfD runs April 2027 to March 2031.
    "cfd_contract_end_year_overrides": {"HPC": 2065, "DRAX_BRIDGE": 2031.25},
    "cfd_contract_starts": "data/public_ownership/financing/legacy_cfd_contract_starts.csv",
    "cfd_contract_term_years_by_round": {
        "Investment Contract": 15, "MGT_IC": 15, "AR1": 15, "AR2": 15, "AR3": 15, "AR4": 15, "AR5": 15, "AR6": 15,
        "DRAX_BRIDGE": 4, "HPC": 35, "POWER_BECCS": 15,
    },
    "cfd_contract_term_years_default": 20,
    # Fix 8 Nov 2026 (user: P3 buys the CCS plants): Net Zero Teesside's dispatchable power agreement,
    # paid from commissioning for this many years.
    "dpa_contract_term_years": 15,
    "cfd_start_years_after_award_undated": 3,
    # Contracts audit (13 Nov 2026): a round's first delivery year floors a commissioning-dated start (AR8's first
    # delivery year starts April 2029; the CP30 cohort commissions from mid-2028).
    "cfd_first_delivery_year_by_round": {"AR8_modelled": 2029.25},
    # Continuity audit (14 Nov 2026): a station holding a Capacity Market agreement runs at least to its end (the
    # agreements in data/market/CP30/gas_station_schedule_2030.csv; GHD for DESNZ 2025 gives gas 25 years, so a 2028
    # 15-year refurbishment agreement keeps a 1997 CCGT to 2043). Empty: no floor.
    "cm_agreement_floor": {"file": "data/market/CP30/gas_station_schedule_2030.csv",
                           "carriers": ["CCGT", "OCGT", "gas_engine"]},
    # Continuity audit (14 Nov 2026): a carrier whose register rows carry no build year of their own takes the fleet's
    # build profile instead of one median year (batteries: 19 GW all dated 2026.5 retired together in 2041, while FES
    # retires them through the 2030s). Rows are given build years in asset-id order so that their MW reproduce the
    # profile's shares. Empty: the median rule.
    "commissioning_profile_by_carrier": {"Battery": "data/public_ownership/financing/battery_build_profile_2030.csv"},
    "capacity_from_year": {"Hinkley Point C": {"from_year": 2031, "capacity_mw": 3260.0}},
    "new_contracts": {"Sizewell B": {"from_year": 2035, "term_years": 20, "strike_gbp_per_mwh": 70.50,
                                     "rebasing_to_model_price_year": 0.9630434782608696}},
    # F19 valuation decision (user, 14 Oct 2026): Drax's published guidance for its bridging CfD
    # (average adjusted EBITDA GBP 100-200 m a year from Drax Power Station, April 2027 to March 2031;
    # Drax plc RNS 10 Feb 2025, unchanged at signing, 5 Nov 2025), the midpoint, for its four 645 MW
    # biomass units; it closes when the contract ends.
    "declared_margins": {"Drax Power Station (biomass units)": {
        "segment_ebitda_gbp": 150.0e6, "segment_capacity_mw": 2580.0, "through_year": 2031.25,
        "rebasing_to_model_price_year": 0.9630434782608696, "close_at_end": True}},
    # F19 (user, 6 Oct 2026): tonnes capped at the UK's actual EfW throughput (Tolvik, 16.82 Mt in 2024).
    "gate_fee_income": {"carrier": "waste_to_energy", "file": "data/public_ownership/financing/efw_gate_fees_2050.json",
                        "case": "central", "plant_efficiency": 0.25, "max_tonnes_per_year": 16.82e6},
    # F29 (user, 26 Oct 2026): workbook v2.0's Approach 2 on each row (module docstring).
    "owner_value_method": OWNER_VALUE_WORKBOOK,
    # Named plant the workbook does not value: the workbook technology whose new-build post-tax
    # WACC (Sc1 Operative column T) discounts it.  Hinkley Point C, user decision 26 Oct 2026.
    "owner_rate_named": {"Hinkley Point C": "Large-scale nuclear"},
    # Carriers the crosswalk gives no workbook technology, for their post-tax operating rate.
    "owner_rate_technology_by_carrier": {"pumped_hydro": "Pumped hydro energy storage (PHES)",
                                         "CHP": "Gas generation - unabated", "oil": "Gas generation - unabated",
                                         "biogas": "Anaerobic digestion (AD)"},
    "close_when_cash_negative": True,
    # F27 item 13 (user, 24 Oct 2026): gas plant's balancing margin after the model year follows
    # NESO's thermal constraint-cost projection relative to the model year (the network case's own
    # series; held at the last year's ratio after it); method flat holds the model year's for life.
    "balancing_income_path": {
        "method": "neso_constraint_cost", "carriers": ["CCGT", "OCGT", "gas_engine"],
        "file": "data/public_ownership/financing/neso_constraint_cost_path.csv",
        "series": "ht_expected_network", "series_by_network_case": {"cp30": "ht_recommended_network"},
    },
    # Decision 48 (user, 3 Oct 2026): solar 38 and offshore 30 (the DESNZ lives the register
    # recovers new build over), onshore 25 (FES E.02 design life); config/defaults/base/public_ownership/acquisition.yaml.
    "retirement_life_years_by_carrier": {
        "wind_onshore": 25, "wind_offshore": 30, "solar_pv": 38, "large_hydro": 60, "marine": 25,
        # Continuity audit (14 Nov 2026): gas 25 (GHD for DESNZ 2025; FES 2025 E.02), kept to any CM agreement end.
        "biomass": 25, "CCGT": 25, "OCGT": 25, "gas_engine": 25, "waste_to_energy": 35, "oil": 25, "CHP": 25,
        "Battery": 15, "battery": 15, "nuclear": 60, "CCS Biomass": 25, "CCS Gas": 25,
    },
    # Sizewell B to 2055 (gov.uk, 8 Jul 2026: lifetime extension with a CfD from 2035).
    # D3 (user, 14 Nov 2026): Sizewell B to FES's 2036 closure; EDF's 2055 is the extended-lives sensitivity.
    "retirement_year_overrides": {"Hinkley Point C": 2065, "Sizewell B": 2036},
    # FES 2050 decision 62 (user, 3 Oct 2026): the repowering rule of the stock-flow
    # (fleet_vintages.RepoweringRule), by carrier. At the carrier's life a share of each row
    # leaves (repowered as new build, outside the legacy fleet); the rest runs to the cap.
    # Empty (the default, and every 2030 config): every row retires at its life, as before.
    "repowering_share_by_carrier": {},
    "max_life_years_by_carrier": {},
    "undated_commissioning_method": "carrier_median_dated",
    # Remaining years for an undated row whose carrier has no dated peer to
    # borrow a commissioning year from: hydro is long-lived, the rest are not.  Pumped hydro
    # runs to the horizon (decision 48: refurbished civil assets FES keeps past 2050).
    "undated_remaining_years_by_carrier": {"large_hydro": 30, "CCGT": 15, "waste_to_energy": 15, "pumped_hydro": 40,
                                           "CCS Biomass": 25},
    "undated_remaining_years_default": 10,
    "valued_carriers": ["wind_onshore", "wind_offshore", "solar_pv", "large_hydro", "marine", "nuclear"],
    "thermal_carriers": ["CCGT", "OCGT", "gas_engine", "biomass", "waste_to_energy", "oil", "CHP", "coal",
                         "CCS Biomass", "CCS Gas"],
    "excluded_carriers": ["EU_market", "EU_import", "load_shedding", "load shedding", "voll", "unserved_energy"],
    # User decision (14 Nov 2026): named plant in the acquisition year's fleet that is not operating at the purchase
    # (CP30's 2030 pumped-storage projects without a final investment decision): never bought, so private in every
    # case and paid as today. Matched on the asset id in a register of the table's acquisition year only.
    "not_operating_at_acquisition": {"file": "data/public_ownership/financing/not_operating_at_acquisition.csv"},
    # User decision (10 Nov 2026): the anchored price path's settings (read by the rules and price_anchors.py; the
    # table itself reaches the valuation as settings["price_anchors"] at run time, a DataFrame).
    "price_anchors": {},
    "reference_owner_rates": [0.067, 0.083],
}

REQUIRED_REGISTER_COLUMNS = {
    "case_id", "asset_id", "component_type", "carrier", "finance_vintage", "support_type", "cfd_round",
    "cfd_award_year", "capacity_mw", "commissioning_year", "market_revenue_gbp", "cfd_support_gbp",
    "cfd_curtailment_compensation_gbp", "roc_support_gbp", "fuel_carbon_gbp", "annual_fixed_om_gbp", "wacc",
    "model_year",
}


def settings_from_config(block: Mapping | None, model_year: int | None = None) -> dict:
    """Merge ``public_ownership.acquisition.valuation`` over the defaults, refusing unknown keys.

    ``model_year`` is the run's year, which the horizon must follow (item Y4: the
    default year when not given; :func:`value_legacy_fleet` checks the register's own).
    """
    block = dict(block or {})
    retired = sorted(set(block) & set(RETIRED_SETTINGS))
    if retired:
        raise ValueError("acquisition.valuation: " + "; ".join(f"{k} {RETIRED_SETTINGS[k]}" for k in retired))
    unknown = sorted(set(block) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(f"acquisition.valuation has unknown keys {unknown}; supported: {sorted(DEFAULT_SETTINGS)}")
    merged = dict(DEFAULT_SETTINGS)
    for key, value in block.items():
        if isinstance(DEFAULT_SETTINGS[key], dict):
            # None switches a mapping off (no named plant, no gate fees); a mapping merges.
            merged[key] = {} if value is None else {**DEFAULT_SETTINGS[key], **dict(value)}
        else:
            merged[key] = value
    from scripts.public_ownership.model_year import DEFAULT_MODEL_YEAR

    # User, 11 Nov 2026: horizon_year "life" runs every plant's cash flows to its own retirement.
    if merged["horizon_year"] != HORIZON_LIFE and \
            int(merged["horizon_year"]) <= int(DEFAULT_MODEL_YEAR if model_year is None else model_year):
        raise ValueError("acquisition.valuation.horizon_year must be after the model year")
    if merged["anchor_streams"] not in ANCHOR_STREAMS:
        raise ValueError(f"acquisition.valuation.anchor_streams={merged['anchor_streams']!r}; supported: {list(ANCHOR_STREAMS)}")
    if int(merged["ro_accreditation_years"]) <= 0 or int(merged["cfd_contract_years_from_award"]) <= 0:
        raise ValueError("acquisition.valuation accreditation and contract lengths must be positive")
    for rate in merged["reference_owner_rates"]:
        if not 0.0 < float(rate) < 1.0:
            raise ValueError("acquisition.valuation.reference_owner_rates must be rates in (0, 1)")
    terms = [merged["cfd_contract_term_years_default"], *merged["cfd_contract_term_years_by_round"].values()]
    if any(float(t) <= 0 for t in terms):
        raise ValueError("acquisition.valuation CfD contract terms must be positive")
    for name, step in merged["capacity_from_year"].items():
        if float(step["capacity_mw"]) <= 0:
            raise ValueError(f"acquisition.valuation.capacity_from_year[{name!r}] needs a positive capacity_mw")
    for name, contract in merged["new_contracts"].items():
        if float(contract["term_years"]) <= 0 or float(contract["strike_gbp_per_mwh"]) < 0:
            raise ValueError(f"acquisition.valuation.new_contracts[{name!r}] needs a positive term and strike")
    gate = merged["gate_fee_income"]
    if gate and not 0.0 < float(gate.get("plant_efficiency", 0.0)) < 1.0:
        raise ValueError("acquisition.valuation.gate_fee_income.plant_efficiency must be in (0, 1)")
    if gate and gate.get("max_tonnes_per_year") is not None and not float(gate["max_tonnes_per_year"]) > 0.0:
        raise ValueError("acquisition.valuation.gate_fee_income.max_tonnes_per_year must be positive")
    if merged["owner_value_method"] not in OWNER_VALUE_METHODS:
        raise ValueError(f"acquisition.valuation.owner_value_method={merged['owner_value_method']!r}; "
                         f"supported: {list(OWNER_VALUE_METHODS)}")
    balancing = merged["balancing_income_path"]
    if balancing and balancing.get("method") not in BALANCING_PATH_METHODS:
        raise ValueError(f"acquisition.valuation.balancing_income_path.method={balancing.get('method')!r}; "
                         f"supported: {list(BALANCING_PATH_METHODS)}")
    return merged


#: FES 2050 decision 62: the valuation carriers whose repowering rule follows each stock-flow technology's.
REPOWERING_CARRIERS = {"Onshore wind": ("wind_onshore",), "Offshore wind": ("wind_offshore",), "Solar PV": ("solar_pv",)}


def group_repowering_settings(settings: Mapping, public_ownership_block: Mapping | None, run_group: str | None) -> dict:
    """A run group's own repowering rule in the valuation (decision 62's named sensitivity, 3 Oct 2026).

    The study's valuation declares the repowering rule once (``repowering_share_by_carrier``,
    ``max_life_years_by_carrier``); a register bound that sets another rule in its
    ``fleet_vintages.repowering`` (``fes_ht_2050_repowering_bound``: half of each onshore cohort)
    values its legacy rows on that rule, mapped to the valuation's carriers
    (:data:`REPOWERING_CARRIERS`). Unchanged when the group sets none.  F28 (d): the central
    valuation declares no rule (no private life extension), so a life-extension toggle group's
    rule applies on its own; its cap comes from the rule (``max_life_years``).
    """
    group = ((public_ownership_block or {}).get("reference_run_groups") or {}).get(run_group) or {}
    rule = (group.get("fleet_vintages") or {}).get("repowering")
    if not rule:
        return dict(settings)
    from scripts.public_ownership.fleet_vintages import parse_repowering

    shares = dict(settings.get("repowering_share_by_carrier") or {})
    caps = dict(settings.get("max_life_years_by_carrier") or {})
    for technology, spec in parse_repowering(rule).items():
        if technology not in REPOWERING_CARRIERS:
            raise ValueError(f"{run_group}: no valuation carrier follows the repowering rule of {technology!r}")
        for carrier in REPOWERING_CARRIERS[technology]:
            shares[carrier], caps[carrier] = float(spec.share), float(spec.max_life_years)
    return {**settings, "repowering_share_by_carrier": shares, "max_life_years_by_carrier": caps}


def group_balancing_path_settings(settings: Mapping, public_ownership_block: Mapping | None,
                                  run_group: str | None) -> dict:
    """The NESO constraint-cost series of a run group's network case (F27 item 13, 24 Oct 2026).

    The path is a ratio to the model year, so it must be the series of the network the group's
    model year was solved on: ``balancing_income_path.series_by_network_case`` maps a group's
    ``network_case`` (``cp30``: NESO's recommended network) to its series; any other group keeps
    ``series`` (the expected network, the CENTRAL ``no_acceleration`` case's). Unchanged when the
    path is off or the group names no mapped network case.
    """
    # A raw config block without the key takes the default path (as settings_from_config will).
    path = dict((settings["balancing_income_path"] if "balancing_income_path" in settings
                 else DEFAULT_SETTINGS["balancing_income_path"]) or {})
    group =((public_ownership_block or {}).get("reference_run_groups") or {}).get(run_group) or {}
    series = (path.get("series_by_network_case") or {}).get(group.get("network_case"))
    if not path or series is None or series == path.get("series"):
        return dict(settings)
    return {**settings, "balancing_income_path": {**path, "series": str(series)}}


def _run_group_of(path) -> str | None:
    """``{run_group}_asset_finance_register.parquet`` (or ``.csv``) -> ``run_group``."""
    name = Path(path).name
    for suffix in ("_asset_finance_register.parquet", "_asset_finance_register.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


# ── the Renewables Obligation run-off ─────────────────────────────────────────

def ro_accreditation_shares(
    stations: pd.DataFrame, model_year: int, horizon_year: int, *,
    accreditation_years: int = 20, closure_date: str = "2037-03-31",
) -> pd.DataFrame:
    """Share of each carrier's accredited DNC still accredited on 1 April of each year.

    ``stations`` is the Ofgem accredited-stations register (``Technology``,
    ``Station DNC in kW``, ``Effective date`` d/m/Y).  A station is accredited
    from its effective date for ``accreditation_years`` or until the scheme
    closes, whichever is first.  Shares are relative to the DNC accredited on
    1 April of ``model_year``, per carrier, with an all-technology column
    ``__all__`` as the fallback for carriers the register does not carry.
    """
    frame = pd.DataFrame({
        "carrier": stations["Technology"].astype(str).str.strip().str.lower().map(RO_TECHNOLOGY_TO_CARRIER),
        "dnc_kw": pd.to_numeric(stations["Station DNC in kW"].astype(str).str.replace(",", ""), errors="coerce"),
        "effective": pd.to_datetime(stations["Effective date"], dayfirst=True, errors="coerce"),
    }).dropna(subset=["dnc_kw", "effective"])
    if frame.empty:
        raise ValueError("RO register has no usable stations")
    close = pd.Timestamp(closure_date)
    end = frame["effective"] + pd.DateOffset(years=int(accreditation_years))
    frame["end"] = end.where(end < close, close)
    years = list(range(int(model_year), int(horizon_year) + 1))
    carriers = sorted(frame["carrier"].dropna().unique()) + [RO_SHARE_ALL]
    accredited = pd.DataFrame(0.0, index=years, columns=carriers)
    for year in years:
        on = pd.Timestamp(f"{year}-04-01")
        live = frame.loc[(frame["effective"] <= on) & (frame["end"] > on)]
        by_carrier = live.groupby("carrier")["dnc_kw"].sum()
        accredited.loc[year, by_carrier.index] = by_carrier.to_numpy()
        accredited.loc[year, RO_SHARE_ALL] = float(live["dnc_kw"].sum())
    base = accredited.loc[int(model_year)]
    shares = accredited.div(base.where(base > 0, np.nan), axis=1).fillna(0.0)
    shares.attrs["accredited_dnc_mw_model_year"] = float(base[RO_SHARE_ALL]) / 1_000.0
    return shares


# ── expiry rules ──────────────────────────────────────────────────────────────

def _base_name(asset_id: str) -> str:
    return str(asset_id).split("::", 1)[0]


def profiled_commissioning(rows: pd.DataFrame, commissioning: pd.Series, profiles: Mapping[str, Path]) -> pd.Series:
    """Build years for a profiled carrier's rows: in asset-id order, MW-weighted, on the profile's cumulative shares.

    A profile describes one year's stock (``battery_build_profile_2030.csv``: the fleet built by 2030), so it re-dates
    rows only in a register whose model year is within its span; a later year's (vintaged) register keeps each
    cohort's own build year.
    """
    out = commissioning.copy()
    model_year = (float(pd.to_numeric(rows["model_year"], errors="coerce").iloc[0])
                  if "model_year" in rows.columns and len(rows) else np.nan)
    for carrier, path in profiles.items():
        mask = rows["carrier"].astype(str).eq(carrier).to_numpy()
        if not mask.any():
            continue
        profile = pd.read_csv(path).groupby("build_year")["mw"].sum().sort_index()
        if np.isfinite(model_year) and model_year > float(profile.index.max()):
            continue
        edges = profile.cumsum() / profile.sum()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(rows["asset_id"].astype(str).to_numpy()[idx], kind="stable")]
        mw = pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0).to_numpy()[order]
        mid = (np.cumsum(mw) - 0.5 * mw) / max(float(mw.sum()), 1e-9)
        years = edges.index.to_numpy()[np.searchsorted(edges.to_numpy(), np.clip(mid, 0.0, 1.0 - 1e-12), side="right")]
        out.iloc[order] = years.astype(float)
    return out


def capacity_market_agreement_ends(path) -> dict[str, float]:
    """``normalised station name -> year its last Capacity Market agreement ends`` (continuity audit, 14 Nov 2026).

    Reads ``cm_agreements`` ("SHB228 565.5 MW (2028-29 T-4, 15 y); ...") from the gas station schedule: an agreement
    for delivery year ``YYYY-YY`` of ``N`` years ends ``YYYY + N`` (its last delivery year ends in that autumn).
    """
    frame = pd.read_csv(_resolve(path) if isinstance(path, str) else path)
    out: dict[str, float] = {}
    for name, text in zip(frame["station_name"], frame.get("cm_agreements", pd.Series(dtype=object))):
        if not isinstance(text, str) or not text.strip():
            continue
        ends = [int(a) + int(n) for a, n in re.findall(r"\((\d{4})-\d{2}\s+T-\d,\s*(\d+)\s*y\)", text)]
        if ends:
            out[normalised_name(name)] = float(max(ends))
    return out


def _contract_base_name(asset_id: str) -> str:
    """The asset name a contract row matches LCCC phases on: the cohort and a "_CfD_part_<round>" suffix dropped."""
    return re.sub(r"_+CfD_part_[A-Za-z0-9]+$", "", _base_name(asset_id))


def _round_code(allocation_round: str) -> str:
    """LCCC's "Allocation Round 4" as the register's "AR4" (other names unchanged)."""
    m = re.fullmatch(r"Allocation Round (\w+)", str(allocation_round).strip())
    return f"AR{m.group(1)}" if m else str(allocation_round).strip()


def retirement_years(rows: pd.DataFrame, settings: Mapping) -> tuple[pd.Series, pd.Series]:
    """Retirement year and the method that produced it, per row.

    Dated rows retire at commissioning plus the carrier's life; named overrides
    win; undated rows take the carrier's median dated commissioning year when
    the carrier has any, otherwise the model year plus the carrier's entry in
    ``undated_remaining_years_by_carrier`` (or ``undated_remaining_years_default``).
    """
    model_year = int(rows["model_year"].iloc[0])
    lives = settings["retirement_life_years_by_carrier"]
    overrides = settings["retirement_year_overrides"]
    profiles = {str(c): _resolve(p) for c, p in (settings.get("commissioning_profile_by_carrier") or {}).items()}
    if profiles:
        commissioning_in = pd.to_numeric(rows["commissioning_year"], errors="coerce")
        commissioning_in = profiled_commissioning(rows, commissioning_in, profiles)
    floor_cfg = settings.get("cm_agreement_floor") or {}
    cm_carriers = set(map(str, floor_cfg.get("carriers") or ()))
    cm_ends = capacity_market_agreement_ends(floor_cfg["file"]) if floor_cfg.get("file") else {}
    commissioning = (commissioning_in if profiles else pd.to_numeric(rows["commissioning_year"], errors="coerce"))
    median_by_carrier = commissioning.groupby(rows["carrier"]).median()
    years = pd.Series(np.nan, index=rows.index, dtype=float)
    method = pd.Series("", index=rows.index, dtype=object)
    for idx, row in rows.iterrows():
        name = _base_name(row["asset_id"])
        carrier = str(row["carrier"])
        life = float(lives.get(carrier, lives.get(carrier.lower(), 25)))
        if name in overrides:
            years[idx], method[idx] = float(overrides[name]), "named_override"
        elif pd.notna(commissioning[idx]):
            years[idx], method[idx] = float(commissioning[idx]) + life, "commissioning_plus_life"
            floor = cm_ends.get(normalised_name(name)) if carrier in cm_carriers else None
            if floor is not None and floor > years[idx]:
                years[idx], method[idx] = float(floor), "capacity_market_agreement_end"
        elif settings["undated_commissioning_method"] == "carrier_median_dated" and pd.notna(median_by_carrier.get(carrier)):
            years[idx], method[idx] = float(median_by_carrier[carrier]) + life, "carrier_median_commissioning_plus_life"
        else:
            remaining = settings["undated_remaining_years_by_carrier"].get(
                carrier, settings["undated_remaining_years_default"])
            years[idx] = model_year + float(remaining)
            method[idx] = "undated_default_remaining_years"
    # Nothing retires before the model year it is observed generating in.
    years = years.clip(lower=model_year + 1)
    return years, method


def cfd_contract_end_years(rows: pd.DataFrame, settings: Mapping) -> pd.Series:
    """Each CfD row's contract end (vintaged registers only), per-round overrides last.

    Contracts audit (13 Nov 2026): a row labelled with a known contract phase (``cfd_contract_id``
    ``<contract>#<phase>``) ends at that phase's expiry (``cfd_strikes.known_contracts``, LCCC's dates); a row on a
    modelled round (``PAR7_<year>``, ``AR8_modelled``) at its build year plus its technology's modelled term; any
    other at the award year plus ``cfd_contract_years_from_award`` (the old rule for every row).
    """
    award = pd.to_numeric(rows["cfd_award_year"], errors="coerce")
    end = award + float(settings["cfd_contract_years_from_award"])
    if "cfd_contract_id" in rows.columns and "model_year" in rows.columns and len(rows):
        from scripts.market.cfd_strikes import known_contracts

        known = known_contracts(int(rows["model_year"].iloc[0]))
        expiry = {f"{c}#{int(p)}": float(e) for c, p, e in zip(known["contract_id"], known["phase"], known["expiry_year"])}
        ids = rows["cfd_contract_id"].fillna("").astype(str)
        hit = ids.map(expiry)
        end = end.where(hit.isna(), hit)
    if {"build_year", "stock_flow_technology"} <= set(rows.columns):
        from scripts.public_ownership.fleet_vintages import modelled_contract_term

        rounds = rows["cfd_round"].fillna("").astype(str)
        modelled = rounds.str.startswith("PAR7_") | rounds.eq("AR8_modelled")
        built = pd.to_numeric(rows["build_year"], errors="coerce")
        for i in np.flatnonzero(modelled.to_numpy() & built.notna().to_numpy()):
            term = modelled_contract_term(str(rows["stock_flow_technology"].iloc[i]), int(built.iloc[i]))
            end.iloc[i] = float(built.iloc[i]) + float(term)
    for round_name, year in settings["cfd_contract_end_year_overrides"].items():
        end[rows["cfd_round"].astype(str).eq(str(round_name))] = float(year)
    is_cfd = rows["support_type"].astype(str).str.strip().str.lower().eq("cfd")
    return end.where(is_cfd, np.nan)


def normalised_name(value: object) -> str:
    """Lower-case letters and digits only: register and LCCC names differ in spacing and stray bytes."""
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower())


def decimal_year(value) -> float:
    """A date as a fractional year (1 April 2027 is 2027.25)."""
    stamp = pd.Timestamp(value)
    days = 366 if stamp.is_leap_year else 365
    return stamp.year + (stamp.dayofyear - 1) / days


def read_contract_starts(source) -> pd.DataFrame:
    """The LCCC contract phases per register asset (``legacy_cfd_contract_starts.csv``), with a decimal start."""
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    required = {"asset_match", "contract_start", "contract_capacity_mw"}
    if missing := required - set(frame.columns):
        raise ValueError(f"CfD contract starts lack {sorted(missing)}")
    frame["key"] = frame["asset_match"].map(normalised_name)
    frame["start_year"] = frame["contract_start"].map(decimal_year)
    frame["contract_capacity_mw"] = pd.to_numeric(frame["contract_capacity_mw"], errors="raise")
    return frame


def _overlap(start: float, end: float, years: np.ndarray) -> np.ndarray:
    """Share of each calendar year inside ``[start, end)``."""
    return np.clip(np.minimum(end, years + 1.0) - np.maximum(start, years), 0.0, 1.0)


def cfd_contract_schedule(rows: pd.DataFrame, settings: Mapping, years: np.ndarray,
                          starts: pd.DataFrame | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Each CfD row's live share of each year, and its contract dates (F19 P3 decision 3).

    A contract runs from its start for its round's term.  Starts come from the LCCC
    phases in ``starts`` (matched on the normalised asset name; each phase weighted
    by its contracted MW), else the middle of the commissioning year, else the award
    year plus ``cfd_start_years_after_award_undated``; per-round end overrides replace
    the end.  The share of the row under contract (contracted MW over the asset's MW,
    at most 1) multiplies the result, so a partly contracted farm is paid its
    top-up on that share only.  The model year is settled as solved: its coverage is
    the contracted share whenever the row holds a CfD.

    Returns ``(coverage, table)``: coverage is rows x years in [0, 1] including the
    share; ``table`` has ``cfd_contract_start_year``, ``cfd_contract_end_year`` (the
    last phase's end), ``cfd_contract_term_years``, ``cfd_contracted_share`` and
    ``cfd_contract_start_basis``.
    """
    n = len(rows)
    coverage = np.zeros((n, len(years)))
    table = pd.DataFrame({
        "cfd_contract_start_year": np.nan, "cfd_contract_end_year": np.nan, "cfd_contract_term_years": np.nan,
        "cfd_contracted_share": np.nan, "cfd_contract_start_basis": "",
    }, index=rows.index)
    is_cfd = rows["support_type"].astype(str).str.strip().str.lower().eq("cfd").to_numpy()
    if not is_cfd.any():
        return coverage, table
    phases = starts.groupby("key") if starts is not None and len(starts) else None
    keys = set(starts["key"]) if phases is not None else set()
    by_round = {str(k): float(v) for k, v in settings["cfd_contract_term_years_by_round"].items()}
    overrides = {str(k): float(v) for k, v in settings["cfd_contract_end_year_overrides"].items()}
    award = pd.to_numeric(rows["cfd_award_year"], errors="coerce").to_numpy(dtype=float)
    built = pd.to_numeric(rows["commissioning_year"], errors="coerce").to_numpy(dtype=float)
    capacity = pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cohort = (pd.to_numeric(rows["cohort_legacy_share"], errors="coerce").to_numpy(dtype=float)
              if "cohort_legacy_share" in rows.columns else np.full(n, np.nan))
    for i, (asset_id, round_name) in enumerate(zip(rows["asset_id"].astype(str), rows["cfd_round"].astype(str))):
        if not is_cfd[i]:
            continue
        term = by_round.get(round_name, float(settings["cfd_contract_term_years_default"]))
        key = normalised_name(_contract_base_name(asset_id))
        group = phases.get_group(key) if key in keys else None
        if group is not None and "allocation_round" in group.columns:
            # Contracts audit (13 Nov 2026): an asset with contracts in two rounds (Viking: AR4 and AR5) has a row
            # per round; each takes its own round's phases.
            own = group[group["allocation_round"].map(_round_code).eq(round_name)]
            group = own if len(own) else group
        if group is not None:
            contracted = float(group["contract_capacity_mw"].sum())
            # A cohort row (asset::legacy) holds its share of the asset's MW.
            asset_mw = capacity[i] / cohort[i] if np.isfinite(cohort[i]) and cohort[i] > 0 else capacity[i]
            share = min(1.0, contracted / asset_mw) if asset_mw > 0 else 1.0
            pieces = list(zip(group["start_year"].astype(float), group["contract_capacity_mw"] / contracted))
            basis = "lccc_contract_phases"
        else:
            share = 1.0
            if np.isfinite(built[i]):
                start, basis = built[i] + 0.5, "commissioning_mid_year"
                first = settings.get("cfd_first_delivery_year_by_round", {}).get(round_name)
                if first is not None and start < float(first):
                    start, basis = float(first), "round_first_delivery_year"
            elif np.isfinite(award[i]):
                start, basis = award[i] + float(settings["cfd_start_years_after_award_undated"]), "award_plus_years"
            else:
                start, basis = float(years[0]), "model_year"
            pieces = [(start, 1.0)]
        ends = [start + term for start, _ in pieces]
        if round_name in overrides:
            ends = [overrides[round_name]] * len(pieces)
            basis += "+end_override"
        live = sum(weight * _overlap(start, end, years.astype(float)) for (start, weight), end in zip(pieces, ends))
        live[0] = 1.0
        coverage[i] = share * live
        table.iloc[i] = [min(s for s, _ in pieces), max(ends), term, share, basis]
    return coverage, table


def dpa_contract_schedule(rows: pd.DataFrame, settings: Mapping, years: np.ndarray) -> np.ndarray:
    """Each row's live share of each year under a dispatchable power agreement (rows x years).

    A DPA runs from the middle of the commissioning year (the model year when undated) for
    ``dpa_contract_term_years``; the model year is settled as solved. Rows without DPA
    support are zero.
    """
    live = np.zeros((len(rows), len(years)))
    if "dpa_support_gbp" not in rows.columns:
        return live
    paid = pd.to_numeric(rows["dpa_support_gbp"], errors="coerce").fillna(0.0).to_numpy(dtype=float) != 0.0
    built = pd.to_numeric(rows["commissioning_year"], errors="coerce").to_numpy(dtype=float)
    term = float(settings["dpa_contract_term_years"])
    for i in np.flatnonzero(paid):
        start = built[i] + 0.5 if np.isfinite(built[i]) else float(years[0])
        live[i] = _overlap(start, start + term, years.astype(float))
        live[i, 0] = 1.0
    return live


def _snapshot_technologies(snapshot: Mapping) -> dict[str, Mapping]:
    return {normalised_name(t["technology"]): t for t in (snapshot or {}).get("technologies", [])}


def workbook_owner_rates(rows: pd.DataFrame, snapshot: Mapping, settings: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Each row's post-tax owners' rate and its source (F29, workbook v2.0), NaN where none applies.

    A row takes its finance technology's ``legacy_owner_rate`` (``Legacy Fleet`` column AF: the
    calibrated offshore rate less the Sc1 hurdle spread); a carrier without one takes
    ``owner_rate_technology_by_carrier``'s; a plant in ``owner_rate_named`` takes the named
    technology's new-build post-tax WACC (``Sc1 Operative`` column T: Hinkley Point C, 6.30%).
    """
    technologies = _snapshot_technologies(snapshot)
    named = {str(k): str(v) for k, v in (settings.get("owner_rate_named") or {}).items()}
    by_carrier = {str(k): str(v) for k, v in (settings.get("owner_rate_technology_by_carrier") or {}).items()}
    finance = (rows["finance_technology"] if "finance_technology" in rows.columns
               else pd.Series(None, index=rows.index, dtype=object))
    rate = np.full(len(rows), np.nan)
    source = np.full(len(rows), "", dtype=object)
    for i, (asset, carrier, technology) in enumerate(zip(rows["asset_id"], rows["carrier"].astype(str), finance)):
        name = _base_name(asset)
        if name in named and normalised_name(named[name]) in technologies:
            rate[i] = float(technologies[normalised_name(named[name])]["post_tax_wacc"])
            source[i] = f"workbook_new_build_post_tax:{named[name]}"
            continue
        label = technology if isinstance(technology, str) and technology.strip() else by_carrier.get(carrier)
        row = technologies.get(normalised_name(label)) if label else None
        if row is not None and row.get("legacy_owner_rate") is not None:
            rate[i] = float(row["legacy_owner_rate"])
            source[i] = f"workbook_operating_post_tax:{row['technology']}"
    return rate, source


def _workbook_cohort(cohorts: Sequence[Mapping], carrier: str, support: str, cfd_round: str, name: str):
    """The workbook legacy cohort a register row belongs to (``Legacy Fleet`` rows 24-38), or None."""
    technology = WORKBOOK_LEGACY_TECHNOLOGY.get(carrier)
    if technology is None:
        return None
    candidates = [c for c in cohorts if str(c.get("technology")) == technology]
    support = support.strip().lower()

    def regime(value):
        return [c for c in candidates if str(c.get("support_regime", "")).strip().lower() == value]

    if technology == "Offshore wind":
        if support == "roc":
            found = regime("ro")
        elif support == "cfd":
            words = OFFSHORE_COHORT_ROUND.get(cfd_round, "AR3")
            found = [c for c in regime("cfd") if words in str(c.get("cohort", ""))]
        else:
            found = regime("merchant")
    elif technology in ("Onshore wind", "Solar PV"):
        found = regime("ro") if support == "roc" else regime("merchant")
    elif technology == "Nuclear":
        found = [c for c in candidates if ("Sizewell" in str(c.get("cohort", ""))) == name.startswith("Sizewell")]
        if not name.startswith("Sizewell"):
            found = [c for c in found if "AGR" in str(c.get("cohort", ""))]
    else:
        found = candidates
    return found[0] if len(found) == 1 else None


def workbook_book_values(rows: pd.DataFrame, snapshot: Mapping, settings: Mapping,
                         expiry: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Each row's net book value for the capital allowances, and its source (F29, workbook v2.0 ``Legacy A1``).

    ``historic capex x capacity x max(0, 1 - age / accounting life)`` with the cohort's capex and
    life; a row the workbook has no cohort for (energy from waste, batteries, marine, a named
    plant such as Hinkley Point C) takes the register's as-new capex (``as_new_capex_gbp``) written
    down over its retirement life (``retirement_life_years_by_carrier``).  The age is the model year
    less the commissioning year, which for an undated row is the valuation's own (its retirement year
    before any closure, less that same life), so an undated row's book value is its remaining share
    of the life the valuation gives it.
    """
    cohorts = list((snapshot or {}).get("legacy_cohorts") or [])
    model_year = float(rows["model_year"].iloc[0])
    lives = settings["retirement_life_years_by_carrier"]
    named = set((settings.get("owner_rate_named") or {}))
    retire = expiry["retirement_year_before_closure"] if "retirement_year_before_closure" in expiry.columns \
        else expiry["retirement_year"]
    commissioning = pd.to_numeric(rows["commissioning_year"], errors="coerce")

    def column(name, default=np.nan):
        return (pd.to_numeric(rows[name], errors="coerce") if name in rows.columns
                else pd.Series(default, index=rows.index)).to_numpy(dtype=float)

    as_new = column("as_new_capex_gbp")
    support = rows["support_type"].fillna("").astype(str) if "support_type" in rows.columns else pd.Series("", index=rows.index)
    rounds = rows["cfd_round"].fillna("").astype(str) if "cfd_round" in rows.columns else pd.Series("", index=rows.index)
    value = np.zeros(len(rows))
    source = np.full(len(rows), "none", dtype=object)
    for i, (idx, asset, carrier) in enumerate(zip(rows.index, rows["asset_id"], rows["carrier"].astype(str))):
        life_retire = float(lives.get(carrier, lives.get(carrier.lower(), 25)))
        cod = commissioning[idx] if pd.notna(commissioning[idx]) else float(retire[idx]) - life_retire
        age = max(0.0, model_year - float(cod))
        name = _base_name(asset)
        cohort = None if name in named else _workbook_cohort(cohorts, carrier, support[idx], rounds[idx], name)
        capacity_kw = float(rows.at[idx, "capacity_mw"]) * 1_000.0
        if cohort is not None:
            accounting = float(cohort["accounting_life_years"])
            value[i] = capacity_kw * float(cohort["historic_capex_gbp_per_kw"]) * max(0.0, 1.0 - age / accounting)
            source[i] = f"workbook_cohort:{cohort['cohort']}"
        elif np.isfinite(as_new[i]):
            # Straight line to the retirement the valuation gives it: the same life that dates an undated row.
            value[i] = as_new[i] * max(0.0, 1.0 - age / life_retire)
            source[i] = "register_as_new_capex"
    return value, source


def is_vintaged_model_year_of(model_year: int) -> bool:
    from scripts.public_ownership.model_year import is_vintaged_model_year

    return is_vintaged_model_year(int(model_year))


NOT_OPERATING_COLUMNS = ("asset_match", "acquisition_year", "model_carrier", "capacity_mw", "reason", "source")


def not_operating_table(settings: Mapping) -> pd.DataFrame:
    """The declared plant not operating at the purchase (``not_operating_at_acquisition.file``), or an empty table."""
    spec = settings.get("not_operating_at_acquisition") or {}
    if not spec.get("file"):
        return pd.DataFrame(columns=list(NOT_OPERATING_COLUMNS))
    table = pd.read_csv(_resolve(spec["file"]))
    missing = set(NOT_OPERATING_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{spec['file']} lacks columns {sorted(missing)}")
    if table[["reason", "source"]].isna().any().any():
        raise ValueError(f"{spec['file']}: every row needs a reason and a source")
    return table


def not_operating_mask(rows: pd.DataFrame, settings: Mapping, *, check: bool = False) -> np.ndarray:
    """Rows of a register in a table row's acquisition year whose asset id contains its ``asset_match``.

    ``check``: in a register of that year holding the row's model carrier, refuse a table row that matches no row of
    it, or matches capacity other than its declared MW (within 1 MW), so a renamed or resized asset cannot silently
    turn the exclusion off (a register without the carrier, such as a test fleet, is not checked).
    """
    table = not_operating_table(settings)
    out = np.zeros(len(rows), dtype=bool)
    if table.empty or "asset_id" not in rows.columns or "model_year" not in rows.columns or rows.empty:
        return out
    year = pd.to_numeric(rows["model_year"], errors="coerce").to_numpy()
    ids = rows["asset_id"].astype(str)
    for r in table.itertuples(index=False):
        hit = (year == float(r.acquisition_year)) & ids.str.contains(str(r.asset_match), regex=False).to_numpy()
        in_year = year == float(r.acquisition_year)
        if check and (in_year & rows["carrier"].astype(str).eq(str(r.model_carrier)).to_numpy()).any():
            carriers = set(rows.loc[hit, "carrier"].astype(str))
            mw = float(pd.to_numeric(rows.loc[hit, "capacity_mw"], errors="coerce").sum())
            # A project split by the legislation year (the 2025-law bound) shows only its legacy part here, so the
            # check refuses no match, another carrier or more MW than declared, not a smaller part.
            if not hit.any() or carriers != {str(r.model_carrier)} or mw > float(r.capacity_mw) + 1.0:
                raise ValueError(f"not operating at acquisition: {r.asset_match!r} matches {mw:.1f} MW of {sorted(carriers)}"
                                 f", declared {float(r.capacity_mw):.1f} MW of {r.model_carrier}")
        out |= hit
    return out


def scope_group(rows: pd.DataFrame, settings: Mapping) -> pd.Series:
    carrier = rows["carrier"].astype(str)
    group = pd.Series(GROUP_EXCLUDED, index=rows.index, dtype=object)
    group[rows["component_type"].astype(str).eq("StorageUnit")] = GROUP_STORAGE
    group[carrier.isin(settings["thermal_carriers"])] = GROUP_THERMAL
    group[carrier.isin(settings["valued_carriers"])] = GROUP_VALUED
    group[carrier.isin(settings["excluded_carriers"])] = GROUP_EXCLUDED
    # F28 (user, 26 Oct 2026): small-scale plant (rooftop solar, micro wind and hydro) is never bought.
    if "ownership_exclusion" in rows.columns:
        group[rows["ownership_exclusion"].notna().to_numpy()] = GROUP_EXCLUDED
    # User decision (14 Nov 2026): plant not operating at the purchase is never bought.
    group[not_operating_mask(rows, settings)] = GROUP_EXCLUDED
    return group


# ── the profile and its present values ────────────────────────────────────────

def present_value(flows: np.ndarray, rate: float | np.ndarray, *, first_period: float = 0.0) -> np.ndarray:
    """PV of ``flows`` (rows x years, year 0 first) at one rate or one rate per row.

    By default year-0 flows are undiscounted: the model year is the acquisition year and its
    flows accrue to the buyer.  ``first_period=1`` discounts every year one period more, end of
    year (workbook v2.0's annuity factors; the owners' value, F29).
    """
    years = np.arange(flows.shape[1], dtype=float) + float(first_period)
    rate = np.asarray(rate, dtype=float)
    if rate.ndim == 0:
        factors = (1.0 + rate) ** -years
        return flows @ factors
    factors = (1.0 + rate[:, None]) ** -years[None, :]
    return (flows * factors).sum(axis=1)


def compensation_in_market_revenue(rows: pd.DataFrame) -> np.ndarray:
    """Per row, the CfD curtailment compensation that ``market_revenue_gbp`` contains.

    Under contract-value bids the compensation is the accepted bid, inside the
    balancing settlement and so inside market revenue, and support revenue is
    CfD plus RO payments alone (``settlement.combine_settlement_revenue``).
    Otherwise support revenue carries it and market revenue does not. A register
    without ``support_revenue_gbp`` is taken to be on contract-value bids, as
    every CP30 run is.
    """
    def _num(column):
        return pd.to_numeric(rows[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    compensation = _num("cfd_curtailment_compensation_gbp")
    if "support_revenue_gbp" not in rows.columns:
        return compensation
    support = _num("support_revenue_gbp")
    contract_payments = _num("cfd_support_gbp") + _num("roc_support_gbp")
    inside_market = np.isclose(support, contract_payments, rtol=1e-9, atol=1.0)
    return np.where(inside_market, compensation, 0.0)


def balancing_margin(rows: pd.DataFrame) -> np.ndarray:
    """Per row, the margin its balancing redispatch earns: the settlement less the variable cost of the energy moved.

    ``balancing_settlement_gbp`` (inside ``market_revenue_gbp``) pays the accepted offers less
    the bids, while fuel, carbon and variable O&M are on the balancing dispatch, so the energy
    redispatched (``balancing_mwh - wholesale_mwh``) is costed at the row's own variable cost
    per MWh dispatched.  A gas unit offering at 1.6 x SRMC keeps 0.6 x SRMC on its turn-up; a
    unit turned down keeps the fuel it saves less its bid.  Zero for a register without the
    balancing columns.
    """
    if not {"balancing_settlement_gbp", "balancing_mwh", "wholesale_mwh"}.issubset(rows.columns):
        return np.zeros(len(rows))

    def _num(column):
        return (pd.to_numeric(rows[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                if column in rows.columns else np.zeros(len(rows)))

    dispatched = _num("balancing_mwh")
    variable = _num("fuel_carbon_gbp") + _num("annual_variable_om_gbp")
    rate = np.divide(variable, dispatched, out=np.zeros(len(rows)), where=dispatched > 0.0)
    return _num("balancing_settlement_gbp") - rate * (dispatched - _num("wholesale_mwh"))


def balancing_income_ratios(spec: Mapping | None, years: np.ndarray) -> np.ndarray | None:
    """Each year's NESO thermal constraint cost over the model year's (F27 item 13), or None under ``flat``.

    ``spec`` is ``balancing_income_path``; its file holds one column per series by ``year``
    (``neso_constraint_cost_path.csv``).  Years after the file's last take its last ratio.
    """
    if not spec or spec.get("method", BALANCING_PATH_FLAT) == BALANCING_PATH_FLAT:
        return None
    table = pd.read_csv(_resolve(spec["file"])).set_index("year")
    series = str(spec["series"])
    if series not in table.columns:
        raise ValueError(f"balancing_income_path.series={series!r} is not a column of {spec['file']}")
    path = pd.to_numeric(table[series], errors="raise")
    model_year = int(years[0])
    if model_year not in path.index or not path[model_year] > 0.0:
        raise ValueError(f"{spec['file']} has no positive {series} value for the model year {model_year}")
    ratio = (path / path[model_year]).reindex(range(model_year, int(path.index.max()) + 1)).ffill()
    return ratio.reindex(years.astype(int)).ffill().to_numpy(dtype=float)


def legacy_cash_flow_profile(
    rows: pd.DataFrame, settings: Mapping, ro_shares: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Per-row annual flows by stream (GBP, model-year real) and the per-row expiry table.

    A non-vintaged register (2030) takes the F19 asset data (CfD dates from the
    contract start, named plant, gate fees, closure: see the module docstring); a
    vintaged one keeps the award-plus-years contract end and no named adjustments.
    """
    from scripts.public_ownership.model_year import is_vintaged_model_year

    model_year = int(rows["model_year"].iloc[0])
    years = np.arange(model_year, int(settings["horizon_year"]) + 1)
    n = len(rows)
    adjust = not is_vintaged_model_year(model_year)
    retire, method = retirement_years(rows, settings)
    alive = (years[None, :] < retire.to_numpy()[:, None]).astype(float)
    repowering = repowering_weights(rows, settings, years) if settings.get("repowering_share_by_carrier") else None
    if repowering is not None:
        # Decision 62: the share that repowers leaves at the life; the rest runs to the cap.
        alive = np.where(repowering["applies"][:, None], repowering["weight"], alive)
        retire = pd.Series(np.where(repowering["applies"], repowering["cap_year"], retire.to_numpy()), index=rows.index)
        method = pd.Series(np.where(repowering["applies"], "repowering_rule", method.to_numpy()), index=rows.index)
    if adjust:
        coverage, contract = cfd_contract_schedule(rows, settings, years, contract_starts(settings))
        cfd_end = contract["cfd_contract_end_year"]
        # Compensation exists while any phase runs; the top-up is on the contracted share.
        share = contract["cfd_contracted_share"].fillna(1.0).to_numpy()
        contract_live = coverage / np.where(share > 0, share, 1.0)[:, None]
    else:
        cfd_end = cfd_contract_end_years(rows, settings)
        contract_live = (years[None, :] < np.nan_to_num(cfd_end.to_numpy(), nan=-np.inf)[:, None]).astype(float)
        coverage, contract = contract_live, None

    def _num(column):
        return pd.to_numeric(rows[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    def _by_year(model_level, path_level):
        """A level held flat, on the declared price path after the model year."""
        out = np.repeat(model_level[:, None], len(years), axis=1)
        out[:, 1:] = path_level[:, None]
        return out

    # Capacity Market revenue (item MR4) is operating income the owner keeps; it is
    # absent from registers built before MR4, and then zero.
    capacity_market = _num("capacity_market_revenue_gbp") if "capacity_market_revenue_gbp" in rows.columns else np.zeros(n)
    # Variable O&M is an operating cost too (item B1: the settlement's fuel and
    # carbon no longer carry the network's VOM, so it is deducted here from the
    # register's own column when present).  annual_fixed_om_gbp includes the
    # generator TNUoS and insurance the register adds (item D4(a)).
    variable_om = _num("annual_variable_om_gbp") if "annual_variable_om_gbp" in rows.columns else np.zeros(n)
    market = _num("market_revenue_gbp")
    on_path = {"market_revenue_path_gbp", "cfd_support_path_gbp"}.issubset(rows.columns)
    # User decision (10 Nov 2026): the anchored path moves each row's wholesale revenue by its technology's price
    # ratio to the model year, and a CfD's top-up the other way (the reference price moves, the strike does not).
    anchored = (str(settings.get("price_path", "")) in ANCHOR_PRICE_PATHS
                and isinstance(settings.get("price_anchors"), pd.DataFrame) and not on_path)
    on_today = anchored and str(settings.get("price_path")) == PRICE_PATH_TODAY
    # Today's prices move the price alone (the other streams stay at the model year's).
    all_streams = anchored and not on_today and str(settings.get("anchor_streams", "price")) == ANCHOR_ALL
    volume_ratio = np.ones((n, len(years)))
    stream_move = np.zeros((n, len(years)))
    if all_streams:
        # User, 11 Nov 2026: each stream scaled by its technology's per-MW path (output, wholesale, balancing,
        # fuel and carbon, Capacity Market). The CfD top-up then follows output and price below.
        r = anchored_stream_ratios(rows["carrier"].astype(str).to_numpy(), years, settings["price_anchors"], model_year)
        wholesale = _num("wholesale_revenue_gbp") if "wholesale_revenue_gbp" in rows.columns else market
        balancing = _num("balancing_settlement_gbp") if "balancing_settlement_gbp" in rows.columns else np.zeros(n)
        price_move = wholesale[:, None] * (r["wholesale_gbp_per_mw"] - 1.0)
        # Gas plant's balancing income keeps its NESO constraint-cost path (F27 item 13), applied below.
        spec_carriers = [str(c) for c in (settings.get("balancing_income_path") or {}).get("carriers", ())]
        on_neso = rows["carrier"].astype(str).isin(spec_carriers).to_numpy()[:, None]
        # The CfD curtailment compensation inside the settlement is its own flow while the contract runs (below):
        # it follows the balancing path there, so only the rest of the settlement moves here (no double count).
        balancing_rest = balancing - compensation_in_market_revenue(rows)
        balancing_ratio = np.where(on_neso, 1.0, r["balancing_gbp_per_mw"])
        stream_move = (balancing_rest[:, None] * (balancing_ratio - 1.0)
                       - _num("fuel_carbon_gbp")[:, None] * (r["fuel_carbon_gbp_per_mw"] - 1.0)
                       + capacity_market[:, None] * (r["capacity_market_gbp_per_mw"] - 1.0))
        volume_ratio = r["mwh_per_mw"]
        # Variable O&M is paid per MWh, so it follows output.
        if "annual_variable_om_gbp" in rows.columns:
            stream_move = stream_move - _num("annual_variable_om_gbp")[:, None] * (volume_ratio - 1.0)
    elif anchored:
        carriers = rows["carrier"].astype(str).to_numpy()
        ratio = (today_price_ratios(carriers, len(years), settings["price_anchors"], model_year) if on_today else
                 anchored_price_ratios(carriers, years, settings["price_anchors"], model_year))
        wholesale = _num("wholesale_revenue_gbp") if "wholesale_revenue_gbp" in rows.columns else market
        price_move = wholesale[:, None] * (ratio - 1.0)
    else:
        price_move = np.zeros((n, len(years)))
    market_path = _num("market_revenue_path_gbp") if on_path else market
    cfd_level = _num("cfd_support_gbp")
    cfd_path = _num("cfd_support_path_gbp") if on_path else cfd_level
    other = capacity_market - _num("fuel_carbon_gbp") - _num("annual_fixed_om_gbp") - variable_om
    # Review 2 (3 Oct 2026, agreed with the 2030 session): the support lines the register books (ancillary
    # services, EfW gate fees, the batteries' sub-hourly revenue, the top-ups) are operating income the
    # owners keep and lose on a sale, so they are in the operating cash; a register without them is unchanged.
    from scripts.public_ownership.support_lines_2050 import COLUMN_GATE_FEE, SUPPORT_COLUMNS

    for column in SUPPORT_COLUMNS:
        if column in rows.columns:
            other = other + _num(column)
    gate_fee = np.zeros(n)
    if adjust and settings.get("gate_fee_income") and COLUMN_GATE_FEE not in rows.columns:
        # F19 P3 decision 3: energy from waste is paid to take the waste it burns.
        gate_fee = gate_fee_income(rows, settings["gate_fee_income"])
    # CfD curtailment compensation exists only while the CfD runs: the unit bids its
    # foregone difference payment, and after expiry it bids like a merchant. Where
    # it sits inside market revenue (contract-value bids) it is carried only while
    # the contract is alive; elsewhere it is not in market revenue to begin with.
    compensation = compensation_in_market_revenue(rows)
    share_matrix = np.zeros((n, len(years)))
    shares = ro_shares.reindex(years).fillna(0.0)
    for i, carrier in enumerate(rows["carrier"].astype(str)):
        column = carrier if carrier in shares.columns else RO_SHARE_ALL
        share_matrix[i] = shares[column].to_numpy()
    merchant_cash = (_by_year(market + other + gate_fee - compensation, market_path + other + gate_fee - compensation)
                     + price_move + stream_move)
    # F27 item 13 (user, 24 Oct 2026): gas plant's balancing margin falls (or rises) with NESO's
    # constraint-cost path after the model year; renewables' turn-down stays flat (it is paid at
    # contract or opportunity value, so fewer constraints leave their value about unchanged).
    spec = settings.get("balancing_income_path") or {}
    ratios = balancing_income_ratios(spec, years) if adjust else None
    on_balancing_path = rows["carrier"].astype(str).isin([str(c) for c in spec.get("carriers", ())]).to_numpy()
    margin = np.where(on_balancing_path, balancing_margin(rows), 0.0) if ratios is not None else np.zeros(n)
    if ratios is not None:
        merchant_cash = merchant_cash - margin[:, None] * (1.0 - ratios)[None, :]
    # F29: the CfD compensation inside net operating cash, which a no-support value leaves out.
    compensation_flow = compensation[:, None] * contract_live * alive
    if all_streams:
        compensation_flow = compensation_flow * balancing_ratio
    if all_streams:
        # A CfD pays strike x output while it runs: the top-up follows output and moves against the wholesale
        # revenue (strike less reference): v x cfd + wholesale x (v - r), with v the output ratio and r the
        # wholesale revenue ratio (v x the price ratio). The RO pays per MWh, so it follows output.
        on_cfd = (cfd_level != 0.0)[:, None]
        cfd_flow = (volume_ratio * cfd_level[:, None]
                    + np.where(on_cfd, wholesale[:, None] * (volume_ratio - (1.0 + price_move / np.where(
                        wholesale[:, None] != 0.0, wholesale[:, None], 1.0))), 0.0))
        flows = {
            "net_operating_cash": merchant_cash * alive + compensation_flow,
            "cfd_top_up": cfd_flow * coverage * alive,
            "roc_income": _num("roc_support_gbp")[:, None] * volume_ratio * share_matrix * alive,
        }
    else:
        flows = {
            "net_operating_cash": merchant_cash * alive + compensation_flow,
            "cfd_top_up": (_by_year(cfd_level, cfd_path) - price_move * (cfd_level != 0.0)[:, None]) * coverage * alive,
            "roc_income": _num("roc_support_gbp")[:, None] * share_matrix * alive,
        }
    # Fix 8 Nov 2026 (user: P3 buys the CCS plants): contract income outside cfd_support_gbp, which the
    # owners keep only while the contract runs. Without it Drax BECCS (carbon CfD) and Net Zero Teesside
    # (DPA) value at about nothing or close as loss-making.
    if "carbon_removal_cfd_revenue_gbp" in rows.columns:
        flows["cfd_top_up"] = flows["cfd_top_up"] + _num("carbon_removal_cfd_revenue_gbp")[:, None] * coverage * alive
    if "dpa_support_gbp" in rows.columns:
        flows["cfd_top_up"] = flows["cfd_top_up"] + (_num("dpa_support_gbp")[:, None]
                                                     * dpa_contract_schedule(rows, settings, years) * alive)
    named = {"capacity_step": np.full(n, np.nan), "new_contract_from_year": np.full(n, np.nan),
             "declared_margin_gbp": np.full(n, np.nan), "closure_year": np.full(n, np.nan),
             "declared_close_year": np.full(n, np.nan), "margin_years": np.zeros((n, len(years)), dtype=bool)}
    if adjust:
        _apply_named_plant(rows, settings, years, flows, alive, named, market, market_path)
    if adjust and settings.get("close_when_cash_negative"):
        # The owner's option: a plant whose year's total flow turns negative after the
        # model year closes then and for good (the model year is settled as solved).
        total = sum(flows.values())
        # A year that is partly on a declared margin is not a closure decision.
        negative = (total[:, 1:] < 0.0) & ~named["margin_years"][:, 1:]
        closed = np.cumsum(negative, axis=1) > 0
        for stream in flows:
            flows[stream][:, 1:] = np.where(closed, 0.0, flows[stream][:, 1:])
        compensation_flow[:, 1:] = np.where(closed, 0.0, compensation_flow[:, 1:])
        first = np.where(negative.any(axis=1), years[1:][np.argmax(negative, axis=1)], np.nan)
        earlier = np.isfinite(first) & (first < retire.to_numpy())
        named["retirement_before_closure"] = retire.to_numpy().copy()
        named["closure_year"] = np.where(earlier, first, np.nan)
        retire = pd.Series(np.where(earlier, first, retire.to_numpy()), index=rows.index)
        method = pd.Series(np.where(earlier, "closed_when_cash_negative", method.to_numpy()), index=rows.index)
    declared = named["declared_close_year"]
    closes = np.isfinite(declared) & (declared < retire.to_numpy())
    if closes.any():
        # A declared margin with close_at_end: the plant closes when its margin (contract) ends.
        named.setdefault("retirement_before_closure", retire.to_numpy().copy())
        named["closure_year"] = np.where(closes, declared, named["closure_year"])
        retire = pd.Series(np.where(closes, declared, retire.to_numpy()), index=rows.index)
        method = pd.Series(np.where(closes, "closed_at_declared_margin_end", method.to_numpy()), index=rows.index)
    expiry = pd.DataFrame({
        "retirement_year": retire.to_numpy(), "retirement_method": method.to_numpy(),
        "cfd_contract_end_year": cfd_end.to_numpy(),
    }, index=rows.index)
    if contract is not None:
        expiry = expiry.join(contract.drop(columns=["cfd_contract_end_year"]))
        expiry["gate_fee_model_year_gbp"] = gate_fee
        expiry["capacity_step_ratio"] = named["capacity_step"]
        expiry["new_contract_from_year"] = named["new_contract_from_year"]
        expiry["declared_margin_model_year_gbp"] = named["declared_margin_gbp"]
        expiry["closure_year"] = named["closure_year"]
        # The life a book value is written down over (an owner's closure is not a shorter life).
        expiry["retirement_year_before_closure"] = named.get("retirement_before_closure", retire.to_numpy())
    expiry["price_path"] = PRICE_PATH_CP30 if on_path else (
        (PRICE_PATH_TODAY if on_today else PRICE_PATH_ANCHORED) if anchored else PRICE_PATH_MODEL)
    expiry["anchor_streams"] = ANCHOR_ALL if all_streams else ("price" if anchored else "none")
    # F27 item 13: the balancing margin each row's path scales, and the path it is on.
    expiry["balancing_margin_model_year_gbp"] = margin
    expiry["balancing_income_path"] = np.where(
        on_balancing_path & (ratios is not None), f"{BALANCING_PATH_NESO}:{spec.get('series')}", BALANCING_PATH_FLAT)
    if repowering is not None:
        expiry["repowering_year"] = np.where(repowering["applies"], repowering["decision_year"], np.nan)
        expiry["repowering_share"] = np.where(repowering["applies"], repowering["share"], np.nan)
        expiry["model_year_operating_share"] = np.where(repowering["applies"], repowering["weight"][:, 0], np.nan)
    expiry.attrs["cfd_compensation_flow"] = compensation_flow
    return flows, expiry


#: The price paths a valuation runs on (lever ``legacy_price_path``): CP30's level after
#: the model year (rows carrying the ``*_path_gbp`` columns), or the model year's own.
PRICE_PATH_CP30, PRICE_PATH_MODEL = "cp30_level", "model_2030"
#: User decision (10 Nov 2026): each technology's capture price on a straight line through the anchors
#: (``price_anchors.py``: today's forward level and every solved year), flat after the last.
PRICE_PATH_ANCHORED = "anchored"
#: User, 11 Nov 2026: today's prices held for the fleet's whole life: each technology at its anchor for today
#: (the anchors' earliest year: today's forward level on the first solved year's capture ratios), flat. How the
#: owners would value the fleet if they expected today's prices to last.
PRICE_PATH_TODAY = "today_level"
#: The price paths that read the anchors table.
ANCHOR_PRICE_PATHS = (PRICE_PATH_ANCHORED, PRICE_PATH_TODAY)
#: The settlement columns a price path replaces, and the row columns it lands in.
PRICE_PATH_COLUMNS = {"market_revenue_gbp": "market_revenue_path_gbp", "cfd_support_gbp": "cfd_support_path_gbp"}
PRICE_PATHS = (PRICE_PATH_CP30, PRICE_PATH_MODEL, PRICE_PATH_ANCHORED, PRICE_PATH_TODAY)


def anchored_price_ratios(carriers, years, anchors: pd.DataFrame, model_year: int) -> np.ndarray:
    """Each row's wholesale price in each year over its model-year price, on the anchored path.

    ``anchors`` holds ``carrier``, ``year``, ``wholesale_gbp_per_mwh``. A carrier's price runs in a straight line
    between its anchor years and holds flat outside them; a carrier without anchors keeps a ratio of 1.
    """
    years = np.asarray(years, dtype=float)
    out = np.ones((len(carriers), len(years)))
    table = anchors.assign(carrier=anchors["carrier"].astype(str)).dropna(subset=["wholesale_gbp_per_mwh"])
    curves = {}
    for carrier, g in table.groupby("carrier"):
        g = g.sort_values("year")
        xs, ys = g["year"].to_numpy(dtype=float), g["wholesale_gbp_per_mwh"].to_numpy(dtype=float)
        base = float(np.interp(float(model_year), xs, ys))
        if base > 0:
            curves[carrier] = np.interp(years, xs, ys) / base
    for i, carrier in enumerate(carriers):
        if carrier in curves:
            out[i] = curves[carrier]
    return out


def today_price_ratios(carriers, n_years: int, anchors: pd.DataFrame, model_year: int) -> np.ndarray:
    """Each row's price at today's anchor over its model-year price, held for every year (user, 11 Nov 2026).

    Today is the anchors' earliest year; a carrier without a price anchor keeps a ratio of 1.
    """
    priced = anchors.dropna(subset=["wholesale_gbp_per_mwh"])
    if priced.empty:
        raise ValueError("legacy_price_path today_level needs the anchors' wholesale_gbp_per_mwh")
    today = int(priced["year"].min())
    return np.repeat(anchored_price_ratios(carriers, [today], anchors, model_year), n_years, axis=1)


#: User, 11 Nov 2026: anchor_streams values, and horizon_year "life" (each plant's own retirement).
ANCHOR_ALL = "all"
ANCHOR_STREAMS = ("price", ANCHOR_ALL)
HORIZON_LIFE = "life"
#: The latest year a "life" horizon runs to (long-lived civil assets).
HORIZON_LIFE_CAP = 2150
#: The anchor columns every stream follows (price_anchors.STREAMS).
STREAM_COLUMNS = ("mwh_per_mw", "wholesale_gbp_per_mw", "balancing_gbp_per_mw", "fuel_carbon_gbp_per_mw",
                  "capacity_market_gbp_per_mw")


def anchored_stream_ratios(carriers, years, anchors: pd.DataFrame, model_year: int) -> dict[str, np.ndarray]:
    """Each row's stream per MW in each year over its model-year value, by its technology (user, 11 Nov 2026).

    ``anchors`` is price_anchors' table with the stream columns. Each stream runs in a straight line between the
    solved years that give it (a technology absent from a later year holds its last solved value) and flat after
    the last; a technology or stream with no model-year value, or a model-year value of zero or below, keeps a
    ratio of 1 (its own model-year figure, as on the price-only path).
    """
    from scripts.public_ownership.price_anchors import canonical

    missing = [c for c in STREAM_COLUMNS if c not in anchors.columns]
    if missing:
        raise ValueError(f"anchor_streams all needs the anchors' stream columns {missing} (price_anchors --register)")
    years = np.asarray(years, dtype=float)
    table = anchors.assign(carrier=anchors["carrier"].map(canonical))
    out = {c: np.ones((len(carriers), len(years))) for c in STREAM_COLUMNS}
    keys = [canonical(c) for c in carriers]
    for col in STREAM_COLUMNS:
        curves = {}
        for carrier, g in table.dropna(subset=[col]).groupby("carrier"):
            g = g.sort_values("year")
            xs, ys = g["year"].to_numpy(dtype=float), g[col].to_numpy(dtype=float)
            if not (xs.min() <= model_year <= xs.max()) and model_year < xs.min():
                continue
            base = float(np.interp(float(model_year), xs, ys))
            if base > 0:
                curves[carrier] = np.interp(years, xs, ys) / base
        for i, key in enumerate(keys):
            if key in curves:
                out[col][i] = curves[key]
    return out


def with_model_year_streams(anchors: pd.DataFrame | None, register: pd.DataFrame, model_year: int,
                            settings: Mapping) -> pd.DataFrame | None:
    """The anchors with the model year's streams per MW taken from the register being valued (user, 11 Nov 2026).

    The anchors table carries only the later solved years' streams (their registers never read it, so the
    workflow has no cycle); the model year's base is this register's own, every row of each technology in the
    private case, as ``price_anchors.stream_anchor`` measures the later years. Unchanged unless
    ``anchor_streams`` is ``all``.
    """
    if anchors is None or str(settings.get("anchor_streams", "price")) != ANCHOR_ALL:
        return anchors
    from scripts.public_ownership.price_anchors import canonical, stream_anchor

    own = stream_anchor(register, int(model_year))[["carrier", "year", *STREAM_COLUMNS]]
    table = anchors.assign(carrier=anchors["carrier"].map(canonical))
    for column in STREAM_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    table.loc[table["year"].astype(int).eq(int(model_year)), list(STREAM_COLUMNS)] = np.nan
    out = table.merge(own, on=["carrier", "year"], how="outer", suffixes=("", "_own"))
    for column in STREAM_COLUMNS:
        out[column] = out[column].fillna(out.pop(f"{column}_own"))
    return out.sort_values(["carrier", "year"]).reset_index(drop=True)


def profile_by_price_path(profiles: Mapping[str, pd.DataFrame], central: pd.DataFrame) -> pd.DataFrame:
    """The cash-flow profile on every value of lever ``legacy_price_path``, one frame (F19, 14 Oct 2026).

    ``profiles`` maps a lever value to the profile it values on; a value without one (no
    CP30-level settlement: a vintaged register) takes ``central``, the path the acquisition
    plan applies then too.  Column ``price_path`` is the lever value.
    """
    frames = []
    for path in PRICE_PATHS:
        frame = (profiles.get(path) if profiles.get(path) is not None else central).copy()
        frame.insert(0, "price_path", path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


#: F19 (27 Oct 2026): every valuation lever that moves the legacy streams, as (profile column,
#: lever, central value).  ``legacy_price_path``'s central is the lever default, read at run time.
LIVES_CENTRAL, LIVES_EXTENDED = "central", "extended"
PROFILE_SETTINGS = (
    ("price_path", "legacy_price_path", None),
    ("asset_lives", "legacy_asset_lives", LIVES_CENTRAL),
    ("capacity_procurement", "capacity_procurement", capacity_procurement.AUCTION),
)
PROFILE_SETTING_COLUMNS = tuple(column for column, _, _ in PROFILE_SETTINGS)


def extended_lives_block(valuation_block: Mapping | None, extension: Mapping | None) -> dict:
    """The ``acquisition.valuation`` block under ``legacy_asset_lives: extended`` (lever of 29 Sep 2026).

    ``extension`` is ``acquisition.extended_asset_lives``: its carriers live ``extra_years``
    longer (a carrier the block names a life for) and its named overrides apply (Sizewell B to
    its extension date), so the flows (and a market-value price) run on.  The acquisition plan
    and the valuation's per-case streams both build the block here.
    """
    block = dict(valuation_block or {})
    extension = dict(extension or {})
    lives = dict(block.get("retirement_life_years_by_carrier") or {})
    for carrier in extension.get("carriers", ()):
        if carrier in lives:
            lives[carrier] = float(lives[carrier]) + float(extension.get("extra_years", 0.0))
    block["retirement_life_years_by_carrier"] = lives
    block["retirement_year_overrides"] = {
        **dict(block.get("retirement_year_overrides") or {}),
        **dict(extension.get("retirement_year_overrides") or {})}
    return block


def profile_by_settings(profiles: Mapping[tuple, pd.DataFrame]) -> pd.DataFrame:
    """Every profile the valuation wrote, one frame, keyed on :data:`PROFILE_SETTING_COLUMNS` (F19, 27 Oct 2026).

    ``profiles`` maps (price path, asset lives, capacity procurement) to the profile valued on
    those settings; the key's values become the frame's leading columns.
    """
    frames = []
    for key, profile in profiles.items():
        frame = profile.copy()
        for position, (column, value) in enumerate(zip(PROFILE_SETTING_COLUMNS, key)):
            frame.insert(position, column, value)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def case_profile(by_settings: pd.DataFrame | None, settings: Mapping | str | None,
                 central: pd.DataFrame) -> pd.DataFrame:
    """The profile a case's own valuation settings value on, from :func:`profile_by_settings`'s frame.

    ``settings`` is the case's lever values (``legacy_price_path``, ``legacy_asset_lives``,
    ``capacity_procurement``; a lever it does not name is at its central value), or a price path
    alone (the frame of 14 Oct 2026).  ``central`` (the valuation's own profile) when no frame is
    given or the case names no price path.  A frame without a setting's column holds only its
    central value, so a case off it is refused rather than given the central streams.
    """
    if by_settings is None or settings is None:
        return central
    wanted = {"price_path": settings} if isinstance(settings, str) else {
        column: settings.get(lever, default) for column, lever, default in PROFILE_SETTINGS}
    price_path = wanted.get("price_path")
    if price_path is None:
        return central
    if price_path not in PRICE_PATHS:
        raise ValueError(f"legacy_price_path={price_path!r}; supported: {list(PRICE_PATHS)}")
    keep = pd.Series(True, index=by_settings.index)
    for column, _, default in PROFILE_SETTINGS:
        value = wanted.get(column, default)
        if value is None:
            continue
        if column in by_settings.columns:
            keep &= by_settings[column].astype(str).eq(str(value))
        elif str(value) != str(default):
            raise ValueError(f"the legacy profile frame carries no {column} column, so it holds no streams for "
                             f"{column}={value!r}; re-run legacy_valuation.py")
    rows = by_settings.loc[keep]
    if rows.empty:
        raise ValueError(f"the legacy profile frame has no rows for {wanted}")
    return rows.drop(columns=[c for c in PROFILE_SETTING_COLUMNS if c in rows.columns]).reset_index(drop=True)


def valuation_setting_keys(case_settings: pd.DataFrame | None, central_path: str) -> list[tuple]:
    """The (price path, asset lives, capacity procurement) points the valuation writes streams for.

    Every price path at the central lives and auction (the frame of 14 Oct 2026), and every point
    a case of ``case_settings`` (the register's ``*_case_settings.csv``) sits at; a lever column the
    table lacks is at its central value.  The central point (``central_path``) comes first.
    """
    central = (central_path, LIVES_CENTRAL, capacity_procurement.AUCTION)
    keys = [central] + [(p, LIVES_CENTRAL, capacity_procurement.AUCTION) for p in PRICE_PATHS if p != central_path]
    if case_settings is not None and len(case_settings):
        columns = []
        for _, lever, default in PROFILE_SETTINGS:
            fallback = central_path if lever == "legacy_price_path" else default
            columns.append(case_settings[lever].fillna(fallback).astype(str) if lever in case_settings.columns
                           else pd.Series(fallback, index=case_settings.index))
        for key in zip(*columns):
            if key not in keys:
                keys.append(key)
    lives = {LIVES_CENTRAL, LIVES_EXTENDED}
    capacity = {capacity_procurement.AUCTION, capacity_procurement.COST_BASED_EXISTING}
    for path, life, cap in keys:
        if path not in PRICE_PATHS or life not in lives or cap not in capacity:
            raise ValueError(f"no legacy streams for (legacy_price_path, legacy_asset_lives, capacity_procurement) "
                             f"= {(path, life, cap)}")
    return keys


def _resolve(path) -> Path:
    path = Path(path)
    return path if path.exists() or path.is_absolute() else PROJECT_ROOT / path


def contract_starts(settings: Mapping) -> pd.DataFrame | None:
    """The configured LCCC contract phases (a path or a frame), or None."""
    source = settings.get("cfd_contract_starts")
    if source is None or (isinstance(source, str) and not source):
        return None
    return read_contract_starts(source if isinstance(source, pd.DataFrame) else _resolve(source))


def attach_price_path(rows: pd.DataFrame, price_level_settlement: pd.DataFrame) -> pd.DataFrame:
    """``rows`` with ``market_revenue_path_gbp`` and ``cfd_support_path_gbp`` from the CP30-level settlement.

    The settlement is joined as the central one is (``settlement.attach_settlement``:
    a cohort row takes its share), so a row's path columns differ from its own
    columns by exactly the re-settlement's change.  A row with no settlement record
    keeps its own values.
    """
    from scripts.public_ownership.settlement import attach_settlement

    needed = {"component_type", "asset_id", "market_revenue_gbp", "support_revenue_gbp", "fuel_carbon_gbp",
              "cfd_support_gbp"}
    if missing := needed - set(price_level_settlement.columns):
        raise ValueError(f"the price-level settlement lacks {sorted(missing)}")
    keys = rows[["component_type", "asset_id"] + (["cohort_legacy_share"] if "cohort_legacy_share" in rows.columns
                                                  else [])].copy()
    joined = attach_settlement(keys, price_level_settlement[sorted(needed)])
    out = rows.copy()
    settled = joined["settlement_status"].eq("settled").to_numpy()
    for source, target in PRICE_PATH_COLUMNS.items():
        own = pd.to_numeric(rows[source], errors="coerce").fillna(0.0).to_numpy()
        out[target] = np.where(settled, pd.to_numeric(joined[source], errors="coerce").fillna(0.0).to_numpy(), own)
    return out


def gate_fee_income(rows: pd.DataFrame, spec: Mapping) -> np.ndarray:
    """Gate-fee income per row: the fee per tonne x the tonnes each MWh burns, on the named carrier.

    Tonnes per MWh = 3.6 GJ / (net calorific value x the plant's efficiency); the fee
    per tonne and the calorific value come from the gate-fee file (WRAP Gate Fees
    Report 2024/25, Tolvik 2024), the efficiency is the build's.  Needs ``dispatch_mwh``.
    """
    data = json.loads(_resolve(spec["file"]).read_text(encoding="utf-8"))
    fee = float(data["gate_fee_gbp_per_tonne"][spec.get("case", "central")]) * float(data.get("cpi_to_2024_factor", 1.0))
    ncv = float(data["mwh_exported_per_tonne"]["net_calorific_value_mj_per_kg"])
    tonnes_per_mwh = 3.6 / (ncv * float(spec["plant_efficiency"]))
    efw = rows["carrier"].astype(str).eq(str(spec.get("carrier", "waste_to_energy"))).to_numpy()
    if efw.any() and "dispatch_mwh" not in rows.columns:
        raise ValueError("gate-fee income needs the rows' dispatch_mwh")
    dispatch = (pd.to_numeric(rows["dispatch_mwh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                if "dispatch_mwh" in rows.columns else np.zeros(len(rows)))
    tonnes = np.where(efw, dispatch * tonnes_per_mwh, 0.0)
    return tonnes * fee * gate_fee_tonnes_scale(tonnes, spec)


def gate_fee_tonnes_scale(tonnes: np.ndarray, spec: Mapping) -> float:
    """The share of the rows' implied tonnes the throughput cap allows (1 without a cap or under it).

    F19 (user, 6 Oct 2026): the fleet cannot burn more waste than the UK processes
    (``max_tonnes_per_year``, Tolvik's 16.82 Mt in 2024); every row's tonnes scale alike.
    """
    cap = spec.get("max_tonnes_per_year")
    total = float(np.sum(tonnes))
    if cap is None or total <= float(cap) or total <= 0.0:
        return 1.0
    return float(cap) / total


def gate_fee_summary(rows: pd.DataFrame, spec: Mapping | None) -> dict:
    """The gate-fee income's audit: implied tonnes, the cap, the scale and the income (model year)."""
    if not spec:
        return {}
    data = json.loads(_resolve(spec["file"]).read_text(encoding="utf-8"))
    ncv = float(data["mwh_exported_per_tonne"]["net_calorific_value_mj_per_kg"])
    tonnes_per_mwh = 3.6 / (ncv * float(spec["plant_efficiency"]))
    efw = rows["carrier"].astype(str).eq(str(spec.get("carrier", "waste_to_energy"))).to_numpy()
    if not efw.any():
        return {}
    dispatch = pd.to_numeric(rows["dispatch_mwh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    tonnes = np.where(efw, dispatch * tonnes_per_mwh, 0.0)
    return {"dispatch_mwh": float(dispatch[efw].sum()), "tonnes_per_mwh": tonnes_per_mwh,
            "implied_tonnes": float(tonnes.sum()), "max_tonnes_per_year": spec.get("max_tonnes_per_year"),
            "tonnes_scale": gate_fee_tonnes_scale(tonnes, spec),
            "income_model_year_gbp": float(gate_fee_income(rows, spec).sum())}


def _apply_named_plant(rows, settings, years, flows, alive, named, market, market_path) -> None:
    """Capacity steps, new contracts and declared margins for named plant (F19 P3 decision 3), in place."""
    base = rows["asset_id"].map(_base_name).to_numpy()
    capacity = pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for name, step in settings["capacity_from_year"].items():
        for i in np.flatnonzero(base == name):
            if int(step["from_year"]) <= int(years[0]) or capacity[i] <= 0:
                continue
            ratio = float(step["capacity_mw"]) / capacity[i]
            later = years >= int(step["from_year"])
            for stream in flows:
                flows[stream][i, later] *= ratio
            named["capacity_step"][i] = ratio
    for name, contract in settings["new_contracts"].items():
        for i in np.flatnonzero(base == name):
            if "wholesale_mwh" not in rows.columns or "wholesale_revenue_gbp" not in rows.columns:
                raise ValueError(f"a new contract for {name} needs the row's wholesale_mwh and wholesale_revenue_gbp")
            energy = float(pd.to_numeric(pd.Series([rows["wholesale_mwh"].iloc[i]]), errors="coerce").fillna(0.0)[0])
            wholesale = float(pd.to_numeric(pd.Series([rows["wholesale_revenue_gbp"].iloc[i]]),
                                            errors="coerce").fillna(0.0)[0])
            # Its wholesale revenue on the price path: the model year's plus the path's change in market revenue.
            level = np.where(years > years[0], wholesale + (market_path[i] - market[i]), wholesale)
            strike = float(contract["strike_gbp_per_mwh"]) * float(contract.get("rebasing_to_model_price_year", 1.0))
            start = float(contract["from_year"])
            live = _overlap(start, start + float(contract["term_years"]), years.astype(float))
            # Two-way: strike on its output less what the market paid for it.
            flows["cfd_top_up"][i] += (strike * energy - level) * live * alive[i]
            named["new_contract_from_year"][i] = start
    for name, margin in settings["declared_margins"].items():
        for i in np.flatnonzero(base == name):
            annual = (float(margin["segment_ebitda_gbp"]) * capacity[i] / float(margin["segment_capacity_mw"])
                      * float(margin.get("rebasing_to_model_price_year", 1.0)))
            covered = np.clip(float(margin["through_year"]) - years, 0.0, 1.0)
            # The margin is the whole plant's cash while it applies (its contract included);
            # outside it the plant is a merchant: no top-up and no ROCs.  With close_at_end
            # (F19 valuation decision, user 14 Oct 2026: Drax) the plant closes when the margin
            # ends, so it has no merchant years at all.
            close_at_end = bool(margin.get("close_at_end", False))
            merchant = 0.0 if close_at_end else flows["net_operating_cash"][i].copy()
            flows["net_operating_cash"][i] = (covered * annual + (1.0 - covered) * merchant) * alive[i]
            flows["cfd_top_up"][i] = 0.0
            flows["roc_income"][i] = 0.0
            named["declared_margin_gbp"][i] = annual
            named["margin_years"][i] = covered > 0.0
            if close_at_end:
                named["declared_close_year"][i] = float(margin["through_year"])
def repowering_weights(rows: pd.DataFrame, settings: Mapping, years: np.ndarray) -> dict:
    """FES 2050 decision 62: each row's operating share by year under the repowering rule.

    For a dated row of a carrier in ``repowering_share_by_carrier`` (life L, cap C, share r):
    a row past the cap at the model year retires the year after it (observed generating in the
    model year); a row past its life has taken the decision and runs to the cap whole; any other
    row operates in full to its life, then ``1 - r`` of it to the cap.  The model year always
    operates (as :func:`retirement_years` clips).  Undated rows and other carriers keep the
    valuation's own retirement year (``applies`` False).
    """
    model_year = int(rows["model_year"].iloc[0])
    shares = {str(k): float(v) for k, v in (settings.get("repowering_share_by_carrier") or {}).items()}
    caps = {str(k): float(v) for k, v in (settings.get("max_life_years_by_carrier") or {}).items()}
    if missing := sorted(set(shares) - set(caps)):
        raise ValueError(f"acquisition.valuation.repowering_share_by_carrier names {missing} without max_life_years_by_carrier")
    lives = settings["retirement_life_years_by_carrier"]
    built = pd.to_numeric(rows["commissioning_year"], errors="coerce").to_numpy(dtype=float)
    carrier = rows["carrier"].astype(str).to_numpy()
    n = len(rows)
    out = {"applies": np.zeros(n, dtype=bool), "weight": np.zeros((n, len(years))), "cap_year": np.full(n, np.nan),
           "decision_year": np.full(n, np.nan), "share": np.full(n, np.nan)}
    for i in range(n):
        c = carrier[i]
        if c not in shares or not np.isfinite(built[i]):
            continue
        life = float(lives.get(c, lives.get(c.lower(), 25)))
        decision, cap, r = built[i] + life, built[i] + caps[c], shares[c]
        cap_year = max(cap, model_year + 1)
        if decision <= model_year:
            weight = (years < cap_year).astype(float)
        else:
            weight = r * (years < decision) + (1.0 - r) * (years < cap_year)
        weight[0] = max(weight[0], 1.0) if years[0] == model_year else weight[0]
        out["applies"][i] = True
        out["weight"][i] = weight
        out["cap_year"][i], out["decision_year"][i], out["share"][i] = cap_year, decision, r
    return out


def value_legacy_fleet(
    register: pd.DataFrame, public_real_rate: float, settings: Mapping, ro_stations: pd.DataFrame,
    snapshot: Mapping | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (per-row valuation, profile by year/group/carrier/stream, summary).

    ``register`` is the asset finance register; only ``case_id == private`` and
    ``finance_vintage == legacy`` rows are valued.  ``snapshot`` (workbook v2.0, schema 4) gives
    the owners' value its method (F29: after tax, the post-tax operating rates, the book values
    for the capital allowances); without it each row's private WACC values the pre-tax flows.
    """
    missing = REQUIRED_REGISTER_COLUMNS - set(register.columns)
    if missing:
        raise ValueError(f"finance register lacks columns: {sorted(missing)}")
    rows = register.loc[register["case_id"].eq("private") & register["finance_vintage"].eq("legacy")].copy()
    if rows.empty:
        raise ValueError("no private-case legacy rows to value")
    if rows["asset_id"].duplicated().any():
        raise ValueError("private legacy rows repeat an asset_id")
    model_year = int(rows["model_year"].iloc[0])
    if settings["horizon_year"] == HORIZON_LIFE:
        # User, 11 Nov 2026: to the last plant's retirement (closures and repowering only shorten a life).
        retire, _ = retirement_years(rows, settings)
        last = float(np.nanmax(pd.to_numeric(retire, errors="coerce").to_numpy(dtype=float)))
        horizon = int(min(HORIZON_LIFE_CAP, max(model_year + 1, np.ceil(last))))
        settings = {**settings, "horizon_year": horizon}
    horizon = int(settings["horizon_year"])
    if horizon <= model_year:
        raise ValueError(f"acquisition.valuation.horizon_year {horizon} must be after the model year {model_year}")
    # Item Y12: the profile must reach the multi-year view's last year (model year + 30),
    # or a later model year's view reads silent zero years (2050: horizon 2070, view to
    # 2080).  The CP30 year's 2070 already covers 2060, so it is unchanged; an extension
    # is declared in the summary.
    configured_horizon = horizon
    horizon = max(horizon, model_year + MULTI_YEAR_VIEW_SPAN_YEARS)
    if horizon != configured_horizon:
        settings = {**settings, "horizon_year": horizon}
    ro_shares = ro_accreditation_shares(
        ro_stations, model_year, horizon, accreditation_years=int(settings["ro_accreditation_years"]),
        closure_date=str(settings["ro_closure_date"]),
    )
    flows, expiry = legacy_cash_flow_profile(rows, settings, ro_shares)
    compensation_flow = expiry.attrs.get("cfd_compensation_flow", np.zeros_like(flows["net_operating_cash"]))
    total = sum(flows.values())
    private_wacc = pd.to_numeric(rows["wacc"], errors="coerce")
    wacc_missing = private_wacc.isna()
    hurdle_rate = private_wacc.fillna(float(np.nanmedian(private_wacc))).to_numpy(dtype=float)
    workbook = snapshot is not None and settings.get("owner_value_method", OWNER_VALUE_WORKBOOK) == OWNER_VALUE_WORKBOOK
    if workbook:
        # F29: workbook v2.0's Approach 2 on each row (module docstring).
        owner_rate, rate_source = workbook_owner_rates(rows, snapshot, settings)
        fill = float(np.nanmedian(owner_rate)) if np.isfinite(owner_rate).any() else float(np.nanmedian(hurdle_rate))
        market_rate = np.where(np.isfinite(owner_rate), owner_rate, fill)
        rate_source = np.where(np.isfinite(owner_rate), rate_source, "workbook_operating_post_tax_imputed_median")
        assumptions = snapshot.get("assumptions") or {}
        tau = float((assumptions.get("corporation_tax_rate") or {}).get("value", 0.25))
        allowance_rate = float((assumptions.get("legacy_capital_allowance_rate") or {}).get("value", 0.18))
        first_period = 1.0
    else:
        market_rate, tau, allowance_rate, first_period = hurdle_rate, 0.0, 0.0, 0.0
        rate_source = np.where(wacc_missing.to_numpy(), "new_build_hurdle_imputed_median", "new_build_hurdle_row_wacc")

    valuation = rows[[
        "asset_id", "component_type", "carrier", "support_type", "cfd_round", "cfd_award_year", "capacity_mw",
        "commissioning_year", "wacc",
    ]].copy()
    not_operating_mask(rows, settings, check=True)
    valuation["scope_group"] = scope_group(rows, settings)
    valuation = valuation.join(expiry)
    valuation["market_wacc_imputed"] = wacc_missing.to_numpy()
    valuation["owner_discount_rate"] = market_rate
    valuation["owner_discount_rate_source"] = rate_source
    for stream in STREAMS:
        valuation[f"{stream}_model_year_gbp"] = flows[stream][:, 0]
    valuation["cfd_curtailment_compensation_model_year_gbp_not_valued"] = (
        pd.to_numeric(rows["cfd_curtailment_compensation_gbp"], errors="coerce").fillna(0.0).to_numpy())
    valuation["capacity_market_revenue_model_year_gbp"] = (
        pd.to_numeric(rows["capacity_market_revenue_gbp"], errors="coerce").fillna(0.0).to_numpy()
        if "capacity_market_revenue_gbp" in rows.columns else 0.0)
    valuation["annual_flow_model_year_gbp"] = total[:, 0]
    valuation["pv_state_gbp"] = present_value(total, float(public_real_rate))

    def owners(matrix):
        return present_value(matrix, market_rate, first_period=first_period)

    pre_tax = owners(total)
    if workbook:
        book, book_source = workbook_book_values(rows, snapshot, settings, expiry)
        allowances = tau * allowance_rate / (market_rate + allowance_rate) * book
    else:
        book, book_source = np.full(len(rows), np.nan), np.full(len(rows), "not_valued", dtype=object)
        allowances = np.zeros(len(rows))
    valuation["pv_market_pre_tax_gbp"] = pre_tax
    valuation["corporation_tax_rate"] = tau
    valuation["net_book_value_gbp"] = book
    valuation["net_book_value_source"] = book_source
    valuation["capital_allowance_pv_gbp"] = allowances
    valuation["pv_market_gbp"] = (1.0 - tau) * pre_tax + allowances
    # The workbook's no-support value: operating cash without the CfD compensation, after tax, with
    # the same allowances (the plant and its tax base are the same; only the support is gone).
    valuation["pv_no_support_gbp"] = ((1.0 - tau) * owners(flows["net_operating_cash"] - compensation_flow)
                                      + allowances)
    for stream in STREAMS:
        valuation[f"pv_state_{stream}_gbp"] = present_value(flows[stream], float(public_real_rate))
        # Item G3: the same stream at owners' rate, after tax, what cancelling it takes from them
        # (the allowances stay with the plant, in its operating cash).
        valuation[f"pv_market_{stream}_gbp"] = (1.0 - tau) * owners(flows[stream]) + (
            allowances if stream == "net_operating_cash" else 0.0)
    for rate in settings["reference_owner_rates"]:
        valuation[f"pv_at_{float(rate):.4f}_gbp"] = present_value(total, float(rate))
    valuation["public_real_rate"] = float(public_real_rate)
    valuation["horizon_year"] = horizon

    years = np.arange(model_year, horizon + 1)
    keys = pd.DataFrame({
        "scope_group": valuation["scope_group"].to_numpy(), "carrier": rows["carrier"].to_numpy(),
        "support_type": rows["support_type"].fillna("").astype(str).to_numpy(),
    })
    pieces = []
    for stream in STREAMS:
        frame = pd.DataFrame(flows[stream], columns=years)
        frame = pd.concat([keys, frame], axis=1).groupby(["scope_group", "carrier", "support_type"], as_index=False).sum()
        long = frame.melt(id_vars=["scope_group", "carrier", "support_type"], var_name="year", value_name="gbp")
        long["stream"] = stream
        pieces.append(long)
    profile = pd.concat(pieces, ignore_index=True)
    profile = profile.loc[profile["gbp"].ne(0.0)].sort_values(["scope_group", "carrier", "support_type", "stream", "year"])
    profile = profile[["scope_group", "carrier", "support_type", "stream", "year", "gbp"]].reset_index(drop=True)

    summary = _summary(valuation, profile, ro_shares, model_year, horizon, float(public_real_rate), settings)
    if not is_vintaged_model_year_of(model_year) and settings.get("gate_fee_income") and "dispatch_mwh" in rows.columns:
        summary["gate_fee_income"] = gate_fee_summary(rows, settings["gate_fee_income"])
    if horizon != configured_horizon:
        summary["horizon_year_configured"] = configured_horizon
        summary["horizon_extension"] = (
            f"horizon extended from {configured_horizon} to {horizon} so the multi-year view (model year + "
            f"{MULTI_YEAR_VIEW_SPAN_YEARS}) has no silent zero years (item Y12)")
    return valuation, profile, summary


def _summary(valuation, profile, ro_shares, model_year, horizon, public_real_rate, settings) -> dict:
    by_group = {}
    for group in GROUPS:
        rows = valuation.loc[valuation["scope_group"].eq(group)]
        if rows.empty:
            continue
        entry = {
            "assets": int(len(rows)), "capacity_mw": float(rows["capacity_mw"].sum()),
            "annual_flow_model_year_gbp": float(rows["annual_flow_model_year_gbp"].sum()),
            "pv_state_gbp": float(rows["pv_state_gbp"].sum()), "pv_market_gbp": float(rows["pv_market_gbp"].sum()),
            "pv_state_by_stream_gbp": {s: float(rows[f"pv_state_{s}_gbp"].sum()) for s in STREAMS},
            "model_year_by_stream_gbp": {s: float(rows[f"{s}_model_year_gbp"].sum()) for s in STREAMS},
            "cfd_curtailment_compensation_model_year_gbp_not_valued": float(
                rows["cfd_curtailment_compensation_model_year_gbp_not_valued"].sum()),
        }
        for rate in settings["reference_owner_rates"]:
            entry[f"pv_at_{float(rate):.4f}_gbp"] = float(rows[f"pv_at_{float(rate):.4f}_gbp"].sum())
        by_group[group] = entry
    valued = valuation.loc[valuation["scope_group"].eq(GROUP_VALUED)]
    flow_by_year = (profile.loc[profile["scope_group"].eq(GROUP_VALUED)].groupby("year")["gbp"].sum()
                    .reindex(range(model_year, horizon + 1), fill_value=0.0))
    named = {}
    for name in settings["retirement_year_overrides"]:
        rows = valuation.loc[valuation["asset_id"].map(_base_name).eq(name)]
        if not rows.empty:
            named[name] = {"pv_state_gbp": float(rows["pv_state_gbp"].sum()),
                           "pv_market_gbp": float(rows["pv_market_gbp"].sum()),
                           "annual_flow_model_year_gbp": float(rows["annual_flow_model_year_gbp"].sum())}
    return {
        "model_year": model_year, "horizon_year": horizon, "public_real_rate": public_real_rate,
        "flows_held_flat_at_model_year_level": True,
        # F19 P3 decision 2: volumes are held; prices after the model year follow the path.
        "price_path": (str(valuation["price_path"].iloc[0]) if "price_path" in valuation.columns
                       else PRICE_PATH_MODEL),
        # The state's value counts the model year undiscounted; the owners' value (F29, the
        # workbook's annuity factors) discounts it one period, end of year.
        "year_zero_undiscounted": True,
        "owners_value_discounting": ("end_of_year" if float(valuation["corporation_tax_rate"].iloc[0]) > 0.0
                                     else "year_zero_undiscounted"),
        "headline_group": GROUP_VALUED,
        "valued_pv_state_gbp": float(valued["pv_state_gbp"].sum()),
        "valued_pv_market_gbp": float(valued["pv_market_gbp"].sum()),
        "valued_gap_state_minus_market_gbp": float(valued["pv_state_gbp"].sum() - valued["pv_market_gbp"].sum()),
        # F29 (26 Oct 2026): the owners' value method and the workbook's purchase bases per group, rows
        # floored at zero as the purchase prices them (A2 market, A2 no-support, A1 book value, A3).
        "owner_value_method": str(settings.get("owner_value_method", OWNER_VALUE_WORKBOOK)),
        "corporation_tax_rate": float(valuation["corporation_tax_rate"].iloc[0]),
        "owners_value_by_basis_gbp": {str(group): workbook_bases(g) for group, g in valuation.groupby("scope_group")},
        "valued_annual_flow_by_year_gbp": {int(y): float(v) for y, v in flow_by_year.items() if v != 0.0},
        "by_group": by_group,
        "named_assets": named,
        "retirement_methods": valuation["retirement_method"].value_counts().to_dict(),
        "market_wacc_imputed_rows": int(valuation["market_wacc_imputed"].sum()),
        "ro_register": {
            "accredited_dnc_mw_on_1_april_model_year": ro_shares.attrs.get("accredited_dnc_mw_model_year"),
            "closure_date": str(settings["ro_closure_date"]),
            "accreditation_years": int(settings["ro_accreditation_years"]),
            "all_technology_share_by_year": {int(y): round(float(s), 4) for y, s in ro_shares[RO_SHARE_ALL].items() if s > 0},
        },
        "cfd_contract_years_from_award": int(settings["cfd_contract_years_from_award"]),
        "cfd_contract_end_year_overrides": dict(settings["cfd_contract_end_year_overrides"]),
        "cfd_contract_dates": ("start plus term (F19)" if "cfd_contract_start_basis" in valuation.columns
                               else "award plus cfd_contract_years_from_award"),
        "value_per_mw_by_carrier": value_per_mw_by_carrier(valuation),
        "closed_when_cash_negative": (
            {c: float(v) for c, v in valuation.loc[valuation["closure_year"].notna()].groupby("carrier")["capacity_mw"]
             .sum().items()} if "closure_year" in valuation.columns else {}),
        "balancing_income_path": _balancing_path_summary(valuation, settings, model_year),
    }


def _balancing_path_summary(valuation: pd.DataFrame, settings: Mapping, model_year: int) -> dict:
    """F27 item 13: the balancing path applied, its ratios to the model year and the margins it scales."""
    spec = dict(settings.get("balancing_income_path") or {})
    on = (valuation["balancing_income_path"].astype(str).ne(BALANCING_PATH_FLAT)
          if "balancing_income_path" in valuation.columns else pd.Series(False, index=valuation.index))
    if not on.any():
        return {"method": BALANCING_PATH_FLAT, "note": "every row's balancing settlement is held at the model year"}
    last = int(pd.read_csv(_resolve(spec["file"]))["year"].max())
    years = np.arange(model_year, max(last, model_year) + 1)
    ratios = balancing_income_ratios(spec, years)
    rows = valuation.loc[on]
    return {
        "method": spec["method"], "series": spec["series"], "file": str(spec["file"]),
        "carriers": list(spec.get("carriers", ())),
        "ratio_to_model_year": {int(y): round(float(r), 4) for y, r in zip(years, ratios)},
        "after_last_year": f"held at the {last} ratio",
        "margin_model_year_gbp_by_carrier": {
            str(c): float(v) for c, v in rows.groupby("carrier")["balancing_margin_model_year_gbp"].sum().items()},
        "renewables": "held at the model year (turn-down is paid at contract or opportunity value)",
    }


#: F29: the support types whose plant the workbook's A3 basis buys at its no-support value.
RO_SUPPORT_TYPES = ("roc", "ro")


def is_ro_row(support_type: pd.Series) -> pd.Series:
    return support_type.fillna("").astype(str).str.strip().str.lower().isin(RO_SUPPORT_TYPES)


def workbook_bases(rows: pd.DataFrame) -> dict[str, float]:
    """F29: workbook v2.0's acquisition bases over ``rows``, each row floored at zero as the purchase prices it.

    ``market_value`` (A2), ``no_support`` (A2 without support), ``book_value`` (A1, the net book
    value) and ``ro_pass_through`` (A3: RO plant at its no-support value, the rest at A2).
    """
    market = rows["pv_market_gbp"].clip(lower=0.0)
    no_support = (rows["pv_no_support_gbp"].clip(lower=0.0) if "pv_no_support_gbp" in rows.columns
                  else pd.Series(np.nan, index=rows.index))
    book = (rows["net_book_value_gbp"].clip(lower=0.0) if "net_book_value_gbp" in rows.columns
            else pd.Series(np.nan, index=rows.index))
    ro = is_ro_row(rows["support_type"]) if "support_type" in rows.columns else pd.Series(False, index=rows.index)
    return {"market_value": float(market.sum()), "no_support": float(no_support.sum()),
            "book_value": float(book.sum()), "ro_pass_through": float(no_support.where(ro, market).sum())}


def value_per_mw_by_carrier(valuation: pd.DataFrame) -> list[dict]:
    """Each carrier's present values per MW (GBP m/MW): the owners' and the state's, with negative rows at zero too."""
    out = []
    for (group, carrier), rows in valuation.groupby(["scope_group", "carrier"]):
        mw = float(rows["capacity_mw"].sum())
        if mw <= 0:
            continue
        entry = {
            "scope_group": group, "carrier": carrier, "capacity_mw": mw,
            "pv_market_gbp_m_per_mw": float(rows["pv_market_gbp"].sum()) / mw / 1e6,
            "pv_market_floored_gbp_m_per_mw": float(rows["pv_market_gbp"].clip(lower=0.0).sum()) / mw / 1e6,
            "pv_state_gbp_m_per_mw": float(rows["pv_state_gbp"].sum()) / mw / 1e6,
        }
        # F29: the workbook's other bases per MW, and the MW-weighted rate the owners' value used.
        for basis, total in workbook_bases(rows).items():
            entry[f"{basis}_gbp_m_per_mw"] = total / mw / 1e6
        if "owner_discount_rate" in rows.columns:
            entry["owner_discount_rate_mw_weighted"] = float(
                (rows["owner_discount_rate"] * rows["capacity_mw"].astype(float)).sum() / mw)
        out.append(entry)
    return out


def transaction_check(valuation: pd.DataFrame, multiples: pd.DataFrame) -> list[dict]:
    """The owners' value per MW beside observed transaction multiples, carrier by carrier (F19 P3 decision 2).

    ``multiples`` is ``data/public_ownership/financing/legacy_transaction_multiples.csv``.  A check
    table, not an input: the purchase is priced from the cash flows.
    """
    per_mw = {row["carrier"]: row for row in value_per_mw_by_carrier(valuation)}
    out = []
    for row in multiples.itertuples():
        model = per_mw.get(row.carrier) or {}
        out.append({
            "carrier": row.carrier, "deal": row.deal, "year": int(row.year),
            "transaction_gbp_m_per_mw": float(row.gbp_m_per_mw),
            "model_owners_value_gbp_m_per_mw": model.get("pv_market_floored_gbp_m_per_mw"),
            "model_to_transaction": (model["pv_market_floored_gbp_m_per_mw"] / float(row.gbp_m_per_mw)
                                     if model else None),
            "model_owners_rate_mw_weighted": model.get("owner_discount_rate_mw_weighted"),
            "source": row.source,
        })
    return out


def is_vintaged_register(register: pd.DataFrame) -> bool:
    from scripts.public_ownership.model_year import is_vintaged_model_year

    return bool(len(register)) and is_vintaged_model_year(int(pd.to_numeric(register["model_year"]).iloc[0]))


def annuitised_purchase_cost(price_gbp: float, rate: float, term_years: float) -> float:
    """Annual cost of a purchase financed at ``rate`` over ``term_years`` (decision 5: 30 years)."""
    return float(price_gbp) * capital_recovery_factor(rate, term_years)


def as_new_capex(rows: pd.DataFrame) -> np.ndarray:
    """Each register row's as-new capex (GBP): power at its effective (calibrated) capex plus any storage energy.

    The book value of a row the workbook has no cohort for (F29, :func:`workbook_book_values`).
    """
    def column(*names):
        for name in names:
            if name in rows.columns:
                return pd.to_numeric(rows[name], errors="coerce").to_numpy(dtype=float)
        return np.full(len(rows), np.nan)

    power = column("effective_capex_power_gbp_per_kw", "capex_power_gbp_per_kw")
    energy = np.nan_to_num(column("energy_capacity_mwh")) * np.nan_to_num(column("capex_energy_gbp_per_kwh"))
    return pd.to_numeric(rows["capacity_mw"], errors="coerce").to_numpy(dtype=float) * 1_000.0 * power + energy * 1_000.0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("finance_register")
    parser.add_argument("snapshot")
    parser.add_argument("out_assets")
    parser.add_argument("out_profile")
    parser.add_argument("out_summary")
    parser.add_argument("--config", default="config")
    parser.add_argument("--run-group", default=None,
                        help="the run group (default: read off the register's file name); a group's own "
                             "fleet_vintages.repowering rule replaces the study's (decision 62)")
    parser.add_argument("--price-level-settlement", default=None,
                        help="the asset settlement at CP30's price level (price_level_settlement.py); with "
                             "--price-path cp30_level the years after the model year take it (F19 P3 decision 2)")
    parser.add_argument("--price-anchors", default=None,
                        help="legacy_price_anchors.csv (price_anchors.py): the anchored path's capture prices")
    parser.add_argument("--price-path", default=None, choices=PRICE_PATHS,
                        help="the central path (default: the legacy_price_path lever's default); the other is "
                             "reported in the summary as the sensitivity")
    parser.add_argument("--transaction-multiples", default=None,
                        help="observed GBP m/MW by carrier, reported beside the owners' value (a check table)")
    parser.add_argument("--out-profile-by-valuation-settings", "--out-profile-by-price-path",
                        dest="out_profile_by_settings", default=None,
                        help="the cash-flow profile at every point of the valuation levers a case sits at (columns "
                             "price_path, asset_lives, capacity_procurement), which the acquisition tariff reads per "
                             "case (F19 valuation decisions, 14 Oct 2026; generalised 27 Oct 2026)")
    args = parser.parse_args(argv)
    po_block = (read_config(args.config) or {}).get("public_ownership") or {}
    block = po_block.get("acquisition") or {}
    run_group = args.run_group or _run_group_of(args.finance_register)
    settings = settings_from_config(block.get("valuation"))
    # FES 2050 decision 62: a register bound's own repowering share (a no-op without a valuation rule).
    settings = group_repowering_settings(settings, po_block, run_group)
    # F27 item 13: the NESO constraint-cost series of the group's own network case.
    settings = group_balancing_path_settings(settings, po_block, run_group)
    with Path(args.snapshot).open(encoding="utf-8") as handle:
        snapshot = yaml.safe_load(handle)
    from scripts.public_ownership.finance import require_snapshot_schema

    require_snapshot_schema(snapshot)
    public_real_rate = float(snapshot["assumptions"]["public_borrowing_real"]["value"])
    from scripts.public_ownership.register_io import read_register

    # The valuation reads the private case's rows only (valuation_inputs), so read only those.
    register = read_register(args.finance_register, cases=["private"])
    # F29: the book value of a row the workbook has no cohort for.
    register["as_new_capex_gbp"] = as_new_capex(register)
    stations = pd.read_csv(_resolve(settings["ro_register"]), low_memory=False)
    path = args.price_path
    if path is None:
        from scripts.public_ownership.levers import LEVER_LEGACY_PRICE_PATH, lever_default

        path = lever_default(LEVER_LEGACY_PRICE_PATH)
    on_path = None
    anchors = pd.read_csv(args.price_anchors) if args.price_anchors else None
    if anchors is not None and len(register):
        anchors = with_model_year_streams(anchors, register, int(register["model_year"].iloc[0]), settings)
    if path == PRICE_PATH_ANCHORED and anchors is None and not is_vintaged_register(register):
        raise ValueError("--price-path anchored needs --price-anchors")
    if args.price_level_settlement:
        on_path = attach_price_path(register, pd.read_csv(args.price_level_settlement, low_memory=False))
    elif path == PRICE_PATH_CP30 and not is_vintaged_register(register):
        raise ValueError("--price-path cp30_level needs --price-level-settlement")
    # F19 (27 Oct 2026): each valuation lever that moves the streams applies here as the acquisition
    # plan applies it to the price: lives through the extended block, cost-based capacity contracts
    # in place of the Capacity Market income.
    lives_settings = group_balancing_path_settings(group_repowering_settings(
        settings_from_config(extended_lives_block(block.get("valuation"), block.get("extended_asset_lives"))),
        po_block, run_group), po_block, run_group)
    capacity_params = capacity_procurement.parameters(
        (po_block.get("lever_parameters") or {}).get(capacity_procurement.PARAMETER_KEY))
    valued_on: dict[tuple, tuple] = {}

    def value_at(key: tuple) -> tuple:
        price_path, lives, capacity = key
        # Without a CP30-level settlement both price paths are the model price (the plan's path then too).
        on_cp30 = price_path == PRICE_PATH_CP30 and on_path is not None
        on_anchors = price_path in ANCHOR_PRICE_PATHS and anchors is not None
        memo = (on_cp30, price_path if on_anchors else None, lives, capacity)
        if memo not in valued_on:
            rows = on_path if on_cp30 else register
            if capacity == capacity_procurement.COST_BASED_EXISTING:
                rows = capacity_procurement.valuation_rows(rows, capacity_params)
            chosen = dict(lives_settings if lives == LIVES_EXTENDED else settings)
            if on_anchors:
                chosen.update(price_path=price_path, price_anchors=anchors)
            valued_on[memo] = value_legacy_fleet(rows, public_real_rate, chosen, stations, snapshot)
        return valued_on[memo]

    from scripts.public_ownership.register_io import settings_path

    case_settings_file = settings_path(args.finance_register)
    case_settings = pd.read_csv(case_settings_file) if case_settings_file.exists() else None
    keys = valuation_setting_keys(case_settings, central_path=path)
    valuation, profile, summary = value_at(keys[0])
    # F19 valuation decisions (user, 14 Oct 2026; generalised 27 Oct 2026): the streams on each point
    # of the valuation levers a case sits at (legacy_price_path, legacy_asset_lives,
    # capacity_procurement), so every case reads its own streams in the tariff (multi-year view,
    # lifetime break-even, three parts), not the central's.
    profiles = {key: value_at(key)[1] for key in keys}
    summary["profiles_by_valuation_settings"] = [
        {**dict(zip(PROFILE_SETTING_COLUMNS, key)),
         "valued_pv_market_gbp": value_at(key)[2]["valued_pv_market_gbp"],
         "valued_pv_state_gbp": value_at(key)[2]["valued_pv_state_gbp"]} for key in keys]
    if on_path is not None:
        # The other path, reported beside the central (the purchase reads the case's own lever).
        other_path = PRICE_PATH_MODEL if path == PRICE_PATH_CP30 else PRICE_PATH_CP30
        alt, alt_profile, alt_summary = value_at((other_path, LIVES_CENTRAL, capacity_procurement.AUCTION))
        summary["price_path_sensitivity"] = {
            "price_path": alt_summary["price_path"],
            "by_group": {g: {k: e[k] for k in ("pv_state_gbp", "pv_market_gbp")} for g, e in alt_summary["by_group"].items()},
            "valued_pv_market_gbp": alt_summary["valued_pv_market_gbp"],
            "value_per_mw_by_carrier": alt_summary["value_per_mw_by_carrier"],
        }
        summary["source_price_level_settlement"] = str(args.price_level_settlement)
    if args.transaction_multiples:
        summary["transaction_check"] = transaction_check(valuation, pd.read_csv(args.transaction_multiples))
    summary["source_finance_register"] = str(args.finance_register)
    for path in (args.out_assets, args.out_profile, args.out_summary):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    valuation.to_csv(args.out_assets, index=False)
    profile.to_csv(args.out_profile, index=False)
    if args.out_profile_by_settings:
        Path(args.out_profile_by_settings).parent.mkdir(parents=True, exist_ok=True)
        profile_by_settings(profiles).to_csv(args.out_profile_by_settings, index=False)
    with Path(args.out_summary).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({
        "valued_pv_state_gbp_bn": round(summary["valued_pv_state_gbp"] / 1e9, 2),
        "valued_pv_market_gbp_bn": round(summary["valued_pv_market_gbp"] / 1e9, 2),
        "valued_flow_model_year_gbp_bn": round(summary["by_group"][GROUP_VALUED]["annual_flow_model_year_gbp"] / 1e9, 2),
    }))


if __name__ == "__main__":
    main()
