"""The FES long-range hydrogen system (task Y6b, plan decision 15; year-generic in stage 5).

Configured by ``hydrogen.fes``. The numbers below are FES 2025 Holistic Transition 2050's.

One GB hydrogen bus (``GB_H2``, carrier ``H2_gas``, as the CP30 build) with:

* **Electrolysers**, links from electricity to ``GB_H2``, at the ETYS nodes FES names for its
  networked electrolysis (``Dem_BB009``, 15.66 GW electric), at FES's own efficiency
  (WS1 row 466 / row 593 = 72.27 / 88.71 = 0.8147, HHV).
* **Hydrogen turbines**, links from ``GB_H2``: the FES ``H2`` generators placed by the thermal
  stage (19.1 GW Direct at the declared schedule's sites, 7.0 GW at FES GSPs) become links at
  the FES-implied efficiency (ES1 generation / WS1 power-generation demand = 10.40 / 27.33 =
  0.381, HHV).
* **Storage**, one store of the WS1 row 603 volume (11.67 TWh), no standing loss by default
  (FES books none; the CP30 value of 1e-4/h on 11.67 TWh would lose ~5 TWh a year).
* **Gas reforming with CCS** (plan decision: a dispatchable link from gas at gas plus CCS
  cost, a hydrogen asset outside the electricity ownership cases): a ``GB_natural_gas`` bus
  fed by a supply generator at the FES gas price, and a link to ``GB_H2`` of 5.38 GW H2 at
  the WS1 efficiency (44.51 / 53.36 = 0.834; DESNZ Hydrogen Production Costs 2021 ATR+CCUS:
  84% HHV). The link carries the residual carbon (95% capture) and the CO2 transport and
  storage fee per MWh of gas.
* **Non-power hydrogen demand**, the WS1 annual volume (industry, transport, commercial,
  residential, losses; 92.23 TWh FY, 91.41 TWh calendar) as a flat load on ``GB_H2``.

Why a flat load rather than an annual constraint. The market solves run in windows (24 h
day-ahead, 1 h or 24 h balancing), so an annual energy constraint cannot bind across them.
The declared treatment is the per-window pro-rata volume: every hour must deliver the
annual volume / 8,760, and the LP still chooses when to MAKE the hydrogen, because the
11.67 TWh store carries its level between windows (``market.hydrogen_store.carry_level``)
with an end-of-window value (``end_of_window_value_gbp_per_mwh_h2``). In a single-window
solve the store plus the flat load is equivalent to an annual volume with a timing-free
offtake profile.

Feasibility on paper (HT 2050): electrolysis at most 15.66 GW x 8,760 h x 0.8147 = 111.8 TWh
H2 plus reforming 5.38 GW x 8,760 h = 47.1 TWh, 158.9 TWh against 119.7 TWh of demand
(91.4 non-power + 28.3 for the turbines, calendar); per hour 12.76 + 5.38 = 18.1 GW of H2
capacity against a 10.4 GW flat offtake. Electrolysis alone (111.8) cannot meet the demand.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from scripts.generators.fes_supply import resolve_node_bus
from scripts.utilities.calendar_year import hours_in_year

logger = logging.getLogger(__name__)

H2_BUS = "GB_H2"
GAS_BUS = "GB_natural_gas"
GAS_SUPPLY = "GB_natural_gas_supply"
REFORMER = "GB_gas_reforming_ccs"
STORE = "GB_H2_storage"
NON_POWER_LOAD = "GB_H2_non_power_demand"
# Natural gas direct combustion factor, t CO2 per MWh thermal (the marginal-cost
# module's 183 kg/MWh, scripts/generators/apply_marginal_costs.py).
GAS_EMISSION_FACTOR_T_PER_MWH = 0.183

# Code defaults = the declared HT 2050 values; the HT50_* scenario states them too.
DEFAULTS = {
    "store_standing_loss_per_hour": 0.0,
    "turbine_variable_om_gbp_per_mwh": 5.0,       # add_hydrogen_system.H2_TURBINE_MARGINAL_COST
    "electrolysis_variable_om_gbp_per_mwh": 0.0,  # add_hydrogen_system.ELECTROLYSIS_MARGINAL_COST
    "non_power_demand_basis": "calendar",         # calendar | fiscal
    "gas_reforming_ccs": {
        "enabled": True,
        "capture_rate": 0.95,                     # DESNZ Hydrogen Production Costs 2021, ATR+CCUS p.13
        "co2_transport_storage_gbp_per_tco2": 34.5,   # DESNZ 2021 p.25: 28 GBP/t (2020) x 1.232 CPI to 2024
        "variable_om_gbp_per_mwh_h2": 0.0,
    },
}


#: ``hydrogen.<key>`` of the system.
CONFIG_KEY = "fes"


def fes_hydrogen_config(hydrogen_config: Optional[dict]) -> dict:
    """The ``hydrogen.fes`` block (given the ``hydrogen`` block), ``{}`` when absent."""
    return dict((hydrogen_config or {}).get(CONFIG_KEY) or {})


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def hydrogen_quantities(targets: dict, basis: str = "calendar") -> Dict[str, float]:
    """The WS1/FLX1/ES1 hydrogen quantities the build uses, from the Y1 targets JSON."""
    h2 = targets["hydrogen"]
    if basis == "calendar":
        demand = h2["annual_twh_calendar"]["demand"]
        non_power = float(sum(v for k, v in demand.items() if k != "Power Generation"))
    elif basis == "fiscal":
        non_power = float(h2["non_power_demand_twh"])
    else:
        raise ValueError(f"non_power_demand_basis must be calendar or fiscal, got {basis!r}")
    reforming = h2["gas_reforming_ccs"]
    return {
        "electrolysis_mw": float(h2["electrolysis_capacity_gw_electric"]) * 1e3,
        "electrolysis_efficiency": float(h2["electrolysis_efficiency"]),
        "turbine_efficiency": float(h2["h2_turbines"]["efficiency_electric_per_h2"]),
        "turbine_mw": float(h2["h2_turbines"]["capacity_mw"]),
        "storage_mwh": float(h2["storage_twh"]) * 1e6,
        "reforming_mw_h2": float(reforming["capacity_gw_h2"]) * 1e3,
        "reforming_efficiency": float(reforming["output_twh"]) / float(reforming["gas_feedstock_twh"]),
        "non_power_demand_mwh": non_power * 1e6,
        "non_power_demand_basis": basis,
    }


def reformer_costs(gas_price: float, carbon_price: float, cfg: dict, efficiency: float) -> Dict[str, float]:
    """Gas supply price and the reformer link's per-MWh-gas CCS cost, GBP."""
    capture = float(cfg["capture_rate"])
    residual_t = GAS_EMISSION_FACTOR_T_PER_MWH * (1.0 - capture)
    captured_t = GAS_EMISSION_FACTOR_T_PER_MWH * capture
    link_cost = residual_t * carbon_price + captured_t * float(cfg["co2_transport_storage_gbp_per_tco2"]) \
        + float(cfg["variable_om_gbp_per_mwh_h2"]) * efficiency
    return {
        "gas_price_gbp_per_mwh_gas": float(gas_price),
        "link_cost_gbp_per_mwh_gas": float(link_cost),
        "hydrogen_cost_gbp_per_mwh_h2": (float(gas_price) + link_cost) / efficiency,
    }


# Transmission licence area of an FES "DNO License Area" (the Direct rows name the TO).
DNO_LICENCE_AREA = {"SSEN (SHEPD)": "SHETL", "SHETL": "SHETL", "SPEN": "SPTL", "SPTL": "SPTL"}


def licence_area(dno) -> str:
    """SHETL, SPTL or NGET for an FES DNO licence-area label (England and Wales default NGET)."""
    return DNO_LICENCE_AREA.get(str(dno).strip(), "NGET")


def electrolyser_sites(fes: pd.DataFrame, buses: pd.DataFrame, pathway: str, year: int) -> pd.DataFrame:
    """FES networked electrolysis (Dem_BB009) by ETYS bus, MW electric, with its licence area."""
    rows = fes[fes["FES Pathway"].eq(pathway) & fes["Building Block ID Number"].eq("Dem_BB009")].copy()
    rows["mw"] = pd.to_numeric(rows[str(year)], errors="coerce").fillna(0.0)
    rows = rows[rows["mw"] > 0]
    rows["bus"] = rows["Node ID"].map(lambda node: resolve_node_bus(node, buses))
    rows["area"] = rows.get("DNO License Area", pd.Series("NGET", index=rows.index)).map(licence_area)
    return rows[["GSP", "Node ID", "bus", "area", "mw"]]


def place_electrolysers(sites: pd.DataFrame) -> pd.Series:
    """MW by bus: placed rows at their node; Direct and unplaced rows spread pro rata over
    the placed nodes of the same licence area (declared: FES names no site for its 6.6 GW of
    Direct electrolysis, so it follows FES's own GSP-level electrolysis in that area)."""
    placed = sites[sites["bus"].notna()]
    if placed.empty:
        raise ValueError("FES electrolysis: no Dem_BB009 row maps to a network bus")
    by_bus = placed.groupby("bus")["mw"].sum()
    area_of_bus = placed.groupby("bus")["area"].agg(lambda a: a.mode().iloc[0])
    for area, mw in sites[sites["bus"].isna()].groupby("area")["mw"].sum().items():
        pool = by_bus[area_of_bus.reindex(by_bus.index).eq(area)]
        if pool.empty:
            pool = by_bus
        by_bus = by_bus.add(mw * pool / pool.sum(), fill_value=0.0)
    return by_bus


def add_fes_hydrogen_system(network, fes: pd.DataFrame, targets: dict, prices: dict,
                                config: Optional[dict] = None, pathway: str = "Holistic Transition",
                                year: int = 2050, log: Optional[logging.Logger] = None):
    """Add the FES hydrogen system of ``pathway`` and ``year`` to ``network`` in place; return an audit dict.

    The non-power offtake is the annual volume spread flat over the modelled year's hours
    (8,784 in a leap year).
    """
    log = log or logger
    cfg = _merge(DEFAULTS, config or {})
    q = hydrogen_quantities(targets, cfg["non_power_demand_basis"])
    for carrier, nice in (("H2_gas", "Hydrogen Gas"), ("electrolysis", "Electrolysis"),
                          ("H2_turbine", "H2 Power Generation"), ("H2_demand", "Hydrogen offtake"),
                          ("natural_gas", "Natural gas (reformer feed)"), ("gas_supply", "Natural gas supply"),
                          ("gas_reforming_ccs", "Gas reforming with CCS")):
        if carrier not in network.carriers.index:
            network.add("Carrier", carrier, nice_name=nice, co2_emissions=0.0)

    # Electrolysers at the FES nodes.
    sites = electrolyser_sites(fes, network.buses, pathway, year)
    unplaced = sites[sites["bus"].isna()]
    if len(unplaced):
        log.info("FES %s electrolysis: %d rows (%.0f MW) without a network node (Direct or unnamed); "
                 "spread over the placed nodes of their licence area: %s",
                 year, len(unplaced), unplaced["mw"].sum(),
                 "; ".join(f"{g} ({a}) {m:,.0f} MW" for g, a, m in zip(unplaced["GSP"], unplaced["area"], unplaced["mw"])))
    by_bus = place_electrolysers(sites)
    by_bus = by_bus * (q["electrolysis_mw"] / by_bus.sum())
    xy = network.buses.loc[by_bus.index, ["x", "y"]]
    network.add("Bus", H2_BUS, carrier="H2_gas", x=float(xy["x"].mean()), y=float(xy["y"].mean()), v_nom=1.0)
    for bus, mw in by_bus.items():
        network.add("Link", f"electrolysis_{bus}", bus0=bus, bus1=H2_BUS, carrier="electrolysis",
                    efficiency=q["electrolysis_efficiency"], p_nom=float(mw), p_nom_extendable=False,
                    marginal_cost=float(cfg["electrolysis_variable_om_gbp_per_mwh"]), capital_cost=0.0)

    # Hydrogen turbines: convert the FES H2 generators to links.
    h2_gens = network.generators[network.generators.carrier.isin(["H2", "Hydrogen"])].copy()
    if h2_gens.empty:
        raise ValueError(f"FES {year} hydrogen: no H2 generators in the network (thermal stage)")
    network.remove("Generator", h2_gens.index.tolist())
    eta_t = q["turbine_efficiency"]
    for name, gen in h2_gens.iterrows():
        network.add("Link", f"H2_turbine_{name}", bus0=H2_BUS, bus1=gen["bus"], carrier="H2_turbine",
                    efficiency=eta_t, p_nom=float(gen["p_nom"]) / eta_t, p_nom_extendable=False,
                    marginal_cost=float(cfg["turbine_variable_om_gbp_per_mwh"]), capital_cost=0.0)

    network.add("Store", STORE, bus=H2_BUS, carrier="H2_gas", e_nom=q["storage_mwh"], e_nom_extendable=False,
                e_cyclic=True, standing_loss=float(cfg["store_standing_loss_per_hour"]),
                marginal_cost=0.0, capital_cost=0.0)

    hours = float(hours_in_year(year))
    flat_mw = q["non_power_demand_mwh"] / hours
    network.add("Load", NON_POWER_LOAD, bus=H2_BUS, carrier="H2_demand", p_set=flat_mw)

    audit = {"electrolysis_mw": float(by_bus.sum()), "electrolysis_buses": int(len(by_bus)),
             "electrolysis_efficiency": q["electrolysis_efficiency"],
             "turbine_mw_electric": float(h2_gens["p_nom"].sum()), "turbine_efficiency": eta_t,
             "turbine_links": int(len(h2_gens)), "storage_mwh": q["storage_mwh"],
             "non_power_demand_mwh": q["non_power_demand_mwh"], "non_power_flat_mw_h2": flat_mw,
             "non_power_demand_basis": q["non_power_demand_basis"]}

    reform_cfg = cfg["gas_reforming_ccs"]
    if reform_cfg.get("enabled", True):
        costs = reformer_costs(prices["gas"], prices["carbon"], reform_cfg, q["reforming_efficiency"])
        gas_mw = q["reforming_mw_h2"] / q["reforming_efficiency"]
        network.add("Bus", GAS_BUS, carrier="natural_gas", x=float(xy["x"].mean()), y=float(xy["y"].mean()), v_nom=1.0)
        network.add("Generator", GAS_SUPPLY, bus=GAS_BUS, carrier="gas_supply", p_nom=gas_mw,
                    marginal_cost=costs["gas_price_gbp_per_mwh_gas"], data_source=f"FES{int(year)}_hydrogen")
        network.add("Link", REFORMER, bus0=GAS_BUS, bus1=H2_BUS, carrier="gas_reforming_ccs",
                    efficiency=q["reforming_efficiency"], p_nom=gas_mw, p_nom_extendable=False,
                    marginal_cost=costs["link_cost_gbp_per_mwh_gas"], capital_cost=0.0)
        audit.update({"reforming_mw_h2": q["reforming_mw_h2"], "reforming_mw_gas": gas_mw,
                      "reforming_efficiency": q["reforming_efficiency"], **costs})

    supply_max_twh = (audit["electrolysis_mw"] * audit["electrolysis_efficiency"]
                      + audit.get("reforming_mw_h2", 0.0)) * hours / 1e6
    turbine_h2_twh = float(targets["hydrogen"]["annual_twh_calendar"]["demand"].get("Power Generation", 0.0))
    audit["feasibility"] = {
        "supply_max_twh_h2": supply_max_twh,
        "demand_twh_h2": q["non_power_demand_mwh"] / 1e6 + turbine_h2_twh,
        "hourly_supply_max_mw_h2": audit["electrolysis_mw"] * audit["electrolysis_efficiency"]
        + audit.get("reforming_mw_h2", 0.0),
        "hourly_flat_offtake_mw_h2": flat_mw,
    }
    if supply_max_twh < audit["feasibility"]["demand_twh_h2"]:
        raise ValueError(f"FES {year} hydrogen balance infeasible on paper: {audit['feasibility']}")
    log.info("FES %s hydrogen: electrolysis %.0f MW at %d buses (eta %.4f); turbines %.0f MW (eta %.4f); "
             "store %.2f TWh; non-power %.2f TWh (%s, %.0f MW flat); reforming %.0f MW H2 at %.2f GBP/MWh_H2; "
             "paper balance %.1f TWh capacity vs %.1f TWh demand",
             year, audit["electrolysis_mw"], audit["electrolysis_buses"], audit["electrolysis_efficiency"],
             audit["turbine_mw_electric"], eta_t, q["storage_mwh"] / 1e6, q["non_power_demand_mwh"] / 1e6,
             q["non_power_demand_basis"], flat_mw, audit.get("reforming_mw_h2", 0.0),
             audit.get("hydrogen_cost_gbp_per_mwh_h2", float("nan")),
             supply_max_twh, audit["feasibility"]["demand_twh_h2"])
    return audit


def future_cost_prices(source_file) -> Dict[str, float]:
    """Gas (GBP/MWh thermal) and total carbon (GBP/t) from a Y1 future-costs source file."""
    import json
    with open(source_file, encoding="utf-8") as handle:
        payload = json.load(handle)
    assumptions = payload["assumptions"]
    carbon = assumptions["carbon"]
    return {"gas": float(assumptions["fuel_prices"]["gas"]),
            "carbon": float(carbon.get("uk_ets", 0.0)) + float(carbon.get("carbon_price_support", 0.0))}
