"""Carrier names of disaggregated demand and consumer flexibility (plan Y6a).

A light module (no pandas-heavy imports) so that the energy validation, the finance layer
and the demand builders share one list.

* **Fixed component loads** are PyPSA ``Load`` components on AC buses whose carrier is
  the component name. They are electricity demand at the meter, inside the ED1 consumer
  total, and the energy validation counts them with the base (carrier "") loads.
* **Consumer flexibility** is PyPSA ``StorageUnit`` components on AC buses (plan Y6a,
  decision 24). Each unit consumes electricity to meet a fixed drain (``inflow < 0``:
  driving, space heat, a DACCS duty) and may shift when; V2G and demand response may also
  inject. They are household- and business-owned assets: **every ownership case must
  exclude them** (:data:`CONSUMER_OWNED_CARRIERS`).
* **DACCS** (decision 25) is a flexible load outside ED1 (ES1 row 611). It is not an
  electricity asset of any ownership case either (:data:`NON_ELECTRICITY_ASSET_CARRIERS`).
"""

# Fixed component loads, carrier = component name (scripts/demand/integrate.py convention).
HEAT_PUMP_LOAD = "heat_pumps"
EV_LOAD = "electric_vehicles"
DISTRICT_HEAT_LOAD = "district_heat"
DATA_CENTRE_LOAD = "data_centres"

# Carriers of loads that the flexibility modules add directly on AC buses (existing code).
LEGACY_COMPONENT_LOAD_CARRIERS = ("heat pump", "EV charging")

DEMAND_COMPONENT_LOAD_CARRIERS = (HEAT_PUMP_LOAD, EV_LOAD, DISTRICT_HEAT_LOAD, DATA_CENTRE_LOAD,
                                  *LEGACY_COMPONENT_LOAD_CARRIERS)

# Consumer flexibility, as StorageUnits on AC buses.
EV_SMART_CHARGING = "EV_smart_charging"
V2G = "V2G_fleet"
HEAT_PUMP_FLEXIBILITY = "HP_flexibility"
DISTRICT_HEAT_FLEXIBILITY = "DH_flexibility"
IC_DSR = "IC_DSR"
RESIDENTIAL_DSR = "residential_DSR"
# Storage heaters (CP30 2030, follow-up F5; scripts/demand/cp30_flexibility.py). FES 2025 FLX1
# books them too; the FES long-range chain does not build them yet.
STORAGE_HEATER_FLEXIBILITY = "SH_flexibility"

CONSUMER_FLEX_CARRIERS = (EV_SMART_CHARGING, V2G, HEAT_PUMP_FLEXIBILITY, DISTRICT_HEAT_FLEXIBILITY,
                          IC_DSR, RESIDENTIAL_DSR, STORAGE_HEATER_FLEXIBILITY)

# DACCS: a flexible load with an annual volume, outside ED1.
DACCS = "DACCS"

# Storage units that are demand, not storage plant: every check of storage capacity, its
# regional split or its market churn leaves them out (F5, 27 Oct 2026).
DEMAND_FLEX_STORAGE_CARRIERS = CONSUMER_FLEX_CARRIERS + (DACCS,)

# Everything the ownership cases must leave out: consumer-owned flexibility and the
# fixed demand components (loads are not assets, listed so a carrier filter is complete).
CONSUMER_OWNED_CARRIERS = CONSUMER_FLEX_CARRIERS + DEMAND_COMPONENT_LOAD_CARRIERS
NON_ELECTRICITY_ASSET_CARRIERS = (DACCS,)

# Which electricity-demand line each carrier belongs to (energy validation breakdown).
DEMAND_LINE_OF_CARRIER = {
    HEAT_PUMP_LOAD: "heat_pumps", "heat pump": "heat_pumps", HEAT_PUMP_FLEXIBILITY: "heat_pumps",
    EV_LOAD: "evs", "EV charging": "evs", EV_SMART_CHARGING: "evs", V2G: "evs",
    DISTRICT_HEAT_LOAD: "district_heat", DISTRICT_HEAT_FLEXIBILITY: "district_heat",
    DATA_CENTRE_LOAD: "data_centres",
    IC_DSR: "demand_response", RESIDENTIAL_DSR: "demand_response",
    # Storage heaters are inside ED1's residential base (the residual): their energy is carved from it.
    STORAGE_HEATER_FLEXIBILITY: "residual",
    DACCS: "daccs",
}
