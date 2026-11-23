"""The GB Capacity Market in calendar 2050 on FES 2025 Holistic Transition (task Y11).

Nothing here is
read by a 2030 rule: the calendar-2030 model is ``capacity_market.py`` and
``build_capacity_market.py``'s default path, which are unchanged.

**No agreement in today's register reaches 2050** (the longest, 15 years from 2030/31,
ends in 2045), so the whole calendar-2050 line is modelled. It reuses the MR4 auction
machinery (:class:`capacity_market.DemandCurve`, :func:`capacity_market.clear_auction`,
:func:`capacity_market.awarded_by_category` and the bid distributions fitted on the two
held T-4 auctions) on a 2050 demand curve and a 2050 supply:

* **Requirement.** De-rated capacity held per GW of ACS peak is calibrated on 2030/31:
  everything the CM contracts for 2030/31 (contract book, modelled T-4, T-1) plus the
  de-rated capacity outside the CM (CfD, RO and DPA plant, merchant renewables) over the
  FES 2025 HT 2030 ACS peak. 2050 applies that ratio to the 2050 peak (``ratio``) or the
  2030 margin in GW (``additive``, a sensitivity).
* **Target.** Requirement less the de-rated capacity that cannot hold an agreement
  (CfD, DPA, RAB plant in its support term, and, centrally, V2G) less the T-1 set-aside.
* **Supply.** Every eligible unit's de-rated capacity by MR4 category: merchant plant
  (renewables whose CfD or RO has ended, CCS and hydrogen plant past their 15-year DPA,
  waste, hydro), storage (LDES on cap and floor enters as a price-taker, as Ofgem's
  interaction rules propose), interconnectors, DSR, and new build (plant commissioned in
  the four years before delivery).
* **De-rating.** NESO's 2030/31 T-4 factors by class (``neso_cm_derating_factors.csv``);
  storage by duration; ``storage_efc_scale`` scales storage and V2G for the saturation
  NESO's EFC method would apply to a 56 GW storage fleet (1.0 central, declared).
* **Calendar weighting.** Calendar 2050 pays half of each of 2049/50 and 2050/51 (the
  supplier charge is levied November to February). Both delivery years are cleared on
  the 2050 fleet, so the weighted total equals one delivery year: the weighting is
  applied formally so the output has the 2030 shape.

The fleet is a stock-flow over the FES 2025 HT ES1 capacity trajectory 2024 to 2050
(``cm_2050_fleet_fes2025_ht.csv``, extracted by :func:`extract_fes_trajectory`), a
declared stand-in until the Y10 vintaged register is wired (:func:`fleet_from_register`).
"""

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from scripts.market import capacity_market as cm
from scripts.public_ownership.model_year import fiscal_year_label

DATA = Path("data/market/capacity_market")
FLEET_FILE = DATA / "cm_2050_fleet_fes2025_ht.csv"
CALIBRATION_FILE = DATA / "cm_2050_requirement_calibration.csv"
DERATING_FILE = Path("data/market/scarcity/neso_cm_derating_factors.csv")
DERATING_AUCTION = "2030-31 (T-4)"
MODEL_YEAR = 2050
PRICE_YEAR = 2025

#: Calendar 2050 pays half of each delivery year it spans (November-to-February levy).
CALENDAR_2050_WEIGHTS = {"2049/2050": 0.5, "2050/2051": 0.5}


def calendar_weights(model_year: int) -> dict[str, float]:
    """Delivery-year weights of a calendar year: half of each delivery year it spans.

    Args:
        model_year: The calendar year.

    Returns:
        ``{"Y-1/Y": 0.5, "Y/Y+1": 0.5}``; for 2030 this is ``capacity_market.CALENDAR_2030_WEIGHTS``.
    """
    y = int(model_year)
    return {f"{y - 1}/{y}": 0.5, f"{y}/{y + 1}": 0.5}


# ── fleet carriers ───────────────────────────────────────────────────────────

#: FES ES1 carrier -> (CM rate key of ``cm_revenue_by_technology``, NESO de-rating class,
#: economic life in years for the stock-flow, 2050 support scheme and its term in years).
#: Lives are the Y2 table's (``technology_economic_lives.csv``); interconnectors 40 years, declared.
CARRIERS = {
    "wind_offshore": ("wind_offshore", "Offshore Wind", 30, "CfD", 20),
    "wind_onshore": ("wind_onshore", "Onshore Wind", 35, "CfD", 20),
    "solar_pv": ("solar_pv", "Solar Photovoltaic", 38, "CfD", 20),
    "marine": ("marine", "Offshore Wind", 25, "CfD", 15),
    "large_hydro": ("large_hydro", "Hydro", 41, "CfD", 15),
    # Every 2050 nuclear unit is supported: HPC on its CfD (35 years), Sizewell C on its RAB and
    # new large nuclear and SMRs on post-AR7 CfDs; the term covers the stock-flow's whole fleet.
    "nuclear": ("nuclear", "Nuclear", 60, "CfD", 99),
    "CCS Gas": ("CCS Gas", "Combined Cycle Gas Turbine (CCGT)", 25, "DPA", 15),
    "Hydrogen": ("Hydrogen", "Combined Cycle Gas Turbine (CCGT)", 25, "DPA", 15),
    "CCS Biomass": ("biomass", "Biomass", 25, "CfD", 15),
    "biomass": ("biomass", "Biomass", 25, "merchant", 0),
    "waste_to_energy": ("waste_to_energy", "Energy from Waste", 35, "merchant", 0),
    "geothermal": ("other", "Hydro", 25, "merchant", 0),
    "gas": ("CCGT", "Combined Cycle Gas Turbine (CCGT)", 25, "merchant", 0),
    "CHP": ("CHP", "Combined Heat and Power (CHP)", 25, "merchant", 0),
    "oil": ("oil", "Oil-fired steam generators", 25, "merchant", 0),
    "Battery": ("battery", "storage", 15, "merchant", 0),
    "LAES": ("long_duration_storage", "storage", 25, "cap_and_floor", 25),
    "CAES": ("long_duration_storage", "storage", 25, "cap_and_floor", 25),
    "Pumped Storage Hydroelectricity": ("long_duration_storage", "storage", 41, "cap_and_floor", 25),
    "interconnector": ("interconnector", "interconnector", 40, "cap_and_floor", 25),
}
#: Support schemes whose units cannot hold a CM agreement while the scheme is live.
NON_CM_SCHEMES = {"CfD", "DPA", "RAB", "RO", "ROC"}
#: Storage durations (hours) where FES's energy/power is not usable (declared; decision for Y6).
DECLARED_DURATION_HOURS = {"CAES": 6.0}
DECLARED_DURATION_NOTE = {
    "CAES": "FES ES1 gives 4 GW with 4 GWh (1 h, about 1,700 cycles a year), which the Y6 research calls a "
            "missing factor; the FES 2025 Economics shortest compressed-air duration, 6 h, is declared until "
            "Y6 fixes the build",
}


# ── FES trajectory (one-off extraction) ──────────────────────────────────────

def extract_fes_trajectory(workbook: str | Path | None = None, pathway: str = "Holistic Transition",
                           years=range(2024, MODEL_YEAR + 1)) -> pd.DataFrame:
    """Capacity by carrier and year from FES 2025 ES1, plus the FLX1 and ED1 rows the CM needs.

    Run by hand (``python scripts/market/build_capacity_market.py --model-year 2050
    --extract-fleet``); the output is ``cm_2050_fleet_fes2025_ht.csv``, one row per
    quantity with a column per year and its workbook rows.

    Args:
        workbook: The FES 2025 Data Workbook.
        pathway: FES pathway.
        years: Years to extract.

    Returns:
        ``quantity, carrier, unit, <year columns>, source, source_page, status``.
    """
    from scripts.FES import fes2025_targets as fes

    workbook = workbook or fes.FES2025_WORKBOOK
    es1 = fes.read_es1(workbook)
    records: dict[tuple, dict] = {}
    rows_used: dict[tuple, set] = {}
    for year in years:
        classified = fes.classify_es1(es1, pathway, int(year))
        keep = classified[classified["role"].isin(["supply", "storage", "interconnector"])
                          & classified["quantity"].isin(["capacity_mw", "energy_gwh"])]
        for (carrier, quantity), group in keep.groupby(["carrier", "quantity"]):
            key = (quantity, carrier)
            records.setdefault(key, {})[str(year)] = float(group["value_std"].sum())
            rows_used.setdefault(key, set()).update(int(r) for r in group["excel_row"])
        # SMRs apart from large nuclear (the support book separates them; the CM does not).
        smr = keep[keep["carrier"].eq("nuclear") & keep["subtype"].eq("Nuclear - Small")
                   & keep["quantity"].eq("capacity_mw")]
        records.setdefault(("capacity_mw", "nuclear_smr"), {})[str(year)] = float(smr["value_std"].sum())
        rows_used.setdefault(("capacity_mw", "nuclear_smr"), set()).update(int(r) for r in smr["excel_row"])
    out = []
    for (quantity, carrier), by_year in sorted(records.items()):
        out.append({"quantity": quantity, "carrier": carrier, "unit": "MW" if quantity == "capacity_mw" else "GWh",
                    **{str(y): by_year.get(str(y), 0.0) for y in years},
                    "source": f"NESO FES 2025 Data Workbook V005, sheet ES1, {pathway}",
                    "source_page": "ES1 rows " + ",".join(str(r) for r in sorted(rows_used[(quantity, carrier)])),
                    "status": "extracted"})
    flx1 = fes.read_flx1(workbook)
    flx = flx1[flx1["Pathway"].map(fes._norm).eq(pathway)]
    for detail, carrier in (("V2G potential at peak", "V2G"), ("V2G maximum potential", "V2G_maximum"),
                            ("Residential Appliance DSR impact at peak", "DSR_residential"),
                            ("I&C DSR impact at peak", "DSR_industrial_commercial"),
                            ("Smart charging impact at peak", "smart_charging")):
        row = flx[flx["Detail"].map(fes._norm).eq(detail)]
        if len(row) != 1:
            raise ValueError(f"FLX1 {detail!r}: expected one row, found {len(row)}")
        row = row.iloc[0]
        out.append({"quantity": "at_peak_mw", "carrier": carrier, "unit": "MW",
                    **{str(y): abs(float(row[str(y)] or 0.0)) * 1e3 for y in years},
                    "source": f"NESO FES 2025 Data Workbook V005, sheet FLX1, {pathway}, '{detail}' (GW, sign dropped)",
                    "source_page": f"FLX1 row {int(row['excel_row'])}", "status": "extracted"})
    ed1 = fes.read_ed1(workbook)
    label = "GBFES Peak Customer Demand: Total Consumption plus Losses"
    peak = ed1[ed1["Pathway"].map(fes._norm).eq(pathway)
               & ed1["Peak/ Annual/ Minimum"].map(fes._norm).str.lower().eq("peak")
               & ed1["Data item"].map(fes._norm).eq(label)]
    if len(peak) != 1:
        raise ValueError(f"ED1 {label!r}: expected one peak row, found {len(peak)}")
    peak = peak.iloc[0]
    peak_row = int(peak["excel_row"])
    peaks = {str(y): fes.to_mw(float(peak[str(y)]), fes._norm(peak["Unit"])) for y in years}
    out.append({"quantity": "acs_peak_mw", "carrier": "demand", "unit": "MW", **peaks,
                "source": f"NESO FES 2025 Data Workbook V005, sheet ED1, {pathway}, '{label}' (ACS, excludes "
                          "electrolysis, after smart charging)",
                "source_page": f"ED1 row {peak_row}", "status": "extracted"})
    return pd.DataFrame(out)


def read_fleet_trajectory(path=FLEET_FILE) -> pd.DataFrame:
    """The extracted FES trajectory, validated for provenance.

    Args:
        path: ``cm_2050_fleet_fes2025_ht.csv``.

    Returns:
        The table.
    """
    table = pd.read_csv(path)
    for column in ("source", "source_page", "status"):
        if table[column].isna().any():
            raise ValueError(f"{path}: rows without {column}")
    return table


def _series(table: pd.DataFrame, quantity: str, carrier: str) -> pd.Series:
    row = table[table["quantity"].eq(quantity) & table["carrier"].eq(carrier)]
    if row.empty:
        return pd.Series(dtype=float)
    years = [c for c in table.columns if re.fullmatch(r"\d{4}", str(c))]
    return pd.Series({int(y): float(row.iloc[0][y]) for y in years})


# ── stock-flow vintages (a stand-in for the Y10 register) ────────────────────

def stock_flow_vintages(capacity: pd.Series, life: int) -> pd.Series:
    """Capacity alive in the last year by build year, from a net capacity path.

    The first year's stock is spread evenly over build years ``first - life + 1 .. first``
    (unknown ages, declared). Each later year builds the net change plus what retires
    that year (units reaching ``life``); a fall in net capacity retires the oldest first.

    Args:
        capacity: MW by year, consecutive.
        life: Economic life in years.

    Returns:
        MW alive in the last year, indexed by build year.
    """
    years = sorted(int(y) for y in capacity.index)
    first, last = years[0], years[-1]
    stock = {y: float(capacity[first]) / life for y in range(first - life + 1, first + 1)}
    for year in years[1:]:
        for built in [b for b in stock if year - b >= life]:
            stock.pop(built)
        alive = sum(stock.values())
        change = float(capacity[year]) - alive
        if change >= 0:
            stock[year] = stock.get(year, 0.0) + change
        else:
            for built in sorted(stock):
                take = min(stock[built], -change)
                stock[built] -= take
                change += take
                if change >= -1e-9:
                    break
    out = pd.Series(stock, dtype=float)
    out = out[out > 1e-9].sort_index()
    if not np.isclose(out.sum(), float(capacity[last]), rtol=1e-9, atol=1e-6):
        raise ValueError("stock-flow vintages do not add up to the last year's capacity")
    return out


def fleet_2050(trajectory: pd.DataFrame | None = None, model_year: int = MODEL_YEAR) -> pd.DataFrame:
    """The 2050 fleet by carrier and build year, from the FES trajectory.

    Args:
        trajectory: :func:`read_fleet_trajectory` output.
        model_year: The snapshot year.

    Returns:
        ``carrier, build_year, capacity_mw, energy_mwh, duration_hours``.
    """
    trajectory = read_fleet_trajectory() if trajectory is None else trajectory
    rows = []
    for carrier, spec in CARRIERS.items():
        cap = _series(trajectory, "capacity_mw", carrier)
        if cap.empty or cap.get(model_year, 0.0) <= 0:
            continue
        cap = cap[cap.index <= model_year]
        vintages = stock_flow_vintages(cap, int(spec[2]))
        energy = _series(trajectory, "energy_gwh", carrier)
        if carrier in DECLARED_DURATION_HOURS:
            duration = DECLARED_DURATION_HOURS[carrier]
        elif not energy.empty and cap[model_year] > 0:
            duration = float(energy.get(model_year, 0.0)) * 1e3 / float(cap[model_year])
        else:
            duration = np.nan
        for built, mw in vintages.items():
            rows.append({"carrier": carrier, "build_year": int(built), "capacity_mw": float(mw),
                         "duration_hours": duration,
                         "energy_mwh": float(mw) * duration if duration == duration else np.nan})
    return pd.DataFrame(rows)


#: Task SETTLE2050: the vintaged register's carriers -> :data:`CARRIERS` keys, where they differ
#: (the register is the HT50 network's: hydrogen turbines are links, imports are EU_market
#: generators at the cables' rating, and biogas engines sit beside waste).
REGISTER_CARRIER_MAP = {"H2_turbine": "Hydrogen", "EU_market": "interconnector", "biogas": "waste_to_energy",
                        "Domestic Battery": "Battery"}


def fleet_from_register(register: pd.DataFrame, carrier_map: Mapping[str, str] | None = None,
                        default_build_year: int | None = None) -> pd.DataFrame:
    """The Y10 vintaged register as the CM fleet: one fleet row per register row.

    Task SETTLE2050 (the reachability audit: two stock-flow implementations): the register
    carries the stock-flow's own build years (``fleet_vintages``) and each row's 2050
    contract, so the CM can clear on it instead of :func:`stock_flow_vintages`. A row's
    ``cfd_live_share`` is carried as ``support_live_share`` (the register's own share of
    the year a CfD or DPA is live), which :func:`classify_fleet` then uses in place of the
    carrier's declared term.

    Args:
        register: Rows with ``carrier``, ``capacity_mw``, ``commissioning_year`` (or
            ``build_year``) and optionally ``energy_capacity_mwh``, ``support_type`` and
            ``cfd_live_share``.
        carrier_map: Register carrier -> :data:`CARRIERS` key, where names differ
            (default :data:`REGISTER_CARRIER_MAP`).
        default_build_year: The build year of a row the register gives none (the cables'
            import generators and consumer-owned batteries are outside the stock-flow);
            None refuses such a row.

    Returns:
        :func:`fleet_2050`-shaped rows, with ``support_type`` carried through when given.
    """
    carrier_map = dict(REGISTER_CARRIER_MAP if carrier_map is None else carrier_map)
    years = pd.to_numeric(register.get("build_year", register.get("commissioning_year")), errors="coerce")
    if years.isna().any():
        if default_build_year is None:
            missing = sorted(register.loc[years.isna(), "carrier"].astype(str).unique())
            raise ValueError(f"register rows without a build year ({missing}); pass default_build_year")
        years = years.fillna(float(default_build_year))
    out = pd.DataFrame({
        "carrier": register["carrier"].astype(str).map(lambda c: carrier_map.get(c, c)),
        "build_year": years.astype("Int64"),
        "capacity_mw": pd.to_numeric(register["capacity_mw"], errors="coerce").fillna(0.0),
    })
    energy = pd.to_numeric(register.get("energy_capacity_mwh"), errors="coerce") if "energy_capacity_mwh" in register else None
    out["energy_mwh"] = energy if energy is not None else np.nan
    out["duration_hours"] = out["energy_mwh"] / out["capacity_mw"].where(out["capacity_mw"] > 0)
    if "support_type" in register:
        out["support_type"] = register["support_type"].astype(str).to_numpy()
    if "cfd_live_share" in register:
        out["support_live_share_given"] = pd.to_numeric(register["cfd_live_share"], errors="coerce").to_numpy()
    unknown = sorted(set(out["carrier"]) - set(CARRIERS))
    if unknown:
        raise ValueError(f"register carriers with no CM treatment: {unknown}; pass carrier_map")
    return out


# ── de-rating ────────────────────────────────────────────────────────────────

def read_derating(path=DERATING_FILE, auction: str = DERATING_AUCTION) -> dict:
    """NESO de-rating factors of one auction, keyed by technology class, plus storage by duration.

    Args:
        path: ``neso_cm_derating_factors.csv``.
        auction: Auction-name prefix.

    Returns:
        ``{"classes": {class: factor}, "storage": Series by duration hours, "interconnector": mean}``.
    """
    table = pd.read_csv(path, encoding="utf-8-sig")
    rows = table[table["Auction Name"].str.startswith(auction)]
    if rows.empty:
        raise ValueError(f"no de-rating factors for {auction}")
    classes = dict(zip(rows["Generating Technology Class"], rows["De Rating Factor"].astype(float)))
    storage = rows[rows["Primary Fuel Type"].eq("Storage")].copy()
    storage["hours"] = storage["Generating Technology Class"].str.extract(r"([\d.]+)h")[0].astype(float)
    interconnector = rows[rows["Primary Fuel Type"].eq("Interconnector")]["De Rating Factor"].astype(float)
    return {"classes": classes, "storage": storage.set_index("hours")["De Rating Factor"].astype(float).sort_index(),
            "interconnector": float(interconnector.mean()), "auction": auction}


def storage_factor(duration_hours: float, storage: pd.Series) -> float:
    """NESO's duration-limited storage factor, interpolated on hours and capped at the longest class."""
    hours = float(duration_hours)
    if not hours == hours or hours <= 0:
        raise ValueError(f"storage needs a positive duration, got {duration_hours!r}")
    return float(np.interp(min(hours, storage.index.max()), storage.index.to_numpy(), storage.to_numpy()))


def derating_factor(carrier: str, duration_hours: float, derating: dict, storage_efc_scale: float = 1.0) -> float:
    """The de-rating factor of a fleet carrier.

    Args:
        carrier: A :data:`CARRIERS` key.
        duration_hours: Storage duration (ignored for generators).
        derating: :func:`read_derating` output.
        storage_efc_scale: Multiplier on storage factors (saturation sensitivity).

    Returns:
        The factor.
    """
    klass = CARRIERS[carrier][1]
    if klass == "storage":
        return storage_factor(duration_hours, derating["storage"]) * float(storage_efc_scale)
    if klass == "interconnector":
        return derating["interconnector"]
    return float(derating["classes"][klass])


# ── settings ─────────────────────────────────────────────────────────────────

V2G_TREATMENTS = ("outside_cm", "in_auction", "not_counted")
REQUIREMENT_RULES = ("ratio", "additive")


@dataclass(frozen=True)
class Settings2050:
    """The declared choices of the 2050 auction (every default is the central case).

    ``v2g``: ``outside_cm`` counts V2G's de-rated contribution against the target and pays
    it nothing (consumer-owned, outside every ownership case); ``in_auction`` enters it as
    DSR; ``not_counted`` ignores it for security. ``fresh_new_build_scale`` scales the
    new-build pipeline as MR4's sensitivity does.
    """

    requirement_rule: str = "ratio"
    v2g: str = "outside_cm"
    storage_efc_scale: float = 1.0
    new_build_lead_years: int = 4
    fresh_new_build_scale: float = 1.0
    demand_curve_half_width_mw: float = 1500.0
    net_cone_gbp_per_kw_year: float = 49.0
    price_cap_gbp_per_kw_year: float = 75.0
    t1_share_of_target: float = 1000.0 / 40900.0
    notes: tuple = field(default=())

    def __post_init__(self):
        if self.v2g not in V2G_TREATMENTS:
            raise ValueError(f"v2g must be one of {V2G_TREATMENTS}")
        if self.requirement_rule not in REQUIREMENT_RULES:
            raise ValueError(f"requirement_rule must be one of {REQUIREMENT_RULES}")


# ── requirement ──────────────────────────────────────────────────────────────

def read_calibration(path=CALIBRATION_FILE) -> dict:
    """The 2030/31 requirement calibration rows, as ``{item: value}``, provenance checked."""
    table = pd.read_csv(path)
    for column in ("source", "status"):
        if table[column].isna().any():
            raise ValueError(f"{path}: rows without {column}")
    return dict(zip(table["item"], table["value"].astype(float)))


def non_cm_derated_2030(assets: pd.DataFrame, derating: dict) -> dict:
    """De-rated capacity outside the 2030/31 CM, from the CP30 2030 register's asset inputs.

    CfD, RO and DPA units (ineligible) and merchant wind and solar (which hold almost no
    agreements) at the 2030/31 factors. Run once to write the calibration file; the test
    recomputes it where the 2030 asset inputs exist.

    Args:
        assets: ``resources/public_ownership/cp30_2030_cp30_network_asset_inputs.csv``.
        derating: :func:`read_derating` output.

    Returns:
        ``{"supported_mw": ..., "merchant_renewable_mw": ...}`` de-rated.
    """
    factor = {"wind_offshore": "Offshore Wind", "wind_onshore": "Onshore Wind", "solar_pv": "Solar Photovoltaic",
              "nuclear": "Nuclear", "biomass": "Biomass", "CCS Biomass": "Biomass", "waste_to_energy": "Energy from Waste",
              "CCS Gas": "Combined Cycle Gas Turbine (CCGT)", "marine": "Offshore Wind"}
    gens = assets[assets["component_type"].eq("Generator") & assets["carrier"].isin(factor)]
    derated = gens["capacity_mw"].astype(float) * gens["carrier"].map(lambda c: derating["classes"][factor[c]])
    supported = gens["support_type"].astype(str).isin(sorted(NON_CM_SCHEMES))
    merchant_res = gens["support_type"].astype(str).eq("merchant") & gens["carrier"].isin(
        ["wind_offshore", "wind_onshore", "solar_pv"])
    return {"supported_mw": float(derated[supported].sum()), "merchant_renewable_mw": float(derated[merchant_res].sum())}


def requirement_mw(calibration: dict, peak_mw: float, rule: str = "ratio") -> dict:
    """De-rated capacity the 2050 system holds, calibrated on 2030/31.

    Args:
        calibration: :func:`read_calibration` output.
        peak_mw: ACS peak of the model year.
        rule: ``ratio`` (capacity per GW of peak) or ``additive`` (the 2030 margin in MW).

    Returns:
        ``requirement_mw`` and the calibration it rests on.
    """
    held_2030 = (calibration["cm_contract_book_2030_31_derated_mw"] + calibration["cm_t4_2030_31_modelled_mw"]
                 + calibration["cm_t1_2030_31_mw"] + calibration["non_cm_supported_derated_2030_mw"]
                 + calibration["non_cm_merchant_renewable_derated_2030_mw"])
    peak_2030 = calibration["acs_peak_2030_mw"]
    ratio = held_2030 / peak_2030
    margin = held_2030 - peak_2030
    value = ratio * peak_mw if rule == "ratio" else peak_mw + margin
    return {"requirement_mw": float(value), "held_2030_31_mw": float(held_2030), "acs_peak_2030_mw": float(peak_2030),
            "ratio": float(ratio), "margin_mw": float(margin), "rule": rule, "acs_peak_mw": float(peak_mw)}


# ── supply ───────────────────────────────────────────────────────────────────

def live_fraction(build_year: float, term_years: int, model_year: int = MODEL_YEAR) -> float:
    """Expected share of calendar ``model_year`` a contract is still live, expiries spread evenly.

    A contract starts evenly through the year beginning ``build_year`` (a calendar
    year, or a delivery year's 1 April start such as 2030.25) and runs ``term_years``,
    so it expires evenly through the year beginning ``build_year + term_years``. The
    result is the expected share of the model year before expiry. The start side is
    not counted: a plant in the snapshot fleet runs the whole model year, so a
    contract that starts during it is taken as live from 1 January (declared).

    Args:
        build_year: Start of the one-year commissioning window.
        term_years: Contract term; zero is never live.
        model_year: The calendar year settled.

    Returns:
        Between 0 and 1.
    """
    if term_years <= 0:
        return 0.0
    # Expiry e = build_year + T + u, u uniform on [0, 1); the live share of [Y, Y+1) is
    # clip(e - Y, 0, 1), whose mean over u is exact through the integral of clip,
    # F(t) = 0 (t <= 0), t^2/2 (0 < t <= 1), t - 1/2 (t > 1).
    def integral(t: float) -> float:
        return 0.0 if t <= 0 else (t * t / 2.0 if t <= 1 else t - 0.5)

    end = float(build_year) + float(term_years) - float(model_year)
    return float(min(max(integral(end + 1.0) - integral(end), 0.0), 1.0))


def classify_fleet(fleet: pd.DataFrame, derating: dict, settings: Settings2050,
                   model_year: int = MODEL_YEAR) -> pd.DataFrame:
    """Each fleet row split into its live-support share (outside the CM) and its CM-eligible share.

    Args:
        fleet: :func:`fleet_2050` or :func:`fleet_from_register` rows.
        derating: :func:`read_derating` output.
        settings: The auction settings.
        model_year: The snapshot year.

    Returns:
        The fleet with ``derating_factor``, ``derated_mw``, ``support_scheme``,
        ``support_live_share``, ``non_cm_derated_mw``, ``eligible_derated_mw`` and ``category``.
    """
    out = fleet.copy()
    out["derating_factor"] = [derating_factor(c, d, derating, settings.storage_efc_scale)
                              for c, d in zip(out["carrier"], out["duration_hours"])]
    out["derated_mw"] = out["capacity_mw"] * out["derating_factor"]
    scheme = out["carrier"].map(lambda c: CARRIERS[c][3])
    term = out["carrier"].map(lambda c: CARRIERS[c][4])
    if "support_type" in out:
        given = out["support_type"].replace({"nan": np.nan, "": np.nan})
        scheme = given.fillna(scheme)
    out["support_scheme"] = scheme
    out["support_live_share"] = [
        live_fraction(int(b), int(t), model_year) if s in NON_CM_SCHEMES else 0.0
        for b, t, s in zip(out["build_year"], term, scheme)]
    if "support_live_share_given" in out:
        # A register fleet (fleet_from_register): the row's own live share where its label gives one.
        given = pd.to_numeric(out["support_live_share_given"], errors="coerce")
        out["support_live_share"] = np.where(given.notna() & scheme.isin(list(NON_CM_SCHEMES)), given,
                                             out["support_live_share"])
    out["non_cm_derated_mw"] = out["derated_mw"] * out["support_live_share"]
    out["eligible_derated_mw"] = out["derated_mw"] - out["non_cm_derated_mw"]
    lead = int(settings.new_build_lead_years)
    new = out["build_year"].astype(int) > model_year - lead
    klass = out["carrier"].map(lambda c: CARRIERS[c][1])
    out["category"] = np.where(klass.eq("interconnector"), "interconnector", np.where(new, "new_build", "existing"))
    out["rate_key"] = out["carrier"].map(lambda c: CARRIERS[c][0])
    return out


def flexibility_rows(trajectory: pd.DataFrame, derating: dict, settings: Settings2050,
                     model_year: int = MODEL_YEAR) -> pd.DataFrame:
    """DSR and V2G at peak as supply or non-CM rows (FES FLX1), de-rated at NESO's DSR factor.

    Smart charging is already inside the ED1 peak, so it is not counted again.
    """
    factor = float(derating["classes"]["DSR"])
    rows = []
    for carrier in ("DSR_residential", "DSR_industrial_commercial", "V2G"):
        mw = float(_series(trajectory, "at_peak_mw", carrier).get(model_year, 0.0))
        scale = settings.storage_efc_scale if carrier == "V2G" else 1.0
        derated = mw * factor * scale
        if carrier == "V2G" and settings.v2g != "in_auction":
            non_cm = derated if settings.v2g == "outside_cm" else 0.0
            eligible = 0.0
        else:
            non_cm, eligible = 0.0, derated
        rows.append({"carrier": carrier, "build_year": model_year, "capacity_mw": mw, "derating_factor": factor * scale,
                     "derated_mw": derated, "support_scheme": "consumer_flexibility", "support_live_share": 0.0,
                     "non_cm_derated_mw": non_cm, "eligible_derated_mw": eligible, "category": "dsr", "rate_key": "dsr"})
    return pd.DataFrame(rows)


# ── the auction ──────────────────────────────────────────────────────────────

def demand_curve(target_mw: float, settings: Settings2050) -> cm.DemandCurve:
    """The 2050 demand curve: the MR4 shape (cap at target - w, net CONE at target, zero at target + w)."""
    w = float(settings.demand_curve_half_width_mw)
    return cm.DemandCurve(float(target_mw), float(target_mw) - w, float(target_mw) + w,
                          float(settings.net_cone_gbp_per_kw_year), float(settings.price_cap_gbp_per_kw_year))


def read_distributions(model: Mapping) -> dict:
    """MR4's fitted bid distributions (``cm_auction_model.json``) as ``{category: (floor, shape)}``."""
    return {c: (float(d["floor_share"]), float(d["shape"])) for c, d in model["distributions"].items()}


def clear_2050(classified: pd.DataFrame, requirement: dict, distributions: dict, settings: Settings2050) -> dict:
    """Clear the 2050 T-4 on its demand curve.

    Args:
        classified: :func:`classify_fleet` rows plus :func:`flexibility_rows`.
        requirement: :func:`requirement_mw` output.
        distributions: :func:`read_distributions` output.
        settings: The auction settings.

    Returns:
        Target, T-1 volume, supply by category, clearing price (auction money, 2025/26),
        awarded MW and the award by category.
    """
    non_cm = float(classified["non_cm_derated_mw"].sum())
    procure = max(requirement["requirement_mw"] - non_cm, 0.0)
    t1 = procure * float(settings.t1_share_of_target) / (1.0 + float(settings.t1_share_of_target))
    target = procure - t1
    supply = classified.groupby("category")["eligible_derated_mw"].sum()
    if "new_build" in supply:
        supply["new_build"] *= float(settings.fresh_new_build_scale)
    supply = supply[supply > 0]
    missing = sorted(set(supply.index) - set(distributions))
    if missing:
        raise ValueError(f"no bid distribution for categories {missing}")
    curve = demand_curve(target, settings)
    if target <= 0:
        result = {"price": 0.0, "awarded_mw": 0.0, "supply_at_cap_mw": float(supply.sum())}
    else:
        result = cm.clear_auction(supply, distributions, curve)
    awarded = (cm.awarded_by_category(supply, distributions, result["price"], result["awarded_mw"], curve.price_cap)
               if result["awarded_mw"] > 0 else supply * 0.0)
    return {"non_cm_derated_mw": non_cm, "procured_mw": procure, "target_mw": target, "t1_mw": t1,
            "supply_mw": {k: float(v) for k, v in supply.items()}, "supply_total_mw": float(supply.sum()),
            "price_gbp_per_kw_year_auction_money": float(result["price"]),
            "awarded_mw": float(result["awarded_mw"]), "shortfall": bool(result.get("shortfall", False)),
            "awarded_by_category_mw": {k: float(v) for k, v in awarded.items()},
            "demand_curve": {"min_at_cap_mw": curve.min_at_cap_mw, "target_mw": curve.target_mw,
                             "max_at_zero_mw": curve.max_at_zero_mw, "net_cone": curve.net_cone,
                             "price_cap": curve.price_cap}}


def revenue_by_carrier(classified: pd.DataFrame, cleared: dict, indexation: float,
                       model_year: int = MODEL_YEAR) -> pd.DataFrame:
    """Calendar-year CM revenue by CM rate key, in the 2030 table's shape (item B7 identity).

    Each category's award (and the T-1, pro rata to the T-4 award) is split across rows
    by their eligible de-rated capacity; contracted connection MW is award / factor. Both
    delivery years are the same auction, so ``calendar_<year>_revenue_gbp`` is one year's
    payment and the table adds up to the calendar total exactly.

    Args:
        classified: The auction's rows.
        cleared: :func:`clear_2050` output.
        indexation: Auction money (2025/26) to report money (2025).
        model_year: The calendar year.

    Returns:
        One row per rate key.
    """
    price = cleared["price_gbp_per_kw_year_auction_money"] * float(indexation)
    rows = classified[classified["eligible_derated_mw"] > 0].copy()
    by_cat = rows.groupby("category")["eligible_derated_mw"].transform("sum")
    award = rows["category"].map(cleared["awarded_by_category_mw"]).fillna(0.0)
    t4_total = float(sum(cleared["awarded_by_category_mw"].values()))
    t1_scale = 1.0 + (cleared["t1_mw"] / t4_total if t4_total > 0 else 0.0)
    rows["awarded_derated_mw"] = rows["eligible_derated_mw"] / by_cat * award * t1_scale
    rows["contracted_connection_mw"] = rows["awarded_derated_mw"] / rows["derating_factor"].where(rows["derating_factor"] > 0)
    rows["payment_gbp"] = rows["awarded_derated_mw"] * 1e3 * price
    label1 = fiscal_year_label(model_year - 1)
    label2 = fiscal_year_label(model_year)
    grouped = rows.groupby("rate_key").agg(derated_mw=("awarded_derated_mw", "sum"),
                                           connection_mw=("contracted_connection_mw", "sum"),
                                           payment_gbp=("payment_gbp", "sum")).reset_index()
    out = pd.DataFrame({
        "carrier": grouped["rate_key"],
        f"derated_mw_{label1}": grouped["derated_mw"],
        f"connection_mw_{label1}": grouped["connection_mw"],
        f"revenue_{label1}_gbp": grouped["payment_gbp"],
        f"revenue_{label2}_gbp": grouped["payment_gbp"],
        f"calendar_{model_year}_revenue_gbp": grouped["payment_gbp"],
    })
    out["gbp_per_kw_connection_year"] = out[f"calendar_{model_year}_revenue_gbp"] / (
        out[f"connection_mw_{label1}"].where(out[f"connection_mw_{label1}"] > 0) * 1e3)
    return out


CASES = {
    "central": {},
    "v2g_in_auction": {"v2g": "in_auction"},
    "v2g_not_counted": {"v2g": "not_counted"},
    "storage_efc_half": {"storage_efc_scale": 0.5},
    "additive_requirement": {"requirement_rule": "additive"},
    "no_fresh_new_build": {"fresh_new_build_scale": 0.0},
}


def run_2050(model: Mapping, cpi: pd.Series, *, trajectory: pd.DataFrame | None = None,
             calibration: dict | None = None, derating: dict | None = None, fleet: pd.DataFrame | None = None,
             settings: Settings2050 | None = None, model_year: int = MODEL_YEAR) -> dict:
    """The calendar-2050 CM: auction, bill total, revenue by carrier and the declared cases.

    Args:
        model: MR4's ``cm_auction_model.json`` (the fitted bid distributions).
        cpi: ONS CPI (:func:`capacity_market.read_cpi`).
        trajectory: The FES trajectory table.
        calibration: The 2030/31 calibration.
        derating: De-rating factors.
        fleet: A fleet (default: the FES stock-flow; the Y10 register through :func:`fleet_from_register`).
        settings: Central settings.
        model_year: The calendar year.

    Returns:
        ``{"summary": dict, "revenue": DataFrame, "classified": DataFrame}``.
    """
    trajectory = read_fleet_trajectory() if trajectory is None else trajectory
    calibration = read_calibration() if calibration is None else calibration
    derating = read_derating() if derating is None else derating
    fleet = fleet_2050(trajectory, model_year) if fleet is None else fleet
    settings = settings or Settings2050()
    distributions = read_distributions(model)
    peak = float(_series(trajectory, "acs_peak_mw", "demand")[model_year])
    indexation = float(cpi[PRICE_YEAR] / cm.cpi_financial_year(cpi, 2025))

    def one(s: Settings2050):
        classified = pd.concat([classify_fleet(fleet, derating, s, model_year),
                                flexibility_rows(trajectory, derating, s, model_year)], ignore_index=True)
        requirement = requirement_mw(calibration, peak, s.requirement_rule)
        cleared = clear_2050(classified, requirement, distributions, s)
        cost = (cleared["awarded_mw"] + cleared["t1_mw"]) * 1e3 * cleared["price_gbp_per_kw_year_auction_money"] * indexation
        return classified, requirement, cleared, cost

    classified, requirement, cleared, cost = one(settings)
    revenue = revenue_by_carrier(classified, cleared, indexation, model_year)
    total = float(revenue[f"calendar_{model_year}_revenue_gbp"].sum())
    if not np.isclose(total, cost, rtol=1e-9, atol=1.0):
        raise ValueError(f"CM revenue by carrier ({total:,.0f}) does not add up to the calendar cost ({cost:,.0f})")
    weights = calendar_weights(model_year)
    cases = {}
    for label, change in CASES.items():
        _, req, clr, cst = one(replace(settings, **change))
        cases[label] = {"changes": change, "requirement_mw": round(req["requirement_mw"], 1),
                        "target_mw": round(clr["target_mw"], 1), "supply_total_mw": round(clr["supply_total_mw"], 1),
                        "price_gbp_per_kw_year_auction_money": round(clr["price_gbp_per_kw_year_auction_money"], 2),
                        "awarded_mw": round(clr["awarded_mw"], 1), "shortfall": clr["shortfall"],
                        f"calendar_{model_year}_cost_gbp": float(cst)}
    by_group = classified.assign(group=np.where(classified["non_cm_derated_mw"] > 0, classified["support_scheme"], "cm_eligible"))
    summary = {
        "model_year": model_year, "price_year": PRICE_YEAR,
        f"calendar_{model_year}_cost_gbp": float(cost),
        "delivery_years": {dy: {"auction_cost_gbp": float(cost), "weight_in_calendar": w} for dy, w in weights.items()},
        "contract_book_gbp": 0.0,
        "contract_book_note": ("No agreement in the NESO register (23 Sep 2026) is live in 2049/50 or 2050/51: the longest "
                               "runs 15 years from 2030/31, to 2044/45. Both delivery years are the modelled auction, "
                               "cleared on the 2050 fleet; agreements from earlier (2046-2049) auctions are taken at "
                               "the same price (steady state, declared)."),
        "requirement": requirement, "auction": cleared,
        "price_gbp_per_kw_year_2025": cleared["price_gbp_per_kw_year_auction_money"] * indexation,
        "indexation_2025_26_to_2025": indexation,
        "derated_mw_by_group": {k: round(float(v), 1) for k, v in
                                by_group.groupby("group")["derated_mw"].sum().items()},
        "non_cm_derated_mw_by_carrier": {k: round(float(v), 1) for k, v in
                                         classified.groupby("carrier")["non_cm_derated_mw"].sum().items() if v > 0},
        "eligible_derated_mw_by_carrier": {k: round(float(v), 1) for k, v in
                                           classified.groupby("carrier")["eligible_derated_mw"].sum().items() if v > 0},
        "settings": {k: v for k, v in settings.__dict__.items() if k != "notes"},
        f"calendar_{model_year}_cost_by_case_gbp": {k: v[f"calendar_{model_year}_cost_gbp"] for k, v in cases.items()},
        "cases": cases,
        "derating_source": f"{DERATING_FILE} ({derating['auction']} factors held to 2050, declared)",
        "fleet_source": "FES 2025 HT ES1 capacity path 2024-2050, stock-flow vintages at the Y2 lives "
                        f"({FLEET_FILE}); a stand-in until the Y10 register is wired (fleet_from_register)",
        "declared": [
            "Requirement: de-rated capacity per GW of ACS peak held at its 2030/31 value (ratio); the 2030 margin in "
            "GW is the additive sensitivity.",
            "DPA plant (gas CCS, hydrogen power; decision 22), CfD plant (post-AR7 CfDs on 20-year terms, nuclear, "
            "BECCS) and RAB plant cannot hold a CM agreement while their support is live; CCS and hydrogen plant past "
            "their 15-year DPA, and renewables past their CfD, enter the auction as existing merchant plant.",
            "LDES on cap and floor and interconnectors enter as price-takers (MR4 existing and interconnector bid "
            "distributions); their CM revenue nets against their floor as in 2030.",
            "V2G (FES FLX1 potential at peak x NESO DSR factor) is consumer-owned and outside every ownership case: "
            "centrally it counts against the target and is paid nothing; the v2g_in_auction case enters it as DSR.",
            "Residential appliance and I&C DSR (FLX1 at peak) enter as DSR; smart charging is already inside the ED1 "
            "peak and is not counted.",
            "Net CONE 49 and price cap 75 GBP/kW/yr (2025/26 money) held in real terms; demand-curve width +/-1.5 GW "
            "as MR4; T-1 at the 2030/31 ratio (1.0 GW per 40.9 GW).",
            "New build is plant commissioned in the four years before delivery (2047-2050), bidding as MR4's new "
            "build; everything else bids as MR4's existing plant.",
            DECLARED_DURATION_NOTE["CAES"],
        ],
    }
    return {"summary": summary, "revenue": revenue, "classified": classified}
