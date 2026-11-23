"""Write the declared CP30 gas station schedule (items MR1 and MR2).

One-off provenance script, run by hand when DUKES, the ELEXON calibration or
the CP30 network changes; its output ``data/market/CP30/gas_station_schedule_2030.csv``
is the versioned input the thermal integration reads (see
``scripts/CP30/gas_schedule.py``).

For every GB gas station in the processed DUKES table it records:

* the carrier (CCGT, OCGT or gas engine), with DUKES labels corrected where
  the station name says otherwise;
* the DUKES capacity and commissioning year (the retirement order);
* the connection bus: nearest bus to the DUKES 5.11 station coordinates, then
  the ETYS BMU-to-node correction the historical build uses, then a short list
  of declared overrides where that still lands on the wrong substation or on a
  busbar that cannot export the station;
* a heat-rate index from the 2024 per-station empirical marginal cost
  (``calibrate_daily_mc.py`` on ELEXON physical notifications and prices),
  relative to the capacity-weighted evidenced CCGT fleet. Stations without
  2024 evidence sit at the fleet level, index 1;
* the retention evidence the schedule keeps stations by (user decision,
  15 Oct 2026; ``scripts/CP30/gas_schedule.py``): the MW under live Capacity
  Market agreements for delivery years 2029/30 and 2030/31 (NESO CM register,
  agreements matched to stations by declared CMU ID), and whether FES keeps
  the station's capacity in 2030.

CCGTs DUKES 5.11 omits but the CM register contracts for 2030 (Sutton Bridge,
mothballed at the DUKES date; the new-build Eggborough CCGT) are declared rows
at their TEC register connection buses.

Usage::

    python scripts/CP30/build_gas_station_schedule.py \
        --network resources/network/CP30_FFR_2030_network_demand_renewables.pkl

    # Refresh only the declared stations and the CM / FES columns of the
    # existing schedule (no network needed; buses and heat rates unchanged):
    python scripts/CP30/build_gas_station_schedule.py --refresh-retention

    # Check (F25.9, rule verify_CP30_gas_station_schedule): the committed schedule
    # is what --refresh-retention would write from the CM register, and every FES
    # claim in FES_2030 holds in the processed FES data. Fails with the refresh
    # command on drift; writes a JSON stamp otherwise. Needs no network.
    python scripts/CP30/build_gas_station_schedule.py --check \
        --fes-data resources/FES/FES_2025_data.csv --stamp STAMP.json
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# The network imports (network_io, spatial_utils) are local to the full build, so
# --check and --refresh-retention need no network stack.

DUKES_TABLE = "resources/generators/DUKES/DUKES_2023_generators.csv"
COORDINATES = "data/generators/dukes_power_station_coordinates.csv"
EMPIRICAL_MC = "resources/marginal_costs/Historical_2024_ETYS_14zone_market_daily_empirical_mc.csv"
OUTPUT = "data/market/CP30/gas_station_schedule_2030.csv"
# Variable O&M taken off the empirical cost before forming the heat-rate
# index: the CCGT value in the future-cost cases (config/defaults/base/marginal_costs.yaml).
CCGT_VOM_GBP_PER_MWH = 2.0

# Not GB, or not a stand-alone gas station in the CP30 "Gas" line.
EXCLUDED = {
    "EP Ballylumford CCGT": "Northern Ireland (SEM), outside the GB model",
    "CoolkeeraghESB C30": "Northern Ireland (SEM), outside the GB model",
    "Fellside CHP": "gas CHP serving Sellafield; CP30 counts gas CHP in its own Gas CHP line",
}
# DUKES carrier labels the station name contradicts.
CARRIER_CORRECTIONS = {
    "Spalding Expansion OCGT": ("OCGT", "an open-cycle unit by name; DUKES lists it under CCGT"),
    "Peterborough Power Station PPL2 Gas Engines": ("gas_engine", "reciprocating gas engines by name; DUKES lists them under OCGT"),
}
# Declared connection buses where the coordinate-plus-BMU mapping is wrong.
BUS_OVERRIDES = {
    "Grain": ("GRAI41", "Grain 400 kV; nearest-bus mapping chose MEDW41, which would carry Grain and Medway (2,120 MW) on 1,390 MVA"),
    "Keadby": ("KEAD41", "Keadby 400 kV main busbar; KEAP41 exports only 500 MVA"),
    "Little Barford": ("EASO41", "Eaton Socon 400 kV, adjacent to the station; the BMU prefix mapping chose WALP41, 67 km away"),
    "EP SHB LTD": ("SHBA41", "South Humber Bank 400 kV, the station's own node (BMU T_SHBA)"),
    "Coryton Energy Company Ltd": ("COSO41", "Coryton South 400 kV, the station's own node (BMU T_COSO)"),
    # TEC register (02 Sep 2025): "Peterborough", River Nene Power, 245 MW,
    # connection site "Walpole 132kV Substation". Mapping had chosen YAXL41,
    # Norfolk 400 kV 95 km away (audit 27 Sep 2026).
    "Peterborough Power Station": ("WALP11", "Walpole 132 kV, the TEC register connection site (Peterborough, River Nene Power, 245 MW); mapping chose YAXL41, 95 km away"),
    "Peterborough Power Station PPL2 Gas Engines": ("WALP11", "Walpole 132 kV, the connection site of the host Peterborough station (TEC register); mapping chose YAXL41, 95 km away"),
}
DECLARED_HEAT_RATE_INDEX = {
    "Keadby 2": (0.90, "no 2024 ELEXON evidence (commissioned 2023); declared at the efficient end of the "
                       "evidenced fleet: a Siemens HL-class unit that SSE reports as GB's most efficient CCGT"),
}

# ── retention evidence (user decision, 15 Oct 2026) ──────────────────────────
CM_REGISTER = "data/market/capacity_market/cm_register_cmu_20260923.csv"
CM_REGISTER_LABEL = "NESO Capacity Market register (CMU export of 23 Sep 2026)"
TEC_REGISTER_LABEL = "NESO TEC register (02 Sep 2025, data/generators/tec-register-02-september-2025.csv)"
CM_DELIVERY_YEARS = {"cm_mw_2029_30": "2029/2030", "cm_mw_2030_31": "2030/2031"}
# Agreements covering at least this share of a station's DUKES capacity cover
# the whole station: the register's connection capacity and DUKES's installed
# capacity differ by a few per cent on the same units (Staythorpe 1,752 against
# 1,828 MW). Below it the agreements name only some units (Connah's Quay units
# 1 and 3; South Humber Bank phase 2), and the rest is uncovered.
WHOLE_STATION_SHARE = 0.9
# CMUs per schedule station, by name and site. The register's grid references
# are too coarse to match by distance (Seabank's falls 20 km from the station,
# Sutton Bridge's nearer Kings Lynn). Grain, Saltend and VPI (Immingham) hold
# their agreements as CHP; DUKES and CP30 count them in the Gas line.
CM_UNITS = {
    "Keadby": ["KEAD29"],
    "Rye House": ["VP25RH"],
    "Little Barford": ["LBA25R"],
    "Connahs Quay": ["CNQP71", "CNQP73"],
    "EP SHB LTD": ["SHB228"],
    "Kings Lynn": ["KLYNA2"],
    "Cottam Development Centre": ["CDCL51"],
    "Didcot B": ["DB525R", "DB625R"],
    "Rocksavage Power Company Ltd": ["IGRK25"],
    "Enfield": ["EECL51"],
    "Damhead Creek": ["VP25DC"],
    "Saltend Power Station": ["S1_R25", "S2_R25", "S3_R25"],
    "Seabank Power Ltd": ["SEAB_1", "SEAB_2"],
    "Shoreham": ["VP25SH"],
    "Great Yarmouth": ["GYA25R"],
    "Coryton Energy Company Ltd": ["IGCE25"],
    "Spalding Energy Company Ltd": ["IGSP25"],
    "VPI": ["IMMT12", "IMMT03"],
    "Marchwood Power": ["MRWD29"],
    "EP Langage Ltd": ["LAGA28"],
    "Severn Power Ltd": ["SV2910", "SV2920"],
    "Staythorpe": ["ST125R", "ST225R", "ST325R", "ST425R"],
    "Grain": ["GRAI46", "GRAI47", "GRAI48"],
    "Pembroke": ["P1125R", "P2125R", "P3125R", "P4125R", "P5125R"],
    "West Burton CCGT": ["WB1-25", "WB2-25", "WB3-25"],
    "Carrington Power": ["CRRG-1", "CRRG-2"],
    "Keadby 2": ["KEA223"],
    "Sutton Bridge": ["29SB01"],
    "Eggborough CCGT": ["EGGPS1"],
    # The register classes Hythe as CCGT; DUKES and the schedule as OCGT (peaker, never retired).
    "Hythe": ["HYTH25"],
}
# Live CCGT agreements for 2029/30 or 2030/31 that are not GB schedule stations.
CM_NOT_IN_SCHEDULE = {
    "EGENMX": "Manx Utilities, Pulrose aeroderivative CCGT, Isle of Man: outside the GB model",
}
# FES 2025 Holistic Transition, building block Gen_BB009 (CCGTs, non CHP), 2030
# (resources/FES/FES_2025_data.csv). "named": FES keeps capacity at the
# station's own GSP or licence-area Direct row; "dropped": that row is zero by
# 2030; every other transmission station sits in FES's Direct(NGET) total
# (19,975 MW in 2030), which names no station ("aggregate").
FES_2030 = {
    "Shoreham": ("named", "FES 2025 HT Gen_BB009 Bolney 460 MW in 2030"),
    "Great Yarmouth": ("named", "FES 2025 HT Gen_BB009 Norwich Main 420 MW in 2030 (zero from 2031)"),
    "Kings Lynn": ("named", "FES 2025 HT Gen_BB009 Walpole (_A) 395 MW in 2030"),
    "Peterhead": ("dropped", "FES 2025 HT Gen_BB009 Direct(SHETL) 1,180 MW to 2029, zero in 2030"),
    "Corby Power Station": ("dropped", "FES 2025 HT Gen_BB009 Grendon zero from 2025"),
}
FES_AGGREGATE = ("aggregate", "inside FES 2025 HT Gen_BB009 Direct(NGET), 19,975 MW in 2030, which names no station")
# The FES rows behind FES_2030 and FES_AGGREGATE, checked against the processed
# FES data by --check: GSP as the FES table names it, and its 2030 MW. A "named"
# or "aggregate" row must be positive, a "dropped" row zero.
FES_BLOCK = "Gen_BB009"
FES_PATHWAY = "Holistic Transition"
FES_2030_ROWS = {
    "Shoreham": ("Bolney", 460.0),
    "Great Yarmouth": ("Norwich Main", 420.0),
    "Kings Lynn": ("Walpole (_A)", 395.0),
    "Peterhead": ("Direct(SHETL)", 0.0),
    "Corby Power Station": ("Grendon", 0.0),
}
FES_AGGREGATE_ROW = ("Direct(NGET)", 19975.0)
# Absolute tolerance on the FES MW claims (the FES table carries three decimals).
FES_MW_TOLERANCE = 0.5
# The columns --refresh-retention re-derives; every other column is the full build's.
RETENTION_COLUMNS = (*CM_DELIVERY_YEARS, "cm_agreements", "fes_2030", "fes_2030_evidence")
# CCGTs that DUKES 5.11 (2025) omits but that hold Capacity Market agreements
# for 2030, at their TEC register connection substations.
DECLARED_STATIONS = [
    {
        "station_name": "Sutton Bridge", "carrier": "CCGT", "capacity_mw": 850.0, "commissioning_year": 1999,
        "bus": "WALP41",
        "bus_source": f"declared: Walpole 400 kV, the TEC register connection site (Sutton Bridge, 850 MW, built; {TEC_REGISTER_LABEL})",
        "grid_ref": "TF 480 199",
        "heat_rate_index": float("nan"),
        "efficiency_evidence": "no per-station evidence (absent from DUKES 5.11, so outside the 2024 ELEXON calibration); class efficiency",
        "source": f"{CM_REGISTER_LABEL}: CMU 29SB01, existing CCGT, 850 MW connection, 2029-30 T-4; {TEC_REGISTER_LABEL}; "
                  "commissioning year 1999 (data/generators/power_stations_locations.csv)",
        "notes": "not in DUKES 5.11 (2025): mothballed at the DUKES date; back under a 2029/30 Capacity Market agreement",
    },
    {
        "station_name": "Eggborough CCGT", "carrier": "CCGT", "capacity_mw": 850.0, "commissioning_year": 2026,
        "bus": "EGGB41",
        "bus_source": f"declared: Eggborough 400 kV, the TEC register connection site (Eggborough CCGT - OCGT - BESS, Eggborough Power Ltd; {TEC_REGISTER_LABEL})",
        "grid_ref": "SE 577 237",
        "heat_rate_index": 0.90,
        "efficiency_evidence": "no ELEXON evidence (new build); declared at the efficient end of the evidenced fleet, as Keadby 2: a 2020s new-build CCGT",
        "source": f"{CM_REGISTER_LABEL}: CMU EGGPS1, new-build CCGT, 850 MW connection, 2026-27 T-4, 15 years to 2040/41, "
                  "substantial completion expected June to September 2026; commissioning year 2026 from that milestone",
        "notes": "new build on the former Eggborough coal station site; not in DUKES 5.11 (2025)",
    },
]


def _dukes_gas_stations() -> pd.DataFrame:
    dukes = pd.read_csv(DUKES_TABLE)
    gas = dukes[dukes["fuel_type"].isin(["CCGT", "OCGT"])].copy()
    gas["station_name"] = gas["station_name"].str.strip()
    gas = gas[~gas["station_name"].isin(EXCLUDED)]
    gas["carrier"] = gas["fuel_type"]
    for name, (carrier, _) in CARRIER_CORRECTIONS.items():
        gas.loc[gas["station_name"].eq(name), "carrier"] = carrier
    gas["commissioning_year"] = pd.to_numeric(gas["status"], errors="coerce").astype("Int64")
    if gas["commissioning_year"].isna().any():
        raise ValueError(f"DUKES gas stations without a year: {gas.loc[gas['commissioning_year'].isna(), 'station_name'].tolist()}")

    coords = pd.read_csv(COORDINATES)
    coords["station_name"] = coords["station_name"].str.strip()
    coords = coords[coords["x_coordinate"] > 0].drop_duplicates("station_name")
    gas = gas.merge(coords[["station_name", "x_coordinate", "y_coordinate"]], on="station_name", how="left")
    if gas["x_coordinate"].isna().any():
        raise ValueError(f"No DUKES 5.11 coordinates for: {gas.loc[gas['x_coordinate'].isna(), 'station_name'].tolist()}")
    return gas


def _assign_buses(gas: pd.DataFrame, network) -> pd.DataFrame:
    from scripts.utilities.spatial_utils import apply_etys_bmu_mapping, map_sites_to_buses

    sites = gas.assign(x_coord=gas["x_coordinate"], y_coord=gas["y_coordinate"], bus=None,
                       site_name=gas["station_name"])
    sites = map_sites_to_buses(network, sites, method="nearest", lat_col="y_coord", lon_col="x_coord",
                               max_distance_km=100)
    nearest = sites["bus"].copy()
    sites = apply_etys_bmu_mapping(sites, network)
    sites["bus_source"] = np.where(
        sites["bus"].eq(nearest),
        "nearest bus to DUKES 5.11 coordinates",
        "nearest bus to DUKES 5.11 coordinates, moved by the ETYS BMU-to-node correction",
    )
    for name, (bus, reason) in BUS_OVERRIDES.items():
        mask = sites["station_name"].eq(name)
        if not mask.any():
            raise ValueError(f"Bus override for unknown station {name}")
        if bus not in network.buses.index:
            raise ValueError(f"Bus override {name} -> {bus}: bus absent from the network")
        sites.loc[mask, "bus"] = bus
        sites.loc[mask, "bus_source"] = f"declared override: {reason}"
    from pyproj import Transformer
    to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    sites["lon"], sites["lat"] = to_wgs84.transform(sites["x_coordinate"].values, sites["y_coordinate"].values)
    return sites


def _heat_rate_index(sites: pd.DataFrame) -> pd.DataFrame:
    empirical = pd.read_csv(EMPIRICAL_MC).groupby("generator")["empirical_mc"].median()
    sites = sites.copy()
    sites["empirical_mc_2024"] = sites["station_name"].map(empirical)
    evidenced = sites["carrier"].eq("CCGT") & sites["empirical_mc_2024"].notna()
    net = sites["empirical_mc_2024"] - CCGT_VOM_GBP_PER_MWH
    weights = sites.loc[evidenced, "capacity_mw"]
    fleet = float((net[evidenced] * weights).sum() / weights.sum())
    sites["heat_rate_index"] = np.where(evidenced, net / fleet, np.nan)
    sites["efficiency_evidence"] = np.where(
        evidenced,
        "2024 ELEXON empirical marginal cost (median of daily values, Historical_2024_ETYS_14zone_market) "
        f"relative to the capacity-weighted evidenced CCGT fleet ({fleet + CCGT_VOM_GBP_PER_MWH:.2f} GBP/MWh)",
        "no per-station evidence; class efficiency",
    )
    for name, (value, reason) in DECLARED_HEAT_RATE_INDEX.items():
        mask = sites["station_name"].eq(name)
        sites.loc[mask, "heat_rate_index"] = value
        sites.loc[mask, "efficiency_evidence"] = reason
    return sites


def _declared_stations() -> pd.DataFrame:
    from pyproj import Transformer

    from scripts.market.capacity_market import os_grid_to_easting_northing

    to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    rows = []
    for station in DECLARED_STATIONS:
        lon, lat = to_wgs84.transform(*os_grid_to_easting_northing(station["grid_ref"]))
        row = {key: value for key, value in station.items() if key != "grid_ref"}
        row.update({"lat": round(lat, 6), "lon": round(lon, 6), "empirical_mc_2024": float("nan")})
        rows.append(row)
    return pd.DataFrame(rows)


def cm_coverage(schedule: pd.DataFrame, register_path: str = CM_REGISTER) -> pd.DataFrame:
    """MW of each station under live CM agreements for 2029/30 and 2030/31, with the agreements named.

    Every live CCGT agreement for either year must belong to a schedule station
    (``CM_UNITS``) or be declared outside the GB model (``CM_NOT_IN_SCHEDULE``).
    """
    from scripts.market.capacity_market import read_agreements, read_register

    agreements = read_agreements(read_register(register_path))
    live = agreements[agreements["delivery_years"].map(lambda years: any(y in years for y in CM_DELIVERY_YEARS.values()))]
    ccgt = live[live["technology"].str.contains("CCGT", na=False)]
    mapped = {cmu for units in CM_UNITS.values() for cmu in units}
    loose = sorted(set(ccgt["cmu_id"]) - mapped - set(CM_NOT_IN_SCHEDULE))
    if loose:
        raise ValueError(f"Live CCGT CM agreements for 2029/30-2030/31 not assigned to a station: {loose}")
    unknown = sorted(set(CM_UNITS) - set(schedule["station_name"]))
    if unknown:
        raise ValueError(f"CM_UNITS names stations absent from the schedule: {unknown}")

    out = schedule.copy()
    for column in CM_DELIVERY_YEARS:
        out[column] = 0.0
    out["cm_agreements"] = ""
    for station, units in CM_UNITS.items():
        rows = live[live["cmu_id"].isin(units)]
        missing = sorted(set(units) - set(rows["cmu_id"]))
        if missing:
            raise ValueError(f"{station}: no live 2029/30-2030/31 agreement for CMU {missing}")
        mask = out["station_name"].eq(station)
        capacity = float(out.loc[mask, "capacity_mw"].iloc[0])
        for column, year in CM_DELIVERY_YEARS.items():
            mw = float(rows.loc[rows["delivery_years"].map(lambda years: year in years), "connection_mw"].sum())
            out.loc[mask, column] = capacity if mw >= WHOLE_STATION_SHARE * capacity else min(mw, capacity)
        out.loc[mask, "cm_agreements"] = "; ".join(
            f"{r.cmu_id} {r.connection_mw:,.1f} MW ({r.auction.split(' ')[0]} {r.auction.split('(')[1].split(')')[0]}, "
            f"{int(r.term_years)} y)" for r in rows.sort_values("cmu_id").itertuples())
    return out


def fes_retention(schedule: pd.DataFrame) -> pd.DataFrame:
    """FES 2025 HT 2030 retention of each CCGT (``FES_2030``); blank for peakers."""
    out = schedule.copy()
    ccgt = out["carrier"].eq("CCGT")
    entries = out.loc[ccgt, "station_name"].map(lambda name: FES_2030.get(name, FES_AGGREGATE))
    out["fes_2030"] = ""
    out["fes_2030_evidence"] = ""
    out.loc[ccgt, "fes_2030"] = entries.str[0]
    out.loc[ccgt, "fes_2030_evidence"] = entries.str[1]
    return out


def add_retention_evidence(schedule: pd.DataFrame, register_path: str = CM_REGISTER) -> pd.DataFrame:
    """Add the declared stations (if absent) and the CM / FES retention columns."""
    declared = _declared_stations()
    schedule = schedule[~schedule["station_name"].isin(declared["station_name"])]
    schedule = pd.concat([schedule, declared[[c for c in declared.columns if c in schedule.columns]]],
                         ignore_index=True)
    schedule = fes_retention(cm_coverage(schedule, register_path))
    columns = [c for c in schedule.columns if c not in ("source", "notes")] + ["source", "notes"]
    return schedule[columns].sort_values(["carrier", "commissioning_year", "station_name"]).reset_index(drop=True)


def build(network_path: str, output: str = OUTPUT) -> pd.DataFrame:
    from scripts.utilities.network_io import load_network

    network = load_network(network_path)
    sites = _heat_rate_index(_assign_buses(_dukes_gas_stations(), network))
    absent = [s["bus"] for s in DECLARED_STATIONS if s["bus"] not in network.buses.index]
    if absent:
        raise ValueError(f"Declared station buses absent from the network: {absent}")
    notes = []
    for name in sites["station_name"]:
        parts = []
        if name in CARRIER_CORRECTIONS:
            parts.append(f"carrier corrected: {CARRIER_CORRECTIONS[name][1]}")
        notes.append("; ".join(parts))
    schedule = pd.DataFrame({
        "station_name": sites["station_name"],
        "carrier": sites["carrier"],
        "capacity_mw": sites["capacity_mw"],
        "commissioning_year": sites["commissioning_year"].astype(int),
        "bus": sites["bus"],
        "bus_source": sites["bus_source"],
        "lat": sites["lat"].round(6),
        "lon": sites["lon"].round(6),
        "empirical_mc_2024": sites["empirical_mc_2024"].round(3),
        "heat_rate_index": sites["heat_rate_index"].round(4),
        "efficiency_evidence": sites["efficiency_evidence"],
        "source": "DUKES 5.11 (2025) via " + DUKES_TABLE + "; coordinates " + COORDINATES,
        "notes": notes,
    })
    schedule = add_retention_evidence(schedule)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(output, index=False)
    return schedule


def _refreshed(schedule: pd.DataFrame, register_path: str = CM_REGISTER) -> pd.DataFrame:
    schedule = schedule.drop(columns=[c for c in RETENTION_COLUMNS if c in schedule.columns])
    return add_retention_evidence(schedule, register_path)


def refresh_retention(output: str = OUTPUT, register_path: str = CM_REGISTER) -> pd.DataFrame:
    """Re-derive the declared stations and the CM / FES columns of the existing schedule."""
    schedule = _refreshed(pd.read_csv(output, dtype={"notes": str}), register_path)
    schedule.to_csv(output, index=False)
    return schedule


def _as_written(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame as a CSV round trip reads it (what the committed file holds)."""
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)), dtype={"notes": str})


#: Station positions are re-derived from the site data on refresh and may move in the 5th-6th decimal place
#: with the geocoding source (about 2 m); 1e-4 degrees (about 11 m) moves no placement.
COORDINATE_ATOL_DEGREES = 1e-4


def schedule_drift(committed: pd.DataFrame, fresh: pd.DataFrame, atol: float = 1e-6) -> list[dict]:
    """Cells where ``fresh`` (what --refresh-retention would write) differs from ``committed``.

    Rows are matched on ``station_name``; a station or a column on one side only is drift.
    ``lat`` and ``lon`` are compared to ``COORDINATE_ATOL_DEGREES``, every other column exactly.
    """
    committed, fresh = _as_written(committed), _as_written(fresh)
    if committed["station_name"].duplicated().any() or fresh["station_name"].duplicated().any():
        raise ValueError("gas station schedule has duplicate station names")
    drift = []
    for column in sorted(set(committed.columns) ^ set(fresh.columns)):
        drift.append({"station_name": None, "column": column,
                      "committed": "present" if column in committed.columns else "absent",
                      "refreshed": "present" if column in fresh.columns else "absent"})
    in_committed, in_fresh = set(committed["station_name"]), set(fresh["station_name"])
    for name in sorted(in_committed ^ in_fresh):
        drift.append({"station_name": name, "column": None,
                      "committed": "present" if name in in_committed else "absent",
                      "refreshed": "present" if name in in_fresh else "absent"})
    common = [c for c in committed.columns if c in fresh.columns and c != "station_name"]
    names = sorted(in_committed & in_fresh)
    a = committed.set_index("station_name").loc[names, common]
    b = fresh.set_index("station_name").loc[names, common]
    for column in common:
        left, right = a[column], b[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            tolerance = COORDINATE_ATOL_DEGREES if column in ("lat", "lon") else atol
            same = np.isclose(left.to_numpy(float), right.to_numpy(float), rtol=0.0, atol=tolerance, equal_nan=True)
        else:
            same = left.fillna("").astype(str).to_numpy() == right.fillna("").astype(str).to_numpy()
        for name in np.asarray(names, dtype=object)[~same]:
            drift.append({"station_name": name, "column": column,
                          "committed": None if pd.isna(left[name]) else str(left[name]),
                          "refreshed": None if pd.isna(right[name]) else str(right[name])})
    return drift


def fes_claim_drift(fes_data: pd.DataFrame) -> list[dict]:
    """``FES_2030_ROWS`` / ``FES_AGGREGATE_ROW`` claims the processed FES table no longer bears out."""
    if set(FES_2030) != set(FES_2030_ROWS):
        raise ValueError(f"FES_2030 and FES_2030_ROWS name different stations: {sorted(set(FES_2030) ^ set(FES_2030_ROWS))}")
    block = fes_data[fes_data["Building Block ID Number"].eq(FES_BLOCK) & fes_data["FES Pathway"].eq(FES_PATHWAY)]
    if block.empty:
        raise ValueError(f"FES data has no {FES_BLOCK} rows for {FES_PATHWAY}")
    values = pd.to_numeric(block["2030"], errors="coerce").groupby(block["GSP"]).sum()
    claims = {name: (FES_2030[name][0], *FES_2030_ROWS[name]) for name in FES_2030}
    claims["(aggregate)"] = (FES_AGGREGATE[0], *FES_AGGREGATE_ROW)
    drift = []
    for name, (status, gsp, mw) in claims.items():
        actual = float(values[gsp]) if gsp in values.index else None
        if actual is None:
            problem = "GSP row absent"
        elif status == "dropped" and abs(actual) > FES_MW_TOLERANCE:
            problem = "declared dropped but FES keeps capacity"
        elif status != "dropped" and actual <= FES_MW_TOLERANCE:
            problem = f"declared {status} but the FES row is zero"
        elif abs(actual - mw) > FES_MW_TOLERANCE:
            problem = "2030 MW differs from the declared value"
        else:
            continue
        drift.append({"station_name": name, "fes_2030": status, "gsp": gsp, "declared_mw_2030": mw,
                      "fes_mw_2030": actual, "problem": problem})
    return drift


def check_retention(output: str = OUTPUT, register_path: str = CM_REGISTER, fes_path: str | None = None) -> dict:
    """Recompute the retention columns of the committed schedule and report every difference.

    Recomputes only what --refresh-retention writes (the declared stations and the CM /
    FES columns, from the CM register and ``FES_2030``) and, with ``fes_path``, checks
    the FES claims against the processed FES data. Buses and heat rates (the full
    build, which needs a network) are not recomputed.
    """
    committed = pd.read_csv(output, dtype={"notes": str})
    try:
        schedule_problems, error = schedule_drift(committed, _refreshed(committed, register_path)), None
    except ValueError as exc:   # an unassigned or lapsed CM agreement: CM_UNITS needs editing
        schedule_problems, error = [], str(exc)
    fes_problems = fes_claim_drift(pd.read_csv(fes_path, low_memory=False)) if fes_path else []
    return {
        "schedule": output, "cm_register": register_path, "fes_data": fes_path,
        "stations": int(len(committed)),
        "retention_error": error,
        "schedule_drift": schedule_problems,
        "fes_claim_drift": fes_problems,
        "ok": error is None and not schedule_problems and not fes_problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--network", help="CP30 network before thermal integration")
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--cm-register", default=CM_REGISTER, help="the NESO CM register (CMU export)")
    parser.add_argument("--refresh-retention", action="store_true",
                        help="only re-derive the declared stations and the CM / FES columns of --output")
    parser.add_argument("--check", action="store_true",
                        help="fail if --refresh-retention would change --output, or an FES claim no longer holds")
    parser.add_argument("--fes-data", help="with --check: the processed FES table (resources/FES/FES_2025_data.csv)")
    parser.add_argument("--stamp", help="with --check: the JSON stamp written when the schedule reproduces")
    args = parser.parse_args()
    if args.check:
        report = check_retention(args.output, args.cm_register, args.fes_data)
        print(f"{args.output}: {report['stations']} stations; {len(report['schedule_drift'])} drifted cells, "
              f"{len(report['fes_claim_drift'])} FES claims not borne out")
        if not report["ok"]:
            print(json.dumps({k: report[k] for k in ("retention_error", "schedule_drift", "fes_claim_drift")},
                             indent=1, default=str), file=sys.stderr)
            raise SystemExit(
                "The gas station schedule's retention columns no longer reproduce. Where an FES claim fails, update "
                "FES_2030 / FES_2030_ROWS; where a CM agreement is unassigned or lapsed, update CM_UNITS; then refresh "
                f"(python scripts/CP30/build_gas_station_schedule.py --refresh-retention --output {args.output} "
                f"--cm-register {args.cm_register}), review the diff and rebuild the CP30 thermal stage.")
        if args.stamp:
            Path(args.stamp).parent.mkdir(parents=True, exist_ok=True)
            Path(args.stamp).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        return
    if args.refresh_retention:
        schedule = refresh_retention(args.output, args.cm_register)
    elif args.network:
        schedule = build(args.network, args.output)
    else:
        parser.error("--network is required unless --refresh-retention is given")
    print(schedule.groupby("carrier")["capacity_mw"].agg(["count", "sum"]))
    print(f"Wrote {len(schedule)} stations to {args.output}; excluded: {sorted(EXCLUDED)}")


if __name__ == "__main__":
    main()
