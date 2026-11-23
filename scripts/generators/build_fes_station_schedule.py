"""Build one year's FES declared station schedule (any FES year).

FES places all of its nuclear, gas CCS, BECCS and transmission hydrogen power at "Direct"
connections with no site (in Holistic Transition 2050: Gen_BB020 Direct(NGET) 14,200 MW;
Gen_BB009 "CCGTs (non CHP)" Direct(NGET) 18,885 and Direct(SHETL) 3,320 MW, which in 2050 is
ES1's gas CCS; Gen_BB023 Direct 17,192 / 1,300 / 605 MW; BECCS inside Gen_BB010 Direct(NGET)).
The generic path would spread these over the "best-connected" buses of each licence area. This
script writes a declared schedule instead, following the CP30 nuclear and gas schedule pattern
(scripts/CP30/nuclear_schedule.py, gas_schedule.py): each row names a site and its bus with the
evidence, and the rows of every (carrier, licence area) sum exactly to the FES Direct total of
the year (the build refuses anything else).

The stations are data: ``data/generators/fes/station_list.csv``, one row per station with its
open and close years (``from_year``, ``to_year``; blank is open-ended) and their evidence
(``dates_source``). For a year, the stations open in it are allocated per (carrier, licence area):

* ``fixed``: the station's ``capacity_mw`` (a named project at its TEC or declared capacity);
* ``share``: the FES total less the fixed rows, spread in proportion to ``share_weight_mw`` (or
  ``capacity_mw`` when no weight is given), so the geography follows evidence and the total
  follows FES;
* ``residual``: one row that takes the FES total less the fixed rows;
* ``es1_capacity``: the year's ES1 capacity of ``es1_carrier``, taken out of the Direct row
  (``partial_of_direct``: BECCS inside the biomass block).

Fixed rows above the FES total are scaled down to it (with a warning). Gas CCS shares
Gen_BB009 with unabated CCGTs before 2050; the CCS total is the block's Direct capacity times
the year's ES1 CCS share (``fes_supply.es1_gas_ccs_share``), as the thermal build splits it.
``source`` and ``notes`` may hold the placeholders ``{direct_mw}``, ``{capacity_mw}``,
``{remainder_mw}``, ``{year}`` and ``{pathway_code}``, filled per year.

The output column ``basis`` says what kind of evidence a row is:

* ``named_project``: a project the NESO TEC register (2 September 2025) or a published
  programme names, at its connection site, at its TEC or declared capacity.
* ``declared_site``: a site this schedule chooses, with the reason (existing nuclear
  licensed sites for the SMR fleet).
* ``site_share``: the residual of the FES total spread over a declared set of sites in
  proportion to a declared weight (the site's existing capacity in the CP30 2030 gas
  schedule, or a TEC project's capacity).

Rule ``build_fes_station_schedule`` (``rules/fes_long_range.smk``) writes
``resources/FES/fes2025_<pathway>_<year>_station_schedule.csv``. By hand::

    python -m scripts.generators.build_fes_station_schedule --year 2050 \\
        --es1-lines resources/FES/fes2025_holistic_transition_2050_es1_lines.csv \\
        --output resources/FES/fes2025_holistic_transition_2050_station_schedule.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.generators.fes_supply import es1_gas_ccs_share  # noqa: E402

logger = logging.getLogger(__name__)

FES_DATA = "resources/FES/FES_2025_data.csv"
STATION_LIST = "data/generators/fes/station_list.csv"
PATHWAY = "Holistic Transition"
PATHWAY_CODE = "HT"

# (FES building block, FES carrier the Direct rows have after the thermal remap) per model carrier
SCHEDULED_BLOCKS = {
    "nuclear": ("Gen_BB020", "nuclear"),
    "CCS Gas": ("Gen_BB009", "CCS Gas"),        # 'CCGTs (non CHP)': gas CCS (and unabated CCGT before 2050)
    # The unabated share of the same block (user decision, 10 Nov 2026, for 2040): the newest existing CCGTs not
    # repowered with capture, by age; no rows in a year whose ES1 has no unabated CCGT (HT 2050).
    "CCGT": ("Gen_BB009", "CCGT"),
    "H2": ("Gen_BB023", "H2"),                  # 'Hydrogen fuelled generation'
    "CCS Biomass": ("Gen_BB010", "biomass"),    # BECCS sits inside BB010 Direct(NGET)
}
REGIONS = {"Direct(NGET)": "NGET", "Direct(SHETL)": "SHETL", "Direct(SPTL)": "SPTL"}
ALLOCATIONS = ("fixed", "share", "residual", "es1_capacity")
REQUIRED_LIST_COLUMNS = {
    "station_name", "carrier", "fes_region", "allocation", "capacity_mw", "share_weight_mw", "es1_carrier",
    "bus", "lat", "lon", "basis", "from_year", "to_year", "dates_source", "source", "status", "notes",
}
OUTPUT_COLUMNS = [
    "station_name", "carrier", "fes_source_carrier", "fes_building_block", "fes_region", "capacity_mw", "bus",
    "lat", "lon", "basis", "share_weight_mw", "partial_of_direct", "source", "status", "notes",
]


def load_station_list(path: str = STATION_LIST) -> pd.DataFrame:
    """Read and check the station list."""
    frame = pd.read_csv(path, keep_default_na=True)
    if missing := REQUIRED_LIST_COLUMNS.difference(frame.columns):
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    unknown = sorted(set(frame["carrier"]) - set(SCHEDULED_BLOCKS))
    if unknown:
        raise ValueError(f"{path}: carriers {unknown} are not scheduled; known {sorted(SCHEDULED_BLOCKS)}")
    bad = sorted(set(frame["allocation"]) - set(ALLOCATIONS))
    if bad:
        raise ValueError(f"{path}: allocations {bad}; expected {ALLOCATIONS}")
    if frame["station_name"].duplicated().any():
        raise ValueError(f"{path}: duplicate stations {frame.loc[frame['station_name'].duplicated(), 'station_name'].tolist()}")
    if frame["source"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError(f"{path}: every station needs a source")
    fixed = frame["allocation"].eq("fixed")
    if pd.to_numeric(frame.loc[fixed, "capacity_mw"], errors="coerce").isna().any():
        raise ValueError(f"{path}: fixed stations need capacity_mw")
    return frame


def stations_open(stations: pd.DataFrame, year: int) -> pd.DataFrame:
    """Stations open in ``year``: ``from_year <= year <= to_year`` (blank bounds are open)."""
    start = pd.to_numeric(stations["from_year"], errors="coerce")
    end = pd.to_numeric(stations["to_year"], errors="coerce")
    mask = (start.isna() | (start <= year)) & (end.isna() | (end >= year))
    return stations[mask]


def fes_direct_totals(fes_path: str, year: int, pathway: str = PATHWAY) -> Dict[tuple, float]:
    """FES Direct capacity of ``year`` by (scheduled carrier, licence area), MW."""
    fes = pd.read_csv(fes_path, low_memory=False)
    fes = fes[fes["FES Pathway"].eq(pathway)]
    column = str(int(year))
    if column not in fes.columns:
        raise ValueError(f"{fes_path} has no {column} column")
    totals = {}
    for carrier, (block, _) in SCHEDULED_BLOCKS.items():
        if carrier == "CCGT":
            continue   # the unabated share of the gas block: derived from it in build_schedule
        rows = fes[fes["Building Block ID Number"].eq(block) & fes["GSP"].isin(list(REGIONS))]
        for gsp, mw in rows.groupby("GSP")[column].sum().items():
            if mw > 0:
                totals[(carrier, REGIONS[gsp])] = float(mw)
    return totals


def _shares(total_mw: float, weights: pd.Series) -> List[float]:
    weights = weights.astype(float)
    shares = (weights / weights.sum() * total_mw).round(3)
    shares.iloc[-1] = round(total_mw - shares.iloc[:-1].sum(), 3)   # exact total
    return shares.tolist()


def _fill(text, **values) -> str:
    if not isinstance(text, str):
        return ""
    return text.format(**values) if "{" in text else text


def _es1_capacity(es1_lines: Optional[pd.DataFrame], carrier: str) -> float:
    if es1_lines is None:
        raise ValueError(f"an es1_capacity station ({carrier}) needs the year's ES1 lines")
    cap = es1_lines[es1_lines["quantity"].astype(str).eq("capacity_mw") & es1_lines["carrier"].astype(str).eq(carrier)]
    return float(cap["value_std"].sum())


def build_schedule(year: int, fes_path: str = FES_DATA, station_list: str = STATION_LIST,
                   es1_lines: Optional[pd.DataFrame] = None, pathway: str = PATHWAY,
                   pathway_code: str = PATHWAY_CODE) -> pd.DataFrame:
    """The declared station schedule of ``year``, conserving every FES Direct total.

    Args:
        year: Modelled year.
        fes_path: Normalised FES building blocks (``FES_2025_data.csv``).
        station_list: The station list.
        es1_lines: The year's classified ES1 lines (the gas CCS share and ``es1_capacity`` rows).
        pathway: FES pathway name; ``pathway_code`` its short code (for the text placeholders).
    """
    totals = fes_direct_totals(fes_path, year, pathway)
    if ("CCS Gas", "NGET") in totals or ("CCS Gas", "SHETL") in totals or ("CCS Gas", "SPTL") in totals:
        share = es1_gas_ccs_share(es1_lines)["share"] if es1_lines is not None else 1.0
        if es1_lines is None:
            logger.warning("No ES1 lines: Gen_BB009 is taken as all gas CCS (true of HT 2050 only)")
        for (carrier, region), block_mw in list(totals.items()):
            if carrier == "CCS Gas":
                totals[("CCS Gas", region)] = block_mw * share
                if share < 1.0:
                    totals[("CCGT", region)] = block_mw * (1.0 - share)
    stations = stations_open(load_station_list(station_list), int(year))
    rows: List[dict] = []
    for (carrier, region), group in stations.groupby(["carrier", "fes_region"], sort=False):
        direct = totals.get((carrier, region), 0.0)
        if direct <= 0:
            logger.info("%s %s: no FES Direct capacity in %s; %d stations unused", carrier, region, year, len(group))
            continue
        alloc = group["allocation"]
        capacity = pd.Series(float("nan"), index=group.index)
        if alloc.eq("es1_capacity").any():
            if not alloc.eq("es1_capacity").all() or len(group) != 1:
                raise ValueError(f"{carrier} {region}: an es1_capacity station must be the group's only row")
            es1_mw = _es1_capacity(es1_lines, str(group["es1_carrier"].iloc[0]))
            if es1_mw > direct + 0.5:
                raise ValueError(f"{carrier} {region}: ES1 {es1_mw:,.1f} MW exceeds FES Direct {direct:,.1f} MW")
            capacity[:] = es1_mw
        else:
            fixed = alloc.eq("fixed")
            fixed_mw = pd.to_numeric(group.loc[fixed, "capacity_mw"]).astype(float)
            residual = direct - float(fixed_mw.sum())
            if residual < -0.5:
                factor = direct / float(fixed_mw.sum())
                logger.warning("%s %s %s: fixed stations %.1f MW exceed FES Direct %.1f MW; scaled by %.4f",
                               carrier, region, year, fixed_mw.sum(), direct, factor)
                fixed_mw = fixed_mw * factor
                residual = 0.0
            capacity[fixed] = fixed_mw
            sharing = group[alloc.isin(["share", "residual"])]
            if len(sharing):
                if alloc.eq("residual").any() and len(sharing) != 1:
                    raise ValueError(f"{carrier} {region}: a residual station shares with no other row")
                weights = pd.to_numeric(sharing["share_weight_mw"], errors="coerce").fillna(
                    pd.to_numeric(sharing["capacity_mw"], errors="coerce")).fillna(1.0)
                capacity[sharing.index] = _shares(max(residual, 0.0), weights)
            elif abs(residual) > 0.5:
                raise ValueError(f"{carrier} {region} {year}: the open stations hold {fixed_mw.sum():,.1f} MW against "
                                 f"FES Direct {direct:,.1f} MW and none takes the rest; add a share or residual "
                                 f"station to {station_list}")
        scheduled_mw = float(capacity.sum())
        for idx, row in group.iterrows():
            mw = float(capacity[idx])
            if mw <= 1e-6:
                continue
            values = dict(direct_mw=direct, capacity_mw=mw, remainder_mw=direct - scheduled_mw, year=year,
                          pathway_code=pathway_code)
            rows.append({
                "station_name": row["station_name"], "carrier": carrier,
                "fes_source_carrier": SCHEDULED_BLOCKS[carrier][1], "fes_building_block": SCHEDULED_BLOCKS[carrier][0],
                "fes_region": region, "capacity_mw": round(mw, 3), "bus": row["bus"], "lat": row["lat"],
                "lon": row["lon"], "basis": row["basis"], "share_weight_mw": row["share_weight_mw"],
                "partial_of_direct": row["allocation"] == "es1_capacity",
                "source": _fill(row["source"], **values), "status": row["status"],
                "notes": _fill(row["notes"], **values),
            })
    schedule = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    check_schedule_totals(schedule, totals)
    return schedule


def check_schedule_totals(schedule: pd.DataFrame, totals: Dict[tuple, float], atol: float = 0.5) -> None:
    """Every full (carrier, region) group equals the FES Direct total; BECCS is a partial take."""
    for (carrier, region), group in schedule.groupby(["carrier", "fes_region"]):
        mw = group["capacity_mw"].sum()
        fes = totals.get((carrier, region), 0.0)
        if carrier == "CCS Biomass":
            if mw > fes + atol:
                raise ValueError(f"{carrier} {region}: schedule {mw:,.1f} MW exceeds FES Direct {fes:,.1f} MW")
        elif abs(mw - fes) > atol:
            raise ValueError(f"{carrier} {region}: schedule {mw:,.1f} MW != FES Direct {fes:,.1f} MW")
    for (carrier, region), fes in totals.items():
        if carrier != "CCS Biomass" and not ((schedule["carrier"] == carrier) & (schedule["fes_region"] == region)).any():
            raise ValueError(f"FES Direct {carrier} {region} ({fes:,.0f} MW) has no open station in the list")


def write_schedule(schedule: pd.DataFrame, output: str) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(output, index=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--pathway", default=PATHWAY)
    parser.add_argument("--pathway-code", default=PATHWAY_CODE)
    parser.add_argument("--fes-data", default=FES_DATA)
    parser.add_argument("--station-list", default=STATION_LIST)
    parser.add_argument("--es1-lines", default=None, help="the year's fes2025_<pathway>_<year>_es1_lines.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    es1 = pd.read_csv(args.es1_lines) if args.es1_lines else None
    schedule = build_schedule(args.year, args.fes_data, args.station_list, es1, args.pathway, args.pathway_code)
    write_schedule(schedule, args.output)
    print(schedule.groupby(["carrier", "fes_region"])["capacity_mw"].agg(["sum", "count"]))


def _run_snakemake(smk) -> None:
    from scripts.utilities.logging_config import setup_logging
    global logger
    logger = setup_logging(smk.log[0])
    schedule = build_schedule(int(smk.wildcards.year), smk.input.fes_data, smk.input.station_list,
                              pd.read_csv(smk.input.es1_lines), smk.params.pathway, smk.params.pathway_code)
    write_schedule(schedule, smk.output.schedule)
    logger.info("Station schedule %s %s: %d stations, %.1f MW", smk.params.pathway, smk.wildcards.year,
                len(schedule), schedule["capacity_mw"].sum())


if "snakemake" in globals():
    _run_snakemake(snakemake)  # noqa: F821
elif __name__ == "__main__":
    main()
