"""Scenario-source registries, the FES raw/normalise split and the source adapters (year-generic plan, stages 1-3).

Tests that need the workflow's own outputs (``resources/FES/FES_*_data.csv``, ``resources/CP30/*.csv``)
read them from ``$PYPSA_GB_RESOURCES`` (default: this checkout's ``resources/``) and skip when absent.
Tests that need the cached NESO downloads skip when ``resources/sources/raw`` has not been filled
(``python -m scripts.sources.fes_raw``).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sources import contract, registry  # noqa: E402
from scripts.sources import fes_raw  # noqa: E402


def _load_legacy_oracle():
    # By path: tests/fixtures/__init__.py imports a module that is not in the repository.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "legacy_fes_gsp_cleaning", PROJECT_ROOT / "tests" / "fixtures" / "legacy_fes_gsp_cleaning.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.legacy_clean_gsp_names


legacy_clean_gsp_names = _load_legacy_oracle()

RESOURCES = Path(os.environ.get("PYPSA_GB_RESOURCES", PROJECT_ROOT / "resources"))
RAW_DIR = fes_raw.DEFAULT_RAW_DIR
FES_EDITIONS = (2020, 2021, 2022, 2023, 2024, 2025)
FES_WORKBOOKS = {
    2020: "data/FES/Data-workbook2020_v1_4.xlsx",
    2021: "data/FES/FES 2021 Data Workbook V08.xlsx",
    2022: "data/FES/Data-workbook2022_V006 (1).xlsx",
    2023: "data/FES/FES 2023 Data Workbook V003.xlsx",
    2024: "data/FES/Future Energy Scenarios 2024 Data Workbook_V006_0.xlsx",
    2025: "data/FES/Future Energy Scenarios 2025 Data Workbook V005_0.xlsx",
}
CP30_WORKBOOK = "data/CP30/Clean Power 2030 Data Workbook_V003.xlsx"
#: Data sheets whose pathway column the workbook test reads (chart sheets reuse the header words).
DATA_SHEETS = ("ES1", "ES3", "ED1", "ED3", "ED5", "ED7", "FL1", "FLX1", "WS1", "WS2", "ES5", "N2")
#: Labels in a pathway column that name no pathway (FES 2021 ED5 has one "All Scenarios" row).
NON_PATHWAY_LABELS = {"All Scenarios"}


def _resource(*parts: str) -> Path:
    path = RESOURCES.joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"{path} not present")
    return path


def _raw_building_blocks(edition: int) -> pd.DataFrame:
    path = fes_raw.raw_path(edition, "building_blocks", RAW_DIR)
    if not path.is_file():
        pytest.skip(f"FES {edition} building blocks not cached ({path})")
    return pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)


# ──────────────────────────────── stage 1: pathways ─────────────────────────

def _workbook_pathway_strings(path: str) -> set[str]:
    import openpyxl
    wb = openpyxl.load_workbook(PROJECT_ROOT / path, read_only=True, data_only=True)
    found: set[str] = set()
    try:
        for sheet in DATA_SHEETS:
            if sheet not in wb.sheetnames:
                continue
            column = None
            for i, row in enumerate(wb[sheet].iter_rows(values_only=True)):
                if column is None:
                    labels = [str(v).strip() if v is not None else "" for v in row]
                    column = next((labels.index(k) for k in ("Pathway", "FES Pathway", "Scenario", "FES Scenario")
                                   if k in labels), None)
                    if column is None and i > 40:
                        break
                    continue
                if column < len(row) and row[column] is not None and str(row[column]).strip():
                    found.add(str(row[column]).strip())
    finally:
        wb.close()
    return found


@pytest.mark.parametrize("edition", FES_EDITIONS)
def test_every_building_block_pathway_resolves(edition):
    blocks = _raw_building_blocks(edition)
    column = next(c for c in blocks.columns if "Scenario" in c or "Pathway" in c)
    names = blocks[column].dropna().unique()
    codes = registry.resolve_all("FES", edition, names)
    in_blocks = {p.code for p in registry.pathways("FES", edition) if p.in_building_blocks}
    assert set(codes.values()) == in_blocks


@pytest.mark.parametrize("source,edition,path", [("FES", e, p) for e, p in FES_WORKBOOKS.items()]
                         + [("CP30", 2024, CP30_WORKBOOK)])
def test_every_workbook_pathway_resolves(source, edition, path):
    if not (PROJECT_ROOT / path).is_file():
        pytest.skip(f"{path} not present")
    names = _workbook_pathway_strings(path)
    assert names, f"no pathway strings found in {path}"
    registry.resolve_all(source, edition, names - NON_PATHWAY_LABELS)


def test_pathway_lookups_and_errors():
    ht = registry.pathway_by_name("FES", 2025, "Holistic Transition")
    assert (ht.code, ht.role, ht.in_building_blocks, ht.slug) == ("HT", "net_zero", True, "holistic_transition")
    assert registry.pathway_by_code("FES", 2025, "tyf").name == "Ten Year Forecast"
    assert registry.pathway_code("FES", 2020, "LW") == registry.pathway_code("FES", 2020, "Leading The Way") == "LW"
    assert registry.pathway_code("FES", 2021, "Central Forecast") == "FYF"
    assert {p.name for p in registry.pathways("CP30", 2024)} == {"Further Flex and Renewables", "New Dispatch"}
    with pytest.raises(registry.UnknownPathway, match="no pathway 'Leading the Way'"):
        registry.pathway_by_name("FES", 2025, "Leading the Way")
    with pytest.raises(registry.UnknownEdition):
        registry.pathways("FES", 2026)
    with pytest.raises(ValueError):
        registry.pathways("TYNDP")


def test_successors_follow_the_counterfactual_renames():
    assert registry.successor("FES", 2024, "CF").code == "FB"
    assert registry.successor("FES", 2023, "FS").code == "CF"
    assert registry.successor("FES", 2021, "SP").code == "FS"
    assert registry.successor("FES", 2024, "FYF").code == "TYF"
    assert registry.successor("FES", 2024, "HT").code == "HT"
    assert registry.successor("FES", 2023, "LW") is None   # FES 2024 redrew the net zero pathways
    assert registry.successor("FES", 2025, "HT") is None


def test_fes2025_targets_take_their_maps_from_the_registry():
    from scripts.FES import fes2025_targets as fes
    assert fes.FES2025_PATHWAYS == registry.pathway_names("FES", 2025)
    assert list(fes.FES2025_PATHWAYS) == ["holistic_transition", "electric_engagement", "hydrogen_evolution",
                                         "falling_behind", "ten_year_forecast"]
    assert len(fes.FES2025_GENERATION_MAP) == 26
    assert fes.FES2025_STORAGE_MAP[("Long Duration Energy Storage", "Pumped Hydro")] == "Pumped Storage Hydroelectricity"
    supply_2024, storage_2024 = fes.edition_line_maps(2024)
    assert {(m.category, m.type, m.subtype): m.carrier for m in supply_2024}[("Thermal", "Low Carbon", "Hydrogen")] == "Hydrogen"
    assert storage_2024[("Storage", "Liquid Air")] == "LAES"
    # Distributed OCGT that FES 2024 lists under Other Thermal is an OCGT: unabated gas (user, 27 Oct 2026).
    assert {(m.category, m.type, m.subtype): m.carrier for m in supply_2024}[("Thermal", "Other Thermal", "OCGT")] == "gas"
    with pytest.raises(registry.UnknownEdition):
        fes.edition_line_maps(2023)


# ──────────────────────────────── stage 1: GSP aliases ──────────────────────

#: Every raw name the old branches touched, plus near-misses (substring and exact rules differ on them).
_SYNTHETIC_GSPS = [
    "Dunbar A", "Dunbar B", "Dunbar", "Cain", "Ferrybridge B (_F)", "Ferrybridge B (_M)", "Dumfries", "Dumfries 11",
    "Crookston", "Crookston A", "Grangemouth", "Grangemouth A", "Kilbowie", "Saltcoats", "Finstown", "Shetland",
    "East Claydon (_B)", "East Claydon (_H)", "Broadford-Skye", "Grange", "Grange 66kV", "Kearsley", "Kearsley 132kV",
    "Roosecote", "Stornoway", "Stornoway 2", "Kirkby (_G)", "Kirkby (_D)", "Ormonde", "Robin Rigg", "Walney 2",
    "Tynemouth 1", "Tynemouth 2", "Hams Hall", "Hart Moor", "Camblesforth", "Drax", "Fiddlers Ferry", "St. Asaph",
    "Axminster_H", "Clacton", "Loudwater", "Sheringham Shoal", "Beddington (_C)", "Beddington (_J)", "Canal Bank",
    "Greenford", "Littlebrook (_C)", "Littlebrook (_J)", "Lodge Road", "New Cross", "New Cross 66kV", "Perivale",
    "Wesley Avenue (Willesden)", "Port Ham", "Kellwood Road", "Long Park", "New Cumnock", "Well Street Paisley",
    "Abernethy ", " Direct(NGET)", "Direct(SHETL)", "Abham",
]


#: FES 2025 has its own exact rows since 27 Oct 2026 (user decision). Its raw names already carry the
#: FES 2024 targets, which the copied substring rules doubled. Old name -> corrected name, pinned.
FES2025_CORRECTED_NAMES = {
    "Camblesforth 66kV (Drax) 66kV (Drax)": "Camblesforth 66kV (Drax)",
    "Canal Bank (Willesden 66kV) (Willesden 66kV)": "Canal Bank (Willesden 66kV)",
    "Clacton (Bramford) (Bramford)": "Clacton (Bramford)",
    "Fiddlers Ferry (Cuerdley 132kV) (Cuerdley 132kV)": "Fiddlers Ferry (Cuerdley 132kV)",
    "Greenford (Willesden 66kV) (Willesden 66kV)": "Greenford (Willesden 66kV)",
    "Hams Hall (Lea Marston 132kV) (Lea Marston 132kV)": "Hams Hall (Lea Marston 132kV)",
    "Lodge Road (St Johns Wood) (St Johns Wood)": "Lodge Road (St Johns Wood)",
    "Loudwater (Amersham) (Amersham)": "Loudwater (Amersham)",
    "New Cross 66kV 66kV": "New Cross 66kV",
    "Ormonde (Heysham) (Heysham)": "Ormonde (Heysham)",
    "Perivale (Willesden 66kV) (Willesden 66kV)": "Perivale (Willesden 66kV)",
    "Robin Rigg (Harker) (Harker)": "Robin Rigg (Harker)",
    "Sheringham Shoal (Norwich) (Norwich)": "Sheringham Shoal (Norwich)",
    "Tynemouth 132kV1kV": "Tynemouth 11kV",
    "Tynemouth 132kV32kV": "Tynemouth 132kV",
    "Walney 2 (Heysham) (Heysham)": "Walney 2 (Heysham)",
}
#: The garbled Tynemouth names had mapped to each other's GSP ids; the corrected names take the ids
#: GSP info gives them (both GSPs are node TYNE11, at the same coordinates).
FES2025_CORRECTED_IDS = {"Tynemouth 11kV": ("TYNE_1", "TYNE_2"), "Tynemouth 132kV": ("TYNE_2", "TYNE_1")}
LEGACY_EDITIONS = tuple(e for e in FES_EDITIONS if e != 2025)


@pytest.mark.parametrize("edition", LEGACY_EDITIONS)
def test_gsp_aliases_reproduce_the_old_branches_on_synthetic_names(edition):
    frame = pd.DataFrame({"GSP": _SYNTHETIC_GSPS, "x": range(len(_SYNTHETIC_GSPS))})
    expected = legacy_clean_gsp_names(edition, frame)
    got = registry.apply_gsp_aliases(frame, edition)
    pd.testing.assert_frame_equal(got, expected)


@pytest.mark.parametrize("edition", LEGACY_EDITIONS)
def test_gsp_aliases_reproduce_the_old_branches_on_the_building_blocks(edition):
    blocks = _raw_building_blocks(edition)[["GSP"]].copy()
    blocks["GSP"] = blocks["GSP"].map(lambda v: v.replace("'", "") if isinstance(v, str) else v)
    expected = legacy_clean_gsp_names(edition, blocks)
    got = registry.apply_gsp_aliases(blocks, edition)
    pd.testing.assert_frame_equal(got, expected)


def test_fes2025_aliases_differ_from_the_old_branches_only_by_the_corrected_names():
    blocks = _raw_building_blocks(2025)[["GSP"]].copy()
    blocks["GSP"] = blocks["GSP"].map(lambda v: v.replace("'", "") if isinstance(v, str) else v)
    expected = legacy_clean_gsp_names(2025, blocks)
    expected["GSP"] = expected["GSP"].replace(FES2025_CORRECTED_NAMES)
    got = registry.apply_gsp_aliases(blocks, 2025)
    pd.testing.assert_frame_equal(got, expected)
    assert all(a.match == "exact" for a in registry.gsp_aliases(2025))
    assert not got["GSP"].isin(list(FES2025_CORRECTED_NAMES)).any()


def test_gsp_aliases_unknown_edition_fails():
    with pytest.raises(registry.UnknownEdition, match="No GSP aliases for FES 2026"):
        registry.apply_gsp_aliases(pd.DataFrame({"GSP": ["Abham"]}), 2026)
    assert [a.action for a in registry.gsp_aliases(2020)] == ["none"]
    assert [(a.action, a.raw, a.canonical) for a in registry.gsp_aliases(2025)] == [
        ("rename", "Dunbar A", "Dunbar"), ("rename", "Dunbar B", "Dunbar"), ("drop", "Stornoway", "")]


# ─────────────────────────── stage 2: raw cache and GSP info ────────────────

def test_raw_cache_reads_offline_and_refetches_a_new_url(tmp_path):
    calls = []

    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    class Session:
        def get(self, url, timeout):
            calls.append(url)
            return Response(f"url,{url}\n".encode())

    fes_raw.POLITE_DELAY_S, delay = 0.0, fes_raw.POLITE_DELAY_S
    try:
        urls = {"building_blocks": {2030: "https://example.invalid/a.csv"}}
        path = fes_raw.fetch_raw(2030, "building_blocks", urls=urls, raw_dir=tmp_path, session=Session())
        fes_raw.fetch_raw(2030, "building_blocks", urls=urls, raw_dir=tmp_path, session=Session())
        assert calls == ["https://example.invalid/a.csv"]
        assert fes_raw.read_sidecar(2030, "building_blocks", tmp_path)["url"] == urls["building_blocks"][2030]
        urls["building_blocks"][2030] = "https://example.invalid/b.csv"
        fes_raw.fetch_raw(2030, "building_blocks", urls=urls, raw_dir=tmp_path, session=Session())
        assert len(calls) == 2 and path.read_bytes().endswith(b"b.csv\n")
        with pytest.raises(fes_raw.RawTableUnavailable):
            fes_raw.fetch_raw(2031, "gsp_info", urls=urls, raw_dir=tmp_path, session=Session())
    finally:
        fes_raw.POLITE_DELAY_S = delay


def test_fes_data_gsp_info_is_the_editions_own_committed_table():
    from scripts.FES import FES_data
    info_2025 = FES_data.load_gsp_info(2025)
    assert len(info_2025) == 388 and info_2025["Comments"].str.contains("GSP_regions").any()
    assert len(FES_data.load_gsp_info(2024)) == 373
    with pytest.raises(FileNotFoundError, match="No GSP info for FES 2026"):
        FES_data.load_gsp_info(2026)


def test_storage_gsp_info_has_no_cross_edition_fallback(monkeypatch):
    import logging
    import scripts.storage.add_storage as add_storage
    log = logging.getLogger("test")
    monkeypatch.setattr(add_storage, "_download_gsp_info",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no API call expected")))
    assert add_storage.FES_GSP_INFO_FILES[2025].endswith("fes2025_regional_breakdown_gsp_info.csv")
    assert len(add_storage.load_gsp_info(2025, log)) == 388
    with pytest.raises(RuntimeError, match="No FES 2026 GSP info"):
        add_storage.load_gsp_info(2026, log)
    assert add_storage.fes_edition_of_file("resources/FES/FES_2025_data.csv") == 2025
    with pytest.raises(ValueError):
        add_storage.fes_edition_of_file("fes.csv")


def _regenerate_fes_data(edition, tmp_path):
    live = _resource("FES", f"FES_{edition}_data.csv")
    for table in ("building_blocks", "building_block_definitions"):
        if not fes_raw.raw_path(edition, table, RAW_DIR).is_file():
            pytest.skip(f"FES {edition} {table} not cached")
    from scripts.FES import FES_data
    out = tmp_path / f"FES_{edition}_data.csv"
    FES_data.build_fes_data(edition, outpath=out)
    return live, out


def test_fes_data_regenerates_fes2024_byte_for_byte(tmp_path):
    live, out = _regenerate_fes_data(2024, tmp_path)
    assert out.read_bytes() == live.read_bytes()


def test_fes_data_regenerates_fes2025_except_the_corrected_names(tmp_path):
    """The workflow file of 26 Oct 2026 was built with the doubled names; only they may differ."""
    live, out = _regenerate_fes_data(2025, tmp_path)
    old = pd.read_csv(live, low_memory=False)
    new = pd.read_csv(out, low_memory=False)
    assert list(old.columns) == list(new.columns) and len(old) == len(new)
    others = [c for c in old.columns if c not in ("GSP", "GSP ID")]
    pd.testing.assert_frame_equal(old[others], new[others])
    pd.testing.assert_series_equal(old["GSP"].replace(FES2025_CORRECTED_NAMES), new["GSP"])
    changed = old["GSP"].ne(new["GSP"])
    # Only the corrected names may differ (none once the workflow file is rebuilt, as it was on 30 Oct 2026).
    assert set(old.loc[changed, "GSP"]) <= set(FES2025_CORRECTED_NAMES)
    id_changed = old["GSP ID"].fillna("").ne(new["GSP ID"].fillna(""))
    pairs = {(g, o, n) for g, o, n in zip(new.loc[id_changed, "GSP"], old.loc[id_changed, "GSP ID"],
                                         new.loc[id_changed, "GSP ID"])}
    assert pairs in (set(), {(g, o, n) for g, (o, n) in FES2025_CORRECTED_IDS.items()})
    # Either way the regenerated Tynemouth rows carry the corrected ids.
    for name, (_, corrected) in FES2025_CORRECTED_IDS.items():
        assert set(new.loc[new["GSP"].eq(name), "GSP ID"]) <= {corrected}


def test_storage_gsp_rows_are_located_by_id_then_exact_name_and_never_by_first_word(tmp_path):
    import logging
    import scripts.storage.add_storage as add_storage
    info = add_storage.load_gsp_info(2025, logging.getLogger("test"))
    locator = add_storage.GspLocator(info)
    newx = info.set_index("GSP ID").loc["NEWX_6", ["Latitude", "Longitude"]].astype(float).tolist()
    # By id, whatever the name says (the doubled FES 2025 name took Newton Stewart's coordinates before).
    assert list(locator.locate("NEWX_6", "New Cross 66kV 66kV")) == newx
    # No id: the exact canonical name; a first word alone matches nothing.
    assert list(locator.locate(None, "New Cross 66kV")) == newx
    assert locator.locate(None, "New") == (None, None)
    assert locator.locate("NOT_AN_ID", "New Cross 66kV") == (None, None)

    fes = tmp_path / "FES_2025_data.csv"
    pd.DataFrame({"FES Pathway": ["Holistic Transition"] * 2, "Building Block ID Number": ["Srg_BB001"] * 2,
                  "GSP": ["New Cross 66kV", "Newhere"], "GSP ID": ["NEWX_6", np.nan], "2030": [5.0, 7.0]}
                 ).to_csv(fes, index=False)
    with pytest.raises(add_storage.GspLocationError, match="Newhere"):   # not swallowed into REPD storage
        add_storage.load_fes_storage_data(str(fes), 2030, "Holistic Transition", pd.DataFrame(),
                                          logging.getLogger("test"))


def test_storage_rule_declares_each_editions_gsp_info():
    text = (PROJECT_ROOT / "rules" / "storage.smk").read_text(encoding="utf-8")
    assert "fes2024_regional_breakdown_gsp_info.csv" not in text
    assert 'f"data/network/ETYS/fes{get_storage_fes_year(scenarios[wc.scenario])}"' in text


# ──────────────────────────────── stage 3: contract ─────────────────────────

def test_contract_writes_reads_and_guards_years(tmp_path):
    capacity = pd.DataFrame({"pathway_code": ["HT", "HT"], "pathway": ["Holistic Transition"] * 2,
                             "year": [2050, 2040], "carrier": ["solar_pv", "solar_pv"], "capacity_mw": [1.0, 2.0],
                             "lines": ["Solar PV"] * 2})
    out = contract.write_source("FES", 2099, {"capacity": capacity}, {"note": "test"}, root=tmp_path)
    assert (out / "capacity.csv").read_text().splitlines()[1].startswith("HT,Holistic Transition,2040,")
    assert contract.read_table("FES", 2099, "capacity", "HT", 2050, root=tmp_path)["capacity_mw"].tolist() == [1.0]
    with pytest.raises(contract.NotPublished, match=r"published years: \[2040, 2050\]"):
        contract.require("FES", 2099, "capacity", "HT", 2045, root=tmp_path)
    with pytest.raises(contract.NotPublished):
        contract.require("FES", 2099, "energy", "HT", 2050, root=tmp_path)
    with pytest.raises(ValueError, match="unexpected"):
        contract.conform("capacity", capacity.assign(extra=1))


# ──────────────────────────────── stage 3: CP30 adapter ─────────────────────

@pytest.fixture(scope="module")
def cp30_tables():
    from scripts.sources.cp30_adapter import build_cp30_tables
    paths = {name: RESOURCES / "CP30" / f"CP30_{file}.csv" for name, file in
             (("es1", "ES1"), ("es5", "ES5_regional"), ("ed1", "ED1"))}
    missing = [p for p in paths.values() if not p.is_file()]
    if missing:
        pytest.skip(f"CP30 CSVs not present: {missing}")
    tables, meta = build_cp30_tables(paths["es1"], paths["es5"], paths["ed1"])
    return tables, meta, paths


@pytest.mark.parametrize("pathway", ["Further Flex and Renewables", "New Dispatch"])
def test_cp30_adapter_equals_the_cp30_readers_for_2030(cp30_tables, pathway):
    from scripts.CP30.energy_validation import cp30_energy_targets
    from scripts.CP30.generation import load_cp30_generation_capacities
    from scripts.CP30.spatial_allocation import read_es5_targets
    tables, meta, paths = cp30_tables
    code = registry.pathway_code("CP30", 2024, pathway)
    pick = lambda name: tables[name][tables[name]["pathway_code"].eq(code) & tables[name]["year"].eq(2030)]  # noqa: E731

    targets = cp30_energy_targets(paths["es1"], pathway, 2030)
    energy = pick("energy").set_index("carrier")
    gen = targets["generation"]
    pd.testing.assert_series_equal(energy.loc[gen.index, "generation_gwh"], gen["generation_gwh"], check_names=False)
    pd.testing.assert_series_equal(energy.loc[gen.index, "gross_gwh"], gen["gross_gwh"], check_names=False)
    assert energy.at["interconnector", "import_gwh"] == targets["import_gwh"]
    assert energy.at["interconnector", "export_gwh"] == targets["export_gwh"]
    demand = pick("demand").set_index("item")["value"]
    assert demand["demand_excluding_electrolysis"] == targets["demand_gwh"]
    assert demand["electrolysis"] == targets["electrolysis_gwh"]
    storage = pick("storage").set_index("carrier")
    for carrier, gwh in targets["storage_gwh"].items():
        assert storage.at[carrier, "generation_gwh"] == gwh

    capacity = pick("capacity").set_index("carrier")["capacity_mw"]
    reference = load_cp30_generation_capacities(paths["es1"], pathway, 2030).set_index("carrier")["capacity_mw"]
    pd.testing.assert_series_equal(capacity.drop("interconnector").sort_index(), reference.sort_index(),
                                   check_names=False)

    es5 = read_es5_targets(paths["es5"], pathway, 2030)
    regional = pick("regional_capacity")
    regional = regional[regional["carrier"].isin(set(es5["carrier"]))]
    merged = es5.merge(regional, left_on=["carrier", "cp30_region"], right_on=["carrier", "region"],
                       suffixes=("_ref", ""), validate="one_to_one")
    assert len(merged) == len(es5) == len(regional)
    assert (merged["capacity_mw"] == merged["capacity_mw_ref"]).all()
    assert meta["unmapped_es1_lines"] == []


def test_cp30_adapter_publishes_only_the_years_cp30_publishes(cp30_tables, tmp_path):
    tables, meta, _ = cp30_tables
    contract.write_source("CP30", 2024, tables, meta, root=tmp_path)
    available = contract.load_availability("CP30", 2024, tmp_path)["tables"]
    assert available["energy"] == {"FFR": [2023, 2030], "ND": [2023, 2030]}
    assert available["regional_capacity"] == {"FFR": [2030], "ND": [2030]}
    assert available["capacity"]["FFR"] == [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2035]
    with pytest.raises(contract.NotPublished):      # ES5 is 2030 only: no silent 2030 rows for 2035
        contract.require("CP30", 2024, "regional_capacity", "FFR", 2035, root=tmp_path)
    with pytest.raises(contract.NotPublished):      # ES1 energy is 2023 and 2030 only
        contract.require("CP30", 2024, "energy", "FFR", 2028, root=tmp_path)
    with pytest.raises(contract.NotPublished):      # ED1 has 2050 rows, the ES1 demand line does not
        contract.require("CP30", 2024, "demand", "FFR", 2050, item="demand_excluding_electrolysis", root=tmp_path)


# ──────────────────────────────── stage 3: FES adapter ──────────────────────

@pytest.fixture(scope="module")
def fes2025_tables():
    from scripts.sources.fes_adapter import build_fes_tables
    if not (PROJECT_ROOT / FES_WORKBOOKS[2025]).is_file():
        pytest.skip("FES 2025 workbook not present")
    return build_fes_tables(2025, include_gsp_blocks=False)


@pytest.mark.parametrize("pathway,year", [("Holistic Transition", 2050), ("Holistic Transition", 2040),
                                          ("Electric Engagement", 2035)])
def test_fes2025_adapter_equals_fes2025_energy_targets(fes2025_tables, pathway, year):
    from scripts.FES.fes2025_targets import FES2025_WORKBOOK, fes2025_energy_targets
    tables, meta = fes2025_tables
    code = registry.pathway_code("FES", 2025, pathway)
    pick = lambda name: tables[name][tables[name]["pathway_code"].eq(code) & tables[name]["year"].eq(year)]  # noqa: E731
    targets = fes2025_energy_targets(PROJECT_ROOT / FES2025_WORKBOOK, pathway, year, demand_basis="system",
                                     include_building_blocks=False, strict=True)
    gen = targets["generation"]
    energy = pick("energy").set_index("carrier")
    nonzero = gen[(gen["generation_gwh"] != 0) | (gen["curtailment_gwh"] != 0)]
    pd.testing.assert_series_equal(energy.loc[nonzero.index, "generation_gwh"], nonzero["generation_gwh"],
                                   check_names=False)
    pd.testing.assert_series_equal(energy.loc[nonzero.index, "curtailment_gwh"], nonzero["curtailment_gwh"],
                                   check_names=False)
    assert energy.at["interconnector", "import_gwh"] == targets["import_gwh"]
    capacity = pick("capacity").set_index("carrier")["capacity_mw"]
    expected_capacity = gen.loc[gen["capacity_mw"] != 0, "capacity_mw"]
    pd.testing.assert_series_equal(capacity.drop("interconnector").sort_index(), expected_capacity.sort_index(),
                                   check_names=False)
    assert capacity["interconnector"] == targets["interconnectors"]["capacity_mw"]
    storage = pick("storage").set_index("carrier")
    for carrier, row in targets["storage"].iterrows():
        assert storage.at[carrier, "power_mw"] == row["capacity_mw"]
        assert storage.at[carrier, "energy_gwh"] == row["energy_gwh"]
    demand = pick("demand").set_index("item")
    assert demand.at["system_excluding_electrolysis", "value"] == targets["demand_gwh"]
    assert demand.at["electrolysis", "value"] == targets["electrolysis_gwh"]
    assert demand.at["consumer_excluding_electrolysis", "fiscal_value"] == \
        targets["demand"]["totals"]["consumer_excluding_electrolysis"]["fy_gwh"]
    assert meta["unmapped_es1_lines"] == []


def test_fes2025_adapter_matches_the_workflow_targets_file(fes2025_tables):
    live = _resource("FES", "fes2025_holistic_transition_2050_targets.csv")
    reference = pd.read_csv(live)
    tables, _ = fes2025_tables
    capacity = tables["capacity"]
    capacity = capacity[capacity["pathway_code"].eq("HT") & capacity["year"].eq(2050)].set_index("carrier")["capacity_mw"]
    ref = reference[reference["section"].eq("supply") & reference["quantity"].eq("capacity_mw")]
    ref = ref[ref["value"] != 0].set_index("item")["value"]
    pd.testing.assert_series_equal(capacity.drop("interconnector").sort_index(), ref.sort_index(), check_names=False)
    demand = tables["demand"]
    demand = demand[demand["pathway_code"].eq("HT") & demand["year"].eq(2050)].set_index("item")
    ref_demand = reference[reference["section"].eq("demand") & reference["quantity"].eq("calendar_gwh")]
    for item, value in ref_demand.set_index("item")["value"].items():
        assert demand.at[item, "value"] == pytest.approx(value, rel=1e-12)


def test_fes2025_adapter_years_and_the_ed1_rounding(fes2025_tables):
    tables, meta = fes2025_tables
    available = contract.availability_of(tables)
    assert available["capacity"]["HT"] == list(range(2024, 2051))
    assert available["capacity"]["TYF"][-1] == 2035
    assert available["demand"]["HT"] == list(range(2024, 2051))   # ED1 starts 2023: calendar 2024 on
    assert meta["demand_skipped"] == []                            # rounding closes within 10 ppm
