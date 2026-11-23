"""Produce a *candidate* CP30-to-ETYS reinforcement crosswalk.

CP30 names projects whereas Appendix B records electrical components.  Similar
place names alone are useful for finding evidence, but are not sufficient to
modify a network.  This utility therefore emits only candidates: a reviewer
must confirm the project/component relationship from a primary project source
before moving any row into ``cp30_n1_reinforcement_manifest.csv`` or the
accelerated-component layer.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ETYS_SHEETS = (
    "B-2-2a", "B-2-2b", "B-2-2c", "B-2-2d",
    "B-3-2a", "B-3-2b", "B-3-2c", "B-3-2d",
)
GENERIC_SITE_NAMES = {
    "central", "east", "north", "south", "west", "new", "old", "main",
    "terminal", "substation", "wind farm", "power station",
}


def _normalise(value: object) -> str:
    """Return an uppercase space-normalised place name."""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def _project_register(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"source_sheet", "Code", "Description", "Required for CP30"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CP30 register is missing columns: {sorted(missing)}")
    required = data.loc[
        data["source_sheet"].eq("N1")
        & data["Required for CP30"].notna()
        & data["Code"].notna()
    , ["Code", "Description"]].copy()
    required.columns = ["project_code", "cp30_description"]
    return required.drop_duplicates("project_code").reset_index(drop=True)


def _site_aliases(path: str | Path, reviewed_aliases_path: str | Path | None = None) -> pd.DataFrame:
    sites = pd.read_csv(path)
    required = {"site_code", "site_name"}
    missing = required - set(sites.columns)
    if missing:
        raise ValueError(f"Substation register is missing columns: {sorted(missing)}")
    sites = sites.loc[:, ["site_code", "site_name"]].dropna().copy()
    sites["site_code"] = sites["site_code"].astype(str).str.upper().str[:4]
    sites["alias"] = sites["site_name"].map(_normalise)
    # Single generic words match narrative text far too easily.
    sites = sites.loc[
        sites["alias"].str.len().ge(5)
        & ~sites["alias"].str.lower().isin(GENERIC_SITE_NAMES)
    ]
    sites["alias_source"] = "substation_coordinates"
    if reviewed_aliases_path:
        reviewed = pd.read_csv(reviewed_aliases_path)
        required_reviewed = {"site_code", "site_name", "alias", "evidence"}
        missing = required_reviewed - set(reviewed.columns)
        if missing:
            raise ValueError(f"Reviewed alias register is missing columns: {sorted(missing)}")
        reviewed = reviewed.loc[:, ["site_code", "site_name", "alias", "evidence"]].copy()
        reviewed["site_code"] = reviewed["site_code"].astype(str).str.upper().str[:4]
        reviewed["alias"] = reviewed["alias"].map(_normalise)
        reviewed["alias_source"] = "reviewed_candidate_alias"
        sites = pd.concat([sites, reviewed.drop(columns="evidence")], ignore_index=True)
    return sites.drop_duplicates(["site_code", "alias"])


def _etys_components(path: str | Path, modelled_year: int) -> pd.DataFrame:
    components = []
    for sheet in ETYS_SHEETS:
        data = pd.read_excel(path, sheet_name=sheet, skiprows=1)
        node_columns = [column for column in data.columns if str(column).replace(" ", "").lower() in {"node1", "node2"}]
        if len(node_columns) != 2 or not {"Year", "Status"}.issubset(data.columns):
            continue
        data = data.rename(columns={node_columns[0]: "node_1", node_columns[1]: "node_2"})
        data["year"] = pd.to_numeric(data["Year"], errors="coerce")
        data = data.loc[
            data["Status"].astype(str).str.casefold().eq("addition")
            & data["year"].le(modelled_year)
            & data["node_1"].notna()
            & data["node_2"].notna()
        , ["node_1", "node_2", "year"]].copy()
        data["etys_sheet"] = sheet
        components.append(data)

    # B-5-1 also contains named HVDC projects.  It uses a planned year rather
    # than an addition/removal status.
    links = pd.read_excel(path, sheet_name="B-5-1", skiprows=1)
    links["year"] = pd.to_numeric(links["Planned from year"], errors="coerce")
    links = links.loc[
        links["Existing"].astype(str).str.casefold().eq("no")
        & links["year"].le(modelled_year)
        & links["Node 1"].notna()
        & links["Node 2"].notna()
    , ["Node 1", "Node 2", "year", "Interconnector Name"]].copy()
    links = links.rename(columns={"Node 1": "node_1", "Node 2": "node_2"})
    links["etys_sheet"] = "B-5-1"
    components.append(links)

    result = pd.concat(components, ignore_index=True, sort=False)
    result["etys_component_name"] = result.get("Interconnector Name", "").fillna("")
    result["node_1"] = result["node_1"].astype(str).str.upper()
    result["node_2"] = result["node_2"].astype(str).str.upper()
    result["node_1_site"] = result["node_1"].str[:4]
    result["node_2_site"] = result["node_2"].str[:4]
    return result


def build_candidate_crosswalk(
    cp30_register: str | Path,
    etys_workbook: str | Path,
    substation_register: str | Path,
    output_csv: str | Path,
    modelled_year: int = 2030,
    reviewed_aliases_path: str | Path | None = None,
) -> pd.DataFrame:
    """Write candidate component matches without asserting electrical equivalence."""
    projects = _project_register(cp30_register)
    aliases = _site_aliases(substation_register, reviewed_aliases_path)
    etys = _etys_components(etys_workbook, int(modelled_year))
    rows = []

    for project in projects.itertuples(index=False):
        description = _normalise(project.cp30_description)
        found = aliases.loc[aliases["alias"].map(lambda alias: alias in description)]
        # A curated phrase such as "NORTH OF KINCARDINE" disambiguates an
        # existing substation whose shorter name ("KINCARDINE") also occurs in
        # the prose.  Keep the reviewed candidate alias and discard only its
        # shorter, otherwise ambiguous, coordinate-register match.
        reviewed_phrases = found.loc[
            found["alias_source"].eq("reviewed_candidate_alias"), "alias"
        ].tolist()
        if reviewed_phrases:
            found = found.loc[
                found["alias_source"].eq("reviewed_candidate_alias")
                | ~found["alias"].map(
                    lambda alias: any(alias != phrase and alias in phrase for phrase in reviewed_phrases)
                )
            ]
        # A component needs two distinct named sites in the CP30 description.
        # This deliberately excludes indirect route/path matches.
        site_codes = set(found["site_code"])
        if len(site_codes) < 2:
            continue
        component_match = etys.loc[
            etys["node_1_site"].isin(site_codes)
            & etys["node_2_site"].isin(site_codes)
            & etys["node_1_site"].ne(etys["node_2_site"])
        ].copy()
        for _, component in component_match.iterrows():
            endpoint_codes = {component["node_1_site"], component["node_2_site"]}
            endpoint_names = found.loc[found["site_code"].isin(endpoint_codes), "site_name"]
            rows.append({
                "project_code": project.project_code,
                "cp30_description": project.cp30_description,
                "matched_cp30_sites": "; ".join(sorted(endpoint_names.unique())),
                "etys_sheet": component["etys_sheet"],
                "etys_node_1": component["node_1"],
                "etys_node_2": component["node_2"],
                "etys_year": int(component["year"]),
                "etys_component_name": component["etys_component_name"],
                "candidate_status": "candidate_pending_primary_project_review",
                "candidate_basis": "Two distinct CP30 place-name matches align with ETYS component endpoints.",
                "review_action": "Confirm project/component identity in a primary NESO or TO project source before modelling.",
            })

    columns = [
        "project_code", "cp30_description", "matched_cp30_sites", "etys_sheet",
        "etys_node_1", "etys_node_2", "etys_year", "etys_component_name",
        "candidate_status", "candidate_basis", "review_action",
    ]
    result = pd.DataFrame(rows, columns=columns).drop_duplicates(
        ["project_code", "etys_sheet", "etys_node_1", "etys_node_2"]
    ).sort_values(["project_code", "etys_sheet", "etys_node_1", "etys_node_2"])
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp30-register", required=True)
    parser.add_argument("--etys-workbook", required=True)
    parser.add_argument("--substation-register", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modelled-year", type=int, default=2030)
    parser.add_argument("--reviewed-aliases", default=None)
    args = parser.parse_args()
    matches = build_candidate_crosswalk(
        args.cp30_register, args.etys_workbook, args.substation_register,
        args.output, args.modelled_year, args.reviewed_aliases,
    )
    print(f"Wrote {len(matches)} conservative candidate component matches to {args.output}")
