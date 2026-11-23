"""Find papers and reports that cite or mention PyPSA-GB, for the docs' Papers list.

Queries free scholarly indexes (no API key):

* **OpenAlex**: works citing the model paper (DOI 10.1016/j.esr.2024.101375) and works whose title, abstract or
  indexed full text mention "PyPSA-GB".
* **Semantic Scholar**: works citing the model paper.

Each hit is listed once (by DOI, else by normalised title) with the sources that found it, and marked ``known`` when
its DOI or title is already on the docs Publications page (``docs/source/reference/publications.md``). Whether a paper *uses* PyPSA-GB or only
cites it is a judgement the indexes cannot make: read the new ones before adding them.

    python scripts/utilities/find_publications.py --out resources/analysis/pypsa_gb_publications.csv

Run it every few months (or from a scheduled CI job) and add the papers that use the model.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

MODEL_DOI = "10.1016/j.esr.2024.101375"
SEARCH_TERM = '"PyPSA-GB"'
USER_AGENT = "PyPSA-GB publication search (https://github.com/andrewlyden/PyPSA-GB)"
PAPERS_PAGE = Path("docs/source/reference/publications.md")


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return {}


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(title).lower()).strip()


def openalex(query: str) -> list[dict]:
    """Every work matching an OpenAlex ``works`` query string (cursor paging)."""
    rows, cursor = [], "*"
    while cursor:
        page = _get(f"https://api.openalex.org/works?{query}&per-page=200&cursor={urllib.parse.quote(cursor)}")
        for work in page.get("results", []):
            rows.append({
                "doi": (work.get("doi") or "").replace("https://doi.org/", "").lower(),
                "title": work.get("display_name") or "",
                "year": work.get("publication_year"),
                "venue": ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                "type": work.get("type") or "",
                "authors": "; ".join(a["author"]["display_name"] for a in work.get("authorships", [])[:6]),
            })
        cursor = (page.get("meta") or {}).get("next_cursor")
        time.sleep(0.2)
    return rows


def semantic_scholar_citations(doi: str) -> list[dict]:
    rows, offset = [], 0
    fields = "title,year,venue,externalIds,authors,publicationTypes"
    while True:
        page = _get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}/citations"
                    f"?fields={fields}&limit=100&offset={offset}")
        data = page.get("data", [])
        for item in data:
            paper = item.get("citingPaper") or {}
            rows.append({
                "doi": ((paper.get("externalIds") or {}).get("DOI") or "").lower(),
                "title": paper.get("title") or "",
                "year": paper.get("year"),
                "venue": paper.get("venue") or "",
                "type": ", ".join(paper.get("publicationTypes") or []),
                "authors": "; ".join(a.get("name", "") for a in (paper.get("authors") or [])[:6]),
            })
        if len(data) < 100:
            return rows
        offset += 100
        time.sleep(1.0)


def known_papers(page: Path) -> tuple[set[str], set[str]]:
    """DOIs and arXiv/title keys already listed on the docs' Publications page."""
    section = page.read_text(encoding="utf-8") if page.exists() else ""
    dois = {d.lower().rstrip(".>") for d in re.findall(r"10\.\d{4,9}/[^\s)>\]]+", section)}
    titles = {_norm_title(t) for t in re.findall(r"\*\*[^*]+\*\*\s*([^.]+)\.", section)}
    return dois, titles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("resources/analysis/pypsa_gb_publications.csv"))
    args = parser.parse_args()

    found: dict[str, dict] = {}

    def add(rows: list[dict], source: str) -> None:
        for row in rows:
            if row["doi"] == MODEL_DOI:
                continue
            key = row["doi"] or _norm_title(row["title"])
            if not key:
                continue
            entry = found.setdefault(key, {**row, "sources": set()})
            entry["sources"].add(source)

    model = _get(f"https://api.openalex.org/works/doi:{MODEL_DOI}")
    openalex_id = (model.get("id") or "").rsplit("/", 1)[-1]
    if openalex_id:
        add(openalex(f"filter=cites:{openalex_id}"), "openalex_cites")
    add(openalex(f"search={urllib.parse.quote(SEARCH_TERM)}"), "openalex_mentions")
    try:
        add(semantic_scholar_citations(MODEL_DOI), "semantic_scholar_cites")
    except Exception as error:   # rate limits without a key: keep the OpenAlex results
        print(f"Semantic Scholar skipped: {error}")

    known_dois, known_titles = known_papers(PAPERS_PAGE)
    table = pd.DataFrame(found.values())
    if table.empty:
        print("No works found")
        return
    table["sources"] = table["sources"].map(lambda s: ", ".join(sorted(s)))
    table["known"] = table["doi"].isin(known_dois) | table["title"].map(_norm_title).isin(known_titles)
    table = table.sort_values(["known", "year"], ascending=[True, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"{len(table)} works ({(~table['known']).sum()} not yet in the docs) -> {args.out}")
    print(table.loc[~table["known"], ["year", "title", "venue", "doi", "sources"]].to_string(index=False))


if __name__ == "__main__":
    main()
