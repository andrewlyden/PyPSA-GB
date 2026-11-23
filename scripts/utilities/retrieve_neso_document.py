"""Download one NESO website document listed in data/neso_documents.yaml and check its SHA-256.

NESO's website terms allow personal copies only, so these documents are fetched by the workflow
instead of being shipped in the repository. Run by rule ``retrieve_neso_document``, or by hand:

    python scripts/utilities/retrieve_neso_document.py "data/CP30/Clean Power 2030 Data Workbook_V003.xlsx"
"""


import hashlib
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "data" / "neso_documents.yaml"


def registry() -> dict[str, dict]:
    """The registered documents, keyed by repository path."""
    documents = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["documents"]
    return {doc["path"]: doc for doc in documents}


def retrieve(path: str, out: Path | None = None) -> Path:
    """Download the registered document ``path`` to ``out`` (default: the path itself)."""
    doc = registry().get(path)
    if doc is None:
        raise KeyError(f"{path} is not in {REGISTRY}")
    target = Path(out) if out is not None else ROOT / path
    request = urllib.request.Request(doc["url"], headers={"User-Agent": "PyPSA-GB data retrieval"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
    except OSError as error:
        raise RuntimeError(
            f"could not download {doc['title']} from {doc['url']} ({error}). Download it by hand from "
            f"{doc['landing_page']} and save it as {path}") from error
    digest = hashlib.sha256(data).hexdigest()
    if digest != doc["sha256"]:
        raise RuntimeError(
            f"{doc['url']} returned a different file (SHA-256 {digest}, expected {doc['sha256']}): NESO may "
            f"have replaced it. Download '{doc['title']}' by hand from {doc['landing_page']} and save it as {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


if __name__ == "__main__":
    if "snakemake" in globals():
        retrieve(snakemake.params.document, Path(snakemake.output.document))  # noqa: F821
    else:
        for argument in sys.argv[1:]:
            print(retrieve(argument))
