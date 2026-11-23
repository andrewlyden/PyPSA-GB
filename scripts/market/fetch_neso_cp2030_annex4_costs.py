"""Fetch and parse NESO Clean Power 2030 Annex 4 commodity assumptions.

The generated JSON is the sole source for the neso_cp2030 future-cost case.
It retains the downloaded Annex PDF, source URL, retrieval time, SHA-256 and
the matched wording so a model run can be reproduced and audited.

Standalone usage:
    python scripts/market/fetch_neso_cp2030_annex4_costs.py

Use --input-pdf to parse a previously saved Annex 4 PDF without downloading.
"""

import argparse
import hashlib
import io
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests

ANNEX4_URL = "https://www.neso.energy/document/346806/download"
THERM_TO_MWH = 0.0293071


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _matched_sentence(text: str, match: re.Match) -> str:
    start = text.rfind(".", 0, match.start()) + 1
    end = text.find(".", match.end())
    return text[start:end + 1].strip() if end >= 0 else text[start:].strip()


def extract_annex4_assumptions(text: str) -> dict:
    """Extract Annex 4's published gas, total-carbon and CPS assumptions."""
    text = _normalise_text(text)
    gas_match = re.search(
        r"gas price of (?:around|approximately)\s*(?P<value>\d+(?:\.\d+)?)\s*p\s*/\s*therm",
        text,
        flags=re.IGNORECASE,
    )
    carbon_match = re.search(
        r"UK carbon price.*?at\s*(?:[^0-9\s])?\s*(?P<value>\d+(?:\.\d+)?)\s*/\s*t\s*CO2",
        text,
        flags=re.IGNORECASE,
    )
    cps_match = re.search(
        r"Carbon Price Support scheme.*?currently set at\s*(?:[^0-9\s])?\s*(?P<value>\d+(?:\.\d+)?)\s*/\s*t\s*CO2",
        text,
        flags=re.IGNORECASE,
    )
    if not gas_match or not carbon_match or not cps_match:
        missing = [
            name for name, match in {
                "gas price (p/therm)": gas_match,
                "total UK carbon price": carbon_match,
                "Carbon Price Support": cps_match,
            }.items() if not match
        ]
        raise ValueError(
            "Could not validate Annex 4 commodity assumptions; missing "
            + ", ".join(missing)
        )

    gas_pence_per_therm = float(gas_match.group("value"))
    total_carbon = float(carbon_match.group("value"))
    carbon_price_support = float(cps_match.group("value"))
    uk_ets = total_carbon - carbon_price_support
    if uk_ets < 0:
        raise ValueError("Annex 4 total carbon price is lower than Carbon Price Support.")

    return {
        "fuel_prices": {
            "gas": round((gas_pence_per_therm / 100.0) / THERM_TO_MWH, 6),
        },
        "carbon": {
            "uk_ets": uk_ets,
            "carbon_price_support": carbon_price_support,
        },
        "extracted_values": {
            "gas_pence_per_therm": gas_pence_per_therm,
            "total_uk_carbon_gbp_per_tco2": total_carbon,
            "therm_to_mwh": THERM_TO_MWH,
        },
        "evidence": {
            "gas": _matched_sentence(text, gas_match),
            "total_carbon": _matched_sentence(text, carbon_match),
            "carbon_price_support": _matched_sentence(text, cps_match),
        },
    }


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Read text from the supplied Annex PDF using pypdf or pdftotext."""
    try:
        from pypdf import PdfReader
    except ImportError:
        # This supports existing Windows environments while the updated Conda
        # environment is being recreated with pypdf installed.
        try:
            result = subprocess.run(
                ["pdftotext", "-", "-"],
                input=pdf_bytes,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Annex 4 extraction requires pypdf or the pdftotext command. "
                "Recreate the project environment or install one of them."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "pdftotext could not read the downloaded Annex 4 PDF."
            ) from exc
        return result.stdout.decode("utf-8", errors="replace")
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf_bytes)).pages)


def download_annex4(url: str) -> bytes:
    # The University/OneDrive environment may expose a stale HTTP proxy. The
    # NESO document is public, so use a direct session rather than inheriting
    # ambient proxy variables that can make a reproducible workflow fail.
    session = requests.Session()
    session.trust_env = False
    response = session.get(
        url,
        headers={"User-Agent": "PyPSA-GB NESO Annex 4 cost-ingestion"},
        timeout=60,
    )
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"Expected a PDF from {url}, received {response.headers.get('content-type')!r}.")
    return response.content


def build_payload(pdf_bytes: bytes, source_url: str) -> dict:
    """Create the auditable future-cost resource from an Annex 4 PDF."""
    extracted = extract_annex4_assumptions(extract_pdf_text(pdf_bytes))
    return {
        "schema_version": 1,
        "case": "neso_cp2030",
        "source": {
            "title": "NESO Clean Power 2030 Annex 4: Costs and benefit analysis",
            "url": source_url,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
            "pages_used": [6, 7],
        },
        "assumptions": {
            "fuel_prices": extracted["fuel_prices"],
            "carbon": extracted["carbon"],
        },
        "extracted_values": extracted["extracted_values"],
        "evidence": extracted["evidence"],
    }


def write_resource(payload: dict, output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=Path("resources/market/neso_clean_power_2030_annex4_costs.json"))
    parser.add_argument("--output-pdf", type=Path, default=Path("resources/market/source/neso_clean_power_2030_annex4.pdf"))
    parser.add_argument("--input-pdf", type=Path, help="Use an already saved Annex 4 PDF instead of downloading it.")
    parser.add_argument("--source-url", default=ANNEX4_URL)
    return parser.parse_args()


def main() -> None:
    if "snakemake" in globals():
        output_json = Path(snakemake.output.costs)
        output_pdf = Path(snakemake.output.pdf)
        source_url = ANNEX4_URL
        pdf_bytes = download_annex4(source_url)
    else:
        args = parse_args()
        output_json, output_pdf, source_url = args.output_json, args.output_pdf, args.source_url
        pdf_bytes = args.input_pdf.read_bytes() if args.input_pdf else download_annex4(source_url)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.write_bytes(pdf_bytes)
    write_resource(build_payload(pdf_bytes, source_url), output_json)


if __name__ == "__main__":
    main()

