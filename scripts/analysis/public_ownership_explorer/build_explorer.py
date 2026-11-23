"""Inject the explorer data into the template and write the Public Ownership Bill Explorer page.

Rule ``build_public_ownership_explorer`` runs ``extract_explorer_data.py`` then this; publishing the
page is a manual step. Usage::

    python scripts/analysis/public_ownership_explorer/build_explorer.py DATA_JSON OUT_HTML [--template HTML]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
DATA_MARKER = "/*__DATA__*/"


def build_page(data_path: Path, template_path: Path, out_path: Path) -> int:
    """Write the page and return its size in bytes.

    Args:
        data_path: the JSON written by ``extract_explorer_data.py``.
        template_path: the HTML template holding exactly one ``/*__DATA__*/`` marker.
        out_path: where the self-contained page goes.
    """
    raw = data_path.read_text(encoding="utf-8")
    json.loads(raw)  # must be valid JSON
    data = raw.replace("</", "<\\/")  # never close the script tag from inside the data
    html = template_path.read_text(encoding="utf-8")
    if html.count(DATA_MARKER) != 1:
        raise SystemExit(f"{template_path} must hold exactly one {DATA_MARKER} marker")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 10 Nov 2026: the model year and pathway the page names (2030 and Clean Power 2030 when the data has none).
    meta = json.loads(raw).get("meta", {})
    for token, value in (("__MODEL_YEAR__", str(meta.get("model_year", 2030))),
                         ("__PATHWAY_SHORT__", meta.get("pathway_short", "Clean Power 2030")),
                         ("__PATHWAY_LONG__", meta.get("pathway_long", "NESO's Clean Power 2030 pathway"))):
        html = html.replace(token, value)
    if int(meta.get("model_year", 2030)) != 2030:  # its own name in a gallery beside the 2030 page
        html = html.replace("<title>Who Pays for Public Ownership</title>",
                            f"<title>Who Pays in {meta['model_year']}</title>", 1)
    out_path.write_text(html.replace(DATA_MARKER, data), encoding="utf-8")
    return out_path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--template", type=Path, default=HERE / "explorer_template.html")
    args = parser.parse_args()
    size = build_page(args.data, args.template, args.out)
    print(f"wrote {args.out} ({size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
