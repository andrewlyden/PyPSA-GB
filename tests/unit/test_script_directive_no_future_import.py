"""No script run through a Snakemake ``script:`` directive may start with ``from __future__ import ...``.

Snakemake prepends its own preamble to such a script, which pushes the future import past the start of the
module and raises ``SyntaxError: from __future__ imports must occur at the beginning of the file``. A unit
test that imports the module directly does not see it; only the workflow run fails (19-20 Oct 2026: the fleet
location check, the secured-capability audit and the pro rata turn-down re-pricing all hit it).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _script_directive_files():
    files = set()
    for smk in [ROOT / "Snakefile", *sorted((ROOT / "rules").glob("*.smk"))]:
        lines = smk.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            if re.match(r"^\s*script:\s*$", line) and i + 1 < len(lines):
                m = re.search(r"[\"']([^\"']+\.py)[\"']", lines[i + 1])
                if m:
                    path = (smk.parent / m.group(1)).resolve()
                    if path.exists():
                        files.add(path)
    return sorted(files)


def test_script_directive_files_are_found():
    assert len(_script_directive_files()) > 50


def test_no_future_import_in_script_directive_files():
    offenders = [str(p.relative_to(ROOT)) for p in _script_directive_files()
                 if re.search(r"^from __future__ import", p.read_text(encoding="utf-8", errors="replace"), re.M)]
    assert not offenders, f"remove `from __future__` from these script-directive files: {offenders}"
