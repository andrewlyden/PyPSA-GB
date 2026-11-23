"""Fingerprint of the project code an output was built with (user, 15 Nov 2026).

Snakemake's metadata records a checksum of each small declared input, but most rules run ``python scripts/...``
without declaring the script, and no rule declares the project modules a script imports. A fix copied in with its old
file time (to avoid a re-solve) therefore left outputs built with the old code looking current. This module
fingerprints the code behind a shell command: every ``scripts/...py`` path and ``-m scripts.x.y`` module it names, plus
every project module those import, transitively (``scripts.*`` only; third-party code is not followed). The results
audit keeps a ledger of the fingerprint each output was built with and flags outputs whose code has changed since.
"""
from __future__ import annotations

import ast
import hashlib
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = re.compile(r"(scripts/[\w./-]+?\.py)\b")
MODULE_FLAG = re.compile(r"-m\s+(scripts(?:\.\w+)+)")


def entry_files(command: str, root: Path = ROOT) -> list[Path]:
    """The project scripts a shell command runs (paths and ``-m`` modules), those that exist."""
    found = [root / p for p in SCRIPT_PATH.findall(command or "")]
    for module in MODULE_FLAG.findall(command or ""):
        found.append(module_file(module, root))
    return sorted({p for p in found if p is not None and p.is_file()})


def module_file(module: str, root: Path = ROOT) -> Path | None:
    """The file of a ``scripts.x.y`` module (``scripts/x/y.py`` or ``scripts/x/y/__init__.py``)."""
    base = root.joinpath(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


@lru_cache(maxsize=None)
def _imports(path: Path, root: Path) -> tuple[Path, ...]:
    """Project modules ``path`` imports (``scripts.*``), statically."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return ()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names if a.name.startswith("scripts."))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("scripts"):
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)   # `from scripts.x import module`
    files = {module_file(name, root) for name in names}
    return tuple(sorted(p for p in files if p is not None))


#: Test and harness tooling, never model code (user decision, 15 Nov 2026): the F20 parity harness's module holds the
#: harness's rule lists, read only by its own test, and a path helper (``parity_path``) two bill scripts import. A
#: classification edit there marked every register stale; it is not a code input of any rule (nor followed further).
TOOLING_PREFIXES = ("scripts/public_ownership/f20_parity/",)


def is_tooling(path: Path, root: Path = ROOT) -> bool:
    try:
        rel = str(Path(path).resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return False
    return rel.startswith(TOOLING_PREFIXES)


def closure(entries: list[Path], root: Path = ROOT) -> list[Path]:
    """``entries`` and every project module they import, transitively (tooling modules left out)."""
    seen: set[Path] = set()
    todo = list(entries)
    while todo:
        path = todo.pop()
        if path in seen or is_tooling(path, root):
            continue
        seen.add(path)
        todo.extend(_imports(path, root))
    return sorted(seen)


def fingerprint(files: list[Path], root: Path = ROOT) -> str:
    """sha256 over each file's repository path and content."""
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def command_fingerprint(command: str, root: Path = ROOT) -> tuple[str | None, list[str]]:
    """The fingerprint of the code a shell command runs, and the files it covers (None when it runs no project code)."""
    files = closure(entry_files(command, root), root)
    if not files:
        return None, []
    return fingerprint(files, root), [str(p.relative_to(root)).replace("\\", "/") for p in files]
