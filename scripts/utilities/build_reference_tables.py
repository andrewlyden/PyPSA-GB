"""Write the documentation's generated reference tables: the configuration keys and the module index.

Two tables, both written to ``docs/source/reference/_generated/`` and included by the reference pages:

* **Configuration keys** (``config_keys_<domain>.md``, one per defaults file): every key in
  ``config/defaults/base.yaml`` and ``config/defaults/base/**`` (file path = key path), with its default
  and the comment that documents it in the file. Values come from the YAML itself; comments are matched
  to keys by a line scan (the block of comment lines just above a key, then its inline comment).
* **Module index** (``modules_<package>.md``, one per package under ``scripts/``): every module with the
  first paragraph of its docstring, read with ``ast`` so no module is imported.

Comments and docstrings are cleaned of working notes before they are published: a parenthetical or
sentence that names an internal planning document, a dated decision note or an internal item code is
dropped (``scrub``). The count of dropped fragments is printed.

Not run by any rule. Run it by hand after changing the defaults or adding modules:

    python scripts/utilities/build_reference_tables.py            # both tables
    python scripts/utilities/build_reference_tables.py --only config
    python scripts/utilities/build_reference_tables.py --check    # fail if a table is out of date
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = ROOT / "config" / "defaults"
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "docs" / "source" / "reference" / "_generated"

# Packages under scripts/ left out of the module index (working material, not part of the model).
EXCLUDED_MODULE_DIRS = {"__pycache__", "slides", "f20_parity"}

MAX_DEFAULT_CHARS = 70
MAX_SUMMARY_CHARS = 400

# ---------------------------------------------------------------------------------------------------
# Cleaning working notes out of published text
# ---------------------------------------------------------------------------------------------------

_MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
_FORBIDDEN = [
    r"docs/[\w\-./]*\.md",                       # internal plans and notes
    r"docs/internal", r"\bruns/",
    r"\b[Uu]ser\s+decision", r"\b[Uu]ser,\s*\d", r"\(user[,)]", r"\b[Uu]ser\s+\d{1,2}\s+(?:" + _MONTHS + r")",
    r"\b[Uu]ser\):", r"\buser\b[^.]*\b20\d\d\b",
    r"\bmeeting notes\b", r"\bresults review\b",
    r"\b[Dd]ecisions?\s+(?:\d+|[A-Z]\d+)\b", r"\b[Dd]ecided\b",
    r"\b(?:F\d{1,2}|OW\d{1,2}|BM1|MR\d|N\d{1,2}|D\d)\b(?![-.]\d)",
    r"\bitem\s+\d+", r"\bPhase\s+[A-Z]\b", r"\bGroup\s+N\b", r"\bPart\s+\d+(?:\.\d+)?[a-z]?\b",
    r"\b[Ss]tage\s+(?:[3-9]|\d[a-z])\b", r"\boption\s+[A-Z]\b",
    # a dated working note: "since 14 Nov 2026", "(10 Nov 2026)", "until 28 Oct 2026"
    r"\b(?i:since|until|from|as of|at|on|before|after)\s+\d{1,2}\s+(?:" + _MONTHS + r")[a-z]*\s+2026\b",
    r"^\s*\(?\s*\d{1,2}\s+(?:" + _MONTHS + r")[a-z]*\s+2026\b",
    r"\(\d{1,2}\s+(?:" + _MONTHS + r")[a-z]*\s+2026\)",
    # any day-month-2026 date: in these comments and docstrings it marks a working note
    r"\b\d{1,2}\s+(?:" + _MONTHS + r")[a-z]*\s+2026\b",
    r"\bF\d{1,2}\.\d+\b", r"\breview\s+(?:of\s+)?\d", r"\brebuild batch\b",
]
_FORBIDDEN_RE = re.compile("|".join(f"(?:{p})" for p in _FORBIDDEN))
_PAREN_RE = re.compile(r"\s*\([^()]*\)")
_SENTENCE_RE = re.compile(r"(?<=[.;!?])\s+(?=[A-Z0-9(`\"'])")

scrub_count = 0


def scrub(text: str) -> str:
    """Drop the parentheticals, then the sentences, that carry working notes; keep the rest."""
    global scrub_count

    def paren(match: re.Match) -> str:
        global scrub_count
        if _FORBIDDEN_RE.search(match.group(0)):
            scrub_count += 1
            return ""
        return match.group(0)

    text = _PAREN_RE.sub(paren, text)
    kept = []
    for sentence in _SENTENCE_RE.split(text):
        if _FORBIDDEN_RE.search(sentence):
            scrub_count += 1
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def md_escape(text: str) -> str:
    """Make free text safe inside a MyST table cell."""
    text = text.replace("\\", "\\\\").replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("$", "\\$").replace("*", "\\*").replace("{{", "{ {")
    if text.count("`") % 2:
        text = text.replace("`", "'")
    return text


def code(value: str) -> str:
    """Inline code safe inside a table cell."""
    value = value.replace("|", "\\|")
    fence = "``" if "`" in value else "`"
    return f"{fence}{value}{fence}"


# ---------------------------------------------------------------------------------------------------
# Configuration keys
# ---------------------------------------------------------------------------------------------------

_KEY_RE = re.compile(r"""^(?P<indent>\s*)(?P<dash>-\s+)?(?P<key>"[^"]+"|'[^']+'|[^\s#:'"{}\[\],&*!|>][^#:]*?)\s*:(?:\s+|$)(?P<rest>.*)$""")


def split_inline_comment(rest: str) -> tuple[str, str]:
    """Split ``value  # comment`` outside quotes."""
    quote = None
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'" and (i == 0 or rest[i - 1] in " [{,:"):
            quote = ch
        elif ch == "#" and (i == 0 or rest[i - 1] in " \t"):
            return rest[:i].rstrip(), rest[i + 1:].strip()
    return rest.rstrip(), ""


def comment_text(line: str) -> str | None:
    """The text of a comment line, or None for a decoration line (``#``, ``# ----``)."""
    text = line.strip()[1:].strip()
    if not text or re.fullmatch(r"[-=#*_~ ]+", text):
        return None
    if text.isupper() and len(text) < 60:     # a section banner ("SOLVER CONFIGURATION")
        return None
    return text


def scan_comments(path: Path) -> dict[tuple[str, ...], str]:
    """Map each key path in the file (relative to the file) to its documenting comment."""
    comments: dict[tuple[str, ...], str] = {}
    stack: list[tuple[int, str]] = []          # (indent, key) of the open mappings
    pending: list[str] = []
    block_scalar_indent: int | None = None
    list_indent: int | None = None              # inside a block sequence of mappings: keys are values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if block_scalar_indent is not None:
            if indent > block_scalar_indent:
                continue
            block_scalar_indent = None
        if line.lstrip().startswith("#"):
            text = comment_text(line)
            if text is not None:
                pending.append(text)
            continue
        if list_indent is not None:
            if indent > list_indent or (indent == list_indent and line.lstrip().startswith("-")):
                pending = []
                continue
            list_indent = None
        match = _KEY_RE.match(line)
        if match is None or match.group("dash"):
            if line.lstrip().startswith("-") and stack:
                list_indent = indent
            pending = []
            continue
        key = match.group("key").strip().strip("\"'")
        value, inline = split_inline_comment(match.group("rest"))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        key_path = tuple(k for _, k in stack) + (key,)
        parts = pending + ([inline] if inline else [])
        if parts:
            comments[key_path] = " ".join(parts)
        pending = []
        if value in ("", ) or value.startswith("&"):
            stack.append((indent, key))
        if re.match(r"^[|>][-+0-9]*$", value):
            block_scalar_indent = indent
    return comments


def render_default(value) -> str:
    if isinstance(value, dict):
        return "(mapping)" if value else code("{}")
    if isinstance(value, list):
        if value and all(isinstance(v, dict) for v in value):
            return f"(list of {len(value)} entries)"
        text = yaml.safe_dump(value, default_flow_style=True, width=10_000).strip()
    elif value is None:
        text = "null"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, str):
        text = value if value else '""'
    else:
        text = str(value)
    if len(text) > MAX_DEFAULT_CHARS:
        if isinstance(value, list):
            return f"(list of {len(value)} items)"
        text = text[:MAX_DEFAULT_CHARS - 3] + "..."
    return code(text)


def walk(data, prefix: tuple[str, ...] = ()):
    """Yield (path, value) for every key, depth first in file order."""
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        path = prefix + (str(key),)
        yield path, value
        if isinstance(value, dict):
            yield from walk(value, path)


def defaults_files() -> list[tuple[Path, tuple[str, ...]]]:
    """(file, key prefix) for base.yaml and every file under base/, in key order."""
    files = [(DEFAULTS / "base.yaml", ())]
    for path in sorted((DEFAULTS / "base").rglob("*.yaml")):
        files.append((path, path.relative_to(DEFAULTS / "base").with_suffix("").parts))
    return sorted(files, key=lambda item: item[1])


def config_tables() -> dict[str, str]:
    """{output file name: markdown} for every defaults file."""
    tables = {}
    for path, prefix in defaults_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        comments = scan_comments(path)
        name = ".".join(prefix) if prefix else "top level"
        rel = path.relative_to(ROOT).as_posix()
        lines = [f"## {code(name) if prefix else 'Top-level keys'}", "",
                 f"File: `{rel}`. Keys below are relative to {code(name) if prefix else 'the top level'}.", ""]
        header = _file_header(path)
        if header:
            lines += [md_escape(header), ""]
        lines += ["| Key | Default | Description |", "|---|---|---|"]
        rows = 0
        for key_path, value in walk(data):
            description = md_escape(scrub(comments.get(key_path, "")))
            lines.append(f"| {code('.'.join(key_path))} | {render_default(value)} | {description} |")
            rows += 1
        if rows == 0:
            lines.append("| (none) | | |")
        stem = "config_keys_" + ("_".join(prefix) if prefix else "top_level")
        tables[stem + ".md"] = "\n".join(lines) + "\n"
    return tables


def _file_header(path: Path) -> str:
    """The comment block at the top of a file, before its first key, if it is followed by a blank line."""
    head = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip().startswith("#"):
            text = comment_text(raw)
            if text is not None:
                head.append(text)
            continue
        if not raw.strip():
            break
        return ""                        # the block documents the first key, not the file
    return scrub(" ".join(head))


# ---------------------------------------------------------------------------------------------------
# Module index
# ---------------------------------------------------------------------------------------------------

def summary(path: Path) -> str:
    """The first paragraph of a module's docstring, cleaned and shortened."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return ""
    doc = ast.get_docstring(tree) or ""
    paragraph = doc.strip().split("\n\n")[0]
    paragraph = " ".join(line.strip() for line in paragraph.splitlines())
    paragraph = scrub(paragraph)
    if len(paragraph) > MAX_SUMMARY_CHARS:
        cut = paragraph[:MAX_SUMMARY_CHARS].rsplit(" ", 1)[0]
        paragraph = cut + " ..."
    return md_escape(paragraph)


def module_tables() -> dict[str, str]:
    """{output file name: markdown}: one table per package directory under scripts/."""
    groups: dict[str, list[Path]] = {}
    for path in sorted(SCRIPTS.rglob("*.py")):
        rel = path.relative_to(SCRIPTS)
        if any(part in EXCLUDED_MODULE_DIRS for part in rel.parts):
            continue
        package = rel.parts[0] if len(rel.parts) > 1 else "(top level)"
        groups.setdefault(package, []).append(path)
    tables = {}
    for package, paths in sorted(groups.items()):
        heading = f"`scripts/{package}/`" if package != "(top level)" else "`scripts/` (top level)"
        lines = [f"## {heading}", ""]
        init = SCRIPTS / package / "__init__.py"
        intro = summary(init) if init.exists() else ""
        if intro:
            lines += [intro, ""]
        lines += ["| Module | Summary |", "|---|---|"]
        for path in paths:
            if path.name == "__init__.py":
                continue
            rel = path.relative_to(ROOT).as_posix()
            lines.append(f"| {code(rel)} | {summary(path) or '(no module docstring)'} |")
        stem = "modules_" + (package if package != "(top level)" else "top_level")
        tables[stem + ".md"] = "\n".join(lines) + "\n"
    return tables


# ---------------------------------------------------------------------------------------------------

def write(tables: dict[str, str], check: bool) -> list[str]:
    stale = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in tables.items():
        target = OUT / name
        old = target.read_text(encoding="utf-8") if target.exists() else None
        if old != text:
            stale.append(name)
            if not check:
                target.write_text(text, encoding="utf-8", newline="\n")
    # conf.py requires a values.json in every _generated folder (it holds the pages' substitutions)
    values = OUT / "values.json"
    content = json.dumps({"generator": "scripts/utilities/build_reference_tables.py", "values": {}}, indent=2) + "\n"
    if not values.exists() or values.read_text(encoding="utf-8") != content:
        stale.append(values.name)
        if not check:
            values.write_text(content, encoding="utf-8", newline="\n")
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=["config", "modules"], help="write one table set only")
    parser.add_argument("--check", action="store_true", help="write nothing; fail if a table is out of date")
    args = parser.parse_args(argv)
    tables: dict[str, str] = {}
    if args.only in (None, "config"):
        tables.update(config_tables())
    if args.only in (None, "modules"):
        tables.update(module_tables())
    stale = write(tables, args.check)
    print(f"{len(tables)} tables; {len(stale)} {'out of date' if args.check else 'written'}; "
          f"{scrub_count} working-note fragments left out")
    if args.check and stale:
        print("out of date: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
