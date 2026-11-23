"""F20 phase 0: the parity harness (scripts/public_ownership/f20_parity) and the output redirection it relies on."""

import json
from pathlib import Path
import re

import pytest
import yaml

from scripts.public_ownership.f20_parity import make_reduced_config as cut
from scripts.public_ownership.f20_parity import run_parity

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "rules" / "public_ownership.smk"


def _rule_blocks(text: str) -> dict[str, str]:
    """``{rule name: its text}`` from a rule file (each rule up to the next top-level line)."""
    blocks, name, lines = {}, None, []
    for line in text.splitlines():
        match = re.match(r"^rule (\w+):", line)
        if match or (name and line and not line[0].isspace() and not line.startswith("#")):
            if name:
                blocks[name] = "\n".join(lines)
            name, lines = (match.group(1), [line]) if match else (None, [])
            continue
        if name:
            lines.append(line)
    if name:
        blocks[name] = "\n".join(lines)
    return blocks


def _directive(block: str, directive: str) -> str:
    out, inside = [], False
    for line in block.splitlines():
        if re.match(r"^    \w+:", line):
            inside = line.strip().startswith(f"{directive}:")
            continue
        if inside:
            out.append(line)
    return "\n".join(out)


def test_every_ownership_rule_is_classified():
    rules = set(_rule_blocks(RULES.read_text(encoding="utf-8")))
    allowed, excluded = set(run_parity.ALLOWED_RULES), set(run_parity.EXCLUDED_RULES)
    assert not allowed & excluded
    assert rules == allowed | excluded, (sorted(rules - allowed - excluded), sorted((allowed | excluded) - rules))


def test_allowed_rules_write_only_into_the_ownership_root():
    blocks = _rule_blocks(RULES.read_text(encoding="utf-8"))
    for name in run_parity.ALLOWED_RULES:
        output = _directive(blocks[name], "output")
        if name == "public_ownership_parity":
            assert not output.strip()
            continue
        assert "{po_resources_path}/public_ownership" in output, name
        for forbidden in ("{resources_path}", "notebooks/", "BILL_DOCS", "CM_OUT", "_PO_SNAPSHOT", "\"resources/"):
            assert forbidden not in output, (name, forbidden)


def test_the_ownership_folder_is_one_variable():
    text = RULES.read_text(encoding="utf-8")
    assert "{resources_path}/public_ownership" not in text
    assert 'po_resources_path = str(config.get("po_resources_path") or resources_path)' in text
    snakefile = (ROOT / "Snakefile").read_text(encoding="utf-8")
    assert 'resources_path = str(config.get("resources_path") or "resources")' in snakefile
    assert "_f20_check_roots(resources_path" in snakefile


def test_check_roots(tmp_path):
    live, run_root = tmp_path / "live" / "resources", tmp_path / "runs" / "new" / "resources"
    with pytest.raises(ValueError, match="parity run only"):
        run_parity.check_roots(live, run_root, None)
    with pytest.raises(ValueError, match="parity run only"):
        run_parity.check_roots(live, None, True)
    with pytest.raises(ValueError, match="must not hold one another"):
        run_parity.check_roots(live, live / "public_ownership", True)
    with pytest.raises(ValueError, match="must not hold one another"):
        run_parity.check_roots(live, tmp_path / "live", True)
    run_parity.check_roots(live, run_root, True)


def test_the_snakemake_command():
    command = run_parity.snakemake_command("C:/live/resources", "C:/runs/new/resources", 4, 20000, True)
    assert command[1:4] == ["-m", "snakemake", "public_ownership_parity"]
    allowed = command[command.index("--allowed-rules") + 1:]
    assert allowed[: len(run_parity.ALLOWED_RULES)] == list(run_parity.ALLOWED_RULES)
    assert command[-3:] == ["resources_path=C:/live/resources", "po_resources_path=C:/runs/new/resources",
                            "f20_parity=true"]
    assert "--dry-run" in command and "--rerun-triggers" in command and "mem_mb=20000" in command


def test_the_environment(tmp_path):
    env = run_parity.parity_env("C:/live/resources", str(tmp_path / "timing"))
    assert env[run_parity.INPUT_ENV] == "C:/live/resources" and env[run_parity.TIMING_ENV].endswith("timing")
    assert run_parity.TIMING_ENV not in run_parity.parity_env("C:/live/resources", None)
    env = run_parity.parity_env("C:/live/resources", None, "C:/runs/new/resources")
    assert env[run_parity.PO_ENV] == "C:/runs/new/resources"
    assert run_parity.PO_ENV not in run_parity.parity_env("C:/live/resources", None)
    assert env["CONDA_AUTO_ACTIVATE_BASE"] == "false"


def test_long_command_rules_run_in_powershell_on_windows():
    """Their commands pass cmd.exe's 8,191 characters with the input tree's absolute paths (14 Oct 2026)."""
    assert set(run_parity.LONG_COMMAND_RULES) <= set(run_parity.ALLOWED_RULES)
    command = run_parity.snakemake_command("C:/live/resources", "C:/runs/new/resources", 4, None, True)
    shell = run_parity.parity_shell()
    if shell is None:
        assert "--set-resources" not in command
    else:
        assert shell.lower().endswith("powershell.exe")
        for rule in run_parity.LONG_COMMAND_RULES:
            assert f"{rule}:shell_exec='{shell}'" in command


def test_parity_path(monkeypatch):
    """Paths named in data (the bill assumptions) are redirected in a parity run only."""
    account = "resources/public_ownership/cp30_2030_cp30_network_public_account_summary.json"
    assert run_parity.parity_path(account, "resources", "resources") == account
    assert run_parity.parity_path(account, "C:/live/resources", "C:/runs/new/resources") ==         "C:/runs/new/resources/public_ownership/cp30_2030_cp30_network_public_account_summary.json"
    assert run_parity.parity_path("resources/market/x.csv", "C:/live/resources", "C:/runs/new/resources") ==         "C:/live/resources/market/x.csv"
    assert run_parity.parity_path("data/x.csv", "C:/live/resources", "C:/runs/new/resources") == "data/x.csv"
    monkeypatch.delenv(run_parity.INPUT_ENV, raising=False)
    monkeypatch.delenv(run_parity.PO_ENV, raising=False)
    assert run_parity.parity_path(account) == account
    # The input tree alone never answers for this layer's own outputs.
    monkeypatch.setenv(run_parity.INPUT_ENV, "C:/live/resources")
    assert run_parity.parity_path(account) == account
    monkeypatch.setenv(run_parity.PO_ENV, "C:/runs/new/resources")
    assert run_parity.parity_path(account).startswith("C:/runs/new/resources/public_ownership/")


def test_run_refuses_a_root_inside_the_input_tree(tmp_path):
    live = tmp_path / "live" / "resources"
    (live / "market").mkdir(parents=True)
    with pytest.raises(ValueError, match="must not hold one another"):
        run_parity.run(live / "parity", live, dry_run=True)
    assert not (live / "parity").exists()


def test_freeze(tmp_path):
    tree = tmp_path / "live"
    po = tree / "resources" / "public_ownership"
    (po / "cp30_2030_cp30_network_price_level_prices").mkdir(parents=True)
    (po / "cp30_2030_cp30_network_exchequer_ledger.json").write_text("{}")
    (po / "cp30_2030_cp30_network_asset_finance_register.csv").write_text("stale")
    (po / "cp30_2030_cp30_network_price_level_prices" / "Jan_wholesale_price.csv").write_text("a\n1\n")
    (po / "fes_ht_2050_exchequer_ledger.json").write_text("{}")
    dest = tmp_path / "reference"
    manifest = run_parity.freeze(tree, dest)
    assert sorted(f["path"] for f in manifest["files"]) == [
        "cp30_2030_cp30_network_exchequer_ledger.json", "cp30_2030_cp30_network_price_level_prices/Jan_wholesale_price.csv"]
    assert (dest / "resources" / "public_ownership" / "cp30_2030_cp30_network_exchequer_ledger.json").exists()
    assert json.loads((dest / run_parity.FREEZE_MANIFEST).read_text())["kind"] == "freeze"
    assert (po / "cp30_2030_cp30_network_asset_finance_register.csv").read_text() == "stale"    # the source is untouched
    with pytest.raises(FileExistsError):
        run_parity.freeze(tree, dest)
    with pytest.raises(ValueError, match="outside the tree"):
        run_parity.freeze(tree, tree / "frozen")


def test_path_replacements(tmp_path):
    run_root, frozen = tmp_path / "run", tmp_path / "frozen"
    run_root.mkdir()
    frozen.mkdir()
    (run_root / run_parity.RUN_MANIFEST).write_text(json.dumps(
        {"path_strings": ["C:/runs/new/resources", "C:/live/resources"]}))
    (frozen / run_parity.FREEZE_MANIFEST).write_text(json.dumps({"path_strings": ["resources"]}))
    pairs = run_parity.path_replacements(run_root, frozen)
    assert ("C:/runs/new/resources", "resources") in pairs and ("C:\\live\\resources", "resources") in pairs
    assert all(new == "resources" for _, new in pairs) and ("resources", "resources") not in pairs
    assert len(pairs[0][0]) >= len(pairs[-1][0])


# ── The reduced config ────────────────────────────────────────────────────────

YAML_TEXT = """\
top: 1
public_ownership:
  enabled: true
  assumptions_snapshot: resources/public_ownership/model.yaml   # an input
  sensitivities:
    named:
      a: {scenario: private}
      b:
        extends: a
        x: 1
      c: {scenario: private}
    # a comment inside
    toggle_grids:
      enabled: true
      max_cases: 6000
      grids:
        p1:
          - {token: one, lever: l1, value: v}
          - {selector: net, lever: networks, options: {private: private, acquired: acquired}}
          - {token: pass, lever: l2, value: w, when: {net: acquired}}
        p2:
          - {token: two, lever: l3, value: v}
      bound_grids:
        p1:
          - {token: three, lever: l4, value: v}
    bounding_pair: false
  # a comment before the next key
  later: [resources/x.csv, data/y.csv]
other:
  path: "resources/CP30/a.csv"
"""


def test_find_and_replace_block():
    lines = YAML_TEXT.split("\n")
    start, end, indent = cut.find_block(lines, ("public_ownership", "sensitivities", "toggle_grids"))
    assert lines[start].strip() == "toggle_grids:" and indent == 4
    assert lines[end - 1].strip().startswith("- {token: three") and lines[end].strip() == "bounding_pair: false"
    out = cut.replace_block(YAML_TEXT, ("public_ownership", "sensitivities", "toggle_grids"), {"enabled": False})
    data = yaml.safe_load(out)
    assert data["public_ownership"]["sensitivities"]["toggle_grids"] == {"enabled": False}
    assert data["public_ownership"]["sensitivities"]["bounding_pair"] is False
    assert "# a comment before the next key" in out and "# a comment inside" in out
    assert out.replace("\n", "\r\n") == cut.replace_block(YAML_TEXT.replace("\n", "\r\n"),
                                                          ("public_ownership", "sensitivities", "toggle_grids"),
                                                          {"enabled": False})
    with pytest.raises(KeyError):
        cut.find_block(lines, ("public_ownership", "absent"))


def test_reduce_grids():
    grids = yaml.safe_load(YAML_TEXT)["public_ownership"]["sensitivities"]["toggle_grids"]
    spec = {"toggle_grids": {"max_cases": 50, "grids": {"p1": ["one", "pass", "gone"]},
                             "bound_grids": {"p1": ["three"]}}}
    out, notes = cut.reduce_grids(grids, spec)
    assert out["max_cases"] == 50 and out["enabled"] is True
    assert [a.get("token") for a in out["grids"]["p1"]] == ["one"]          # pass needs its selector
    assert "p2" not in out["grids"] and out["bound_grids"]["p1"][0]["token"] == "three"
    assert any("pass dropped" in n for n in notes) and any("'gone'" in n or "gone" in n for n in notes)
    assert any("p2: no axis kept" in n for n in notes)


def test_reduce_named_keeps_what_a_kept_case_extends():
    named = yaml.safe_load(YAML_TEXT)["public_ownership"]["sensitivities"]["named"]
    assert list(cut.reduce_named(named, ["b"])) == ["a", "b"]
    assert cut.reduce_named(named, None) == named
    with pytest.raises(ValueError, match="does not have"):
        cut.reduce_named(named, ["zz"])


def test_rewrite_resource_paths():
    out, count = cut.rewrite_resource_paths(YAML_TEXT, "C:/live/resources/")
    assert count == 3
    data = yaml.safe_load(out)
    assert data["public_ownership"]["assumptions_snapshot"] == "C:/live/resources/public_ownership/model.yaml"
    assert data["public_ownership"]["later"] == ["C:/live/resources/x.csv", "data/y.csv"]
    assert data["other"]["path"] == "C:/live/resources/CP30/a.csv"
    assert "# an input" in out


def test_prune_case_selections():
    po = {"reference_run_groups": {"twin": {"case_selection": {"kinds": ["scenario"], "cases": ["a", "gone"]}},
                                   "plain": {"model_year": 2030}}}
    assert cut.prune_case_selections(po, {"a"}) == {"twin": {"kinds": ["scenario"], "cases": ["a"]}}
    assert cut.prune_case_selections(po, {"a", "gone"}) == {}


def test_the_tree_config_cuts_to_a_small_list_that_keeps_every_non_grid_case():
    """On this tree's own config: only the grid block changes, and the cut validates."""
    from config.config_loader import load_defaults
    from scripts.public_ownership.reference_runs import case_kind, configured_case_specs

    texts = cut.read_defaults_texts(ROOT / "config" / "defaults")
    out, record = cut.build(texts, yaml.safe_load(cut.DEFAULT_SPEC.read_text(encoding="utf-8")))
    full, reduced = cut.assemble(texts), cut.assemble(out)
    assert full == load_defaults(ROOT / "config")         # the tool reads the tree as the loader does
    assert set(record["changed_files"]) <= {cut.SENSITIVITIES_FILE, cut.RUN_GROUPS_FILE}
    grids_full = full["public_ownership"]["sensitivities"].pop("toggle_grids")
    grids_reduced = reduced["public_ownership"]["sensitivities"].pop("toggle_grids")
    full_twins = {g: b.pop("case_selection", None) for g, b in full["public_ownership"]["reference_run_groups"].items()}
    reduced_twins = {g: b.pop("case_selection", None)
                     for g, b in reduced["public_ownership"]["reference_run_groups"].items()}
    reduced_po = cut.assemble(out)["public_ownership"]
    assert full == reduced                               # nothing else moved
    assert grids_reduced != grids_full and grids_reduced["max_cases"] == 600
    for group, selection in reduced_twins.items():      # a twin keeps its own kinds; only dropped cases go
        if selection is not None:
            assert set(selection.get("cases") or []) <= set(full_twins[group].get("cases") or [])
    specs = configured_case_specs(reduced_po)
    kinds = {case_kind(s) for s in specs}
    assert {"scenario", "one_at_a_time", "named", "preset", "bound", "grid"} <= kinds
    assert len(specs) < 300 and not [n for n in record["notes"] if "has no" in n]
