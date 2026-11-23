# Contributing

Contributions are welcome: bug reports, data updates, fixes and new features. This page covers the set-up, the checks
a change must pass, and how to document it.

## Set-up

1. Fork the repository on GitHub and clone your fork.
2. Create the environment and activate it:

   ```bash
   conda env create -f envs/pypsa-gb.yaml
   conda activate pypsa-gb
   ```

3. Install the git hooks once:

   ```bash
   pre-commit install
   ```

4. Work on a branch (`git checkout -b fix/short-description`) and open a pull request against `master`.

## Pre-commit hooks

`.pre-commit-config.yaml` runs on every commit:

| Hook | What it checks |
|---|---|
| `check-added-large-files` | refuses any added file over 10 MB (`--maxkb=10240`) |
| `check-merge-conflict` | no conflict markers left |
| `check-yaml` | YAML parses (with `--unsafe`, since the config uses anchors and merge keys) |
| `ruff` | lint errors (see below) |

Large inputs belong on a Zenodo record or behind a download rule, not in git history. NESO website documents are
never committed: register them in `data/neso_documents.yaml` (see {doc}`/reference/data/maintenance`).

Ruff's settings are in `pyproject.toml`: line length 120, target Python 3.11, and the error rules only (`E9`, `F63`,
`F7`, `F82`: syntax errors, undefined names, invalid comparisons). `snakemake` is declared a builtin, because Snakemake
injects it into every script run by a rule. Run it by hand with `ruff check .`.

## Tests

Run the tests from the repository root with the `pypsa-gb` environment active. `pytest.ini` sets `testpaths = tests`;
`workflow` is registered there and the other markers in `tests/conftest.py`. `tests/README.md` describes the layout.

```bash
python -m pytest                                              # everything
python -m pytest tests/unit/test_spatial_utils.py             # one module
python -m pytest                                                     # what CI runs
```

| Marker | Meaning |
|---|---|
| `smoke` | quick sanity check |
| `unit` | one function or module |
| `integration` | several components together |
| `workflow` | part of a pipeline test |
| `slow` | may be skipped in quick runs |
| `requires_data` | needs specific files under `data/` |
| `requires_solver` | needs Gurobi or HiGHS |
| `requires_outputs` | reads workflow outputs under `resources/`; skips on a fresh clone |

When adding tests:

- Put function-level tests in `tests/unit/` as `test_<module>.py`, end-to-end checks in `tests/workflow/`.
- Build small inputs in the test or in `tmp_path`; never write files into `tests/`.
- Guard any read of `resources/` with `tests.helpers.requires_outputs`, and mark solver tests `requires_solver`.

## Continuous integration

`.github/workflows/tests.yml` runs on pushes to `master` and `release/**` branches and on pull requests. On Ubuntu with
the `envs/pypsa-gb-stable.yaml` environment it:

1. lints with `ruff check .`;
2. runs `python -m pytest` (tests that need workflow outputs skip);
3. dry-runs the workflow on a fresh checkout,
   `python -m snakemake -n --quiet rules --config scenario=Test_Market --cores 1`, which fails if any declared input
   of the default scenario is neither tracked nor made by a rule.

## Changing the model

- **Configuration:** a new setting goes in the base defaults (`config/defaults/base/...`, file path = key path) with
  a comment saying what it is, its units and its source, and the code default must equal it. Read configuration only
  through `config/config_loader.py`. Renaming a key means adding it to `RENAMED_KEYS` so the old path is refused
  with the new one named. Regenerate the key reference afterwards
  (`python scripts/utilities/build_reference_tables.py`).
- **Scenarios:** a new scenario writes `targets` and `method`. Check it with `python scripts/validate_scenarios.py`
  and `snakemake --config scenario=<id> -n`.
- **Rules:** one rule per step, inputs and outputs declared, a log file per rule, and the script under the matching
  `scripts/` folder. A rule that reads the defaults declares them as inputs. `rules/STYLE_GUIDE.md` has the details.
- **Data:** add the raw file under `data/`, the processing script under `scripts/`, the rule under `rules/`, and a
  row on {doc}`/reference/data/data_sources` and {doc}`/reference/licences`.
- **Errors:** fail with a message that names the cause. Do not fall back silently to other data.

## Documentation

The documentation is Sphinx with MyST Markdown under `docs/source/`. Build it with the requirements in
`docs/requirements.txt` (Read the Docs uses `.readthedocs.yaml`; notebooks need `pandoc`):

```bash
python -m sphinx -b html docs/source docs/build/html
```

- Pages under `model/` follow the order: What it does, Method and assumptions, Data, Configuration, Validation.
- Every configuration key named on a page must exist.
- Results from solved runs belong on `studies/` pages only. Their figures come from generated `_generated/values.json`
  files, used as `{{ name }}` substitutions.
- After changing the defaults or adding modules, regenerate the reference tables with
  `python scripts/utilities/build_reference_tables.py` (`--check` fails if a table is out of date).

## Reporting issues

Open an issue on [GitHub](https://github.com/andrewlyden/PyPSA-GB/issues) with the command you ran, the scenario
entry, the full error and the rule's log file from `logs/`. See also {doc}`troubleshooting`.

## Licence

Contributions are released under the project's MIT licence.
