# PyPSA-GB tests

Run everything from the repository root with the `pypsa-gb` environment active.
Many tests read paths relative to the root (`data/...`, `config/...`).

## Layout

```
tests/
├── conftest.py        # shared fixtures, marker registration, sys.path setup
├── test_smoke.py      # imports and basic sanity checks
├── fixtures/          # shared fixture modules (fes_fixtures.py)
├── helpers/           # test helpers (requires_outputs guard)
├── unit/              # about 240 test modules, one or more per script
│   └── conftest.py    # loads the FES fixtures from fixtures/
└── workflow/          # 3 modules: scenario validation, historical and future pipelines
```

`pytest.ini` sets `testpaths = tests`, so a bare `pytest` collects all three layers.

## Running

```bash
pytest                                  # everything
pytest tests/test_smoke.py              # smoke
pytest tests/unit                       # unit
pytest tests/workflow                   # workflow
pytest tests/unit/test_spatial_utils.py # one module
pytest -k "cfd and not nuclear"         # by name
```

## Markers

Markers are registered in `tests/conftest.py` (and `workflow` in `pytest.ini`).

| Marker | Meaning |
|--------|---------|
| `smoke` | Quick sanity check |
| `unit` | Tests a single function or module |
| `integration` | Needs several components together |
| `workflow` | Part of a workflow (pipeline) test |
| `slow` | Slow; may be skipped in quick runs |
| `requires_data` | Needs specific input files under `data/` |
| `requires_solver` | Needs an optimisation solver (Gurobi or HiGHS) |
| `requires_outputs` | Reads workflow outputs under `resources/` |

## Tests that need generated outputs

`resources/` is produced by the Snakemake workflow and is not in the repository.
Tests that read it use `tests.helpers.requires_outputs(path, ...)`, which applies the
`requires_outputs` marker and skips the test while any named path is missing. On a
fresh clone these tests skip; after the relevant workflow outputs are built they run.
The workflow pipeline tests detect outputs the same way and skip when none are found.

```python
from tests.helpers import requires_outputs

@requires_outputs("resources/analysis/CP30_2030_ro_scheme_summary.json")
def test_ro_summary_reports_both_relief_cases():
    ...
```

## Continuous integration

CI (`.github/workflows/tests.yml`, on pull requests and on pushes to master and release branches) runs the tests that need neither a solver nor workflow outputs:

```bash
pytest
```

## Adding tests

- Put function-level tests in `unit/` as `test_<module>.py`; end-to-end checks in `workflow/`.
- Build small inputs in the test (or `tmp_path`); never write files into `tests/`.
- Guard any read of `resources/` with `requires_outputs`, and mark solver-dependent
  tests `requires_solver`.
