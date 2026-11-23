# Parity harness

Work that speeds up the 2030 bill and ownership layer must not change a
number.  This folder is how that is checked: a frozen reference, a reduced config that runs the whole layer
in minutes, old and new runs into their own roots, and an exact comparison.

| File | What it does |
| --- | --- |
| `../compare_outputs.py` | Compares two output trees file by file (CSV, Parquet, the register through `register_io`, JSON); exact unless `--atol`/`--rtol`; exit 1 on any difference |
| `make_reduced_config.py` | Cuts `config/defaults.yaml` to the reduced case list (`reduced_config.yaml`), in place with a backup, and reports its coverage |
| `reduced_config.yaml` | The cut: every non-grid case (118) plus a few axes of each grid, 141 cases and 75 register keys on 13 Oct 2026 |
| `run_parity.py` | `freeze` a reference, `run` the layer into a root, `compare` two roots |
| `../stage_timing.py` | Opt-in timing of the hot loops (`PYPSA_GB_STAGE_TIMING`); `run` turns it on and writes `ROOT/stage_times.md` |

Every command below runs from the parity tree's root (`C:\tmp\pypsa-f20`) with the `pypsa-gb` interpreter:

```powershell
$py = "<path to the pypsa-gb environment>\python.exe"
$live = "<path to the main working tree>"
```

Keep run roots outside both trees and outside OneDrive (`C:/f20_parity/...`).  Never junction or symlink
anything into the live tree's `resources/`.  Never run a parity run while Snakemake is running in the live
tree: the run reads the live outputs and would see half-written files.

## (a) Freeze the full reference (after tonight's bill run)

When the live bill run has finished and no Snakemake process is left in the live tree:

```powershell
& $py scripts/public_ownership/f20_parity/run_parity.py freeze --from $live --to C:/f20_parity/reference_2026-10-13 --dry-run
& $py scripts/public_ownership/f20_parity/run_parity.py freeze --from $live --to C:/f20_parity/reference_2026-10-13
```

It copies `resources/public_ownership/cp30_2030*` (every 2030 group, the P2/P3 bid groups and the CP.32
price-level twins; `--prefixes` to change) except the superseded CSV registers of 29 Sep, about 3 GB.  It
reads the live tree only.  `freeze_manifest.json` records every file, the live commit, a digest of its
uncommitted changes and the hashes of its three config files; `config_snapshot/` keeps the config itself.

## (b) The reduced config, run old and new

**1. Bring the parity tree to the live code.**  The reference and the old run must come from the same code
as the live run, plus phase 0 (which changes no number).  The live tree will gain three patches today (bill
wiring re-pointed at the new months, the I3 deferral, the CP.32 twins with their reduced case list); merge
them into this tree first.  Check with the config hashes: `config/defaults.yaml.f20_full.bak` (after step 2)
and `config/scenarios.yaml` must hash as in `freeze_manifest.json` (`source.config_sha256`).

**2. Cut the config** (in place; the full one is kept as `config/defaults.yaml.f20_full.bak`):

```powershell
& $py scripts/public_ownership/f20_parity/make_reduced_config.py build --in-place --input-resources "$live/resources"
& $py scripts/public_ownership/f20_parity/make_reduced_config.py report --out C:/f20_parity/reduced_coverage.json
```

The cut changes only `public_ownership.sensitivities.toggle_grids` (and a twin's `case_selection.cases`,
dropping any case the cut drops), so it does not depend on what today's patches put in the config.  With
`--input-resources` every `resources/...` value in the config (the workbook snapshot, the institution's
marginal costs and FTI settlement, the CP30 files) points at the live tree, in its 8.3 form (no spaces).  `report` must show
`lever_values_missing: []` and `comparators_missing: []`, and both balancing groups with cases.

**3. Old run** (before any phase 1 code is in the tree):

```powershell
& $py scripts/public_ownership/f20_parity/run_parity.py run C:/f20_parity/runs/old --input-resources "$live/resources" --dry-run
& $py scripts/public_ownership/f20_parity/run_parity.py run C:/f20_parity/runs/old --input-resources "$live/resources" --label old
```

Read `C:/f20_parity/runs/old/snakemake_dry_run.log` first: every job must be one of the ownership rules
(`run_parity.ALLOWED_RULES`) and every output under `C:/f20_parity/runs/old/resources/public_ownership`.
A `MissingInputException` naming a live file means the live tree lacks a solved input the config asks for.

**4. Phase 1 code goes in; new run** into its own root, same command with `runs/new` and `--label new`.

**5. Put the full config back** before committing anything:

```powershell
& $py scripts/public_ownership/f20_parity/make_reduced_config.py restore
```

What `run` does: Snakemake on `rule public_ownership_parity` (every ownership output `all` asks for, less
the reader-facing reports and the 2050 groups, plus the cost-calibration check) with
`--rerun-triggers mtime`, `--allowed-rules` the ownership rules only, and
`--config resources_path=<live resources> po_resources_path=ROOT/resources f20_parity=true`; environment
`PYPSA_GB_INPUT_RESOURCES=<live resources>` and `PYPSA_GB_STAGE_TIMING=ROOT/stage_timing`, the conda env
first on PATH.  It refuses a root that holds outputs (`--resume` to continue one), a root inside the live
tree, a cut config with `--full` and a full config without it.  `ROOT/run_manifest.json` records the code
state, config hashes, command and exit code; `ROOT/stage_times.md` is the stage-time table.

Fixes of 14 Oct 2026 (the first end-to-end runs):

- **Paths named in data.**  The bill assumptions name the battery top-up's account as a literal
  `resources/public_ownership/cp30_2030_public_account_summary.json`
  (`sourced_externals.*.derivation.cost_from.public_account`).  `run_parity.parity_path` redirects such a path
  (`resources/public_ownership/...` to the run root, any other `resources/...` to the input tree): in the rules
  (`_po_bill_support_inputs`) from `resources_path`/`po_resources_path`, in the projector from the environment
  (`PYPSA_GB_PO_RESOURCES`, set by `run`, beside `PYPSA_GB_INPUT_RESOURCES`).  Unset, every path is as before.
- **Files found by naming.**  `decompose_public_bm_bids.py` (the month files) and the ledger's state-bids
  re-trading (the forecast bills in `analysis/`) read `PYPSA_GB_INPUT_RESOURCES` when it is set.
- **Long commands.**  With the input tree's 8.3 path (51 characters) the settlement, price-level and
  strike-adjustment commands pass cmd.exe's 8,191 characters.  `run` gives those rules
  (`LONG_COMMAND_RULES`) `shell_exec=powershell.exe` through `--set-resources`; PowerShell passes up to 32,767.
  Git bash cannot: it silently truncates a native program's arguments at about 8,150 characters.  The
  environment sets `CONDA_AUTO_ACTIVATE_BASE=false`, or the profile's conda hook puts the base env first.
- **The cost-calibration check fails on today's register** (19 cohorts, max factor difference 0.136, on the
  live register too), so a run exits 1 on it.  Pass `--snakemake-args --keep-going` so the rest completes;
  `stage_times.md` is written on a failed run too (the manifest's `exit_code` says whether every stage ran).

## (c) Compare

```powershell
& $py scripts/public_ownership/f20_parity/run_parity.py compare C:/f20_parity/runs/old C:/f20_parity/runs/new --report C:/f20_parity/runs/new_vs_old.csv
```

Exact by default; exit 0 means every file matched.  Each root's own path strings (from its manifest: the
run root, the live root in both its forms) are rewritten to `resources` on both sides first, so a path a
script writes into its output is not a difference.  Anything after the two roots goes to
`compare_outputs.py` (`--atol`, `--rtol`, `--include`, `--exclude`, `--keys GLOB=col1,col2`,
`--ignore-keys REGEX`, `--ignore-columns REGEX`, `--json`).  The tool alone works on any two trees:

```powershell
& $py scripts/public_ownership/compare_outputs.py C:/f20_parity/runs/old/resources C:/f20_parity/runs/new/resources --quiet
```

**Before switching: one full-size comparison.**  Restore the config, then
`run C:/f20_parity/runs/full_new --input-resources "$live/resources" --full`, and
`compare C:/f20_parity/reference_2026-10-13 C:/f20_parity/runs/full_new`.  That holds only while this tree's
code (less F20) is the code that made the reference; otherwise make a full old run instead.  A full run
takes hours (about 4 h before F20), so run it when the laptop is free.

The guards stay in every pass: `verify_public_ownership_ownership_tables` and
`verify_public_ownership_cost_calibration` are in the run's DAG, the ledger checks the three-party identity
and the resource invariant, the public account `check_all_cases_repriced`.  Compare the
`*_ownership_tables_check.json` stamps too: any change to a bid case's register rows would force the 26
bid-month re-solves in the live tree.

## Stage timing on any stage

```powershell
$env:PYPSA_GB_STAGE_TIMING = "C:/f20_parity/timing_adhoc"     # unset (or 0) = off, the default
# ... run a stage by hand or through Snakemake ...
& $py scripts/public_ownership/stage_timing.py summarise C:/f20_parity/timing_adhoc --out C:/f20_parity/timing_adhoc.md
```

One CSV per process; labels are inclusive and may nest (a timed helper inside a timed function counts in
both).  Timed: the tariff's `revenue_requirement`, `build_bill`, `reconcile`, `multi_year` and the other
per-case steps; the ledger's `build` and each pass over `CaseRows` (`three_party_columns`,
`household_totals`, `hpc_schedule_by_case`, `party_flow_rows`, `state_borrowing_gbp` ...) and
`measure_dispatch`; the bridge's `_technology_savings` and `_bridge`; the financing's per-key
`_key_account`, `_key_frame` (the tiling), `_book_ldes_floor_to_rows` with `public_account._floor_parts_original`
(the per-row floor loop), `RegisterWriter.write/close`; the public account's steps (laps in
`build_public_account`: the per-row maps are `per_row_lever_maps_and_booking`); `StrikeAdjustments.add`,
`per_row_savings` and `SavingsStream.__iter__`; and the register reads (`register_io.read_register`,
`iter_register_cases`, `CaseRows._case`).  A multiprocessing worker (phase 1) must call
`stage_timing.flush()` itself: workers exit without running `atexit`.

Phase 1 (13 Oct 2026) adds the labels of its own steps: `tariff.RevenueRequirementMemo.__call__` (with
`tariff.revenue_requirement` inside it counting only the cases computed), `bridge._SavingsMemo.savings` and
`bridge._new_cfd_saving_by_case`, `ledger._register_requests` and `register_io.CaseRows.prefetch` (the ledger's
one sweep; its passes then read memoised quantities), `register_io.RegisterReader.read` (the tariff's batches with
private read once, and every `CaseRows` read).  Every row carries the process's peak memory (`peak_mb`, the
`Peak (MB)` column of `stage_times.md`): the old and new runs' peaks per rule are what the rules' `mem_mb` should be
set from (phase 1 set conservative estimates on the bridge, the tariff and the ledger, and left the financing,
strike adjustments and public account at their 29 Sep values).

## Phase 2: one computation per fingerprint (27 Oct 2026)

`scripts/public_ownership/fingerprints.py` writes down what each stage's per-case result depends on.  The
register body (`body_key`): the register levers, the gated levers, the market-bill flag and, on the market
bill, the LDES floor basis and the base case (the two things the financing step books per case after the
key's account).  The stages then compute once per key and relabel:

| Stage | Once per | Key |
| --- | --- | --- |
| CfD savings (strike adjustments, public account) | `savings_key` | body, comparator, strike basis, amortisation |
| Strike adjustments | reads only the first case of each savings key (`SavingsStream.savings_batches`) | as above |
| Bridge | `bridge_key` | body, the summary's comparator |
| Tariff revenue requirement | `nationalised_tariff.revenue_requirement_key` + the call's arguments | body, `beccs_removal_funding` |
| Ledger register requests | `CaseRows(share_key=)` | body (every request reads the body alone) |

A register without its lever columns (a hand-built test frame) shares nothing.  Each stage recomputes the
first shared case and one in 50 after it in full and raises `fingerprints.FingerprintMismatch` on any
difference; `PYPSA_GB_F20_SAMPLE_EVERY` changes the rate (1: every shared case), `PYPSA_GB_F20_SHARE=0`
turns sharing off (each case computed as before).  The lever-read audit is
`tests/unit/test_f20_phase2_fingerprints.py`.
