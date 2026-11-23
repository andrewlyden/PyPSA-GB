# Data maintenance

How to bring an input up to date when its publisher releases a new version. Most inputs are named explicitly in a
rule or a registry, so an update is: add the new file, point the registry or rule at it, rebuild, and check the
result against the previous version.

## Release cycles

| Source | Frequency | Typical release |
|---|---|---|
| FES (NESO) | Annual | July |
| ETYS (NESO) | Annual | November |
| DUKES (DESNZ) | Annual | July (Chapter 5) |
| REPD (DESNZ) | Quarterly | January, April, July, October |
| TEC register (NESO) | Updated continuously | Monthly snapshots |
| ERA5 weather cutouts | As needed | 2010-2024 are on Zenodo |

## FES

FES data comes from two places: the NESO data portal (building blocks, building-block definitions, GSP information),
whose download addresses are in `data/FES/FES_api_urls.yaml`, and the FES data workbook in `data/FES/`.

1. Add the new edition's portal addresses under each table in `data/FES/FES_api_urls.yaml`, keyed by edition.
2. Add the data workbook to `data/FES/` and register it in `WORKBOOKS` in `scripts/sources/fes_adapter.py` (rule `FES_data`
   refuses an edition whose workbook is not registered).
3. Add the edition's GSP table, `data/network/ETYS/fes{edition}_regional_breakdown_gsp_info.csv`, and its GSP
   name clean-up rows in `data/FES/gsp_aliases.csv`. An edition with no alias rows fails; there is no silent
   fallback to an earlier edition.
4. Add the edition's pathways to `data/scenario_sources/pathways.csv`.
5. Rebuild the processed table: `snakemake resources/FES/FES_<edition>_data.csv --cores 1`.

For a new long-range model year (rather than a new edition), list what that year needs with:

```bash
python -m scripts.sources.new_fes_year --year <year>
```

## ETYS

1. Add the new Appendix B workbook to `data/network/ETYS/`. If it is a NESO website document that may not be
   redistributed, register it in `data/neso_documents.yaml` instead (path, URL, landing page, SHA-256) so rule
   `retrieve_neso_document` downloads it.
2. Add the edition to `ETYS_FILES` in `scripts/network_build/etys_file_registry.py` (file and sheet names).
3. Select it with `network.etys.year` in a scenario, or in `config/defaults/base/network.yaml` to change the
   default.
4. Rebuild a network and compare bus, line and transformer counts with the previous edition. Check the upgrade
   audit, `resources/network/{scenario}_etys_upgrade_audit.csv`, for rows that could not be represented.

## DUKES

DUKES Table 5.11 is read from `data/generators/DUKES_5.11_2025.xlsx` (one worksheet per year). A new release
replaces that file name in the rules that read it (`rules/generator_data_prep.smk`, `rules/generators.smk`, `rules/consumer_bill.smk`).
New stations without a location need an entry in `data/generators/dukes_power_station_coordinates.csv` or
`data/generators/coordinate_corrections.csv`.

## REPD

The default extract is `data/renewables/repd-q2-jul-2025.csv`. Earlier extracts stay for the historical builds that
name them. A new extract is added beside them and named in the rules and in the `fleet.station_registry.repd_file`
setting. REPD sometimes renames columns or technology labels between extracts; compare the headers before switching.

## TEC register

The generator data preparation rules read `data/generators/tec-register-02-september-2025.csv`
(`rules/generator_data_prep.smk`). Add a new snapshot beside it and change the path there.

## NESO website documents

Documents published on NESO's website (as opposed to the data portal) are fetched, not shipped. Each is listed in
`data/neso_documents.yaml` with its path, download URL, landing page, title and SHA-256. When NESO publishes a new
version:

1. Download it and compute its checksum (`sha256sum <file>`, or `certutil -hashfile <file> SHA256` on Windows).
2. Add or update the entry in `data/neso_documents.yaml`.
3. Run `python scripts/utilities/retrieve_neso_document.py "<path>"` to check that the entry downloads and verifies.

## Weather cutouts

Cutouts for 2010-2024 are downloaded from the Zenodo record by `Snakefile_cutouts`. Other years are built from ERA5
through atlite, which needs Copernicus Climate Data Store credentials. Years are chosen in
`config/cutouts_config.yaml`:

```bash
snakemake -s Snakefile_cutouts --configfile config/cutouts_config.yaml --cores 4
```

Cutouts are written to `resources/atlite/cutouts/uk-<year>.nc`; a copy in `data/atlite/cutouts/` is used first if
present.

## ELEXON data

ELEXON BMRS data used for calibration (physical notifications, market index prices, bid-offer data) is cached by
`scripts/utilities/elexon_cache.py`:

```bash
python scripts/utilities/elexon_cache.py status     # what is cached
python scripts/utilities/elexon_cache.py pack       # pack CSV caches into the compressed archive
python scripts/utilities/elexon_cache.py restore    # restore from the archive
```

The calibration scripts try the cache before calling the ELEXON API.

## Checking an update

1. Dry-run the workflow: `snakemake -n` lists what the change will rebuild.
2. Run the unit tests: `python -m pytest`.
3. Rebuild a small scenario (for example `Test_Market`) and compare capacity by carrier with a copy of the network
   kept from before the update:

```python
import pypsa

old = pypsa.Network("Test_Market_before.nc")          # a copy kept before the update
new = pypsa.Network("resources/network/Test_Market.nc")
diff = new.generators.groupby("carrier").p_nom.sum().sub(
    old.generators.groupby("carrier").p_nom.sum(), fill_value=0)
print(diff[diff.abs() > 1])                            # MW
```

4. For CP30 and FES builds, read the validation outputs (see {doc}`/reference/outputs`): they compare the build's
   capacity and energy with the publication it follows.

## Large files

The pre-commit hook refuses files over 10 MB (see {doc}`/development/contributing`). Large inputs belong on a
Zenodo record or behind a download rule, not in git history.
