# Data Maintenance

Guide for updating data files when new versions are released.

## Update Schedule

| Data Source | Frequency | Typical Release |
|-------------|-----------|-----------------|
| FES | Annual | July |
| DUKES | Annual | March (provisional), July (final) |
| REPD | Quarterly | Jan, Apr, Jul, Oct |
| TEC Register | Monthly | 1st of month |
| ETYS | Annual | November |
| ERA5 Cutouts | As needed | N/A |

## Updating FES Data

### When New FES is Released

1. **Download new data**:
   - Visit [NESO Data Portal](https://www.nationalgrideso.com/future-energy/future-energy-scenarios)
   - Download the Data Workbook (Excel)

2. **Update API configuration**:
   ```yaml
   # data/FES/FES_api_urls.yaml
   FES_2025:
     base_url: "https://api.neso.energy/..."
     capacity_endpoint: "/fes/capacity"
     demand_endpoint: "/fes/demand"
   ```

3. **Process new data**:
   ```bash
   snakemake resources/FES/FES_2025_data.csv -j 1 -R process_fes_data
   ```

4. **Update scenario defaults**:
   ```yaml
   # config/defaults.yaml
   FES_year: 2025
   ```

### Validating FES Update

```python
import pandas as pd

# Compare old and new
old = pd.read_csv("resources/FES/FES_2024_data.csv")
new = pd.read_csv("resources/FES/FES_2025_data.csv")

# Check capacity totals
print("2024 FES - 2035 Wind:", old[(old.year==2035) & (old.technology=='Wind')].capacity_mw.sum())
print("2025 FES - 2035 Wind:", new[(new.year==2035) & (new.technology=='Wind')].capacity_mw.sum())
```

## Updating DUKES Data

### When New DUKES is Released

1. **Download from GOV.UK**:
   - [DUKES Chapter 5](https://www.gov.uk/government/statistics/electricity-chapter-5-digest-of-united-kingdom-energy-statistics-dukes)
   - Download Table 5.11 (Major power producers)

2. **Place in data directory**:
   ```bash
   mv DUKES_5.11_2026.xlsx data/generators/
   ```

3. **Update configuration**:
   ```yaml
   # config/defaults.yaml
   dukes_file: "DUKES_5.11_2026.xlsx"
   ```

4. **Re-process generators**:
   ```bash
   snakemake resources/generators/DUKES/DUKES_2026_generators.csv -j 1 -R process_dukes
   ```

### DUKES Data Mapping

When new power stations appear:

1. **Check coordinates**: Verify grid reference is correct
2. **Map fuel type**: Add to `data/generators/fuel_mapping.yaml` if needed
3. **Verify capacity**: Cross-reference with TEC register

## Updating REPD Data

### Quarterly Update Process

1. **Download from GOV.UK**:
   - [REPD](https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract)

2. **Replace existing file**:
   ```bash
   mv repd-q3-oct-2025.csv data/renewables/
   ```

3. **Update reference**:
   ```yaml
   # config/defaults.yaml
   repd_file: "repd-q3-oct-2025.csv"
   ```

4. **Re-process renewables**:
   ```bash
   snakemake -R prepare_renewable_sites -j 1
   ```

### Handling REPD Changes

REPD format occasionally changes. Check:
- Column names haven't changed
- Technology categories are consistent
- Coordinate format is the same

## Updating TEC Register

### Monthly Update

1. **Download from NESO**:
   - [TEC Register](https://www.nationalgrideso.com/industry-information/codes/connection-codes/tec-register)

2. **Replace file**:
   ```bash
   mv tec-register-august-2025.csv data/generators/
   ```

3. **Update reference** in config

## Updating ETYS Network

### Annual Update (Major)

1. **Download Appendices**:
   - [ETYS Publication](https://www.nationalgrideso.com/research-and-publications/electricity-ten-year-statement-etys)
   - Get Appendix B Excel files

2. **Extract network data**:
   ```bash
   python scripts/extract_etys_network.py data/network/ETYS/ETYS_2024_appendix_b.xlsx
   ```

3. **Validate**:
   ```bash
   python scripts/validate_network_topology.py
   ```

4. **Update upgrades**:
   - Review planned reinforcements
   - Update `data/network/ETYS/ETYS_upgrades.xlsx`

## Generating New Weather Cutouts

### When to Generate

- Modeling a new historical year
- Updated ERA5 data available
- Different geographic scope needed

### Process

1. **Set up CDS API**:
   ```bash
   # Create ~/.cdsapirc
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR-API-KEY
   ```

2. **Generate cutout**:
   ```bash
   snakemake -s Snakefile_cutouts resources/atlite/GB_2023.nc -j 2
   ```

3. **Validate**:
   ```python
   import atlite
   cutout = atlite.Cutout("resources/atlite/GB_2023.nc")
   print(cutout)
   print(f"Time range: {cutout.data.time.min().values} to {cutout.data.time.max().values}")
   ```

## Validation After Updates

### Quick Validation

```bash
# Validate all scenarios
python scripts/validate_scenarios.py

# Check specific data
python scripts/validate_data_integrity.py
```

### Full Regression Test

After major updates, run a known scenario and compare:

```python
import pypsa

# Reference (before update)
n_ref = pypsa.Network("resources/network/HT35_solved_reference.nc")

# New (after update)
n_new = pypsa.Network("resources/network/HT35_solved.nc")

# Compare key metrics
print(f"System cost change: {(n_new.objective - n_ref.objective)/n_ref.objective*100:.2f}%")

# Generation by carrier
gen_ref = n_ref.generators.groupby('carrier').p_nom.sum()
gen_new = n_new.generators.groupby('carrier').p_nom.sum()
print("\nCapacity changes:")
print((gen_new - gen_ref).sort_values())
```

## Version Control

### Best Practices

1. **Commit data updates separately**:
   ```bash
   git add data/generators/DUKES_5.11_2026.xlsx
   git commit -m "data: Update DUKES to 2026 edition"
   ```

2. **Tag releases**:
   ```bash
   git tag -a "data-2026Q1" -m "Data update Q1 2026"
   ```

3. **Document changes**:
   - Update `docs/readthedocs/source/development/release_notes.md`

### Large Files

For large data files (>100MB), consider:
- Git LFS for versioning
- External hosting with download scripts
- Documentation of manual download steps

## Troubleshooting

### Data Format Changed

If a data source changes format:

1. Check the processing script for hardcoded column names
2. Update column mappings in `config/` if applicable
3. Test with `--dry-run` first

### Missing Data Points

If data is missing for specific years:

1. Check if the year is covered by the source
2. Consider interpolation for intermediate years
3. Document any assumptions made

### Coordinate Issues

If generators map to wrong locations:

1. Verify coordinate system (OSGB36 vs WGS84)
2. Check for data entry errors in source
3. Use `scripts/validate_coordinates.py` to identify issues
