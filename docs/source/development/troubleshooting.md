# Troubleshooting

Common issues and their solutions.

## Installation Issues

### Conda Environment Creation Fails

**Symptom**: Error during `conda env create`

**Solutions**:

1. Update conda first:
   ```bash
   conda update -n base conda
   ```

2. Try mamba (faster resolver):
   ```bash
   conda install -n base mamba
   mamba env create -f envs/pypsa-gb.yaml
   ```

3. Clear package cache:
   ```bash
   conda clean --all
   ```

### Gurobi License Not Found

**Symptom**: `GurobiError: No Gurobi license found`

**Solutions**:

1. Check license file location:
   - Windows: `C:\Users\<username>\gurobi.lic`
   - Linux/Mac: `~/gurobi.lic`

2. Regenerate license (if academic):
   ```bash
   grbgetkey YOUR-LICENSE-KEY
   ```

3. Use HiGHS instead:
   ```yaml
   solver:
     name: "highs"
   ```

### Snakemake Not Found

**Symptom**: `snakemake: command not found`

**Solutions**:

1. Ensure environment is activated:
   ```bash
   conda activate pypsa-gb
   which snakemake
   ```

2. Reinstall snakemake:
   ```bash
   conda install -n pypsa-gb snakemake
   ```

---

## Workflow Issues

### "Missing Input Files"

**Symptom**: Snakemake reports missing input files

**Solutions**:

1. Check the file exists:
   ```bash
   ls path/to/expected/file
   ```

2. Generate the missing file:
   ```bash
   snakemake path/to/missing/file -j 1
   ```

3. Check for typos in scenario name

### Rule Fails with Exit Code 1

**Symptom**: Rule fails, exit code 1

**Solutions**:

1. Check the log file:
   ```bash
   cat logs/<rule_name>/<scenario>.log
   ```

2. Re-run with verbose output:
   ```bash
   snakemake target.nc -j 1 --verbose
   ```

3. Check Python errors in log

### "No Rule to Produce Target"

**Symptom**: `No rule to produce <target>`

**Solutions**:

1. Check target filename matches pattern
2. Verify scenario is defined in `scenarios.yaml`
3. Check for wildcards mismatch

---

## Network Issues

### All Generators at Same Bus

**Symptom**: All generators mapped to one bus (e.g., `INDQ41`)

**Cause**: Coordinate system mismatch (WGS84 vs OSGB36)

**Solution**: The issue is in spatial mapping. Check:
```python
from scripts.spatial_utils import detect_coordinate_system

# Check bus coordinates
print(detect_coordinate_system(network.buses[['x', 'y']]))
# Should output: 'OSGB36'
```

### Network Not Connected

**Symptom**: Isolated buses, disconnected regions

**Solution**:
```python
import pypsa

n = pypsa.Network("resources/network/my_network.nc")

# Check connectivity
from pypsa.topology import find_disconnected_buses
disconnected = find_disconnected_buses(n)
print(f"Disconnected buses: {disconnected}")
```

### Negative Line Impedance

**Symptom**: Warnings about negative impedance

**Solution**: Usually data issue. Check raw network files for data entry errors.

---

## Solver Issues

### Infeasible Optimization

**Symptom**: Solver returns "infeasible" or "unbounded"

**Causes & Solutions**:

1. **Demand exceeds supply**:
   ```python
   n = pypsa.Network("resources/network/pre_solve.nc")
   print(f"Generation: {n.generators.p_nom.sum()/1000:.1f} GW")
   print(f"Peak demand: {n.loads_t.p_set.sum(axis=1).max()/1000:.1f} GW")
   ```

2. **Network disconnected**: See above

3. **Zero capacity lines**:
   ```python
   print(n.lines[n.lines.s_nom == 0])
   ```

4. **Missing generators**: Check if integration ran correctly

### Solver Runs Forever

**Symptom**: Solver doesn't converge

**Solutions**:

1. Add time limit:
   ```yaml
   solver:
     TimeLimit: 3600  # 1 hour
   ```

2. Use barrier method:
   ```yaml
   solver:
     method: 2
     crossover: 0
   ```

3. Reduce problem size:
   - Use reduced network
   - Shorten solve period
   - Cluster network

### Out of Memory

**Symptom**: Process killed, memory error

**Solutions**:

1. Reduce parallel jobs:
   ```bash
   snakemake target.nc -j 1
   ```

2. Use network clustering

3. Reduce timesteps:
   ```yaml
   solve_period:
     start: "2035-01-01"
     end: "2035-01-02"  # Shorter period
   ```

---

## Data Issues

### FES Data Not Found

**Symptom**: Error loading FES data

**Solutions**:

1. Check FES year is available:
   ```bash
   ls resources/FES/
   ```

2. Regenerate FES data:
   ```bash
   snakemake resources/FES/FES_2024_data.csv -j 1 -R
   ```

### DUKES Coordinates Missing

**Symptom**: Generators without coordinates

**Solutions**:

1. Check for manual coordinate file:
   ```bash
   ls data/generators/manual_coordinates.csv
   ```

2. Run coordinate geocoding:
   ```bash
   python scripts/geocode_missing_coordinates.py
   ```

### Weather Cutout Missing

**Symptom**: "Cutout not found" for renewables year

**Solutions**:

1. Check available cutouts:
   ```bash
   ls resources/atlite/
   ```

2. Generate missing cutout:
   ```bash
   snakemake -s Snakefile_cutouts resources/atlite/GB_2019.nc -j 2
   ```

3. Use existing weather year:
   ```yaml
   renewables_year: 2019  # Year with available cutout
   ```

---

## Logging Issues

### Empty Log Files

**Symptom**: Log files exist but are empty

**Cause**: Scripts not using `snakemake.log[0]`

**Solution**: This was fixed in v0.1. If still occurring:
```python
# In script's main block
log_path = snakemake.log[0] if snakemake.log else "fallback"
logger = setup_logging(log_path)
```

### "Logger Not Defined"

**Symptom**: `NameError: name 'logger' is not defined`

**Solution**:
```python
from scripts.logging_config import setup_logging
logger = setup_logging("my_script")
```

---

## Getting Help

### Before Asking

1. Check this troubleshooting guide
2. Search existing [GitHub Issues](https://github.com/andrewlyden/PyPSA-GB/issues)
3. Check the relevant log file

### Information to Include

When reporting an issue:

1. **Environment**:
   ```bash
   conda list | grep -E "pypsa|snakemake|pandas"
   python --version
   ```

2. **Command run**:
   ```bash
   snakemake target.nc -j 4
   ```

3. **Full error message** (copy from terminal)

4. **Log file contents** (from `logs/`)

5. **Configuration snippet** (relevant scenario)

### Quick Diagnostic

Run this to collect system info:

```python
import sys
import pypsa
import pandas
import snakemake

print(f"Python: {sys.version}")
print(f"PyPSA: {pypsa.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"Snakemake: {snakemake.__version__}")
```
