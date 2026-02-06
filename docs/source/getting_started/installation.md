# Installation

This guide covers the complete installation of PyPSA-GB.

## Step 1: Install Conda

If you don't have Conda installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download).

Verify your installation:

```bash
conda --version
```

## Step 2: Clone the Repository

Clone PyPSA-GB from GitHub:

```bash
git clone https://github.com/andrewlyden/PyPSA-GB.git
cd PyPSA-GB
```

```{note}
The repository is approximately 2GB due to included data files. This may take some time depending on your internet connection.
```

## Step 3: Create the Conda Environment

Create the `pypsa-gb` environment using the provided environment file:

```bash
conda env create -f envs/pypsa-gb.yaml
```

Activate the environment:

```bash
conda activate pypsa-gb
```

```{tip}
Add `conda activate pypsa-gb` to your shell profile to automatically activate it in new terminals.
```

## Step 4: Install a Solver

PyPSA-GB requires an optimization solver. You have two options:

### Option A: Gurobi (Recommended)

Gurobi offers free academic licenses and provides excellent performance.

1. Register for an [academic license](https://www.gurobi.com/downloads/end-user-license-agreement-academic/)
2. Download and install Gurobi
3. Activate your license:
   ```bash
   grbgetkey YOUR-LICENSE-KEY
   ```

### Option B: HiGHS (Open Source)

HiGHS is included in the conda environment. No additional setup required.

To use HiGHS, set in your scenario configuration:

```yaml
solver:
  name: "highs"
```

## Step 5: Verify Installation

Test that everything is working:

```bash
# Check Python and key packages
python -c "import pypsa; print(f'PyPSA version: {pypsa.__version__}')"
python -c "import snakemake; print('Snakemake OK')"

# Validate scenarios
python scripts/validate_scenarios.py

# Dry run to check workflow
snakemake -n -p
```

## Step 6: Download Weather Data

For renewable generation profiles, you need ERA5 weather data "cutouts". Pre-generated cutouts for common years are available.
  1. Configure the CDS API (one-time setup)
To generate new cutouts (requires CDS API credentials):

```bash
# Configure CDS API (one-time setup)
# See: https://cds.climate.copernicus.eu/api-how-to
```
 2. To generate weather data for a specific year, update the configuration file:

```bash
config/cutouts_config.yaml
```
Set the desired year in the year_to_generate field, for example:

```bash
years_to_generate:
    - 2020
```
 3. Generate the Cutouts
 
```bash
# Generate cutouts for a specific year
snakemake -s Snakefile_cutouts -j 2 --config year=2019
```

```{warning}
Cutout generation downloads large weather datasets and can take several hours.
```

## Directory Structure

After installation, your directory should look like:

```
PyPSA-GB/
├── config/              # Configuration files
│   ├── config.yaml      # Active scenarios
│   ├── scenarios.yaml   # Scenario definitions
│   └── defaults.yaml    # Default values
├── data/                # Input data (versioned)
├── resources/           # Generated outputs
├── scripts/             # Python modules
├── rules/               # Snakemake rules
├── envs/                # Conda environment
├── Snakefile            # Main workflow
└── Snakefile_cutouts    # Weather data workflow
```

## Troubleshooting

### Common Issues

**Conda environment creation fails**

Try updating conda first:
```bash
conda update -n base conda
conda env create -f envs/pypsa-gb.yaml
```

**Gurobi license not found**

Ensure your license file is in the correct location:
- Linux/macOS: `~/gurobi.lic`
- Windows: `C:\Users\<username>\gurobi.lic`

**Snakemake not found**

Ensure the environment is activated:
```bash
conda activate pypsa-gb
which snakemake  # Should show path in conda env
```

## Next Steps

- {doc}`quickstart` - Run your first model
- {doc}`first_scenario` - Create a custom scenario
