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

Create the `pypsa-gb-stable` environment using the provided stable environment file:

```bash
conda env create -f envs/pypsa-gb-stable.yaml
```

Activate the environment:

```bash
conda activate pypsa-gb-stable
```

```{tip}
Add `conda activate pypsa-gb-stable` to your shell profile to automatically activate it in new terminals.
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

# Dry run to check workflow
snakemake --cores 4 -n -p
```

## Optional: Configure an ENTSO-E API Key

PyPSA-GB can use hourly day-ahead electricity prices from the
[ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) for
interconnector pricing when `interconnectors.pricing.source` is set to
`"entsoe"`.

This is optional. If no ENTSO-E token is configured, PyPSA-GB falls back to the
derived interconnector pricing method.

1. Register for a free ENTSO-E Transparency Platform account and request API
   access.
2. Add your token to a `.env` file in the project root:

   ```bash
   ENTSOE_API_TOKEN=your-token-here
   ```

   Alternatively, set it as an environment variable:

   ```bash
   # Linux/macOS
   export ENTSOE_API_TOKEN=your-token-here

   # Windows PowerShell
   $env:ENTSOE_API_TOKEN = "your-token-here"
   ```

3. To use ENTSO-E prices, set the interconnector pricing source in your
   scenario or defaults configuration:

   ```yaml
   interconnectors:
     pricing:
       source: "entsoe"
   ```

## Step 6: Download Weather Data

For renewable generation profiles, you need ERA5 weather data "cutouts". PyPSA-GB uses a **tiered acquisition strategy** to minimize download times:

1. **Data directory** - Check `data/atlite/cutouts/` for cached copy (instant)
2. **Zenodo** - Download pre-built cutouts from [Zenodo repository](https://zenodo.org/records/18325225) (~5-10 minutes per year, years 2010-2024)
3. **ERA5 API** - Full download via atlite as fallback (~2-4 hours per year)

### Quick Start (Recommended)

For years 2010-2024, cutouts are automatically downloaded from Zenodo:

```bash
# Edit the configuration to specify which years you need
nano config/cutouts_config.yaml

# Set years_to_generate, for example:
# years_to_generate:
#   - 2020
#   - 2021

# Generate cutouts (will download from Zenodo for 2010-2024)
snakemake -s Snakefile_cutouts --cores 1
```

**No CDS API credentials required** for years 2010-2024 when using Zenodo!

### Manual ERA5 Download (Advanced)

For years outside 2010-2024 or if you prefer direct ERA5 download:

1. **Configure CDS API** (one-time setup):
   ```bash
   # Register at: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)
   # Get your API key from: [https://cds.climate.copernicus.eu/how-to-api](https://cds.climate.copernicus.eu/how-to-api)
   # Create ~/.cdsapirc with your credentials
   ```

2. **Disable Zenodo** (optional) in `config/cutouts_config.yaml`:
   ```yaml
   zenodo:
     enabled: false  # Force ERA5 download
   ```

3. **Generate cutouts**:
   ```bash
   snakemake -s Snakefile_cutouts --cores 1
   ```

```{tip}
Zenodo downloads are much faster (~minutes vs hours) and don't require API credentials. Use this method unless you need years outside 2010-2024.
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
conda env create -f envs/pypsa-gb-stable.yaml
```

**Gurobi license not found**

Ensure your license file is in the correct location:
- Linux/macOS: `~/gurobi.lic`
- Windows: `C:\Users\<username>\gurobi.lic`

**Snakemake not found**

Ensure the environment is activated:
```bash
conda activate pypsa-gb-stable
which snakemake  # Should show path in conda env
```

## Next Steps

- {doc}`quickstart` - Run your first model
- {doc}`first_scenario` - Create a custom scenario
