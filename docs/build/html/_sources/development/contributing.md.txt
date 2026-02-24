# Contributing

Thank you for your interest in contributing to PyPSA-GB!

## Getting Started

### Prerequisites

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/PyPSA-GB.git
   cd PyPSA-GB
   ```
3. Set up the development environment:
   ```bash
   conda env create -f envs/pypsa-gb.yaml
   conda activate pypsa-gb
   ```
4. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/andrewlyden/PyPSA-GB.git
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Follow the code style guidelines below
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Validate scenarios
python scripts/validate_scenarios.py

# Run a test scenario
snakemake resources/network/Test_reduced_solved.nc -j 4

# Check for errors
python -m pytest tests/ -v
```

### 4. Commit Your Changes

Use clear commit messages:

```bash
git add .
git commit -m "feat: Add support for new storage technology"
```

**Commit message prefixes**:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `data:` - Data file updates
- `refactor:` - Code restructuring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

### 5. Push and Create PR

```bash
git push origin feature/my-new-feature
```

Then create a Pull Request on GitHub.

## Code Style

### Python Style

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use descriptive variable names

```python
# Good
def calculate_curtailment(
    network: pypsa.Network,
    carrier: str = "wind_offshore"
) -> float:
    """Calculate curtailment for a carrier."""
    ...

# Bad
def calc(n, c="wind"):
    ...
```

### Docstrings

Use NumPy-style docstrings:

```python
def map_sites_to_buses(
    network: pypsa.Network,
    sites_df: pd.DataFrame,
    method: str = 'nearest'
) -> pd.DataFrame:
    """
    Map generator sites to nearest network buses.
    
    Parameters
    ----------
    network : pypsa.Network
        PyPSA network with bus coordinates
    sites_df : pd.DataFrame
        Sites with 'lat' and 'lon' columns
    method : str, optional
        Mapping method: 'nearest' (default) or 'voronoi'
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'bus' column added
        
    Raises
    ------
    ValueError
        If sites_df lacks coordinate columns
        
    Examples
    --------
    >>> mapped = map_sites_to_buses(network, wind_farms)
    >>> mapped.groupby('bus').p_nom.sum()
    """
```

### Logging

Use the centralized logging configuration:

```python
from scripts.logging_config import setup_logging

# At module level (for imports)
logger = setup_logging("module_name")

# In main block (for Snakemake)
if __name__ == "__main__":
    snk = globals().get('snakemake')
    log_path = snk.log[0] if snk and snk.log else "module_name"
    logger = setup_logging(log_path)
```

## Adding New Features

### New Data Source

1. Add raw data to `data/` directory
2. Create processing script in `scripts/`
3. Add Snakemake rule in `rules/`
4. Update documentation
5. Add validation checks

### New Technology/Carrier

1. Add to `scripts/carrier_definitions.py`:
   ```python
   'my_new_tech': {
       'nice_name': 'My New Technology',
       'color': '#HEXCODE',
       'co2_emissions': 0.0,
       'renewable': True
   }
   ```
2. Update integration modules to handle the carrier
3. Add test cases
4. Document in data reference

### New Network Model

1. Create topology files in `data/network/`
2. Update `scripts/build_network.py` to load it
3. Add to configuration options
4. Document in network models guide

## Snakemake Rules

### Rule Style

```python
rule my_rule:
    """Brief description of what this rule does."""
    input:
        network="resources/network/{scenario}_network.nc",
        data="data/my_data.csv"
    output:
        "resources/network/{scenario}_with_feature.nc"
    log:
        "logs/my_rule/{scenario}.log"
    params:
        param1=lambda wildcards: config['scenarios'][wildcards.scenario].get('param1', 'default')
    script:
        "../scripts/my_script.py"
```

### Rule Guidelines

- Include `log:` directive for every rule
- Use wildcards consistently
- Document complex rules
- Keep scripts focused on one task

## Testing

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_scenario_detection.py -v

# With coverage
python -m pytest tests/ --cov=scripts
```

### Writing Tests

```python
# tests/test_my_feature.py
import pytest
from scripts.my_module import my_function

def test_my_function_basic():
    """Test basic functionality."""
    result = my_function(input_data)
    assert result == expected_output

def test_my_function_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

## Documentation

### Building Docs Locally

```bash
cd docs/readthedocs
make html
# Open build/html/index.html
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Add cross-references with `{doc}` and `{ref}`
- Include diagrams for complex concepts

## Reporting Issues

### Bug Reports

Include:
1. PyPSA-GB version (git commit hash)
2. Python and key package versions
3. Operating system
4. Full error message and traceback
5. Minimal reproduction steps
6. Relevant configuration snippets

### Feature Requests

Include:
1. Clear description of the feature
2. Use case / motivation
3. Proposed implementation (if any)
4. Impact on existing functionality

## Code Review

All contributions go through code review:

1. **Functionality**: Does it work as intended?
2. **Tests**: Are there adequate tests?
3. **Documentation**: Is it documented?
4. **Style**: Does it follow guidelines?
5. **Performance**: Any performance concerns?

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Contact

- **Lead Developer**: Dr Andrew Lyden
- **Email**: andrew.lyden@ed.ac.uk
- **GitHub Issues**: For bug reports and feature requests
