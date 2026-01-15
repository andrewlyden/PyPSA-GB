# PyPSA-GB Testing Suite

A streamlined, focused testing regime for the PyPSA-GB energy system model.

---

## 🚀 Quick Start

```bash
# BEFORE COMMIT - Quick sanity check (<30 seconds)
pytest tests/test_smoke.py -v

# BEFORE PUSH - Core functionality (~2 minutes)
pytest tests/unit/ -v

# BEFORE RELEASE - Full validation (~5 minutes)
pytest tests/ -v
```

---

## 📋 Current Test Files (17 total)

### Root Level (4 files)
| File | Purpose | Time | Priority |
|------|---------|------|----------|
| `test_smoke.py` | Import checks, basic sanity | 30s | 🔴 Critical |
| `test_marginal_costs.py` | Fuel/carbon cost calculations | 10s | 🟡 Important |
| `test_time_utils.py` | Timeseries resampling/alignment | 5s | 🟡 Important |
| `conftest.py` | Shared fixtures for all tests | N/A | 🔴 Critical |

### Unit Tests (10 files)
| File | Purpose | Time | Priority |
|------|---------|------|----------|
| `test_scenario_detection.py` | Historical vs Future routing | 5s | 🔴 Critical |
| `test_spatial_utils.py` | Coordinate conversion, bus mapping | 10s | 🔴 Critical |
| `test_build_network.py` | Network construction & validation | 15s | 🔴 Critical |
| `test_carrier_definitions.py` | Technology colors/emissions | 3s | 🟡 Important |
| `test_add_storage.py` | Battery/pumped hydro integration | 8s | 🟡 Important |
| `test_integrate_thermal_generators.py` | CCGT/nuclear/coal integration | 20s | 🟡 Important |
| `test_integrate_renewable_generators.py` | Wind/solar/hydro integration | 20s | 🟡 Important |
| `test_solve_network.py` | Optimization solver configuration | 10s | 🟡 Important |
| `test_cluster_network.py` | Network aggregation | 15s | 🟢 Useful |
| `test_etys_network.py` | ETYS network specifics | 10s | 🟢 Useful |
| `test_finalize_network.py` | Network finalization & metadata | 5s | 🟢 Useful |
| `test_map_renewable_profiles.py` | Profile-to-site mapping | 8s | 🟢 Useful |
| `test_map_to_buses.py` | Interconnector bus mapping | 8s | 🟢 Useful |

### Workflow Tests (3 files)
| File | Purpose | Time | Priority |
|------|---------|------|----------|
| `test_scenario_validation.py` | Config validation | 5s | 🟡 Important |
| `test_historical_scenario_pipeline.py` | DUKES/REPD workflow | 30s | 🟢 Useful |
| `test_future_scenario_pipeline.py` | FES workflow | 30s | 🟢 Useful |

---

## 🏗️ Test Structure

```
tests/
├── conftest.py                              # Shared fixtures ✅
├── test_smoke.py                            # Quick sanity checks ✅
├── test_marginal_costs.py                   # Economic calculations ✅
├── test_time_utils.py                       # Timeseries utilities ✅
│
├── unit/                                    # Function-level tests (14 files)
│   ├── conftest.py                          # Unit test fixtures ✅
│   ├── test_scenario_detection.py           # ⭐ CRITICAL - Data routing
│   ├── test_spatial_utils.py                # ⭐ CRITICAL - Bus mapping
│   ├── test_build_network.py                # ⭐ CRITICAL - Network construction
│   ├── test_carrier_definitions.py          # Technology definitions
│   ├── test_add_storage.py                  # Storage integration
│   ├── test_integrate_thermal_generators.py # Thermal generation
│   ├── test_integrate_renewable_generators.py # Renewable generation
│   ├── test_solve_network.py                # Optimization
│   ├── test_cluster_network.py              # Network aggregation
│   ├── test_etys_network.py                 # ETYS specifics
│   ├── test_finalize_network.py             # Network finalization
│   ├── test_map_renewable_profiles.py       # Profile mapping
│   └── test_map_to_buses.py                 # Interconnector mapping
│
└── workflow/                                # End-to-end tests (3 files)
    ├── test_scenario_validation.py          # Config validation ✅
    ├── test_historical_scenario_pipeline.py # Historical workflow ✅
    └── test_future_scenario_pipeline.py     # Future workflow ✅
```

---

## 🔧 Testing Commands

### By Speed

```bash
# Fast (<1 min) - COMMIT check
pytest tests/test_smoke.py \
       tests/unit/test_scenario_detection.py \
       tests/unit/test_spatial_utils.py -v

# Medium (~2 min) - PUSH check
pytest tests/unit/ -v

# Full (~5 min) - RELEASE check
pytest tests/ -v
```

### By Functionality

```bash
# Critical path (prevents most failures)
pytest tests/unit/test_scenario_detection.py \
       tests/unit/test_spatial_utils.py \
       tests/unit/test_build_network.py -v

# Generation integration
pytest tests/unit/test_integrate_*_generators.py -v

# Storage & demand
pytest tests/unit/test_add_storage.py -v

# Network manipulation
pytest tests/unit/test_cluster_network.py \
       tests/unit/test_etys_network.py \
       tests/unit/test_finalize_network.py -v

# Optimization
pytest tests/unit/test_solve_network.py -v

# End-to-end workflows
pytest tests/workflow/ -v
```

### With Coverage

```bash
# Unit test coverage
pytest tests/unit/ --cov=scripts --cov-report=html
# Open htmlcov/index.html

# Full coverage
pytest tests/ --cov=scripts --cov=rules --cov-report=term-missing
```

### Continuous Integration

```bash
# Pre-commit hook (fast)
pytest tests/test_smoke.py -x --tb=short

# GitHub Actions (comprehensive)
pytest tests/ -v --junitxml=test-results.xml
```

---

## 🎯 What Each Test Catches

| Test | Catches These Critical Bugs |
|------|---------------------------|
| `test_smoke` | Import errors, missing dependencies, basic setup issues |
| `test_scenario_detection` | ⚠️ Wrong data source (DUKES vs FES) → incorrect capacities |
| `test_spatial_utils` | ⚠️ Generators at wrong buses → infeasibility/load shedding |
| `test_build_network` | ⚠️ Missing buses/lines → disconnected network |
| `test_carrier_definitions` | Wrong colors, missing carriers, incorrect emissions |
| `test_add_storage` | Over-scaled storage, wrong efficiency, missing pumped hydro |
| `test_integrate_thermal` | Wrong thermal capacity, missing plants, incorrect routing |
| `test_integrate_renewable` | Missing renewable sites, wrong profiles, capacity errors |
| `test_solve_network` | Solver config errors, wrong objective, optimization failures |
| `test_cluster_network` | Lost capacity during aggregation, disconnected regions |
| `test_marginal_costs` | Wrong fuel costs, incorrect carbon pricing |
| `test_time_utils` | Resampling errors, misaligned timeseries |

---

## 💡 Best Practices

### Before Committing
1. **Always run smoke tests first** - Catches import/setup issues in 30 seconds
2. **Run affected tests** - If you modified generators, run `test_integrate_*_generators.py`
3. **Check critical path** - scenario_detection + spatial_utils + build_network (1 min)

### Common Failure Patterns
| Symptom | Likely Cause | Test to Run |
|---------|--------------|-------------|
| Infeasible optimization | Wrong bus mapping | `test_spatial_utils.py` |
| Missing generators | Data routing error | `test_scenario_detection.py` |
| Network disconnected | Network building error | `test_build_network.py` |
| Wrong capacities | FES integration error | `test_integrate_*_generators.py` |

### Debugging Tips
- **Use `-v` flag** - Shows individual test names
- **Use `-x` flag** - Stops at first failure
- **Use `--tb=short`** - Shorter tracebacks
- **Use `-k pattern`** - Run tests matching pattern (e.g., `-k "thermal or renewable"`)
- **Check fixtures** - `conftest.py` has ready-to-use networks and scenarios

---

## 📊 Continuous Integration Setup

### GitHub Actions Example

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  smoke-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 2
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Smoke tests
        run: pytest tests/test_smoke.py -v

  unit-tests:
    runs-on: ubuntu-latest
    needs: smoke-tests
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Unit tests
        run: pytest tests/unit/ -v --cov=scripts --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  workflow-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      - name: Workflow tests
        run: pytest tests/workflow/ -v
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest tests/test_smoke.py -x --tb=short
if [ $? -ne 0 ]; then
    echo "❌ Smoke tests failed. Fix errors before committing."
    exit 1
fi
echo "✅ Smoke tests passed"
```

---

## 📚 Additional Resources

- **Test fixtures**: See `conftest.py` for ready-to-use networks and scenarios
- **Test patterns**: Look at existing tests for examples of good test structure
- **Coverage reports**: Run `pytest --cov` and open `htmlcov/index.html` to see what's tested
- **Cleanup docs**: See `CLEANUP_RECOMMENDATIONS.md` for rationale behind test selection

---

## 🔄 Test Maintenance

### Adding New Tests
1. **Choose the right location**:
   - Core functions → `unit/`
   - End-to-end workflows → `workflow/`
2. **Follow naming convention**: `test_<module_name>.py`
3. **Use fixtures**: Import from `conftest.py` rather than creating new test data
4. **Keep tests focused**: One concept per test function

### Updating Tests
- When you change functionality, update the corresponding test
- Run the affected tests to verify they still pass
- Update docstrings if test purpose changes

### Removing Tests
- Only remove tests if functionality is permanently removed
- Document why in commit message
- Check no other tests depend on removed fixtures

---

**Last Updated:** January 2026  
**Maintainer:** PyPSA-GB Development Team
