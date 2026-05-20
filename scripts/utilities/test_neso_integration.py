"""Integration test: apply NESO boundary limits to Test_Rolling_Market network."""
import pypsa
import yaml
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("test_neso")

# Add project root to path
sys.path.insert(0, str(Path.cwd()))
from scripts.solve.solve_network import apply_outage_schedule

# Load network (use the complete pre-solve network with datetime snapshots)
nc_path = "resources/network/Test_Rolling_Market_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"
network = pypsa.Network(nc_path)
print(f"Network: {len(network.buses)} buses, {len(network.lines)} lines")
print(f"Snapshots: {network.snapshots[0]} to {network.snapshots[-1]} ({len(network.snapshots)} steps)")

# Truncate to January 2020 for the test (31 days × 24 hours = 744 snapshots)
network.set_snapshots(network.snapshots[:744])
print(f"Truncated to: {network.snapshots[0]} to {network.snapshots[-1]} ({len(network.snapshots)} steps)")

# Load scenario config
with open("config/scenarios.yaml", encoding="utf-8") as f:
    scenarios = yaml.safe_load(f)
scenario_config = scenarios.get("Test_Rolling_Market", {})
print(f"\nOutage config: {scenario_config.get('transmission', {}).get('outage_schedule', {})}")

# Apply outage schedule
result = apply_outage_schedule(network, scenario_config, logger)
print(f"\nResult: {result}")

# Check what was applied
if hasattr(network.lines_t, 's_max_pu') and not network.lines_t.s_max_pu.empty:
    smax = network.lines_t.s_max_pu
    print(f"\nLines with time-varying s_max_pu: {len(smax.columns)}")
    for col in smax.columns:
        print(f"  {col}: min={smax[col].min():.3f}, max={smax[col].max():.3f}, "
              f"mean={smax[col].mean():.3f}")
else:
    print("\nNo time-varying s_max_pu applied")
