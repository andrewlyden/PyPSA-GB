"""Verify NESO boundary mapping line IDs exist in the ETYS network."""
import pypsa
import yaml
from pathlib import Path

# Find a suitable ETYS network
networks = list(Path("resources/network").glob("*network.nc"))
if not networks:
    networks = list(Path("resources/network").glob("*.nc"))
print(f"Available networks: {[n.name for n in networks[:10]]}")

# Try to load the Validation_2020 or Test_Rolling_Market network
target = None
for n in networks:
    if "Validation_2020" in n.name or "Test_Rolling" in n.name or "Historical_2020" in n.name:
        target = n
        break
if not target and networks:
    # Pick any ETYS network
    for n in networks:
        if "etys" in n.name.lower() or "ETYS" in n.name:
            target = n
            break
if not target and networks:
    target = networks[0]

print(f"\nUsing network: {target}")
network = pypsa.Network(str(target))
print(f"Lines: {len(network.lines)}, Buses: {len(network.buses)}")

# Load boundary mapping
with open("data/network/neso_boundary_mapping.yaml", encoding="utf-8") as f:
    mapping = yaml.safe_load(f)

for bname, bdef in mapping["boundaries"].items():
    lines = bdef.get("lines", [])
    valid = [l for l in lines if l in network.lines.index]
    missing = [l for l in lines if l not in network.lines.index]
    total_s_nom = network.lines.loc[valid, "s_nom"].sum() if valid else 0
    print(f"\n{bname}: {len(valid)}/{len(lines)} lines found, total_s_nom={total_s_nom:.0f} MVA")
    if missing:
        print(f"  MISSING: {missing}")
    for lid in valid:
        s = network.lines.loc[lid]
        print(f"  {lid}: s_nom={s['s_nom']:.0f} MVA, bus0={s['bus0']}, bus1={s['bus1']}")
