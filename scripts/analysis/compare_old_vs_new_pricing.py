"""Final comparison: old (derived) vs new (median carrier_average) pricing for unmatched generators."""
import sys
sys.path.insert(0, ".")

import pypsa
import yaml
import pandas as pd
import logging

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("cmp")

n = pypsa.Network("resources/network/Validation_2020.nc")

# Load the ELEXON mapping to identify matched vs unmatched
bmu_map = pd.read_csv("resources/generators/Validation_2020_bmu_mapping.csv")
matched_gens = set(bmu_map["generator_name"].unique()) & set(n.generators.index)
unmatched_gens = set(n.generators.index) - matched_gens

# Compute OLD derived prices (with NEW tuned overrides)
from scripts.market.market_utils import _compute_component_prices
with open("config/defaults.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
bal = cfg["market"]["balancing"]
old_offer, old_bid = _compute_component_prices(
    n.generators, bal.get("carrier_overrides", {}),
    bal.get("default_offer_markup", 0.10), bal.get("default_bid_discount", 0.10),
)

# Compute NEW prices (full pipeline with carrier_average)
from scripts.market.market_utils import calculate_bid_offer_prices
new_offer, new_bid, _, _ = calculate_bid_offer_prices(
    n, cfg["market"], logger, scenario_id="Validation_2020"
)

# Compare for unmatched generators only (matched get individual ELEXON prices)
unmatched_list = sorted(unmatched_gens)
carriers_of_interest = ["CCGT", "OCGT", "wind_onshore", "wind_offshore", 
                        "nuclear", "coal", "large_hydro", "biomass", "oil",
                        "solar_pv", "embedded_wind", "embedded_solar"]

print(f"{'Carrier':25s} | {'# Unmatched':>11s} | {'OLD offer':>10s} {'NEW offer':>10s} {'Δ offer':>10s} | "
      f"{'OLD bid':>10s} {'NEW bid':>10s} {'Δ bid':>10s}")
print("-" * 115)

for carrier in carriers_of_interest:
    um = [g for g in unmatched_list if n.generators.loc[g, "carrier"] == carrier]
    if not um:
        continue
    cap = n.generators.loc[um, "p_nom"].sum()
    oo = old_offer[um].median()
    no = new_offer[um].median()
    ob = old_bid[um].median()
    nb = new_bid[um].median()
    delta_o = no - oo
    delta_b = nb - ob
    print(f"  {carrier:23s} | {len(um):6d} ({cap:5.0f}MW) | £{oo:8.1f} £{no:8.1f} "
          f"{'↑' if delta_o > 0.5 else '↓' if delta_o < -0.5 else '='}{abs(delta_o):8.1f} | "
          f"£{ob:8.1f} £{nb:8.1f} "
          f"{'↑' if delta_b > 0.5 else '↓' if delta_b < -0.5 else '='}{abs(delta_b):8.1f}")

print(f"\n{'='*80}")
print("KEY IMPROVEMENTS FOR BM COST ACCURACY")
print(f"{'='*80}")

# CCGT unmatched: these drive most BM offer costs
ccgt_um = [g for g in unmatched_list if n.generators.loc[g, "carrier"] == "CCGT"]
ccgt_cap = n.generators.loc[ccgt_um, "p_nom"].sum()
print(f"\n  CCGT ({len(ccgt_um)} unmatched, {ccgt_cap:.0f} MW):")
print(f"    OLD offer (derived): £{old_offer[ccgt_um].median():.1f}/MWh")
print(f"    NEW offer (ELEXON median): £{new_offer[ccgt_um].median():.1f}/MWh")
print(f"    OLD bid (derived): £{old_bid[ccgt_um].median():.1f}/MWh")
print(f"    NEW bid (ELEXON median): £{new_bid[ccgt_um].median():.1f}/MWh")
print(f"    → Offers closer to actual BM; bids now show CCGTs are cheap to turn down")

# Wind onshore bids: drive BM curtailment costs
wind_um = [g for g in unmatched_list if n.generators.loc[g, "carrier"] == "wind_onshore"]
wind_cap = n.generators.loc[wind_um, "p_nom"].sum()
print(f"\n  Wind onshore ({len(wind_um)} unmatched, {wind_cap:.0f} MW):")
print(f"    OLD bid: £{old_bid[wind_um].median():.1f}/MWh")
print(f"    NEW bid: £{new_bid[wind_um].median():.1f}/MWh")
print(f"    → Nearly identical (derived override was already tuned to ELEXON)")

# Summary: total capacity getting different prices
changed = 0
unchanged = 0
for g in unmatched_list:
    if abs(new_offer[g] - old_offer[g]) > 0.5 or abs(new_bid[g] - old_bid[g]) > 0.5:
        changed += 1
    else:
        unchanged += 1
print(f"\n  Generators with changed prices: {changed}/{changed+unchanged} "
      f"({100*changed/(changed+unchanged):.0f}%)")
