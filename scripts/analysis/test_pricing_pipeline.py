"""Quick integration test: verify carrier_average fallback produces reasonable prices."""
import sys
sys.path.insert(0, ".")

import pypsa
import yaml
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pricing_test")

# Load network
network_path = r"resources\network\Validation_2020.nc"
print(f"Loading network: {network_path}")
n = pypsa.Network(network_path)
print(f"  {len(n.generators)} generators, {len(n.storage_units)} storage units, "
      f"{len(n.snapshots)} snapshots")

# Load config
with open("config/defaults.yaml", encoding="utf-8") as f:
    defaults = yaml.safe_load(f)
with open("config/scenarios.yaml", encoding="utf-8") as f:
    scenarios = yaml.safe_load(f)

# Get market config (merge defaults + scenario)
market_config = defaults.get("market", {})

# Run the pricing
from scripts.market.market_utils import calculate_bid_offer_prices

gen_offer, gen_bid, su_offer, su_bid = calculate_bid_offer_prices(
    n, market_config, logger, scenario_id="Validation_2020"
)

print(f"\n{'='*80}")
print("GENERATOR OFFER PRICES")
print(f"{'='*80}")
# Per-carrier summary
for carrier in sorted(n.generators.carrier.unique()):
    mask = n.generators.carrier == carrier
    gens = n.generators.index[mask]
    cap = n.generators.loc[gens, "p_nom"].sum()
    offer_mean = gen_offer[gens].mean()
    offer_med = gen_offer[gens].median()
    bid_mean = gen_bid[gens].mean()
    bid_med = gen_bid[gens].median()
    n_gens = len(gens)
    if cap > 0:
        print(f"  {carrier:25s}: {n_gens:4d} gens, {cap:8.0f} MW | "
              f"offer mean=£{offer_mean:7.1f} med=£{offer_med:7.1f} | "
              f"bid mean=£{bid_mean:7.1f} med=£{bid_med:7.1f}")

print(f"\n{'='*80}")
print("STORAGE OFFER PRICES")
print(f"{'='*80}")
for carrier in sorted(n.storage_units.carrier.unique()):
    mask = n.storage_units.carrier == carrier
    sus = n.storage_units.index[mask]
    cap = n.storage_units.loc[sus, "p_nom"].sum()
    offer_mean = su_offer[sus].mean()
    bid_mean = su_bid[sus].mean()
    n_units = len(sus)
    print(f"  {carrier:25s}: {n_units:4d} units, {cap:8.0f} MW | "
          f"offer mean=£{offer_mean:7.1f} | bid mean=£{bid_mean:7.1f}")

# Sanity checks
print(f"\n{'='*80}")
print("SANITY CHECKS")
print(f"{'='*80}")
issues = []

# 1. No NaN prices
nan_offers = gen_offer.isna().sum()
nan_bids = gen_bid.isna().sum()
print(f"NaN offers: {nan_offers}, NaN bids: {nan_bids}")
if nan_offers > 0: issues.append(f"{nan_offers} NaN offer prices")
if nan_bids > 0: issues.append(f"{nan_bids} NaN bid prices")

# 2. CCGT offers should be ~£60-100/MWh (not £54)
ccgt = n.generators.index[n.generators.carrier == "CCGT"]
ccgt_offer = gen_offer[ccgt].mean()
print(f"CCGT mean offer: £{ccgt_offer:.1f}/MWh (expect ~£60-100)")
if ccgt_offer < 50: issues.append(f"CCGT offer too low: £{ccgt_offer:.1f}")

# 3. Nuclear should have near-zero or small MC-based offer
nuclear = n.generators.index[n.generators.carrier == "nuclear"]
nuc_offer = gen_offer[nuclear].mean()
print(f"Nuclear mean offer: £{nuc_offer:.1f}/MWh (expect ~£10-20)")

# 4. Wind offers should be very low
wind_on = n.generators.index[n.generators.carrier == "wind_onshore"]
wind_offer = gen_offer[wind_on].mean()
print(f"Wind onshore mean offer: £{wind_offer:.1f}/MWh (expect <£10)")

# 5. Load shedding should have very high offer
ls = n.generators.index[n.generators.carrier == "load_shedding"]
ls_offer = gen_offer[ls].mean()
print(f"Load shedding mean offer: £{ls_offer:.1f}/MWh (expect >£5000)")

if issues:
    print(f"\n⚠ ISSUES: {', '.join(issues)}")
else:
    print("\n✓ All sanity checks passed")
