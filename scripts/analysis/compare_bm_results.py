"""Quick comparison: Test_Rolling_Market BM results vs ELEXON benchmarks."""
import sys
sys.path.insert(0, ".")
import pandas as pd

# ── Model results (48h slice: Jan 7-8 2020) ──────────────────────────────────
bm_cost_48h = 5_175_617  # £
increase_vol_48h = 145_825  # MWh
decrease_vol_48h = 145_825  # MWh

# Annualise (48h → 8784h for leap year 2020)
scale = 8784 / 48
bm_cost_annual = bm_cost_48h * scale
increase_vol_annual = increase_vol_48h * scale
decrease_vol_annual = decrease_vol_48h * scale

# ── ELEXON 2020 benchmarks ───────────────────────────────────────────────────
# From ELEXON Insights / BSC reports
elexon_bm_net_cost_2020 = 1_400_000_000  # ~£1.4bn
elexon_bm_offer_vol_2020 = 19_000_000    # ~19 TWh (CCGT-dominated)
elexon_bm_bid_vol_2020 = 16_000_000      # ~16 TWh (wind curtailment)

# ── Comparison ────────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST_ROLLING_MARKET (48h) → ANNUAL EXTRAPOLATION vs ELEXON 2020")
print("=" * 70)
print(f"{'Metric':40s} {'Model (annualised)':>18s} {'ELEXON 2020':>15s} {'Ratio':>8s}")
print("-" * 85)

def fmt(val, unit=""):
    if val > 1e9:
        return f"£{val/1e9:.2f}bn{unit}"
    elif val > 1e6:
        return f"£{val/1e6:.0f}m{unit}" if "£" in str(val) or unit == "£" else f"{val/1e6:.1f} TWh"
    else:
        return f"{val:,.0f}{unit}"

rows = [
    ("BM net cost", bm_cost_annual, elexon_bm_net_cost_2020, "£"),
    ("Increase volume (MWh)", increase_vol_annual, elexon_bm_offer_vol_2020, ""),
    ("Decrease volume (MWh)", decrease_vol_annual, elexon_bm_bid_vol_2020, ""),
]

for label, model, elexon, unit in rows:
    ratio = model / elexon if elexon else float('nan')
    if unit == "£":
        m_str = f"£{model/1e6:,.0f}m"
        e_str = f"£{elexon/1e6:,.0f}m"
    else:
        m_str = f"{model/1e6:,.1f} TWh"
        e_str = f"{elexon/1e6:,.1f} TWh"
    print(f"  {label:38s} {m_str:>18s} {e_str:>15s} {ratio:>7.2f}x")

# ── Previous run comparison ───────────────────────────────────────────────────
# From the Validation_2020 full-year run with OLD pricing
old_bm_cost_annual = 155_000_000  # £155m from the bm_deep_dive analysis
old_ratio = old_bm_cost_annual / elexon_bm_net_cost_2020

print(f"\n{'='*70}")
print("IMPROVEMENT vs PREVIOUS RUN")
print(f"{'='*70}")
print(f"  Old BM cost (full year):    £{old_bm_cost_annual/1e6:.0f}m (ratio: {old_ratio:.2f}x of ELEXON)")
print(f"  New BM cost (annualised):   £{bm_cost_annual/1e6:.0f}m (ratio: {bm_cost_annual/elexon_bm_net_cost_2020:.2f}x of ELEXON)")
print(f"  Improvement:                {bm_cost_annual/old_bm_cost_annual:.1f}x increase in BM costs")

# ── Per-carrier sanity ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("PER-CARRIER BM COSTS (48h)")
print(f"{'='*70}")
carrier_costs = {
    "CCGT": 4_203_239,
    "wind_onshore": 1_018_864,
    "Pumped Storage": 311_920,
    "large_hydro": -352_993,
}
for carrier, cost in carrier_costs.items():
    annual = cost * scale
    print(f"  {carrier:25s}: £{cost:>12,.0f} (48h) → £{annual/1e6:>8,.0f}m/yr")

# BM cost per hour
print(f"\n  BM cost per hour: £{bm_cost_48h/48:,.0f}")
print(f"  (ELEXON 2020 avg: £{elexon_bm_net_cost_2020/8784:,.0f}/hr)")
