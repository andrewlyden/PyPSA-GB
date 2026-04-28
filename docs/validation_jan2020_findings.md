# Validation & Improvement Plan

Updated: 2026-04-20
Scenario: `Validation_Jan2020` (Full ETYS, January 2020, 744h, two-stage market)

---

## 1. Model Architecture (context for interpreting results)

The model runs a **copperplate wholesale** (Stage 1, unconstrained) followed by a **constrained BM** (Stage 2, full network). Both stages serve identical demand — the BM is **pure constraint resolution** with no energy balancing, forecast errors, or plant trips.

Key consequence: the copperplate pushes ALL constraint work into the BM, producing larger redispatch volumes than reality (where participants self-resolve constraints at day-ahead). This inflated constraint volume approximately equals total real BM volume — a coincidental match for the wrong reasons.

**Validation is automated** via two Snakemake rules:
- `validate_bm_results` — compares model vs ELEXON actuals (BOALF volumes, system prices, B1610 dispatch)
- `validate_neso_constraints` — compares model vs NESO thermal constraint costs and DA boundary flows

Both produce CSV + HTML dashboard outputs in `resources/market/` and `resources/analysis/`.

---

## 2. Current Score Card (Jan 2020)

| Dimension | Rating | Key numbers |
|-----------|--------|-------------|
| Energy balance | A | Perfect — gen = demand every timestep |
| Wholesale price | A | 1.00x (£34.60 vs £34.68 N2EX) |
| IC flows | A | 1.00x (fixed via ESPENI `p_set`) |
| Wind dispatch (post-BM) | B− | +9.4% (48h test with stacked factors) |
| CCGT dispatch | B | 0.89x; 3.85x BM increase (structural) |
| Nuclear dispatch | B− | Monthly total 1.00x but hourly r = 0.04 |
| Coal dispatch | C+ | 0.86x; merit order ambiguity with CCGT |
| BM cost vs NESO thermal | B− | £46.2M model vs £70.5M NESO = 0.66x (Jan) |
| Congestion realism | C+ | 33kV artifacts fixed; outage schedule implemented |
| Biomass in BM | D+ | 5.9% of gen but 0 MWh BM increase |

---

## 3. NESO Constraint Benchmarks

Source: NESO Thermal Constraint Costs + DA Constraint Flows (automated via `validate_neso_constraints`).

### 3.1 Thermal costs — model vs NESO

| Period | Model BM | NESO thermal | Ratio |
|--------|----------|--------------|-------|
| **Jan 2020** | £46.2M | £70.5M | **0.66x** |
| Calendar 2020 (pre-fix) | ~£126M | £450.3M | 0.28x |

### 3.2 NESO cost by boundary (Jan 2020)

| Boundary | Cost | Share | Primary cause |
|----------|------|-------|---------------|
| SSE-SP | £34.8M | 49% | Dynamic security limit (floor ~2,050 MW) |
| SCOTEX (B6) | £22.7M | 32% | Structural + outage |
| SSHARN | £12.8M | 18% | Outage-driven |

The boundary ranking **flips** annually: SSHARN dominates calendar 2020 (£238M, 53%) due to summer outages.

### 3.3 DA boundary utilisation (Jan 2020)

| Boundary | NESO mean util | NESO ≥90% periods | Model mean flow |
|----------|---------------|-------------------|-----------------|
| SSHARN | 77% | 36% | 473 MW |
| SCOTEX | 72% | 23% | 1,521 MW |
| SSE-SP | 17%† | 71% | 129 MW |

†SSE-SP mean is low because the dynamic limit swings 2,050–99,999 MW. At 71% of periods the limit is at its security floor.

---

## 4. Remaining Improvements

### Priority 1 — Wholesale dispatch (high impact)

| # | Improvement | Current | Target | Effort |
|---|-------------|---------|--------|--------|
| 1 | **Nuclear hourly availability** — remove monthly smoothing from ESPENI fleet-aggregate (Option A) or use per-station ELEXON PN (Option B) | r = 0.04 | r > 0.80 (A) / > 0.90 (B) | Low / Medium |
| 2 | **Biomass must-run** — set `p_min_pu = 0.80` or use ROC-based negative MC for CHP units | 0 MWh BM increase | Match ~1,363 MW actual | Medium |
| 3 | **Coal-gas MC separation** — increase carbon pass-through to widen the £1.11 gap | Coal r = 0.69 | r > 0.80 | Low |
| 4 | **Wind factor cross-validation** — test seasonal/inter-annual stability of stacked factors (0.645 offshore, 0.805 onshore) | +9.4% (48h) | ±8% full year | Medium |

### Priority 2 — Network & congestion (medium impact)

| # | Improvement | Current | Target | Effort |
|---|-------------|---------|--------|--------|
| 5 | **Enable outage schedule** — use real NESO outage data for historical scenarios to capture SSHARN/SSE-SP outage-driven costs | 0.66x Jan, 0.28x annual | ≥0.55x annual | Low (config change) |
| 6 | **ERRO1T\_KIIN1-\_0** investigation — 94% congested, 132 MVA; verify against ETYS Appendix B | Possibly wrong s_nom | Correct or override | Low |
| 7 | **Nodal price investigation** — £14.95 mean vs £29.64 SBP = 0.50x | 0.50x | >0.67x | Medium |

### Priority 3 — BM calibration (after P1 rerun)

| # | Improvement | Notes |
|---|-------------|-------|
| 8 | **Bid/offer price tuning** — re-evaluate CCGT 1.50x markup and carrier overrides after wholesale fixes | Secondary to dispatch accuracy |
| 9 | **OCGT over-dispatch** — currently 5.27x; expect cascade improvement from hydro + wind fixes | Monitor only |

---

## 5. Structural Limitations (will not fix)

These are inherent to the copperplate→constrained architecture:

- **No energy balancing** — no forecast errors, plant trips, or demand deviations between stages. ~50% of real BM volume has no analogue.
- **No ancillary services** — frequency response, STOR, inertia, voltage support are not modelled. This limits PSH and battery utilisation.
- **Copperplate over-dispatch** — wholesale freely pushes Scottish wind south, forcing larger BM redispatch than reality. CCGT BM increase ~3.85x is a structural floor.

Possible future mitigations (not planned): partial relaxation of wholesale s_nom, zonal wholesale pricing, or self-dispatch heuristics.

---

## 6. Validation Targets

### Dispatch accuracy (primary)

| Metric | Current | Target |
|--------|---------|--------|
| Wind (post-BM) | +9.4% (48h) | ±8% |
| CCGT | 0.89x | ±5% |
| Coal | 0.86x | ±8% |
| Nuclear hourly r | 0.04 | >0.80 |
| Large hydro | 0.69x | >0.80x |
| Pumped hydro gen | 0.23x | >0.50x |

### BM cost targets (secondary)

| Metric | Current | Target |
|--------|---------|--------|
| Model/NESO thermal (Jan) | 0.66x | 0.8–1.2x |
| Model/NESO thermal (annual) | 0.28x | ≥0.55x (with outages) |
| CCGT BM increase | 3.85x | <2.0x (structural floor) |
| PSH BM increase | 0.44x | >0.30x ✓ |

---

## 7. How to Run Validation

```bash
# Run full market workflow (includes validation)
snakemake -j 4

# Outputs (for historical market scenarios):
resources/market/{scenario}_bm_validation.csv        # ELEXON comparison
resources/market/{scenario}_neso_validation.csv       # NESO comparison
resources/analysis/{scenario}_bm_validation.html      # ELEXON dashboard
resources/analysis/{scenario}_neso_validation.html    # NESO dashboard
resources/analysis/{scenario}_market_dashboard.html   # Market overview

# Quick check of solved network
python -c "
import pypsa
n = pypsa.Network('resources/market/Validation_Jan2020_balancing.nc')
print(n.generators.groupby('carrier')['p_nom'].sum().sort_values(ascending=False))
print(f'Objective: {n.objective}')
"
```

### Key validation data (cached in `data/validation/`)

| File | Source | Coverage |
|------|--------|----------|
| `thermal_constraint_costs_19-20.xlsx` | NESO API | Aug 2019 – Mar 2020 |
| `thermal_constraint_costs_20-21.xlsx` | NESO API | Apr 2020 – Mar 2021 |
| `day_ahead_constraint_flows_limits.csv` | NESO API | Oct 2019 – present |

Additional years download automatically on first run.
