# Market Dispatch

PyPSA-GB can run a two-stage electricity market simulation after the standard
network build. The market workflow separates the unconstrained wholesale market
from the constrained physical dispatch that is needed after transmission limits
are restored.

The market mode is useful when you want to analyse:

- a single GB wholesale price from copperplate dispatch;
- balancing mechanism (BM) redispatch caused by network constraints;
- constraint costs and congested network boundaries;
- historical BM validation against ELEXON and NESO data;
- dispatch and revenue impacts of CfD and ROC subsidy assumptions.

The implementation lives in `rules/market.smk` and `scripts/market/`.

## Workflow Overview

Market dispatch starts from the final network produced by the normal PyPSA-GB
workflow:

```{mermaid}
flowchart TB
    FINAL["Finalized network<br/>resources/network/{scenario}.nc"]
    WS["Stage 1: wholesale market<br/>copperplate dispatch"]
    BM["Stage 2: balancing mechanism<br/>anchored constrained redispatch"]
    ANALYSIS["Analysis<br/>dashboard, summary, notebooks"]
    VALIDATION["Historical validation<br/>ELEXON BM and NESO constraints"]

    FINAL --> WS
    WS --> BM
    BM --> ANALYSIS
    BM --> VALIDATION
```

Stage 1 relaxes line and transformer capacities to a very large value. This
creates a copperplate dispatch and a uniform wholesale price.

Stage 2 reloads the original constrained network and anchors each participating
asset to its wholesale position. The solver can move assets up or down only
through explicit increase and decrease variables, priced by BM offer and bid
prices.

## Enabling Market Mode

Enable the workflow per scenario in `config/scenarios.yaml`:

```yaml
My_Market_Scenario:
  description: "Example two-stage market run"
  modelled_year: 2035
  network_model: "ETYS"
  FES_scenario: "Holistic Transition"

  solve_period:
    enabled: true
    start: "2035-01-01 00:00"
    end: "2035-01-07 23:00"

  market:
    enabled: true
    wholesale:
      mode: "rolling_day_ahead"
      window_hours: 24
      carry_soc: true
    balancing:
      mode: "rolling"
      window_hours: 1
      bid_offer_source: "derived"
      fix_interconnectors: true
```

The full default configuration is in `config/defaults.yaml` under `market:`.
Scenario-level values override these defaults.

## Running

Run the normal workflow. Market scenarios are detected automatically when
`market.enabled: true`.

```bash
snakemake -j 4
```

To build a specific market output:

```bash
snakemake resources/analysis/My_Market_Scenario_market_dashboard.html -j 4
```

For development, it is often faster to target the individual stage you are
debugging:

```bash
# Stage 1 only
snakemake resources/market/My_Market_Scenario_wholesale.nc -j 4

# Stage 2
snakemake resources/market/My_Market_Scenario_balancing.nc -j 4

# Analysis notebook
snakemake resources/analysis/My_Market_Scenario_market_notebook.ipynb -j 1
```

Use `market.wholesale_only: true` if you only want the wholesale solve and
wholesale analysis notebook.

## Running A Wholesale-Only Market

Wholesale-only mode runs Stage 1 without the balancing mechanism. Use it when
you want a copperplate market schedule, a uniform wholesale price, and a quick
check of merit-order dispatch before adding network-constrained redispatch.

Enable it with both `market.enabled: true` and `market.wholesale_only: true`:

```yaml
My_Wholesale_Only_Scenario:
  description: "Historical 2024 wholesale-only market run"
  modelled_year: 2024
  network_model: "ETYS"

  solve_period:
    enabled: true
    start: "2024-01-01 00:00"
    end: "2024-01-07 23:00"

  market:
    enabled: true
    wholesale_only: true
    wholesale:
      mode: "rolling_day_ahead"
      window_hours: 24
      carry_soc: true
      transmission_relaxation: 1.0e6
```

Then run the normal workflow:

```bash
snakemake -j 4
```

Or target the wholesale outputs directly:

```bash
snakemake resources/market/My_Wholesale_Only_Scenario_wholesale.nc -j 4
snakemake resources/analysis/My_Wholesale_Only_Scenario_wholesale_notebook.ipynb -j 1
```

In wholesale-only mode, the workflow includes:

| Output | Meaning |
|--------|---------|
| `resources/market/{scenario}_wholesale.nc` | Solved copperplate network |
| `resources/market/{scenario}_wholesale_dispatch.csv` | Generator wholesale schedule |
| `resources/market/{scenario}_wholesale_storage.csv` | Storage wholesale schedule |
| `resources/market/{scenario}_wholesale_links.csv` | Link and interconnector schedule |
| `resources/market/{scenario}_wholesale_price.csv` | Uniform demand-bus wholesale SMP |
| `resources/analysis/{scenario}_wholesale_notebook.ipynb` | Generated wholesale analysis notebook |

The workflow skips the BM outputs, market dashboard, BM validation, and NESO
constraint validation because no constrained redispatch is solved.

Choose the wholesale mode based on the question:

| Setting | Use when |
|---------|----------|
| `mode: "single"` | You want the fastest run or full-period storage foresight. |
| `mode: "rolling_day_ahead"` | You want a day-ahead style schedule with limited storage foresight. |

For historical price comparison, the generated wholesale notebook can overlay
the model SMP with ELEXON MID data when the relevant market data is available.
For future scenarios, use the wholesale outputs as modelled counterfactual
prices rather than validation against observed market prices.

## Stage 1: Wholesale Market

The wholesale stage is implemented by `scripts/market/solve_wholesale.py`.
It removes internal transmission congestion by setting every line and
transformer rating to `market.wholesale.transmission_relaxation`.

```yaml
market:
  wholesale:
    mode: "single"
    transmission_relaxation: 1.0e6
```

Available modes:

| Mode | Description |
|------|-------------|
| `single` | Solve the whole selected period in one optimisation. Storage has full-period foresight. |
| `rolling_day_ahead` | Solve independent rolling windows, normally 24 hours. Storage state of charge can carry between windows. |

Rolling day-ahead mode is usually a better representation of operational
market scheduling, especially for storage:

```yaml
market:
  wholesale:
    mode: "rolling_day_ahead"
    window_hours: 24
    carry_soc: true
```

The wholesale stage writes:

| Output | Meaning |
|--------|---------|
| `resources/market/{scenario}_wholesale.nc` | Solved copperplate network |
| `resources/market/{scenario}_wholesale_dispatch.csv` | Generator dispatch by snapshot |
| `resources/market/{scenario}_wholesale_storage.csv` | Storage dispatch by snapshot |
| `resources/market/{scenario}_wholesale_links.csv` | Link dispatch by snapshot |
| `resources/market/{scenario}_wholesale_price.csv` | Uniform wholesale price time series |

Wholesale prices are reported for demand buses only. Internal DC link buses can
have directional shadow-price artefacts even in a copperplate solve, so they are
excluded from the reported spread.

### Optional Unit Commitment

The wholesale stage can use a unit commitment overlay even when the rest of the
scenario uses LP mode. This is slower, but can produce more realistic thermal
schedules.

```yaml
market:
  wholesale:
    unit_commitment:
      enabled: true
      min_p_nom_mw: 50.0
```

Carrier-specific commitment parameters are configured under
`market.wholesale.unit_commitment.carrier_parameters`.

## Stage 2: Balancing Mechanism

The balancing stage is implemented by `scripts/market/solve_balancing.py`.
It solves the physical network with original constraints and minimises
redispatch cost:

```text
physical_dispatch = wholesale_dispatch + increase - decrease
cost = offer_price * increase + bid_price * decrease
```

Bid prices use the ESO-cost convention: a positive bid price means it costs the
ESO money to turn the asset down.

Available modes:

| Mode | Description |
|------|-------------|
| `full_period` | Solve the whole period in one constrained optimisation. |
| `rolling` | Solve shorter constrained windows anchored to the wholesale positions. |

For BM-style operation, a one-hour rolling window is common:

```yaml
market:
  balancing:
    mode: "rolling"
    window_hours: 1
```

Storage state of charge is carried between rolling BM windows automatically.

The balancing stage writes:

| Output | Meaning |
|--------|---------|
| `resources/market/{scenario}_balancing.nc` | Solved constrained network |
| `resources/market/{scenario}_balancing_dispatch.csv` | Physical generator dispatch |
| `resources/market/{scenario}_balancing_storage.csv` | Physical storage dispatch |
| `resources/market/{scenario}_redispatch_summary.csv` | Increase/decrease volume and cost by asset |
| `resources/market/{scenario}_constraint_costs.csv` | Constraint cost summary by carrier |
| `resources/market/{scenario}_congestion.csv` | Congested line and transformer diagnostics |
| `resources/market/{scenario}_price_comparison.csv` | Wholesale price and constrained nodal price comparison |

Like wholesale reporting, BM nodal price spread calculations use demand buses
only.

## Bid And Offer Prices

Configure the price source with `market.balancing.bid_offer_source`.

| Source | Use case |
|--------|----------|
| `auto` | Prefer ELEXON data for historical scenarios when available; otherwise use derived prices. |
| `derived` | Compute prices from marginal costs and configured markups or absolute carrier prices. |
| `elexon` | Use historical ELEXON BMRS bid/offer data. Valid only for historical scenarios. |
| `csv` | Load user-supplied offer and bid CSV files. |

### Derived Prices

Derived pricing starts from generator marginal costs:

```yaml
market:
  balancing:
    bid_offer_source: "derived"
    default_offer_markup: 0.10
    default_bid_discount: 0.10
```

Carrier overrides can use either markup mode or absolute prices:

```yaml
market:
  balancing:
    carrier_overrides:
      CCGT:
        mode: "markup"
        offer_markup: 0.85
        bid_discount: 0.10
      nuclear:
        mode: "absolute"
        offer_price: 999.0
        bid_price: 150.0
      wind_offshore:
        mode: "absolute"
        offer_price: 0.0
        bid_price: 90.0
```

Absolute prices are useful for technologies where BM behaviour is not well
represented by short-run marginal cost, such as nuclear, subsidised renewables,
and storage.

### ELEXON Prices

Historical scenarios can use ELEXON BMRS bid and offer data:

```yaml
market:
  balancing:
    bid_offer_source: "elexon"
    elexon:
      fallback: "carrier_average"
      price_ladders:
        enabled: false
```

When ELEXON mode is selected, the workflow retrieves or reuses cached BOD data,
builds a BMU-to-generator mapping, and writes per-scenario files under
`resources/market/{scenario}/elexon/`.

If `price_ladders.enabled: true`, the BM solve can use ELEXON price/volume
blocks rather than a single averaged bid and offer price.

### CSV Prices

Use CSV mode for custom studies:

```yaml
market:
  balancing:
    bid_offer_source: "csv"
    csv:
      offer_file: "path/to/offers.csv"
      bid_file: "path/to/bids.csv"
```

CSV inputs should be aligned with the scenario snapshots and model asset names.

## BM Participation

By default, all generators and storage units can redispatch. You can restrict
participation under `market.balancing.participation`.

```yaml
market:
  balancing:
    participation:
      generators:
        mode: "elexon_mapped"
        behavior: "priced_out"
        min_p_nom_mw: 50.0
      storage_units:
        mode: "all"
        behavior: "priced_out"
```

Participation modes:

| Mode | Meaning |
|------|---------|
| `all` | Every asset in that class can redispatch. |
| `none` | No assets in that class participate. |
| `elexon_mapped` | Only assets with direct BMU mappings participate. |

Non-participant behaviour controls how the solver handles assets that would be
needed for feasibility:

| Behaviour | Meaning |
|-----------|---------|
| `priced_out` | Allow emergency redispatch at penalty prices. |
| `fallback_priced` | Allow redispatch at the normal fallback prices. |
| `fixed` | Lock the asset at its wholesale position. This is stricter but can make the problem infeasible. |

## Interconnectors

Set `market.balancing.fix_interconnectors: true` to hold interconnector links at
their wholesale position in the BM solve. This is the default and reflects a
first-order assumption that cross-border schedules are not redispatched by the
GB BM.

```yaml
market:
  balancing:
    fix_interconnectors: true
```

Set it to `false` only for sensitivity analysis where interconnector
redispatch is intentionally allowed.

## Subsidies And Revenue Tracking

Subsidy-aware marginal costs and market revenue tracking are separate options.

During generator integration, `subsidy_tracking.enabled: true` stores
`support_type` attributes such as `CfD`, `ROC`, and `merchant`. If
`marginal_costs.subsidies.enabled: true`, these attributes affect renewable
dispatch order before the market solve.

```yaml
subsidy_tracking:
  enabled: true

marginal_costs:
  subsidies:
    enabled: true

market:
  revenue_tracking:
    enabled: true
    include_cfd: true
    include_roc: true
```

Revenue tracking computes post-solve CfD difference payments and ROC income.
It requires both market mode and subsidy tracking.

## Analysis Outputs

The `analyze_market_results` rule creates:

| Output | Meaning |
|--------|---------|
| `resources/analysis/{scenario}_market_dashboard.html` | Interactive Plotly dashboard |
| `resources/analysis/{scenario}_market_summary.json` | Machine-readable metrics |

The generated dashboard compares wholesale and physical dispatch, redispatch
volumes, constraint costs, congestion, and price spreads.

You can also generate a per-scenario notebook:

```bash
snakemake resources/analysis/My_Market_Scenario_market_notebook.ipynb -j 1
```

For wholesale-only scenarios:

```bash
snakemake resources/analysis/My_Market_Scenario_wholesale_notebook.ipynb -j 1
```

## Historical Validation

For historical market scenarios (`modelled_year <= 2024`), extra validation
rules compare the model with observed market and constraint data.

| Rule | Output | Purpose |
|------|--------|---------|
| `validate_bm_results` | `{scenario}_bm_validation.csv`, `{scenario}_bm_validation.html` | Compare model BM volumes, costs, prices, and dispatch with ELEXON actuals. |
| `validate_bm_calibration` | `{scenario}_bm_calibration.csv`, `{scenario}_bm_calibration.html` | Compare model bid/offer prices and redispatch volumes with ELEXON carrier-level medians. |
| `validate_neso_constraints` | `{scenario}_neso_validation.csv`, `{scenario}_neso_validation.html` | Compare boundary flows and constraint costs with NESO constraint data. |

Example targets:

```bash
snakemake resources/analysis/Historical_2020_ETYS_market_bm_validation.html -j 4
snakemake resources/analysis/Historical_2020_ETYS_market_neso_validation.html -j 4
```

Network access may be needed the first time historical ELEXON or NESO data is
downloaded. Subsequent runs use local caches where available.

## Choosing Modes

Use these starting points:

| Goal | Suggested settings |
|------|--------------------|
| Fast smoke test | `wholesale.mode: "single"`, `balancing.mode: "full_period"`, short `solve_period` |
| Operational market study | `wholesale.mode: "rolling_day_ahead"`, `balancing.mode: "rolling"`, `balancing.window_hours: 1` |
| Future scenario without historical BM data | `bid_offer_source: "derived"` |
| Historical validation | `bid_offer_source: "elexon"` or `auto`, with ELEXON validation targets |
| Wholesale price only | `market.wholesale_only: true` |

## Common Issues

### ELEXON requested for a future scenario

`bid_offer_source: "elexon"` is valid only for historical scenarios. Use
`derived` for future years, or `auto` if you want PyPSA-GB to choose the best
available source.

### BM solve is infeasible

Common causes are overly strict participation settings, fixed non-participants,
or a network that cannot physically serve demand without redispatching assets
that have been excluded.

Start by using:

```yaml
market:
  balancing:
    participation:
      generators:
        mode: "all"
        behavior: "priced_out"
      storage_units:
        mode: "all"
        behavior: "priced_out"
```

Then tighten the participation filters once the scenario solves.

### Wholesale price spread looks too high

Use the `*_wholesale_price.csv` output rather than directly averaging all
entries in `network.buses_t.marginal_price`. The reporting code filters to
demand buses because internal DC link buses can retain directional price
differences.

### Redispatch volume is unexpectedly large

Check the wholesale and BM modes first. A copperplate wholesale dispatch can
schedule remote generation behind real network constraints, especially Scottish
wind and English thermal generation. That is expected to produce redispatch.
Use `*_congestion.csv`, `*_constraint_costs.csv`, and the market dashboard to
identify the binding regions and assets.

### Runs are slow

Market mode solves at least two optimisation problems per scenario. Rolling
mode solves many smaller problems. Reduce the `solve_period`, use a Reduced or
Zonal network for testing, or start with `single` and `full_period` modes before
moving to rolling runs.
