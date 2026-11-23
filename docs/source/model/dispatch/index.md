# Dispatch

How the model decides what runs, hour by hour, on a finished network. There are two ways to do it.

- **A single network-constrained solve.** One least-cost optimisation over the solve period with every
  transmission limit in force. It answers "what is the cheapest feasible dispatch?" and gives a
  price at every bus. This is the default ({doc}`single_solve`).
- **A two-stage market.** Stage 1 clears a GB wholesale market with no internal network limits, as
  the day-ahead market does: one GB price, and a position (schedule) for every unit. Stage 2 is the
  balancing mechanism: the system operator moves units off those positions, paying offers to turn
  up and bids to turn down, until the dispatch fits the network. The cost of those moves is the
  constraint cost. Set `market.enabled: true` in a scenario to use it ({doc}`wholesale`,
  {doc}`balancing`).

The two-stage market separates two questions the single solve answers together: what the market
schedules, and what it costs to make that schedule physically possible. That is why the consumer
bill is built on it.

```{mermaid}
flowchart TD
    N["Finalised network<br/>(finalize_network)"]
    N -->|"market.enabled: false"| S["Single solve<br/>(solve_network)<br/>LOPF on the full network"]
    S --> SO["Solved network<br/>and results tables"]
    N -->|"market.enabled: true"| W["Stage 1: wholesale<br/>(solve_wholesale_market)<br/>copperplate, one GB price"]
    W --> P["Positions of every unit,<br/>GB price, hydrogen value"]
    P --> B["Stage 2: balancing mechanism<br/>(solve_balancing_mechanism)<br/>full network, bids and offers"]
    N -.->|"reloaded with<br/>real ratings"| B
    B --> R["Redispatch volumes,<br/>constraint costs,<br/>interconnector re-trading"]
    R --> A["analyze_market_results<br/>market summary<br/>and accounting"]
    R --> T["Turn-down re-pricing<br/>(monthly studies)<br/>cheapest first and<br/>pro rata by zone"]
```

The pages in this section:

| Page | What it covers |
|------|----------------|
| {doc}`single_solve` | `finalize_network` and `solve_network`: the one-stage LOPF, solver settings, load shedding, numerical conditioning |
| {doc}`wholesale` | Stage 1: the copperplate, rolling day-ahead windows and look-ahead, scarcity pricing, the reported GB price |
| {doc}`balancing` | Stage 2: anchoring to Stage 1, the bid and offer objective, what is held, cost measures, turn-down allocation |
| {doc}`bids_offers_support` | How offers and bids are priced, and how CfD, RO and merchant support enter dispatch |
| {doc}`interconnector_trading` | Cross-border cables in the balancing stage: fixed, or re-traded at a counter-trade price |
| {doc}`central_dispatch` | The `central` dispatch overlay: the operator schedules and corrects at resource cost |
| {doc}`solvers` | The solver and its settings, choices by run size, and what to do when a solve is slow or fails |

```{toctree}
:maxdepth: 2

single_solve
wholesale
balancing
bids_offers_support
interconnector_trading
central_dispatch
solvers
```
