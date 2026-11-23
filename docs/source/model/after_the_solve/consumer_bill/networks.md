# Network charges and balancing

## What it does

These lines pay for the wires and for running the system in real time. In Ofgem's cap all of them are network costs
(Annex 3: TNUoS, DUoS and BSUoS).

| Line | Basis |
| --- | --- |
| DUoS | Ofgem's current allowance rolled forward on the pathway's distribution investment and demand |
| TNUoS | NESO's five-year view of TNUoS tariffs |
| BSUoS: constraints | Modelled: the balancing stage's accepted bids and offers |
| BSUoS: other services | NESO's ancillary services carried to the year by the pathway's system needs |
| BSUoS: voltage and stability | NESO's voltage and inertia actions |
| Interconnector cap and floor | Modelled: each regulated cable's Stage 1 congestion rent against its cap and floor (a credit or a charge) |

## Method and assumptions

### DUoS: distribution

Distribution Use of System charges pay the 14 distribution network operators' allowed revenue under Ofgem's price
control (RIIO-ED2 to March 2028, then ED3). The roll-forward is `scripts/consumer_bill/duos_cp30_growth.py`:

1. **Base.** The GB average of Ofgem's cap for Oct-Dec 2026 (DNO charging year 2026/27): £42.39 a year fixed plus
   £27.10/MWh. It carries no Supplier of Last Resort or bad-debt adders.
2. **ED2 to 2027/28** on the ED2 Finance Annex's revised tables (Appendix 8): the ratio of final-determination revenue
   for 2027/28 to 2026/27, both in 2020/21 prices.
3. **ED3 on the pathway's investment.** From the 31 March 2028 regulatory asset value (RAV), each year adds CP30's
   distribution capex (CP.13; Baringa built it from ED2) and depreciates at ED2's final-year rate. Allowed revenue
   moves with the return and tax allowance on the average RAV, the depreciation, and fast money with the pathway's
   peak demand.
4. **ED3 return.** RAV × (ED3's allowed return − ED2's) is added to allowed revenue (Ofgem's early view, 4.35% real
   against 3.93%).
5. **Charging base.** The unit part of the household's charge is spread over the pathway's demand growth (GB system
   demand excluding electrolysis); the per-site part moves with revenue alone.
6. Calendar 2030 is 3/12 of 2029/30 and 9/12 of 2030/31, rebased to 2025 money.

Sensitivities: no real growth in ED3; ED2's own RAV additions continued with peak demand; the charging base held.

### TNUoS: transmission

Transmission Network Use of System charges pay the transmission owners' allowed revenue (RIIO-ET3 from April 2026).
For a household it is mostly a fixed daily residual charge plus a small locational charge on use between 16:00 and
19:00. The line takes NESO's five-year view of TNUoS tariffs (2027/28-2031/32, version 1.0, 9 October 2026): the
domestic residual per site per day and the non-half-hourly locational rate, applied to the household's peak-time use,
blended to the calendar year and deflated to 2025.

NESO's revenue is the transmission owners' January 2026 submissions, which put Norwich-Tilbury and Sea Link on their
own 2031 dates. The forecast case's `tnuos_network_delivery` key names a network-delivery case in
`data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` (`project_future_bill.apply_tnuos_network_delivery`),
so a delayed network can lower onshore TO income (NESO's revenue sensitivity 1 as a proxy). CP30's own network costs
are a bound, not a replacement: they count only investment added since 2023, annuitised, while TNUoS recovers a return
on and of the whole asset base; and offshore network costs reach bills mostly through CfD strikes, not TNUoS.

### BSUoS: constraints (modelled)

Balancing Services Use of System charges recover what NESO spends running the system. The thermal constraint part
comes from the model: Stage 2 redispatches the Stage 1 schedule around the network's limits, and the cost of the
accepted offers and bids is the constraint cost (`scripts/consumer_bill/extract_future_balancing_costs.py`;
{doc}`/model/dispatch/balancing`):

$$
\text{constraint cost} = \text{offers} + \text{bids} - \text{feasibility slack} - \text{non-participant turn-down} + \text{interconnector re-trading}
$$

- **CfD plant turned down** is paid its lost difference payment through its bid (contract-value bids), so it is
  already inside the accepted bids and is not added again.
- **Interconnector re-trading** is the change to the cables' Stage 1 schedules in balancing, valued at the trade
  prices ({doc}`/model/dispatch/interconnector_trading`); NESO's constraint-management category carries the same
  actions.
- **Turn-down basis.** The balancing LP turns wind down cheapest first; NESO shares a zone's turn-down across its wind.
  Each month is reported on both bases (`scripts/market/turn_down_allocation.py`), and
  `market.balancing.turn_down_allocation.mode` picks the one the bill uses (`closest_to_neso` in the base: whichever is
  nearer NESO's published cost for the network case). The bill reads per-unit balancing results only through the
  helpers that apply that choice.

The cost is spread over the GSP charging volume and carries the household's losses, as Ofgem's cap treats BSUoS
({doc}`recovery`).

### BSUoS: other services, and voltage and stability

**Other services.** Each ancillary service is its FY2024/25 outturn carried to the year by a CP30 driver
(`scripts/public_ownership/ancillary_services_2030.py`, table `data/public_ownership/financing/ancillary_services_2030.csv`):

| Service | Driver |
| --- | --- |
| Stability network services | Contracts live in the year plus CP30's added short-circuit need (CP.21), at the Phase 2 and 3 unit cost |
| Commercial reactive power | Today's book plus CP.21's added absorption and injection, at the Voltage 2026 tender's unit cost |
| Response, reserve | Scaled by CP30's requirement against NESO's Operability Strategy Report (March 2025) |
| Restoration | Scaled for the restoration standard |
| Voltage and inertia actions in the BM, obligatory reactive power | Held at outturn: CP30 holds them itself |

NESO's internal costs and the remaining trade categories are held at outturn per MWh of final demand. The
plant that provides each service is credited with it (which lowers the battery top-up); synchronous condensers and
other providers outside the modelled fleet are paid in the line but credited to no modelled asset.

**Voltage and stability.** NESO's voltage and inertia actions at FY2024/25 outturn. The model's DC power flow has no
voltage or inertia constraints, so this cost has no other home in the bill.

### Interconnector cap and floor

Regulated interconnectors keep their congestion rent between a floor and a cap; rent above the cap is returned to
consumers, and a shortfall below the floor is paid by them, through TNUoS. The line takes each cable's Stage 1
congestion rent, applies its regime's cap and floor (RPI-indexed), and spreads the net through the TNUoS demand
residual. NESO's TNUoS forecast carries no interconnector adjustment of its own in 2029/30 or 2030/31, so the two do
not overlap. The cables are CP30's set (`data/interconnectors/cp30_interconnector_composition.csv`). MaresConnect has
a cap and floor in principle but its levels are set only at Ofgem's Final Project Assessment; until then they are
derived from Greenlink's per MW of cable, with NeuConnect's as the sensitivity.

### Assumptions to keep in view

- **Interconnector re-trading** is a large part of the constraint line and rests on how the balancing stage prices
  foreign plant ({doc}`/model/dispatch/interconnector_trading`).
- **TNUoS** rests on the transmission owners' January 2026 revenue forecasts; NESO's later tariffs will update them.
- **ED3** has no determinations: the growth and the return are declared assumptions.
- **Ancillary and stability costs**: the stability unit cost is the largest uncertainty; part of the need could
  instead be met by transmission-owner assets through TNUoS.

## Data

- Ofgem, default tariff cap, Annex 3 network cost allowance v1.25 (Oct-Dec 2026).
- Ofgem, RIIO-ED2 final determinations (November 2022) and finance annex; ED3 sector-specific methodology decision,
  finance annex (May 2026).
- NESO, Five-Year View of TNUoS Tariffs 2027/28-2031/32, version 1.0, 9 October 2026 (tables 14, 15, 23, 38, 39 and B).
- NESO, balancing services and constraint cost outturn, FY2024/25; Operability Strategy Report (March 2025).
- NESO, Clean Power 2030 data workbook (CP.13, CP.21, CP.31) and Annexes 3 and 4.
- Ofgem, interconnector cap and floor regime handbook; Window 3 Initial Project Assessment (November 2024).

## Configuration

- `model_linked_rules.bsuos_constraints` and `model_linked_rules.interconnector_cap_and_floor`;
  `sourced_externals.tnuos`, `.duos`, `.bsuos_other_services`, `.bsuos_voltage_and_stability_constraints` in the
  assumptions file.
- The forecast case's `tnuos_network_delivery` key.
- `market.balancing.turn_down_allocation.mode` for the constraint line's turn-down basis.

## Validation

- **Constraint cost against NESO.** The model's NESO-comparable redispatch cost for each network case is compared with
  NESO's published figure for that case (`data/CP30/validation_references/neso_balancing_benchmarks.csv`); see {doc}`/studies/cp30_2030`. CP30's own
  thermal constraint cost for the pathway (CP.31) is a further reference.
- **Against Ofgem's cap.** Each network line is set beside the cap's current allowance in {doc}`validation`.

The worked example below is one run's constraint build-up (GBP million, 2024 money):

```{include} _generated/network_constraint_build_up.md
```
