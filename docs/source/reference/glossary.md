# Glossary

Terms and abbreviations used in this documentation and in the configuration.

```{glossary}
:sorted:

Balancing mechanism (BM)
  The market in which the system operator buys and sells energy close to real time to keep supply and demand
  balanced and power flows within network limits. In PyPSA-GB it is Stage 2 of the market: a network-constrained
  redispatch from the wholesale positions, priced with bids and offers. See {doc}`/model/dispatch/balancing`.

Bid
  A price at which a unit is paid (or pays) to reduce its output, or increase its demand, in the balancing
  mechanism. A renewable unit that loses support when turned down bids negatively: the system operator pays it.

Offer
  A price at which a unit is paid to increase its output, or reduce its demand, in the balancing mechanism.

BMU
  Balancing Mechanism Unit: the unit in which a generator, storage site or demand trades in the BM and is settled by
  Elexon. PyPSA-GB maps BMUs to its own generators for calibration and validation.

BSUoS
  Balancing Services Use of System charge: the charge through which NESO recovers balancing costs from users of the
  transmission system.

Boundary
  A cut through the transmission network separating two regions (for example B6, the Anglo-Scottish boundary). NESO
  publishes each boundary's transfer capability; PyPSA-GB can limit the sum of flows across it.

Secured capability
  The power a boundary can carry after the worst credible fault (an N-1 or N-D condition). It is lower than the sum of
  the circuits' thermal ratings. See {doc}`/model/network/constraints`.

CfD
  Contract for Difference: a support contract that pays a generator the difference between a strike price and a
  market reference price per MWh, and takes it back when the reference is higher. Awarded in allocation rounds
  (AR1, AR2, ...) and administered by the Low Carbon Contracts Company (LCCC).

Contract value
  The support a CfD unit loses per MWh it is turned down: the strike price less the reference price. PyPSA-GB can
  use it as the unit's BM bid.

RO, ROC
  Renewables Obligation, the support scheme that preceded CfDs (closed to new capacity in 2017). Accredited stations
  earn Renewables Obligation Certificates (ROCs) per MWh, at a banding by technology; a ROC is worth about the
  buy-out price, so a ROC unit bids negatively to avoid curtailment.

Copperplate
  A network model with unlimited transmission capacity: every bus sees the same price. Stage 1 of the market solves
  on a copperplate to represent the GB wholesale market, which has a single national price.

Central dispatch
  A dispatch in which the system operator schedules every unit at its cost with full knowledge of the network,
  rather than a market followed by a balancing mechanism. See {doc}`/model/dispatch/central_dispatch`.

CP30
  NESO's Clean Power 2030 advice to government (November 2024) and its data workbook. PyPSA-GB's 2030 study follows
  its Further Flex and Renewables (FFR) pathway.

ES1, ES5, ED1
  Tables of the CP30 data workbook: ES1 national generation capacity and energy by technology; ES5 capacity by
  region; ED1 demand. FES data workbooks use the same table codes.

Curtailment
  Renewable output that was available but not used. Economic curtailment happens in the wholesale stage when output
  is not worth using; constraint curtailment (turn-down) happens in the BM when the network cannot carry it.

Turn-down
  A unit reducing output in the BM at its bid, usually wind behind a congested boundary.

Redispatch
  Changing the wholesale positions of units to respect network limits: turning some down (bids) and others up
  (offers). Its cost is the constraint cost.

ETYS
  Electricity Ten Year Statement: NESO's annual statement of the GB transmission network. Appendix B lists every
  circuit and transformer, with ratings and planned changes, and is the source of the ETYS network model.

FES
  Future Energy Scenarios: NESO's annual set of pathways for GB energy to 2050 (for example Holistic Transition,
  Electric Engagement), published with capacity and demand by technology and Grid Supply Point.

GSP
  Grid Supply Point: a connection between the transmission network and a distribution network. FES and CP30 publish
  distribution-connected capacity and demand by GSP.

NOA
  Network Options Assessment: NESO's annual recommendation of which network reinforcements to proceed with. Its
  boundary capabilities and seasonal factors are used for the secured boundary limits.

Interconnector re-trading
  Changing an interconnector's scheduled flow in the BM (counter-trading) to relieve a GB constraint, paid at the
  difference between GB and foreign prices plus any premium. See {doc}`/model/dispatch/interconnector_trading`.

LDES
  Long-duration energy storage (in CP30: pumped hydro and other stores of many hours). New LDES is supported by a
  cap-and-floor regime.

SoC
  State of charge: the energy held in a storage unit (MWh). Rolling-window solves carry it from one window to the
  next.

TNUoS
  Transmission Network Use of System charge: the charge through which the transmission owners' allowed revenue is
  recovered from generators and suppliers.

TEC
  Transmission Entry Capacity: the capacity a generator is allowed to export to the transmission network, listed in
  NESO's TEC register.

VoLL
  Value of lost load: the cost assigned to unserved demand (GBP/MWh). It sets the price of the load-shedding
  generators that keep a solve feasible. Configured as `solve.voll`.

Wholesale market
  Stage 1 of the market: a copperplate dispatch that gives each unit its scheduled position and the system marginal
  price. See {doc}`/model/dispatch/wholesale`.

DUKES
  Digest of UK Energy Statistics (DESNZ). Table 5.11 lists power stations; it is the source of historical thermal
  capacity.

REPD
  Renewable Energy Planning Database (DESNZ): site-level renewable and storage projects with status and location.

ESPENI
  A cleaned half-hourly dataset of GB electricity demand and generation by source (Wilson et al.), used for
  historical demand.

Scenario, family, study
  A scenario is one model run (a network, a year and its settings). A family is a scenario solved as twelve monthly
  scenarios. A study is a set of families and scenarios in `config/studies/<study>/`.

Targets, method
  The two keys every scenario carries: `targets` says which publication its numbers follow (source, edition,
  pathway); `method` says how the fleet is built (`brownfield`, `gsp_block` or `historical`). See
  {doc}`/model/scenarios_configuration`.
```
