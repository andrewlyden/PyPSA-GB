# Recovery: from system cost to a household

## What it does

Most bill lines start as a system cost in pounds: what all CfDs paid, what balancing cost, what the Capacity Market's
agreements cost. A household's share is that cost spread over the volume it is levied on, multiplied by the household's
3.1 MWh. This page explains the volumes, losses, prices, periods and VAT.

The volumes are declared, not computed by each line, in the case's recovery file
(`data/market/consumer_bill_forecast/<year>_recovery_demand_assumptions.json`). The projector refuses to run without it
and checks it against the model run. Each line's volume and factors are written to the bill summary
(`line_recovery`), and `scripts/consumer_bill/line_recovery.py` converts between household and system pounds with them,
so every account converts a line the same way.

## Method and assumptions

### The volumes

The model's GB electrical load already contains network losses: it equals NESO's "GB Total System Demand", which
includes distribution and transmission losses, so it is a transmission-entry volume, not a metered one. The recovery
file therefore builds two volumes from it (`recovery_basis: loss_inclusive_two_volume`):

| Volume | Derivation | Lines recovered over it |
| --- | --- | --- |
| Metered supply | GB electrical load ÷ 1.14 + electrolysers' draw | Battery revenue top-up; the Exchequer's per-household key |
| Obligation-eligible supply | Metered supply × the share left after the energy-intensive-industry exemption | Renewables Obligation, Feed-in Tariffs |
| GSP charging volume | GB electrical load ÷ 1.14 × 1.085 + electrolysers' draw | BSUoS (three lines), LDES cap and floor, AAHEDC, LCCC/ESC, Capacity Market |
| Eligible GSP volume | GSP charging volume × the eligible share | CfD, Power BECCS CfD, Net Zero Teesside DPA |

- **1.14**, system demand over metered supply: FES 2025's FY2024 GB system demand (the same definition as the model's
  load) over Ofgem's licensed supply (range 1.125 to 1.165).
- **1.085**, metered supply to the GSP charging volume: NESO's 2024/25 BSUoS chargeable volume over Ofgem's licensed
  supply (range 1.08 to 1.10). It is the volume BSUoS and the CfD and Capacity Market supplier charges are levied on.
- **The eligible share** ({{ eligible_share }}) is LCCC's reconciled eligible demand over eligible plus
  energy-intensive-industry demand, 2024-25.
- Hydrogen produced by electrolysers pays nothing: electrolysers pay on their electricity draw.

### The recovery equation

A household's share of a system cost $C$, recovered over a volume $V$, is

$$
\text{household line} = \frac{C \times f}{V} \times 3.1\ \text{MWh} \times L \times a
$$

- **$C$, the system cost**, in pounds a year in its own price year.
- **$f$, the price-year factor**, carries $C$ to 2025 money ({{ rebasing_factor }} for a model cost in 2024 money).
- **$V$, the volume it is levied on**, in MWh. $C \times f / V$ is then the levy rate in £/MWh that a supplier pays on
  every MWh in the base: the CfD's interim levy rate, a BSUoS tariff.
- **$L$, the household's loss factor**, {{ loss_factor }} (Ofgem's current single-rate average, Annex 2 v1.30, `3c`),
  on every line charged at a GSP rate. A household's supplier takes more from the grid than its meter records. The RO
  and FiT are levied per MWh supplied and carry no loss uplift ($L$ = 1).
- **$a$, an allocation factor**, where a charge does not fall on all energy alike: the Capacity Market's household
  factor ({{ cm_household_factor }}, which already includes losses, so $L$ = 1 for it; {doc}`capacity_market`).

Why divide by the levy base, not by total demand: a levy is charged only on what it is charged on. The CfD is levied on
eligible supply, which excludes energy-intensive industry; spreading the same cost over a smaller base raises the rate
everyone else pays.

The same arithmetic run backwards turns a household figure into system pounds, so each line must be converted with
its own volume.

### Prices

The bill is in **2025 money** throughout, and each input is converted to it once:

| Input | Its price basis | Conversion to 2025 |
| --- | --- | --- |
| Model costs and prices (fuel, carbon, the hourly price, constraint costs) | 2024 real (`price_year: 2024`) | ONS CPIH annual average, 132.9 to 138.0 (× {{ rebasing_factor }}) |
| CfD strikes | 2012 money in the contracts | CPI to 2024 (× 1.39334, which reproduces DESNZ's published AR7 conversion: £65.45 in 2012 money = £91.20 in 2024 money), then the CPIH step to 2025 |
| Published charges | Their own year or cap period | Their own declared index; Ofgem's Oct-Dec 2026 figures come down to 2025 money by about 0.973 |

Model quantities and prices are combined in 2024 money, which is consistent because the strikes are brought to 2024
first, and the result moves to 2025 in one step. Strikes use CPI, as the contracts index on CPI; the 2024-to-2025 step
uses CPIH, the index the bill is reported in.

### Periods

The bill is a calendar year (`benchmark.bill_calendar_year`). Many published charges are set for fiscal years (April
to March) or cap periods, so each published line must declare how its period maps onto the calendar year
(`period_basis`), or the projector refuses it:

| Period basis | Meaning | Lines in 2030 |
| --- | --- | --- |
| `calendar_year` | Already a calendar-year figure | Capacity Market, ECO, battery top-up |
| `calendar_blend` | Fiscal years blended by their months in the year (for 2030: three months of 2029/30, nine of 2030/31) | TNUoS, DUoS, FiT, Warm Home Discount |
| `period_average` | An average over a span that contains the year | Net Zero Teesside DPA |
| `held_flat_real` | The latest published value held flat in real terms, naming the period it comes from | Supplier costs, smart meters, Sizewell C RAB, NCC, AAHEDC, LCCC/ESC, BSUoS other services, voltage and stability |
| `extrapolated_declared` | Extrapolated by a declared rule from a named period | none in 2030 |

### VAT

VAT at 5% (the domestic rate for electricity) is applied once, to the ex-VAT total. A temporary 0% rate on electricity
(October 2026 to March 2027) is not carried into future years.

## Data

- Ofgem, default tariff cap, Annex 2 v1.30, `3c Electricity losses`.
- FES 2025 (FY2024 system demand); NESO BSUoS settlement data 2024/25; Ofgem FiT licensed supply.
- LCCC, reconciled eligible and energy-intensive-industry demand, 2024-25.
- ONS CPIH (L522) and CPI (D7BT).

## Configuration

`recovery_demand_assumptions` of the forecast case names the recovery file. The file is refreshed from the solved
months by `scripts/consumer_bill/refresh_recovery_demand.py`; each line's recovery entry is declared with the line
(`line_recovery` in the summary).

## Validation

The projector refuses a recovery file whose system demand disagrees with the run, volumes that do not reconcile, and a
household loss factor different from the one the procurement losses line uses. The worked example below is one run's
volumes and its CfD recovery:

```{include} _generated/recovery_volumes.md
```

```{include} _generated/recovery_cfd_example.md
```
