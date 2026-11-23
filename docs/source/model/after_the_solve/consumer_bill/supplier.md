# Supplier costs

## What it does

These lines pay for running a supplier: serving customers, bad debt, industry bodies, and the supplier's margin and
headroom. In Ofgem's cap they are operating costs, EBIT and headroom.

## Method and assumptions

### Supplier costs and margin

Ofgem's supplier allowances from the cap model v1.31 (Oct-Dec 2026), for a single-rate, direct-debit customer ("other
payment method"), GB average of the 14 regions, at 3,100 kWh from Ofgem's nil-consumption and 2,500 kWh values (a
standing charge plus a unit rate): core operating cost, debt-related costs and industry charges. The direct-debit
levelisation (Ofgem Annex 9 v1.11, GB mean) is added.

The allowances and the levelisation are rebased by × {{ supplier_to_report_money }} to 2025 money and held flat in real
terms. EBIT and headroom are not carried as the cap period's cash: they are Ofgem's percentages applied to the bill's
own costs, as the cap model builds them:

- EBIT: 1.44% of costs, plus £11.23 per dual-fuel customer, halved per fuel;
- headroom: 1.46% of costs excluding networks, plus EBIT.

The cap's adjustment allowance is nil in Oct-Dec 2026.

### Smart-meter net cost change

Ofgem's SMNCC at 3,100 kWh from its nil and 2,500 kWh values. It is negative since October 2025, after the July 2025
restructuring moved smart-metering costs into cost to serve and industry charges.

### Assumptions to keep in view

- **Held flat in real terms.** These costs scale with customers, not with the power system, so the pathway does not
  move them. Known pressures, none modelled:
  - *Debt-related costs* are the most uncertain, and the risk is upward: household energy debt is at a record, and
    Ofgem's proposed Debt Relief Scheme would be funded by a temporary charge on other customers' bills.
  - *Operating costs*: Ofgem's 2025 review cut them; its programme of work for 2026 and 2027 plans no new
    operating-cost review. Its work on debt and affordability, standing charges and cost allocation could move costs
    between the standing charge and the unit rate.
  - *Smart metering*: Ofgem will update its smart-meter cost model.
- **Payment method.** A standard-credit customer pays more, through higher core operating cost and debt-related
  allowances.

## Data

- Ofgem, Default tariff cap level v1.31 (Oct-Dec 2026): sheets `3m CO`, `3n DRC`, `3o IC`, `3k EBIT`, `3l HAP`,
  `3i SMNCC`; Annex 9 v1.11 (levelisation).
- Ofgem, Energy price cap programme of work for 2026 and 2027 (August 2026); operating cost and debt allowances
  decision (May 2025).
- NEA, *Clearing the Decks* (June 2026), on energy debt and the Debt Relief Scheme.

## Configuration

`sourced_externals.supplier_costs_and_margin` (its `derivation` holds the allowances, the levelisation and the EBIT and
headroom percentages) and `sourced_externals.smart_meter_net_cost_change` in the assumptions file.

## Validation

Every allowance is Ofgem's own, so the check is that each cap component has a bill line ({doc}`validation`). The worked
example below is one run's build-up:

```{include} _generated/supplier_components.md
```
