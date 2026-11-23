# Validation of the bill

The bill is checked three ways: line by line against Ofgem's current price cap (is anything missing?), against
Ofgem's 2020 cap by rebuilding a historical year (does the method reproduce a real bill?), and by the checks the
projector enforces on every run.

## Against Ofgem's current cap

Every electricity component of Ofgem's cap for Oct-Dec 2026 (cap model v1.31, Annex 2 v1.30, Annex 3 v1.25, Annex 4
v1.25, Annex 9 v1.11) was checked against the bill. No electricity cost in the cap is missing from the bill:

| Cap component | Bill line(s) |
| --- | --- |
| Direct fuel: forward index, shaping, re-hedging, imbalance, transaction, additional risk, losses | Energy, shaping, procurement allowances, losses ({doc}`wholesale`) |
| Direct fuel: CfD (and the LCCC operational levy) | CfD, LCCC/ESC levies ({doc}`cfd`) |
| Direct fuel: backwardation | Not included: forward premium zero centrally, with sensitivities |
| Capacity Market (and ESC administration) | Capacity Market, LCCC/ESC levies ({doc}`capacity_market`) |
| Adjustment allowance | Nil in Oct-Dec 2026 |
| Policy costs: RO, FiT, WHD, AAHEDC, NCC, nuclear RAB, ECO | One line each ({doc}`policy_levies`) |
| Network costs: TNUoS, DUoS, BSUoS | TNUoS, DUoS, three BSUoS lines ({doc}`networks`) |
| Core operating cost, debt-related costs, industry charges, EBIT, headroom, levelisation | Supplier costs and margin ({doc}`supplier`) |
| Smart-meter net cost change | Its own line |

Differences that remain, by design:

| Difference | Treatment |
| --- | --- |
| Backwardation (about £28 a year in Oct-Dec 2026) | Kept at zero; the forward-premium sensitivities bracket it |
| Payment method | The bill is a direct-debit household; standard credit pays more |
| VAT | The bill keeps 5%; the temporary 0% (October 2026 to March 2027) is not carried forward |
| Bill Discount Scheme, from April 2027 | Below £1 a household; not added |
| Renewables Obligation | Gross, where the cap is net of the temporary 75% Exchequer relief, which ends in March 2029 |

## Against Ofgem's 2020 cap: the backtest

The 2020 backtest rebuilds a year that has happened. The `consumer_bill_fast` family of the 2020 validation study
(`config/studies/validation_2020/`, scenarios `ConsumerBill_<Mon>2020_Fast`, annual group
`ConsumerBill_2020_MonthlyFast`) solves 2020 hourly as a two-stage market, and rule `backtest_consumer_bill`
(`scripts/consumer_bill/build_2020_backtest.py`) compares Ofgem's 2020 annual cap, component by component, with named
reconstruction paths:

| Component | Reconstruction | Inputs |
| --- | --- | --- |
| Wholesale | The model's spot prices carried through a procurement bridge to Ofgem's forward index; the bridge is fitted on other historical years and validated out of sample | Rules `build_pypsa_spot_procurement_bridge`, `build_ofgem_forward_procurement_bridge`, `align_historical_procurement` |
| CfD and RO | The model's support payments on its own dispatch, reconciled contract by contract (CfD, against LCCC) and station by station (RO, against Ofgem's certificate outturn), then recovered as the levies are | Rules `reconcile_annual_lccc_cfd_model_dispatch`, `reconcile_annual_roc_model_dispatch`, `build_annual_support_to_bill_bridge` |
| Networks | Allowed revenue through the network tariffs, with a physical loss proxy from the solved networks | Rules `build_network_cost_to_bill_bridge`, `aggregate_annual_physical_network_loss_proxy` |
| Everything else | Ofgem's published allowance, retained | The 2020 cap snapshot `data/market/ofgem_price_cap/2020_annual` |

A component without a reconstruction keeps Ofgem's published value and is labelled so; a missing candidate is blank,
never zero, and an unmapped contract is reported as not represented, never filled with a technology average. The
backtest writes `resources/analysis/<scenario>_consumer_bill_backtest.csv`, its summary JSON and an HTML report. The
2020 dispatch it rests on is validated in {doc}`/studies/validation_2020`.

The CfD settlement arithmetic is checked on its own: on LCCC's own generation, strike minus reference times output
reproduces LCCC's published 2020 payments within 0.01% (`scripts/consumer_bill/reconcile_lccc_cfd_model_dispatch.py`).

## Checks on every run

- **Energy.** The pathway's available renewable energy and its demand are checked against NESO's totals at build
  (`scripts/CP30/energy_validation.py`).
- **Consistency.** The projector refuses a recovery-demand file whose system demand disagrees with the run, a published
  line without a period basis, a CfD rule that differs from the balancing bids', a Capacity Market pin that differs
  from the auction builder (rule `verify_capacity_market_pin`), and a bill line without a group.
- **Results audit.** After a bills run, `scripts/public_ownership/results_audit.py` checks that no output is older than
  its inputs or its code and that no fallback was used.
