## `public_ownership.consumer_bill`

File: `config/defaults/base/public_ownership/consumer_bill.yaml`. Keys below are relative to `public_ownership.consumer_bill`.

| Key | Default | Description |
|---|---|---|
| `forecast_case` | `CP30_2030_cp30_network` | How each public-ownership case reaches the consumer bill. Every value is validated by scripts/public_ownership/reference_runs.py and only the listed treatments are implemented. The consumer_bill_forecasts case the per-case bill is projected through. Its source_scenario_pattern must equal the 2030 group's scenario_pattern, so the private case reproduces that projection exactly. |
| `cfd_repricing` | (mapping) |  |
| `cfd_repricing.at_cost_electricity_leg_carriers` | `[CCS Biomass]` | At-cost re-pricing charges these carriers only the electricity leg's share of required revenue (Power BECCS: capture is paid off the bill through the carbon-removal contract). |
| `cfd_repricing.eligible_award_years_from` | `2025` | Contracts from rounds awarded in or after this year are re-priced (route R1); capacity awarded earlier but commissioned later keeps its strike and its saving goes to the public account (route R2). Award years by round: award_years. |
| `cfd_repricing.award_years` | `data/public_ownership/financing/cfd_round_award_years.csv` |  |
| `cfd_repricing.contract_term_years` | `20` | CfD term (years) for lever cfd_amortisation: contract_term spreads the cohort's capital charge change over this term, economic_life over the technology's life. AR7 and later: 20 years for offshore, floating and onshore wind and solar (DESNZ AR7 final response, July 2025). Earlier contracts keep their own terms: Power BECCS 15 (DESNZ business model update, Dec 2023), AR1-AR6 and the FIDeR Investment Contracts 15, Hinkley Point C 35. |
| `cfd_repricing.contract_term_years_by_round` | (mapping) |  |
| `cfd_repricing.contract_term_years_by_round.POWER_BECCS` | `15` |  |
| `cfd_repricing.contract_term_years_by_round.AR1` | `15` |  |
| `cfd_repricing.contract_term_years_by_round.AR2` | `15` |  |
| `cfd_repricing.contract_term_years_by_round.AR3` | `15` |  |
| `cfd_repricing.contract_term_years_by_round.AR4` | `15` |  |
| `cfd_repricing.contract_term_years_by_round.AR5` | `15` |  |
| `cfd_repricing.contract_term_years_by_round.AR6` | `15` |  |
| `cfd_repricing.contract_term_years_by_round.Investment Contract` | `15` |  |
| `cfd_repricing.contract_term_years_by_round.MGT_IC` | `15` |  |
| `cfd_repricing.contract_term_years_by_round.HPC` | `35` |  |
| `cfd_repricing.strike_basis` | `settled_bill_strike_table` | Re-price against the strike each contract settles at in the rebuilt bill (the published round strike, or the declared future strike for unassigned capacity), never a carrier-wide fallback. |
| `ldes_cap_and_floor` | (mapping) | LDES cap and floor: floor = required revenue at the case's floor rate, top-up = max(0, floor - market revenue). A policy line in every case, private included, recovered over final demand like a network charge. |
| `ldes_cap_and_floor.enabled` | `true` |  |
| `ldes_cap_and_floor.carriers` | `[LAES]` |  |
| `ldes_cap_and_floor.recovery_denominator` | `final_demand_mwh` |  |
| `ldes_cap_and_floor.floor_return` | (mapping) | Ofgem's scheme floors at a return "similar to the cost of debt for both debt and equity investors", so the floor re-annuitises the capital charge at a debt-like real rate: BBB- debt for a private project, two notches better (BBB+) under P1's guarantee, and the state's own rate (the case WACC, null) under ownership. Rates: financing workbook v2.0's real CPI cost of debt by rating (snapshot cost_of_debt_by_rating, Assumptions G120:G128). The floor at the technology WACC is reported as the upper sensitivity (ldes_floor_topup_at_wacc_gbp). |
| `ldes_cap_and_floor.floor_return.basis` | `cost_of_debt_like` |  |
| `ldes_cap_and_floor.floor_return.real_rate_by_case` | (mapping) |  |
| `ldes_cap_and_floor.floor_return.real_rate_by_case.private` | `0.05674` |  |
| `ldes_cap_and_floor.floor_return.real_rate_by_case.public_control` | `0.05086` |  |
| `ldes_cap_and_floor.floor_return.real_rate_by_case.public_ownership` | `null` |  |
| `capacity_market_sensitivity` | (mapping) | Sensitivity, reported not applied: the Capacity Market line less the financing saving of the plant the auction pays (new-build gas and batteries). |
| `capacity_market_sensitivity.enabled` | `true` |  |
| `sizewell_c_rab_public_rate` | (mapping) | Sizewell C RAB levy at the public rate (scripts/public_ownership/sizewell_c_rab.py). In every public case (never private or a comparator) the return on the privately financed capital is refinanced at the public rate the network account uses (its public_rate_basis, no construction premium). The existing state share of capital is kept: existing share = (state_equity_share x equity + NWF debt) / (equity + total debt) = (0.449 x 8.5 + 36.6) / (8.5 + 41.6) = 80.7%, so P1 (a 30% state share) moves nothing. Saving = private equity x its required return + private debt x its rate - private capital x public rate, over the whole return (capital x rab_real_rate). Private capital: 55.1% x 8.5 = GBP 4.68 bn of equity and the GBP 5.0 bn of commercial debt. Sources, NAO, Sizewell C, HC 33, nominal amounts: rab_real_rate          construction-period WACC agreed with investors, 6.73% CPIH-real (para 2.14, Appendix One para 17) state_equity_share     DESNZ's share of the equity (para 2.7) equity_gbp_bn          "up to GBP 8.5 billion" of equity (summary) nwf_debt_gbp_bn        the National Wealth Fund's "up to GBP 36.6 billion" of debt (para 2.8) total_debt_gbp_bn      NWF debt plus GBP 5 bn of commercial debt with a Bpifrance guarantee equity_return_nominal  post-tax nominal equity IRR at the lower regulatory threshold, 12.0-13.0% (para 2.23), midpoint; deflated at cpih_inflation to 10.29% real The debt rate is derived so the capital structure earns 6.73% real: (50.1 x 6.73% - 8.5 x 10.29%) / 41.6 = 6.00% real.  return_share 1.0 takes the whole levy as return on capital (during construction the allowance has no depreciation and its tax and other blocks are small; NAO Figure 8), an upper bound on the saving. 2030 only: a vintaged register (2050) books Sizewell C's RAB itself and is never adjusted. |
| `sizewell_c_rab_public_rate.enabled` | `true` |  |
| `sizewell_c_rab_public_rate.rab_real_rate` | `0.0673` |  |
| `sizewell_c_rab_public_rate.return_share` | `1.0` |  |
| `sizewell_c_rab_public_rate.state_equity_share` | `0.449` |  |
| `sizewell_c_rab_public_rate.equity_gbp_bn` | `8.5` |  |
| `sizewell_c_rab_public_rate.nwf_debt_gbp_bn` | `36.6` |  |
| `sizewell_c_rab_public_rate.total_debt_gbp_bn` | `41.6` |  |
| `sizewell_c_rab_public_rate.equity_return_nominal` | `0.125` |  |
| `sizewell_c_rab_public_rate.cpih_inflation` | `0.02` |  |
| `label_corrections` | (mapping) | Label corrections applied in the register; the network's label is kept in network_support_type. Tidal stream holds a CfD in every GB project but the network labels it merchant. Gas and battery shortfalls belong to the Capacity Market, so their saving is classed to that route, not to the wholesale or CfD line. Solar CfD labels come from the rebuilt fleet's vintage schedule and are not overridden. |
| `label_corrections.support_type` | (mapping) |  |
| `label_corrections.support_type.marine` | `CfD` |  |
| `label_corrections.route_classification` | (mapping) |  |
| `label_corrections.route_classification.CCGT` | `capacity_market` | new gas is paid by the Capacity Market or a strategic reserve |
| `label_corrections.route_classification.Battery` | `capacity_market` |  |
| `label_corrections.route_classification.battery` | `capacity_market` |  |
| `label_corrections.route_classification.CCS Biomass` | `dispatchable_cfd` | BECCS sits on a dispatchable CfD (the CCS levy line), never merchant |
| `label_corrections.solar_cfd_share` | `network_labels` |  |
