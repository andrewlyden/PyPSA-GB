## `public_ownership.sensitivities`

File: `config/defaults/base/public_ownership/sensitivities.yaml`. Keys below are relative to `public_ownership.sensitivities`.

| Key | Default | Description |
|---|---|---|
| `structural` | (list of 6 items) | Public-ownership sensitivities: reported as a range around each scenario, the scenario being the headline. Snakemake runs every one through the post-solve rules from one case list, reference_runs.expanded_cases(). Structural axes: each expands one at a time from every scenario where the lever bites (bounding_pair below is false, so there is no all-alternatives bounding case; `bounds` gives the best and worst cases). |
| `one_at_a_time` | (list of 12 items) | One lever at a time, no bounding pair. A bare name takes every other value of the lever; {name: [values]} takes those values only. Retired levers and values fail with a pointer (levers.RETIRED_LEVERS / RETIRED_VALUES). |
| `named` | (mapping) | Named multi-lever positions. "extends" inherits another named sensitivity's settings. |
| `named.conservative_stake` | (mapping) | Conservative readings of P1 and P2: the institution alone, nothing else changed. |
| `named.conservative_stake.scenario` | `public_control` |  |
| `named.conservative_stake.networks` | `private` |  |
| `named.conservative_stake.route_booking` | `sensitivity_only` |  |
| `named.conservative_stake.surplus_recycling` | `public_cfd_at_cost` |  |
| `named.conservative_stake.shortfall_funding` | `levy_on_eligible_supply` |  |
| `named.conservative_developer` | (mapping) |  |
| `named.conservative_developer.scenario` | `public_ownership` |  |
| `named.conservative_developer.cfd_strike_basis` | `financing_saving_only` |  |
| `named.conservative_developer.cfd_amortisation` | `contract_term` |  |
| `named.conservative_developer.route_booking` | `sensitivity_only` |  |
| `named.conservative_developer.surplus_recycling` | `public_cfd_at_cost` | The surplus recycled through the at-cost CfD and a shortfall levied on eligible supply (P2's base rebates gross). |
| `named.conservative_developer.shortfall_funding` | `levy_on_eligible_supply` |  |
| `named.legacy_acquired_contracts_cancelled_on_bills` | (mapping) | The acquired plant's CfD top-up and ROC income cancelled rather than honoured: the purchase price excludes those streams and the former owners are paid their value at their own WACC, on the bill or by the Exchequer. Tariff cases, like the legacy_acquired cases they vary. Full nationalisation is exempt (parse_sensitivities refuses it). |
| `named.legacy_acquired_contracts_cancelled_on_bills.scenario` | `public_ownership` |  |
| `named.legacy_acquired_contracts_cancelled_on_bills.legacy_fleet` | `acquire_valued_group` |  |
| `named.legacy_acquired_contracts_cancelled_on_bills.legacy_contracts_under_acquisition` | `cancelled_residual_on_bills` |  |
| `named.legacy_acquired_contracts_cancelled_on_exchequer` | (mapping) |  |
| `named.legacy_acquired_contracts_cancelled_on_exchequer.scenario` | `public_ownership` |  |
| `named.legacy_acquired_contracts_cancelled_on_exchequer.legacy_fleet` | `acquire_valued_group` |  |
| `named.legacy_acquired_contracts_cancelled_on_exchequer.legacy_contracts_under_acquisition` | `cancelled_residual_on_exchequer` |  |
| `named.legacy_and_thermal_acquired_contracts_cancelled_on_bills` | (mapping) |  |
| `named.legacy_and_thermal_acquired_contracts_cancelled_on_bills.scenario` | `public_ownership` |  |
| `named.legacy_and_thermal_acquired_contracts_cancelled_on_bills.legacy_fleet` | `acquire_all` |  |
| `named.legacy_and_thermal_acquired_contracts_cancelled_on_bills.legacy_contracts_under_acquisition` | `cancelled_residual_on_bills` |  |
| `named.legacy_and_thermal_acquired_contracts_cancelled_on_exchequer` | (mapping) |  |
| `named.legacy_and_thermal_acquired_contracts_cancelled_on_exchequer.scenario` | `public_ownership` |  |
| `named.legacy_and_thermal_acquired_contracts_cancelled_on_exchequer.legacy_fleet` | `acquire_all` |  |
| `named.legacy_and_thermal_acquired_contracts_cancelled_on_exchequer.legacy_contracts_under_acquisition` | `cancelled_residual_on_exchequer` |  |
| `named.fully_nationalised` | (mapping) | P3, financing workbook Sc3: the state buys the existing fleet, legacy plant at its acquisition price, and the acquisition tariff replaces the market bill. The wires stay private (buying them is the networks axis of P3's toggle grid). Hinkley Point C is bought as an operating plant at its CfD value; completing it at the public rate is the _hpc_completion case below. |
| `named.fully_nationalised.scenario` | `public_ownership` |  |
| `named.fully_nationalised.pre_2025_cfd_capacity` | `public_repriced` |  |
| `named.fully_nationalised.legacy_fleet` | `acquire_all` |  |
| `named.fully_nationalised.acquisition_transaction_costs` | `two_pct` |  |
| `named.fully_nationalised_market_value` | (mapping) | Purchase-price and term variants. |
| `named.fully_nationalised_market_value.extends` | `fully_nationalised` |  |
| `named.fully_nationalised_market_value.purchase_price_basis` | `market_value` |  |
| `named.fully_nationalised_book_value` | (mapping) | The financing workbook's other acquisition bases, on every acquired row: A1 net book value, A2 without CfD and RO support (the Northern Rock principle, a legal lower bound) and A3, RO plant at its no-support value (a policy case embedding a transfer from RO shareholders). |
| `named.fully_nationalised_book_value.extends` | `fully_nationalised` |  |
| `named.fully_nationalised_book_value.purchase_price_basis` | `book_value` |  |
| `named.fully_nationalised_no_support_value` | (mapping) |  |
| `named.fully_nationalised_no_support_value.extends` | `fully_nationalised` |  |
| `named.fully_nationalised_no_support_value.purchase_price_basis` | `no_support` |  |
| `named.fully_nationalised_ro_pass_through` | (mapping) |  |
| `named.fully_nationalised_ro_pass_through.extends` | `fully_nationalised` |  |
| `named.fully_nationalised_ro_pass_through.purchase_price_basis` | `ro_pass_through` |  |
| `named.fully_nationalised_thermal_at_zero` | (mapping) |  |
| `named.fully_nationalised_thermal_at_zero.extends` | `fully_nationalised` |  |
| `named.fully_nationalised_thermal_at_zero.thermal_basis` | `zero` |  |
| `named.fully_nationalised_market_value_price_path_cp30` | (mapping) | The legacy valuation on CP30's CP.32 price level, against the model's own 2030 price (the central). |
| `named.fully_nationalised_market_value_price_path_cp30.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_price_path_cp30.legacy_price_path` | `cp30_level` |  |
| `named.fully_nationalised_market_value_price_path_today` | (mapping) | The fleet valued as if today's forward prices lasted its whole life: each technology at its anchor for today (Ofgem's 2026 direct-fuel index on the 2030 capture ratios), flat. A higher purchase price. |
| `named.fully_nationalised_market_value_price_path_today.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_price_path_today.legacy_price_path` | `today_level` |  |
| `named.fully_nationalised_market_value_capacity_cost_based` | (mapping) | Capacity without an auction under P3. The tariff pays no Capacity Market agreement to plant the state owns, so the bill line does not move; the existing units are valued on cost-based contracts (going-forward fixed cost less their own margin, floored at zero) instead of their Capacity Market income, a lower purchase price, and only if the regime is announced before the purchase. |
| `named.fully_nationalised_market_value_capacity_cost_based.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_capacity_cost_based.capacity_procurement` | `cost_based_existing` |  |
| `named.fully_nationalised_market_value_offshore_deal_multiples` | (mapping) | Legacy offshore wind priced at observed UK operating offshore wind stake sales per MW (lever_parameters.legacy_offshore_transaction_multiple) instead of the model's cash-flow (owners') value of about 2.31 GBP m/MW; everything else still at owners' value. A higher price. |
| `named.fully_nationalised_market_value_offshore_deal_multiples.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_offshore_deal_multiples.legacy_offshore_price` | `transaction_multiple` |  |
| `named.fully_nationalised_term_15y` | (mapping) |  |
| `named.fully_nationalised_term_15y.extends` | `fully_nationalised` |  |
| `named.fully_nationalised_term_15y.acquisition_term_years` | `15` |  |
| `named.gbe_value_to_bills` | (mapping) | Coherence rules for every proposition bundle, scenario or sensitivity: (i)   ownership implies contract form: 100% state ownership takes strike_at_cost amortised over the asset's economic life (the state holds the asset for its life); a minority stake keeps the auction-cleared strike over the contract term (there is a commercial counterparty); (ii)  whoever takes the surplus carries the shortfall: retained_by_treasury pairs with shortfall_exchequer, public_cfd_at_cost with levy_on_eligible_supply; (iii) everything the proposition owns is booked the same way: if the state owns the new-build gas and BECCS, the Capacity Market and CCS-levy savings are bill lines (routes_booked); (iv)  nothing retrospective in a headline: pre_2025_cfd_capacity stays private except under full nationalisation, where the state has bought the counterparty and re-pricing re-opens no contract. P1 is the public_control scenario. This twin sends the state's position to bills through the at-cost CfD instead of to the Exchequer; at a 30% stake that position can be a deficit, so it can raise the net bill. It prices the recycling decision; rule (ii) ties the shortfall to the surplus, so two levers move together. |
| `named.gbe_value_to_bills.scenario` | `public_control` |  |
| `named.gbe_value_to_bills.surplus_recycling` | `public_cfd_at_cost` |  |
| `named.gbe_value_to_bills.shortfall_funding` | `levy_on_eligible_supply` |  |
| `named.fully_nationalised_market_value_term_15y` | (mapping) | P2 is the public_ownership scenario; its twins are one-lever sensitivities generated by the expansion, e.g. wires acquired                    -&gt; public_ownership@networks_acquired PWLB margins, gilt flat           -&gt; public_ownership@public_rate_pwlb_\* / _gilt_flat sovereign yield B                 -&gt; public_ownership@yield_b socialised risk at other premia   -&gt; public_ownership@systematic_premium_\* P3 ("the public power system") is the market-value family above; its central case is fully_nationalised_market_value_central_dispatch below, and fully_nationalised_market_value is P3 with the market machinery kept. The CfD strike, amortisation and route levers have no effect once legacy_fleet is acquire_all: the state has bought the counterparty and the acquisition tariff replaces the recovery they describe, so P3's design choices are the purchase price and the term. market_value (what an owner would sell for) is P3's central price. It is cheaper than state_present_value because it discounts at the owner's higher WACC. A control premium over it is the lever generation_control_premium (a P3 toggle-grid axis). |
| `named.fully_nationalised_market_value_term_15y.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_term_15y.acquisition_term_years` | `15` |  |
| `named.fully_nationalised_market_value_term_30y` | (mapping) | The central repays the purchase asset by asset over each asset's remaining life (acquisition_term_years: remaining_life); this toggle uses one 30-year annuity for the whole purchase. The lifetime present value at the public rate is the same; the 2030s carry more of it centrally. |
| `named.fully_nationalised_market_value_term_30y.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_term_30y.acquisition_term_years` | `30` |  |
| `named.fully_nationalised_market_value_central_dispatch` | (mapping) | P3's central case: the public power system run by a nationalised operator dispatching centrally at resource cost (institution I3) on its own solved year. It differs from fully_nationalised_market_value (I1, market machinery kept) in the dispatch institution alone; the institution comparison takes that twin as its I1 and moves four measured lines. Joins `all` only when institution_comparison's scenarios are in run_scenarios. |
| `named.fully_nationalised_market_value_central_dispatch.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_central_dispatch.dispatch_institution` | `nationalised_dispatch` |  |
| `named.fully_nationalised_market_value_hpc_completion` | (mapping) | Hinkley Point C completed by the state (sunk cost to the vendors, remaining capex at the public cash rate, no construction premium) in place of its purchase as an operating plant at its CfD value. |
| `named.fully_nationalised_market_value_hpc_completion.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_hpc_completion.in_construction` | `public_completion` |  |
| `named.fully_nationalised_market_value_systematic_premium` | (mapping) | Scenario 3's public-risk and gilt-response sensitivities: gilts plus the workbook's systemic risk premium (v6.2: a flat 1.9%, off the bill), and gilts rising with debt to GDP (B). |
| `named.fully_nationalised_market_value_systematic_premium.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_systematic_premium.systematic_premium` | `social_flat_19` |  |
| `named.fully_nationalised_market_value_systematic_premium_1pc` | (mapping) | Scenario 1 carries none by design, so it has no such case. |
| `named.fully_nationalised_market_value_systematic_premium_1pc.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_systematic_premium_1pc.systematic_premium` | `social_flat_10` |  |
| `named.systematic_premium_1pc` | (mapping) |  |
| `named.systematic_premium_1pc.scenario` | `public_ownership` |  |
| `named.systematic_premium_1pc.systematic_premium` | `social_flat_10` |  |
| `named.systematic_premium_1pc9` | (mapping) |  |
| `named.systematic_premium_1pc9.scenario` | `public_ownership` |  |
| `named.systematic_premium_1pc9.systematic_premium` | `social_flat_19` |  |
| `named.fully_nationalised_market_value_yield_b` | (mapping) |  |
| `named.fully_nationalised_market_value_yield_b.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_yield_b.sovereign_yield` | `B` |  |
| `named.fully_nationalised_market_value_efw_acquired` | (mapping) | The purchase's scope: energy-from-waste is left with its owners centrally (waste infrastructure paid mostly by councils' gate fees); this buys it with the rest. And Hinkley Point C left out of the purchase, private on its own CfD (the central buys it operating; _hpc_completion has the state finish it). |
| `named.fully_nationalised_market_value_efw_acquired.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_efw_acquired.efw_acquisition` | `included` |  |
| `named.fully_nationalised_market_value_hpc_left_private` | (mapping) |  |
| `named.fully_nationalised_market_value_hpc_left_private.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_hpc_left_private.in_construction` | `left_private` |  |
| `named.fully_nationalised_market_value_prices_held_at_2030` | (mapping) | The existing fleet valued on the model's 2030 prices held flat, beside the anchored path through today and every solved year. |
| `named.fully_nationalised_market_value_prices_held_at_2030.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_prices_held_at_2030.legacy_price_path` | `model_2030` |  |
| `named.fully_nationalised_market_value_legacy_levies_to_taxation` | (mapping) | The two background levers on P3 (they expand one at a time for private, P1 and P2): domestic legacy levies to taxation, and BECCS's carbon-removal leg levied on bills. Each is compared with private at the same point. |
| `named.fully_nationalised_market_value_legacy_levies_to_taxation.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_legacy_levies_to_taxation.legacy_levy_funding` | `domestic_to_taxation` |  |
| `named.fully_nationalised_market_value_beccs_removal_on_bills` | (mapping) |  |
| `named.fully_nationalised_market_value_beccs_removal_on_bills.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_beccs_removal_on_bills.beccs_removal_funding` | `bill_levy` |  |
| `named.networks_acquired_rav_at_statute` | (mapping) | Network compensation at RAV set by statute (premium 0), beside the evidence-based central premium of 0.50, on P2 with the wires and on P3. Both report the break-even premium beside the premium paid. |
| `named.networks_acquired_rav_at_statute.scenario` | `public_ownership` |  |
| `named.networks_acquired_rav_at_statute.networks` | `acquired` |  |
| `named.networks_acquired_rav_at_statute.network_acquisition_premium` | `statutory_at_rav` |  |
| `named.fully_nationalised_market_value_rav_at_statute` | (mapping) |  |
| `named.fully_nationalised_market_value_rav_at_statute.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_rav_at_statute.networks` | `acquired` | P3's base keeps the wires private |
| `named.fully_nationalised_market_value_rav_at_statute.network_acquisition_premium` | `statutory_at_rav` |  |
| `named.strike_at_cost_over_life` | (mapping) | P1's "cut to cost": the state's share costed over the plant's life (the auction's basis), not the 20-year contract term. |
| `named.strike_at_cost_over_life.scenario` | `public_control` |  |
| `named.strike_at_cost_over_life.cfd_strike_basis` | `at_cost` |  |
| `named.strike_at_cost_over_life.cfd_amortisation` | `economic_life` |  |
| `named.stake_networks_acquired_rav_premium_25pct` | (mapping) | The wires bought at a 25% premium over RAV, on each scenario that can buy them (P1, P2, P3), beside the central 50% and the statutory 0%. |
| `named.stake_networks_acquired_rav_premium_25pct.scenario` | `public_control` |  |
| `named.stake_networks_acquired_rav_premium_25pct.networks` | `acquired` |  |
| `named.stake_networks_acquired_rav_premium_25pct.network_acquisition_premium` | `quarter` |  |
| `named.networks_acquired_rav_premium_25pct` | (mapping) |  |
| `named.networks_acquired_rav_premium_25pct.scenario` | `public_ownership` |  |
| `named.networks_acquired_rav_premium_25pct.networks` | `acquired` |  |
| `named.networks_acquired_rav_premium_25pct.network_acquisition_premium` | `quarter` |  |
| `named.fully_nationalised_market_value_rav_premium_25pct` | (mapping) |  |
| `named.fully_nationalised_market_value_rav_premium_25pct.extends` | `fully_nationalised_market_value` |  |
| `named.fully_nationalised_market_value_rav_premium_25pct.networks` | `acquired` |  |
| `named.fully_nationalised_market_value_rav_premium_25pct.network_acquisition_premium` | `quarter` |  |
| `named.gbe_announced` | (mapping) | Every lever value no other case uses gets one named case, on the parent where it bites, moving that one lever (plus the networks treatment it needs) off its parent's base. GB Energy as announced, a realism check on P2: P2's financing and contract levers, but the state holds only a 30% stake in the post-2025 offshore wind cohort (P1's stake mechanics and contract form) and 8 GW of local solar and onshore wind (P2's, at cost over life); everything else private (lever_parameters.gbe_announced). The partnership twin holds the 8 GW at 50% public. |
| `named.gbe_announced.scenario` | `public_ownership` |  |
| `named.gbe_announced.new_build_generation` | `gbe_announced` |  |
| `named.gbe_announced_partnership_half` | (mapping) |  |
| `named.gbe_announced_partnership_half.scenario` | `public_ownership` |  |
| `named.gbe_announced_partnership_half.new_build_generation` | `gbe_announced_partnership_half` |  |
| `named.developer_storage_private` | (mapping) | P2 with its new-build storage left private. |
| `named.developer_storage_private.scenario` | `public_ownership` |  |
| `named.developer_storage_private.new_build_storage` | `private` |  |
| `named.recycling_cost_recovery_contracting_on_developer` | (mapping) | P2's surplus under the four recycling rules no base or preset takes. The shortfall stays with the Exchequer (the base's), so each moves one lever. |
| `named.recycling_cost_recovery_contracting_on_developer.scenario` | `public_ownership` |  |
| `named.recycling_cost_recovery_contracting_on_developer.surplus_recycling` | `cost_recovery_contracting` |  |
| `named.recycling_rebate_per_mwh_on_developer` | (mapping) |  |
| `named.recycling_rebate_per_mwh_on_developer.scenario` | `public_ownership` |  |
| `named.recycling_rebate_per_mwh_on_developer.surplus_recycling` | `rebate_per_mwh_eligible_supply` |  |
| `named.recycling_offset_policy_levies_on_developer` | (mapping) |  |
| `named.recycling_offset_policy_levies_on_developer.scenario` | `public_ownership` |  |
| `named.recycling_offset_policy_levies_on_developer.surplus_recycling` | `offset_policy_levies` |  |
| `named.recycling_no_assumption_on_developer` | (mapping) |  |
| `named.recycling_no_assumption_on_developer.scenario` | `public_ownership` |  |
| `named.recycling_no_assumption_on_developer.surplus_recycling` | `no_assumption` |  |
| `named.networks_acquired_rav_premium_low` | (mapping) | P2 with the wires acquired: the low premium over RAV (20%, listed market-to-RAV), and the wires' return retained by the Treasury instead of reaching TNUoS and DUoS. |
| `named.networks_acquired_rav_premium_low.scenario` | `public_ownership` |  |
| `named.networks_acquired_rav_premium_low.networks` | `acquired` |  |
| `named.networks_acquired_rav_premium_low.network_acquisition_premium` | `low` |  |
| `named.networks_acquired_return_retained` | (mapping) |  |
| `named.networks_acquired_return_retained.scenario` | `public_ownership` |  |
| `named.networks_acquired_return_retained.networks` | `acquired` |  |
| `named.networks_acquired_return_retained.network_incidence` | `return_retained` |  |
| `presets` | (mapping) | Presets: "who pays". A preset moves only the incidence levers (levers.INCIDENCE_LEVERS), so it leaves its parent's resource saving unchanged to GBP 1, which the three-party ledger enforces. A lever with no meaning for a case (the acquisition in P1, the contract form under an acquisition, the network incidence while the wires are private) stays at the parent's value; a preset that then moves nothing makes no case. Every base is bill-payers first, so: as_designed            the base itself: no new case todays_rules           today's incidence, per case (levers_by_case): P1 retains the state's surplus at the Treasury; P2 recycles through the at-cost CfD and levies a shortfall on eligible supply. P3's is its base (the purchase on the tariff), so it makes no P3 case competitive_neutrality the state behaves like a private owner, a company paying dividends to its shareholder: surplus kept at the Treasury, tax equivalents, its own deficits levied on bills (the shortfall rule is inert under retention: nothing is routed) treasury_first         the state charges market prices (the auction strike) and keeps its financing margin and surpluses; on P3 it charges the private comparator's market bill and the Treasury keeps the whole margin over the tariff's cost (tariff_pricing: market_bill) purchase_on_taxation   P3 only: the acquisition annuity funded from general taxation |
| `presets.as_designed` | (mapping) |  |
| `presets.as_designed.description` | `The base of each scenario: bill-payers first.` |  |
| `presets.as_designed.applies_to` | (list of 4 items) |  |
| `presets.as_designed.levers` | `{}` |  |
| `presets.todays_rules` | (mapping) |  |
| `presets.todays_rules.description` | `Each scenario's earlier incidence (its old central case, with the w...` |  |
| `presets.todays_rules.applies_to` | (list of 4 items) |  |
| `presets.todays_rules.levers` | `{}` |  |
| `presets.todays_rules.levers_by_case` | (mapping) | A target without an entry takes its scenario's (state bids at cost: P2's). |
| `presets.todays_rules.levers_by_case.public_control` | (mapping) |  |
| `presets.todays_rules.levers_by_case.public_control.surplus_recycling` | `retained_by_treasury` |  |
| `presets.todays_rules.levers_by_case.public_control.shortfall_funding` | `exchequer` |  |
| `presets.todays_rules.levers_by_case.public_ownership` | (mapping) |  |
| `presets.todays_rules.levers_by_case.public_ownership.surplus_recycling` | `public_cfd_at_cost` |  |
| `presets.todays_rules.levers_by_case.public_ownership.shortfall_funding` | `levy_on_eligible_supply` |  |
| `presets.todays_rules.levers_by_case.public_ownership@fully_nationalised_market_value` | (mapping) |  |
| `presets.todays_rules.levers_by_case.public_ownership@fully_nationalised_market_value.acquisition_financing` | `tariff` |  |
| `presets.competitive_neutrality` | (mapping) |  |
| `presets.competitive_neutrality.description` | `The state behaves like a private owner: dividends to the Treasury, ...` |  |
| `presets.competitive_neutrality.applies_to` | (list of 4 items) |  |
| `presets.competitive_neutrality.levers` | (mapping) |  |
| `presets.competitive_neutrality.levers.surplus_recycling` | `retained_by_treasury` |  |
| `presets.competitive_neutrality.levers.shortfall_funding` | `exchequer` |  |
| `presets.competitive_neutrality.levers.tax_equivalent` | `on` |  |
| `presets.competitive_neutrality.levers.state_deficit_funding` | `bill_levy` |  |
| `presets.treasury_first` | (mapping) |  |
| `presets.treasury_first.description` | `The state charges market prices and the Treasury keeps the margin.` |  |
| `presets.treasury_first.applies_to` | (list of 4 items) |  |
| `presets.treasury_first.levers` | (mapping) |  |
| `presets.treasury_first.levers.cfd_strike_basis` | `auction_clearing` |  |
| `presets.treasury_first.levers.surplus_recycling` | `retained_by_treasury` |  |
| `presets.treasury_first.levers.shortfall_funding` | `exchequer` |  |
| `presets.treasury_first.levers.route_booking` | `sensitivity_only` |  |
| `presets.treasury_first.levers.network_incidence` | `return_retained` |  |
| `presets.treasury_first.levers.tariff_pricing` | `market_bill` |  |
| `presets.purchase_on_taxation` | (mapping) |  |
| `presets.purchase_on_taxation.description` | `P3's acquisition annuity funded from general taxation instead of th...` |  |
| `presets.purchase_on_taxation.applies_to` | `[public_ownership@fully_nationalised_market_value]` |  |
| `presets.purchase_on_taxation.levers` | (mapping) |  |
| `presets.purchase_on_taxation.levers.acquisition_financing` | `exchequer` |  |
| `bounds` | (mapping) | Best and worst cases, from assumption ranges only, under each base's own incidence. Each assumption lever takes its most (best) or least (worst) favourable value for the scenario's household total, where it bites; one value per lever, so no mutually exclusive settings (the lending margins are values of one lever: the worst case takes the dearest, PWLB standard, +100 bp on the cash rate). The social premium per unit of beta is none (Green Book) at best and the market ERP at worst; the ERP convention's worst is Fixed TMR (lower private rates, a smaller saving); the purchase's worst is the state's own value, the dearest basis. Yield B is in the worst case for P1 and P2, not P3, whose lending-margin stress already covers the rate. Hinkley Point C bought as an operating plant is the favourable option, so the best case keeps the central. A bound that moves a comparator (all_cases) lever gets a private comparator case at the same point (the best case moves the LDES floor basis, the worst the private WACC basis), which the ledger measures against. The contested gilt-stock externality and the marginal cost of public funds enter the worst cases' household totals only (the Exchequer ledger). Ranges: financing workbook v2.0. |
| `bounds.applies_to` | (list of 3 items) |  |
| `bounds.rule` | `most and least favourable coherent assumption values, one per lever...` |  |
| `bounds.best` | (mapping) |  |
| `bounds.best.state_overhead` | `low` |  |
| `bounds.best.systematic_premium` | `none` |  |
| `bounds.best.interconnector_price` | `low` |  |
| `bounds.best.ldes_floor_basis` | `technology_wacc` |  |
| `bounds.best.acquisition_transaction_costs` | `none` |  |
| `bounds.worst` | (mapping) |  |
| `bounds.worst.state_overhead` | `high` |  |
| `bounds.worst.systematic_premium` | `market_erp` |  |
| `bounds.worst.sovereign_yield` | `B` |  |
| `bounds.worst.public_rate_basis` | `pwlb_standard` |  |
| `bounds.worst.private_wacc_basis` | `fixed_tmr_shielded` |  |
| `bounds.worst.purchase_price_basis` | `state_present_value` |  |
| `bounds.worst.in_construction` | `public_completion` |  |
| `bounds.worst.interconnector_price` | `high` |  |
| `bounds.worst.acquisition_transaction_costs` | `two_pct` |  |
| `bounds.worst.state_capex_overrun` | `plus_15pct` | Longer lives raise the purchase price (+2.9 bn in 2030) but lengthen the repayments (-6 a household in 2030) and raise the purchase's lifetime break-even by more than the price (+5.7 bn), so it is not a downside for S3; it stays a bound-grid toggle (lives-extended). S1 holds a stake and builds nothing). Each was only a toggle before, so the worst case left them out. |
| `bounds.worst.state_om_premium` | `plus_10pct` |  |
| `bounds.exclude` | (mapping) |  |
| `bounds.exclude.public_ownership@fully_nationalised_market_value` | `[sovereign_yield]` |  |
| `bounds.exclude.public_control` | `[systematic_premium, sovereign_yield]` | Scenarios 1 and 2: no gilt-response bound. Scenario 1 carries none by design. |
| `bounds.exclude.public_ownership` | `[sovereign_yield]` |  |
| `toggle_grids` | (mapping) | Toggle grids for the explorer: every combination of every axis on each scenario, under the base incidence. Axis kinds (levers.parse_grid_axes): {token, lever, value}            a switch, off (0) or on (1); {selector, lever, options}       one lever over several values, option token -&gt; value; the parent's own value is the base option; {token, lever, value, when}      a switch that exists only under the named selector option(s), null in the position vector elsewhere, so it never repeats a case. A case id joins the switches that are on and &lt;selector&gt;-&lt;option&gt; for each selector off its base, in declared order (public_control@grid_stake-15_networks-acquired_strike-auction). Each case carries its position vector in the case catalogue ({axis: 0 \| 1 \| null \| option}); a position another case already accounts for (a one-at-a-time, named or preset case at the same settings) is carried by that case as a grid membership, and each parent carries its own all-base position, so every position maps to one case. P1   2 switches x networks {private, acquired} x pre-2025 x strike 3 = 4 x 2 x 2 x 3 = 48 (no rate switch: the workbook's Sc2 rate does not use the lending margin) P2   5 switches x networks 2 x pre-2025 = 32 x 2 x 2 = 128 P3   5 switches x networks {private, acquired (x premium high)} = 32 x 3 = 96 272 positions, less the three parents. P2 with state bids at cost has no grid (it keeps its presets). P3 has no CfD axes: its tariff costs every owned asset at cost. max_cases is the warning threshold, not a cap. |
| `toggle_grids.enabled` | `true` |  |
| `toggle_grids.max_cases` | `6000` |  |
| `toggle_grids.grids` | (mapping) |  |
| `toggle_grids.grids.public_control` | (list of 5 entries) |  |
| `toggle_grids.grids.public_ownership` | (list of 7 entries) |  |
| `toggle_grids.grids.public_ownership@fully_nationalised_market_value` | (list of 7 entries) | No strike axis: the state owns 100%, so there is no co-owner, and under bill-payers first every strike rule gives the same bill (whatever the strike leaves with the state is rebated gross). The P3 group tariffs the P3 grid (reference_runs.same_fleet_tariff_cases): no P3 axis changes which plant the state owns. |
| `toggle_grids.bound_grids` | (mapping) | Bound grids: every assumption the best and worst bounds move (sensitivities.bounds), as an exact grid on each scenario, so the explorer can switch them one by one between the two bounds. A lever both bounds move (the overhead, the interconnector price) is a selector with three options. The private WACC basis and the LDES floor move private too, so each position is measured against private at the same point on them. A bound grid is named &lt;parent&gt;#bounds and its case ids are &lt;parent&gt;_bgrid_...; it is a separate exact space from the toggle grid above. P1  2 x 2 x 3 x 2               =    24 (no premium or margin axis: the Sc2 rate uses neither) P2  2 x 2 x 3 x 2 x 2 x 2 x 2   =   192 P3  2 x 2 x 3 x 3 x 3 x 2 x 2 x 2 x 2 x 2 = 3,456 (no yield path: the lending-margin stress covers its rate, sensitivities.bounds.exclude) |
| `toggle_grids.bound_grids.public_control` | (list of 3 entries) |  |
| `toggle_grids.bound_grids.public_ownership` | (list of 7 entries) |  |
| `toggle_grids.bound_grids.public_ownership@fully_nationalised_market_value` | (list of 10 entries) | Scenario 3's bound grid keeps its ten axes (3,456 positions; the two delivery switches would make it 13,824), so its all-worst corner is not its worst case: the worst case carries the delivery risks as well (finding 7). |
| `bounding_pair` | `false` | false: structural levers expand one at a time only, with no all-alternatives bounding case. |
