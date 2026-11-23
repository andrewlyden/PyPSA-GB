## `marginal_costs`

File: `config/defaults/base/marginal_costs.yaml`. Keys below are relative to `marginal_costs`.

Marginal costs of generators (GBP/MWh electrical unless stated). Historical scenarios (modelled_year &lt;= 2024) use built-in historical lookup tables; future scenarios use the settings below unless a scenario overrides them.

| Key | Default | Description |
|---|---|---|
| `nuclear_mc` | `10.0` | Marginal costs of generators (GBP/MWh electrical unless stated). Historical scenarios (modelled_year &lt;= 2024) use built-in historical lookup tables; future scenarios use the settings below unless a scenario overrides them. Nuclear marginal cost (GBP/MWh); overrides per-generator empirical values, since ELEXON nuclear bid prices reflect strategic behaviour, not short-run cost. |
| `fuel_prices` | (mapping) | Fallback fuel prices (GBP/MWh thermal) when neither historical tables nor FES prices are available. |
| `fuel_prices.gas` | `35.0` |  |
| `fuel_prices.coal` | `30.0` |  |
| `fuel_prices.oil` | `75.0` |  |
| `fuel_prices.biomass` | `45.0` |  |
| `carbon_price` | `85.0` | Fallback carbon price (GBP/tCO2). |
| `use_historical_data` | `true` | Use built-in historical lookup tables for modelled_year &lt;= 2024 |
| `use_fes_prices` | `true` | Use FES-extracted prices for future scenarios (&gt;2024) |
| `future_costs` | (mapping) | Future commodity-price cases (GBP real 2024 modelling assumptions). Total carbon cost is UK ETS + Carbon Price Support. |
| `future_costs.enabled` | `true` |  |
| `future_costs.case` | `central` | low \| central \| high \| neso_cp2030 |
| `future_costs.carbon_price_support_end` | `null` | Date from which Carbon Price Support no longer applies (null = it continues). A case whose modelled period starts on or after it keeps only its UK ETS price (apply_marginal_costs._apply_carbon_price_support_end). null matches CP30's own SRMCs (CP.25), which carry the full carbon price including CPS. Option: "2028-04-01", HM Treasury's end of CPS for power (written statement HLWS1528), about -6 GBP/MWh on a CCGT. |
| `future_costs.gas_emission_factor_basis` | `gross_cv` | Calorific basis of the gas emission factor: gross_cv (0.183 t/MWh) \| net_cv (0.2016 t/MWh, CP30's; for net-CV efficiencies). |
| `future_costs.gas_price_override_pence_per_therm` | `null` | Gas price in p/therm (GCV) replacing the case's own; null = the case's. Code default: apply_marginal_costs.DEFAULT_GAS_PRICE_OVERRIDE_PENCE_PER_THERM. |
| `future_costs.price_year` | `2024` |  |
| `future_costs.currency` | `GBP` |  |
| `future_costs.efficiency_defaults` | (mapping) | Electrical efficiency by class (share 0-1). In an enabled future-cost case these take precedence over generic network placeholders; unit_efficiency_overrides take highest priority. |
| `future_costs.efficiency_defaults.CCGT` | `0.52` |  |
| `future_costs.efficiency_defaults.OCGT` | `0.38` |  |
| `future_costs.efficiency_defaults.CHP` | `0.42` |  |
| `future_costs.efficiency_defaults.gas_engine` | `0.4` |  |
| `future_costs.efficiency_defaults.coal` | `0.36` |  |
| `future_costs.efficiency_defaults.Coal` | `0.36` |  |
| `future_costs.efficiency_defaults.oil` | `0.3` |  |
| `future_costs.efficiency_defaults.Oil` | `0.3` |  |
| `future_costs.efficiency_defaults.biomass` | `0.35` |  |
| `future_costs.efficiency_defaults.Biomass` | `0.35` |  |
| `future_costs.efficiency_defaults.waste_to_energy` | `0.25` |  |
| `future_costs.efficiency_defaults.waste` | `0.25` |  |
| `future_costs.efficiency_defaults.CCS Gas` | `0.5` |  |
| `future_costs.efficiency_defaults.CCS Biomass` | `0.35` |  |
| `future_costs.efficiency_defaults.Hydrogen` | `0.55` |  |
| `future_costs.efficiency_defaults.H2` | `0.55` |  |
| `future_costs.unit_efficiency_overrides` | `{}` | Exact generator-name: efficiency overrides |
| `future_costs.cases` | (mapping) |  |
| `future_costs.cases.low` | (mapping) |  |
| `future_costs.cases.low.fuel_prices` | (mapping) | GBP/MWh thermal except hydrogen fuel input |
| `future_costs.cases.low.fuel_prices.gas` | `20.0` |  |
| `future_costs.cases.low.fuel_prices.coal` | `25.0` |  |
| `future_costs.cases.low.fuel_prices.oil` | `60.0` |  |
| `future_costs.cases.low.fuel_prices.biomass` | `35.0` |  |
| `future_costs.cases.low.fuel_prices.waste` | `10.0` |  |
| `future_costs.cases.low.fuel_prices.hydrogen` | `45.0` |  |
| `future_costs.cases.low.carbon` | (mapping) |  |
| `future_costs.cases.low.carbon.uk_ets` | `35.0` | GBP/tCO2 |
| `future_costs.cases.low.carbon.carbon_price_support` | `18.0` | GBP/tCO2 |
| `future_costs.cases.low.variable_om` | (mapping) | GBP/MWh electrical |
| `future_costs.cases.low.variable_om.CCGT` | `2.0` |  |
| `future_costs.cases.low.variable_om.OCGT` | `3.0` |  |
| `future_costs.cases.low.variable_om.gas_engine` | `3.0` |  |
| `future_costs.cases.low.variable_om.CCS Gas` | `12.0` |  |
| `future_costs.cases.low.variable_om.CCS Biomass` | `18.0` |  |
| `future_costs.cases.low.variable_om.Hydrogen` | `5.0` |  |
| `future_costs.cases.central` | (mapping) |  |
| `future_costs.cases.central.fuel_prices` | (mapping) |  |
| `future_costs.cases.central.fuel_prices.gas` | `30.0` |  |
| `future_costs.cases.central.fuel_prices.coal` | `30.0` |  |
| `future_costs.cases.central.fuel_prices.oil` | `75.0` |  |
| `future_costs.cases.central.fuel_prices.biomass` | `45.0` |  |
| `future_costs.cases.central.fuel_prices.waste` | `15.0` |  |
| `future_costs.cases.central.fuel_prices.hydrogen` | `70.0` |  |
| `future_costs.cases.central.carbon` | (mapping) |  |
| `future_costs.cases.central.carbon.uk_ets` | `55.0` |  |
| `future_costs.cases.central.carbon.carbon_price_support` | `18.0` |  |
| `future_costs.cases.central.variable_om` | (mapping) |  |
| `future_costs.cases.central.variable_om.CCGT` | `2.0` |  |
| `future_costs.cases.central.variable_om.OCGT` | `3.0` |  |
| `future_costs.cases.central.variable_om.gas_engine` | `3.0` |  |
| `future_costs.cases.central.variable_om.CCS Gas` | `15.0` |  |
| `future_costs.cases.central.variable_om.CCS Biomass` | `20.0` |  |
| `future_costs.cases.central.variable_om.Hydrogen` | `6.0` |  |
| `future_costs.cases.high` | (mapping) |  |
| `future_costs.cases.high.fuel_prices` | (mapping) |  |
| `future_costs.cases.high.fuel_prices.gas` | `45.0` |  |
| `future_costs.cases.high.fuel_prices.coal` | `40.0` |  |
| `future_costs.cases.high.fuel_prices.oil` | `100.0` |  |
| `future_costs.cases.high.fuel_prices.biomass` | `55.0` |  |
| `future_costs.cases.high.fuel_prices.waste` | `25.0` |  |
| `future_costs.cases.high.fuel_prices.hydrogen` | `100.0` |  |
| `future_costs.cases.high.carbon` | (mapping) |  |
| `future_costs.cases.high.carbon.uk_ets` | `80.0` |  |
| `future_costs.cases.high.carbon.carbon_price_support` | `18.0` |  |
| `future_costs.cases.high.variable_om` | (mapping) |  |
| `future_costs.cases.high.variable_om.CCGT` | `2.0` |  |
| `future_costs.cases.high.variable_om.OCGT` | `3.0` |  |
| `future_costs.cases.high.variable_om.gas_engine` | `3.0` |  |
| `future_costs.cases.high.variable_om.CCS Gas` | `18.0` |  |
| `future_costs.cases.high.variable_om.CCS Biomass` | `24.0` |  |
| `future_costs.cases.high.variable_om.Hydrogen` | `8.0` |  |
| `future_costs.cases.neso_cp2030` | (mapping) | Gas and carbon from NESO Clean Power 2030 Annex 4 (file written by scripts/market/fetch_neso_cp2030_annex4_costs.py); other fuels and VOM from fallback_case. |
| `future_costs.cases.neso_cp2030.source_file` | `resources/market/neso_clean_power_2030_annex4_costs.json` |  |
| `future_costs.cases.neso_cp2030.fallback_case` | `central` |  |
| `carrier_correction_factors` | (mapping) | Carrier correction factors: multiply formula marginal costs by factors derived from historical ELEXON calibration (e.g. CCGT about 0.62). File written by scripts/generators/export_correction_factors.py. |
| `carrier_correction_factors.enabled` | `false` | Off: historical correction factors are not extrapolated to future cases. |
| `carrier_correction_factors.file` | `data/market/historical_carrier_correction_factors.csv` |  |
| `carrier_correction_factors.source_year` | `closest` | "closest" \| "median" \| specific year (e.g. 2023) |
| `carrier_correction_factors.fallback` | `1.0` | Factor when carrier not in file |
| `carrier_correction_factors.allow_future_historical_factors` | `false` | true applies historical ELEXON factors to future cases too (a sensitivity only). |
| `empirical_calibration` | (mapping) | Empirical calibration: per-generator thermal and renewable marginal costs from ELEXON data; carrier correction factors are derived inline. |
| `empirical_calibration.enabled` | `false` |  |
| `empirical_calibration.window_days` | `7` | Rolling window half-width for thermal calibration |
| `empirical_calibration.carrier_mc_adjustments` | (mapping) | Additive marginal-cost adjustments per carrier (GBP/MWh), applied after empirical calibration. Widens the coal-gas gap the switch-on method compresses: at about GBP 25/tCO2 the carbon cost of coal (341 kgCO2/MWh_th at 0.36 efficiency) and CCGT (183 kgCO2/MWh_th at 0.49) differs materially, but calibration gives only about GBP 1 because both turn on at similar prices (2020 data). |
| `empirical_calibration.carrier_mc_adjustments.coal` | `15.0` | +GBP 15/MWh carbon cost premium |
| `byproduct_adjustments` | (mapping) | By-product adjustments for carriers whose fuel is paid for by a by-product (empty for most scenarios). Modes: gate_fee_covers_fuel_and_carbon, heat_credit. Energy from waste is paid 92-110 GBP a tonne to take residual waste (letsrecycle.com EfW gate fee index, 2025) and must burn it under disposal contracts; the UK ETS charges only its fossil CO2 from 2028, which the gate fee covers, so its electricity costs VOM only. Gas CHP sells heat: DUKES 2026 chapter 7 gives 79% of 2025 CHP output as qualifying, two thirds of it heat, so about 1.1 MWh of heat per MWh of electricity; the credit is that heat at the cost of an 85% efficient gas boiler, fuel plus carbon. |
| `byproduct_adjustments.waste_to_energy` | (mapping) |  |
| `byproduct_adjustments.waste_to_energy.mode` | `gate_fee_covers_fuel_and_carbon` |  |
| `byproduct_adjustments.waste_to_energy.source` | `https://www.letsrecycle.com/prices/efw-landfill-rdf/efw-landfill-rd...` |  |
| `byproduct_adjustments.CHP` | (mapping) |  |
| `byproduct_adjustments.CHP.mode` | `heat_credit` |  |
| `byproduct_adjustments.CHP.heat_fuel` | `gas` |  |
| `byproduct_adjustments.CHP.heat_to_power` | `1.1` |  |
| `byproduct_adjustments.CHP.boiler_efficiency` | `0.85` |  |
| `byproduct_adjustments.CHP.source` | `https://assets.publishing.service.gov.uk/media/6a69fa1e1bcde4220570...` |  |
| `byproduct_adjustments.CHP.seasonal` | (mapping) | Seasonal heat credit: heat demand = base_load_share (process heat, an assumption, share 0-1) + the rest in proportion to monthly heating degree days (Jan-Dec, 2010-24 ERA5 means), normalised to an annual mean of 1, so annual averages are unchanged. apply_to_min_output also scales the must-run floor (p_min_pu) by it. |
| `byproduct_adjustments.CHP.seasonal.enabled` | `true` |  |
| `byproduct_adjustments.CHP.seasonal.base_load_share` | `0.5` |  |
| `byproduct_adjustments.CHP.seasonal.apply_to_min_output` | `true` |  |
| `byproduct_adjustments.CHP.seasonal.monthly_heating_degree_days` | (list of 12 items) |  |
| `biofuel_subsidy` | (mapping) | Biofuel ROC deduction: dedicated biomass, biogas, landfill gas and sewage gas generators earn ROCs per MWh, which lowers their effective marginal cost below the fuel-cost formula (biomass about GBP 70-80/MWh without it, when it never dispatches; with it, it runs as baseload, as in observed ESPENI dispatch of about 2,200 MW in 2020). Waste-to-energy is excluded (gate-fee economics, see byproduct_adjustments). |
| `biofuel_subsidy.enabled` | `false` |  |
| `biofuel_subsidy.require_support_label` | `false` | true: only generators explicitly labelled ROC take the deduction (CfD or merchant biomass is never treated as ROC on carrier alone). |
| `biofuel_subsidy.roc_labelled_follow_subsidies` | `true` | Units explicitly labelled ROC take the deduction whenever support.dispatch_costs.enabled is on, even with this block disabled: banding x buyout price, as ROC wind and solar forgo, floored at mc_floor. |
| `biofuel_subsidy.default_roc_bandings` | (mapping) | ROC/MWh by carrier, used when a generator's own ro_banding is missing or zero. |
| `biofuel_subsidy.default_roc_bandings.biomass` | `1.5` | Ofgem dedicated biomass band |
| `biofuel_subsidy.default_roc_bandings.advanced_biofuel` | `1.5` | Same band as dedicated biomass |
| `biofuel_subsidy.default_roc_bandings.biogas` | `0.5` | Ofgem standard band for AD biogas |
| `biofuel_subsidy.default_roc_bandings.landfill_gas` | `0.2` | Minimal ROC support (declining sector) |
| `biofuel_subsidy.default_roc_bandings.sewage_gas` | `0.5` | Ofgem standard band for sewage gas |
| `biofuel_subsidy.mc_floor` | `5.0` | Option: roc_buyout_price: 50.05 (GBP/ROC); omitted, support.dispatch_costs.roc_buyout_prices for the modelled year is used. Floor on the deducted marginal cost (GBP/MWh), so ROC income cannot make it negative and force 100% dispatch. |
