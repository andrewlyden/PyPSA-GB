# FES scenarios (GSP blocks)

A future scenario on NESO's Future Energy Scenarios (`targets.source: FES`, `method: gsp_block`), outside the long-range
studies, takes its annual demand from the FES workbook, its regional split from the FES building blocks and its hourly
shape from a historical year.

## Method and assumptions

### Annual energy

The national total is FES consumer demand from the edition's workbook, sheet ED1, for the pathway and
`modelled_year` (`get_ed1_consumer_demand_gwh` in `scripts/demand/load.py`). It is the ED1 consumer total where the
sheet has one; otherwise the sum of four rows:

- residential;
- industrial, excluding electrolysis (or the industrial row less its networked-electrolysis deduction);
- commercial;
- direct transmission customers.

It is gross consumer demand: embedded generation is not deducted, because the model builds it as generators, and
transmission and distribution losses are not included. Electrolysis is excluded; it is built as its own load in
{doc}`../flexibility/hydrogen`.

### Regional split

The building block Dem_BB003 gives demand by GSP. Its total is lower than consumer demand, because it is demand seen at
the GSP, net of distributed generation, so it is used only for shares: each GSP's Dem_BB003 value is scaled by
(ED1 consumer total / Dem_BB003 total). The GSPs are then placed on the network ({doc}`placement`; `pro_rata` bus
capacity rule).

### Hourly shape

One national shape is scaled to each GSP's annual energy, so every GSP takes the same shape. `demand.timeseries`
chooses it:

| `demand.timeseries` | Shape |
|---|---|
| `ESPENI` (default) | the `TOTAL_ESPENI` series of `demand_year` ({doc}`historical`) |
| `eload` | the eLOAD model's hourly UK profile, 2010 or 2050 |
| `desstinee` | the DESSTINEE model's hourly UK profile, 2010 or 2050 |

For eLOAD and DESSTINEE (`data/demand/egy_7649_mmc1.xlsx`), the scenario's optional top-level `profile_year` picks the
profile; without it the 2050 profile is used for `modelled_year` 2040 and later, and the 2010 profile otherwise.

The source year is placed on the modelled year by day of year and time of day, or by month, day and time when exactly
one of the two is a leap year ({doc}`index`). The shape is normalised to sum to one over the year and multiplied by
each GSP's annual energy, so energy is exact.

## Data

| Input | File | Used for |
|---|---|---|
| FES data workbook, sheet ED1 | `data/FES/` (downloaded by rule `retrieve_neso_document`) | national consumer demand |
| FES building blocks | `resources/FES/FES_{edition}_data.csv` | Dem_BB003 GSP shares |
| ESPENI | `data/demand/espeni.csv` | default shape |
| eLOAD / DESSTINEE | `data/demand/egy_7649_mmc1.xlsx` | optional shapes |

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `targets` | (scenario) | `{source: FES, edition, pathway}`: the workbook and pathway |
| `demand_year` | (scenario) | ESPENI year of the shape |
| `demand.timeseries` | `ESPENI` | `ESPENI`, `eload` or `desstinee` |
| `profile_year` | (by year) | eLOAD or DESSTINEE profile: 2010 or 2050 |

## Validation

- The build logs the ED1 total, the Dem_BB003 total and the scale factor between them.
- Every placement step checks that total demand is conserved (`assert_demand_conserved` in `load.py`).

## Limitations

- One national shape at every GSP, and the shape of a historical year: electrified heat and transport do not change
  when in the day or year demand falls. The CP30 and FES long-range builds construct these components explicitly
  ({doc}`cp30`, {doc}`fes_long_range`).
- Day-of-year mapping does not keep weekdays: a source Monday can land on any day of the week.
- Network losses are not in the demand.
