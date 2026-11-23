# Policy levies and support schemes

## What it does

These lines pay for closed and current support schemes, social programmes and industry levies. Two are modelled in
full (the Renewables Obligation and the LDES cap and floor); the rest apply published rates or forecasts.

| Line | Method | Source |
| --- | --- | --- |
| Renewables Obligation | ROC-eligible stations still accredited in the year (Ofgem register) × the model's balancing-stage load factors × banding × notional ROC value, over obligation-eligible supply | Ofgem RO register; `2030_ro_scheme_parameters.json` |
| Feed-in Tariffs | DESNZ projection of scheme cost under CPI indexation, over eligible supply, blended to the calendar year | DESNZ FiT indexation analytical annex (January 2026) |
| Sizewell C RAB | Ofgem's nuclear RAB allowance (£4.63/MWh supplied, Oct-Dec 2026) held flat in real terms | Ofgem Annex 4 v1.25, `3l nRAB` |
| Warm Home Discount | OBR forecast of levy receipts, half to electricity, per MWh of domestic supply | OBR, March 2026 EFO |
| LDES cap and floor | Top-up of new long-duration storage to its floor, less its modelled income, over the GSP charging volume | Ofgem LDES cap and floor; model revenue |
| Net Zero Teesside Power DPA | The East Coast Cluster full business case's operating budget over eligible supply, an upper bound | DESNZ ECC full business case (June 2025) |
| Power BECCS CfD (electricity) | Settled with the CfDs ({doc}`cfd`), booked on its own line | Model dispatch; DESNZ business model |
| Battery revenue top-up | Tops merchant batteries up to their required revenue after all modelled income; a declared line, no such scheme exists | Model revenue |
| Network Charging Compensation | Ofgem allowance (£1.04/MWh supplied) held flat in real terms | Ofgem Annex 4 v1.25, `3k NCC` |
| AAHEDC | NESO 2026/27 tariff held flat in real terms | NESO AAHEDC tariff |
| LCCC and ESC operating levies | DESNZ 2028/29 operational cost levy rates | DESNZ consultation (November 2025) |
| ECO | Removed from bills from April 2026 | Budget 2025 |

## Method and assumptions

### Renewables Obligation

The Renewables Obligation (RO) is the closed support scheme for renewable stations accredited between 2002 and 2017.
Each station earns Renewables Obligation Certificates (ROCs) for every MWh it generates, at a banding set by technology
(offshore wind 2 ROCs/MWh, onshore 0.9, solar 1.2, biomass 1.5, marine up to 5). Suppliers must present a set number
of ROCs per MWh they supply, or pay the buy-out price into a fund that is recycled to those who presented
certificates. Each station's support lasts 20 years from accreditation, and the scheme closes in March 2037. In
Ofgem's cap it is a policy cost (Annex 4, sheet `3b RO`).

The line (`scripts/consumer_bill/build_ro_eligible_fleet.py`, rule `build_future_ro_scheme_cost`, parameters in
`data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json`) values the certificates the year's fleet would
earn:

$$
\text{RO line} = \frac{\text{ROCs} \times \text{GB share} \times \text{notional ROC value}}{\text{obligation-eligible supply}} \times 3.1\ \text{MWh} \times (1 - \text{relief})
$$

- **Which stations still earn ROCs.** The Ofgem register of accredited stations, filtered to the share of the calendar
  year each station's 20-year accreditation still covers, less fuelled stations that have stopped issuing
  certificates. The model's networks carry the operating fleet from REPD, which has no accreditation dates, so the
  register decides which capacity earns ROCs; the model supplies how much it generates.
- **How much they generate.** The load factor of the model's own ROC-labelled units in the balancing stage. ROC plant
  bids minus its certificate value, so it is rarely curtailed. Biomass uses CP30's implied load factor instead of the
  model's.
- **Bandings** are observed per station from Ofgem's certificate register where the station has been issued
  certificates, otherwise the carrier default.
- **The value of a ROC** is the buy-out price plus the recycle value, £{{ ro_value }} in 2025 money: the
  {{ ro_base_year }} values carried forward with RPI, because the buy-out is RPI-indexed and therefore flat in real
  RPI terms.
- **Northern Ireland.** NI's ROCs trade in the same UK market and DESNZ sets one UK obligation, so NI stations stay in
  the certificate volume and GB suppliers carry the GB share ({{ ro_gb_share_pct }}%, DESNZ's 2026/27 obligation
  calculation).
- **Recovery** is over obligation-eligible metered supply. The RO is levied per MWh supplied, with no loss uplift.
- **Exchequer relief.** The Autumn Budget 2025 funds 75% of the domestic RO from taxation for three years from April
  2026; the cost returns to bills from April 2029. Calendar 2030 is the first full year of gross recovery, and the
  central case carries no relief (`ro_relief_case: relief_ends_march_2029`). If the relief continued, the line would be
  a quarter of its value.

Assumptions to keep in view: certificates are estimated from capacity and load factor, not from issuance; the buy-out
is RPI-indexed while the bill is in CPIH terms; and the British Industrial Competitiveness Scheme (from April 2027)
would exempt more industry from the RO, FiT and Capacity Market bases (a 5.5% sensitivity).

### LDES cap and floor

Ofgem's cap and floor for long-duration storage guarantees new projects a minimum revenue (the floor) and shares
revenue above a cap; the floor is paid by consumers through TNUoS. The line (`model_linked_rules.ldes_cap_and_floor`;
`scripts/public_ownership/public_account.py`) tops each new long-duration store up to its floor, the annual required
revenue with capital at the cost of debt for its rating, less every income the model gives it: wholesale and
balancing revenue, Capacity Market payments (its own award), ancillary services and sub-hourly trading. Existing
pumped hydro is not eligible.

The line assumes every new store in the pathway operates, and draws the floor, in the modelled year, so it is an upper
bound where projects come online later.

### The other lines

- **Feed-in Tariffs.** DESNZ's projected scheme cost (the scheme closed in 2019 and costs fall from 2030 as contracts
  expire), blended to the calendar year and spread over eligible supply.
- **Sizewell C RAB.** Ofgem's current rate held flat in real terms; the NAO expects the charge to keep rising until the
  station opens, so this is a floor.
- **Warm Home Discount.** OBR's forecast of levy receipts, half to electricity as in Ofgem's cap, over domestic supply
  only (held at Ofgem's 2025 figure).
- **Net Zero Teesside Power DPA.** The business case's operating budget for 2028/29 to 2030/31, a third a year, over
  eligible supply. It is an upper bound: the budget also covers Exchequer-funded transport and storage support, and
  no official figure for the DPA alone exists.
- **Battery revenue top-up.** Merchant batteries are topped up to break even after every income (wholesale,
  balancing, Capacity Market, ancillary services and sub-hourly trading). No such scheme exists; the line keeps the
  pathway's battery fleet whole.
- **Network Charging Compensation, AAHEDC, LCCC and ESC levies.** Current rates held flat in real terms; each matches
  Ofgem's cap. **ECO** is off bills from April 2026.

### How these lines are grouped

The RO, FiT, Warm Home Discount, Sizewell C RAB, Network Charging Compensation, AAHEDC and ECO are Ofgem policy costs
(Annex 4). The Power BECCS CfD, the DPA and the LCCC/ESC levies are grouped with the CfD as wholesale, the LDES cap and
floor with network costs (it is recovered through TNUoS), and the battery top-up with policy costs; see
{doc}`index`.

## Data

- Ofgem, Renewables Obligation register and certificate register; Annex 4 v1.25 (`3b RO`, `3k NCC`, `3l nRAB`).
- DESNZ, RO obligation calculation 2026/27; FiT indexation analytical annex (January 2026); ECC full business case
  (June 2025); LCCC and ESC operational cost levy consultation (November 2025).
- OBR, Economic and Fiscal Outlook, March 2026, table 3.20 (RO, WHD, Sizewell C receipts).
- NESO, AAHEDC tariff; Ofgem, LDES cap and floor regime.

## Configuration

`model_linked_rules.ldes_cap_and_floor` and `sourced_externals` (including `battery_revenue_top_up` and `remaining_ccs`, the DPA) of the assumptions file;
the RO through `ro_scheme_parameters` and `ro_relief_case` of the forecast case.

## Validation

- **RO against OBR.** OBR's forecast of RO receipts for 2029-30 and 2030-31 is converted to the calendar year in 2025
  money and compared with the model's UK certificate cost.
- **Against Ofgem's cap.** The FiT, Sizewell C, Network Charging Compensation, AAHEDC and LCCC/ESC lines are Ofgem's
  current allowances, carried forward; the RO is compared gross, as the cap is net of the temporary relief.
- **Sizewell C against OBR.** OBR's forecast of the levy for 2029-30 and 2030-31 is compared with the held-flat rate on
  the year's supply.

The worked example below is one run's RO by carrier:

```{include} _generated/ro_by_carrier.md
```
