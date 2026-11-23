# ENTSO-E day-ahead prices, 2015 to 2024

Hourly day-ahead prices of the markets GB trades with, shipped so that a run needs no ENTSO-E API token.

**Source:** ENTSO-E Transparency Platform, Day-ahead Prices (Regulation (EU) No 543/2013, Article 12.1.D),
<https://transparency.entsoe.eu/>. Retrieved through the platform's API with `entsoe-py`
(`python scripts/interconnectors/fetch_entsoe_prices.py --year <year> --archive`, 18 Nov 2026).

**Attribution:** data from the ENTSO-E Transparency Platform. ENTSO-E's terms of use ask users to name the platform
as the source. Day-ahead prices are published there by the power exchanges (NEMOs) as primary owners of the data,
and may fall outside ENTSO-E's list of data open for free re-use; anyone republishing them should check the
platform's current terms.

## Files

`day_ahead_prices_<year>_eur_per_mwh.csv`: one row per UTC hour (`datetime`), one column per market, EUR/MWh as
published. Gaps are left empty; nothing is converted, capped or filled.

| Column | Bidding zone |
|---|---|
| France | FR |
| Belgium | BE |
| Netherlands | NL |
| Germany | DE_LU, which exists from October 2018: no column for 2015 to 2017, a quarter of 2018 |
| Norway | NO_2 |
| Denmark | DK_1 |
| Ireland | IE_SEM (92% of 2015's hours, over 99% of every later year) |

The rule fills a market's remaining gaps with its mean for the year, so Germany's January to September 2018 is that
mean. A cached file fetched from the API earlier can differ in a small share of hours (2015 to 2024 against this
model's earlier caches: 0 to 0.7% of hours; 2020 identical) because ENTSO-E revises published values.

## Use

Rule `fetch_entsoe_day_ahead_prices` reads the shipped year when the file exists and the API only for a year that is
not here. Either way it converts to GBP at `interconnectors.pricing.eur_to_gbp`, caps at `price_floor` and
`price_cap`, interpolates gaps of up to 6 hours and fills up to 24 hours forward and back, and writes
`resources/interconnectors/entsoe_day_ahead_prices_<year>.csv`. Years before 2015 come from
`data/interconnectors/entsoe_legacy/`.
