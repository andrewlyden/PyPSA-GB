# ENTSO-E pre-2015 day-ahead price inputs

The current ENTSO-E Transparency API does not serve day-ahead prices before
2015. For historical comparator scenarios, download the relevant year from
ENTSO-E's legacy archive and normalise the verified source exports into this
canonical file:

`day_ahead_prices_2013_eur_per_mwh.csv`

The CSV must have exactly 8,760 UTC hourly records for 2013, no gaps, and these
columns. Prices are in EUR/MWh before the model's configured conversion to
GBP/MWh.

```text
datetime,France,Belgium,Netherlands,Norway,Denmark,Ireland
2013-01-01T00:00:00Z,...,...,...,...,...,...
```

The source exports and any time-zone conversion used to create this canonical
file must be recorded alongside it (for example in
`day_ahead_prices_2013_source.md`). The workflow deliberately rejects missing
countries, gaps, duplicated timestamps, or an incorrect year; it will never
substitute a price profile from another year.

Use ENTSO-E's legacy archive for France, Belgium, Netherlands, Norway and
Denmark. Ireland had the pre-2018 Single Electricity Market rather than an
Irish day-ahead bidding-zone price: use the legacy SEM-O *Market Results
(System Load SMP and Shadow Price)* EA2 export, aggregate its two half-hourly
SMP values to an hourly EUR/MWh mean, and document this in the source note.

Official ENTSO-E archive: <https://transparency.entsoe.eu/content/static_content/Static-content/legacy-data/year-selection.html>

SEMO's historical-market guidance: <https://www.ucc.ie/en/eel/posts/how-to-download-legacy-electricity-market-data-from-sem-o.html>
