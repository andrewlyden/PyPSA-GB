# Data licences and attribution

PyPSA-GB's code is released under the MIT licence (`LICENSE` at the repository root). The input data under `data/`
comes from third parties under their own terms. This page lists those terms and the attribution each publisher asks
for. If you publish results or redistribute a copy of the repository, keep these attributions.

## Summary

| Source | Publisher | Licence | In the repository? |
|---|---|---|---|
| BMRS data (bid-offer data, physical notifications, market index prices, balancing services) | Elexon | BMRS Open Data Licence | Yes (caches and extracts under `data/market/`) |
| NESO data portal datasets (FES building blocks, GSP information, TEC register, Capacity Market register, balancing costs, constraint limits) | NESO | NESO Open Data Licence, checked per dataset | Yes |
| NESO website documents (data workbooks, ETYS appendices, reports) | NESO | NESO website terms (personal copies) | See [below](#neso-website-documents) |
| Ofgem publications (price cap models and annexes, RO data) | Ofgem | Open Government Licence v3.0 | Yes |
| Ofgem RO certificate register | Ofgem | Open Government Licence v3.0 | Yes, reduced to the columns the model reads |
| LCCC CfD data | Low Carbon Contracts Company | Open Government Licence v3.0 | Yes |
| DUKES | DESNZ | Open Government Licence v3.0 | Yes |
| REPD | DESNZ | Open Government Licence v3.0 | Yes |
| ESPENI demand | Wilson et al. (Zenodo) | CC BY-NC 4.0 | Yes |
| ERA5 weather cutouts | Copernicus Climate Change Service (ECMWF) | Licence to use Copernicus products | No: downloaded (Zenodo or the Climate Data Store) |

## Elexon BMRS

Elexon's Balancing Mechanism Reporting Service data is used under the BMRS Open Data Licence. Attribution:

> Contains BMRS data © Elexon Limited copyright and database right [year].

Replace `[year]` with the year of the data you use.

## NESO data portal

Datasets downloaded from the NESO data portal are used under the NESO Open Data Licence. The licence is stated on
each dataset's page, and each dataset the model reads was checked individually. Attribution:

> Supported by National Energy SO Open Data.

The NESO Open Data Licence excludes personal data. For that reason the release copy of the Capacity Market register
(`data/market/capacity_market/cm_register_cmu_20260923.csv`) leaves out the trading contacts' telephone and email
columns; no script reads them.

## NESO website documents

Documents NESO publishes on its website rather than on the data portal (for example data workbooks and ETYS
appendices) fall under NESO's website terms, which allow personal copies only. They are not redistributed: the
workflow downloads them when a rule first needs them. `data/neso_documents.yaml` lists each one with its download
address, landing page and SHA-256 checksum; rule `retrieve_neso_document` fetches the file to its path and refuses
it if the checksum differs. If NESO moves a document, download it by hand from the landing page and save it at the
listed path. Fetched this way: the Clean Power 2030 data workbook, the FES data workbooks for 2020 to 2024, the ETYS
Appendix B workbooks for 2022 to 2024 and the regional breakdown of FES 2024 data.

Two NESO documents are still shipped, because NESO no longer publishes the exact files the model was built on and a
download could not reproduce the results:

- `data/FES/Future Energy Scenarios 2025 Data Workbook V005_0.xlsx`: NESO's link now serves version V006.
- `data/network/ETYS/Regional breakdown of FES23 data (ETYS 2023 Appendix E).xlsb`: NESO now publishes it only
  inside zip files, none byte-identical to this one. It is read for ETYS 2023 network builds and for the GSP weights of
  historical demand.

## Ofgem and LCCC

Ofgem publications (the default tariff cap model and its annexes, Renewables Obligation data) and Low Carbon
Contracts Company data (CfD contracts and generation) are used under the Open Government Licence v3.0. Attribution:

> Contains public sector information licensed under the Open Government Licence v3.0.

The Ofgem RO certificate register export (`data/market/ofgem_renewables_obligation/certificates_ro_current.csv`)
is shipped reduced to the columns the model reads (accreditation reference, station name, scheme, country,
technology, output period, certificates and certificate numbers, MWh per certificate, compliance period) and to Great
Britain's schemes (RO and ROS). Generator owners' names are left out.

## DUKES and REPD

The Digest of UK Energy Statistics (Table 5.11, power stations) and the Renewable Energy Planning Database are
published by the Department for Energy Security and Net Zero under the Open Government Licence v3.0. Use the OGL
attribution above.

## ESPENI

The ESPENI half-hourly electricity dataset (Wilson et al., Zenodo concept DOI
[10.5281/zenodo.3884858](https://doi.org/10.5281/zenodo.3884858)) is licensed CC BY-NC 4.0 (checked on its Zenodo
record, version 8.0.8). The licence allows non-commercial use with attribution; cite the dataset as its record asks.
Commercial users must obtain ESPENI under other terms or replace `data/demand/espeni.csv`.

## ERA5 weather cutouts

Weather cutouts are built from ERA5 reanalysis data (Copernicus Climate Change Service). ERA5 is available under the
licence to use Copernicus products, which asks users to acknowledge the source, for example:

> Generated using Copernicus Climate Change Service information [year].

Cutouts are not in the repository. Pre-built cutouts for 2010-2024 are downloaded from the PyPSA-GB Zenodo record
([10.5281/zenodo.18325225](https://zenodo.org/records/18325225)); other years are built from the Climate Data Store
with your own credentials.

## Other sources

The sources table on {doc}`/reference/data/data_sources` lists further inputs (the Reduced network, emissions factors,
technical parameters from the literature) and their licences.
