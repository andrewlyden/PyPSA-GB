# Publications

Where PyPSA-GB has been used: papers, reports and software whose own analysis runs on the model or its data. Each
was checked against its text, abstract or code (search of November 2026); works that only cite the model are not
listed. Every work that cites the PyPSA-GB paper (Lyden et al., 2024, *Energy Strategy Reviews*) is on
[its Google Scholar page](https://scholar.google.com/scholar?q=%22PyPSA-GB%3A+An+open-source+model+of+Great+Britain%27s+power+system+for+simulating+future+energy+scenarios%22),
under "Cited by". To add yours, open an issue or pull request on GitHub.

- **Kordilas, M., Mumovic, D., Schwartz, Y., Cooke, R., Mordak, S. and Chalabi, Z., 2026.** Temporal myopia in
  building Life Cycle Assessment? Granular versus coarse dynamics in climate change and grid decarbonisation.
  *Building and Environment*, 295, 114419. <https://doi.org/10.1016/j.buildenv.2026.114419>
  *Use:* future hourly UK electricity mixes to 2050, for the carbon of a building's electricity use.
- **McCallion, A., 2026.** Can a network congestion-optimised grid connection queue reduce balancing costs?
  Oxford Institute for Energy Studies, Paper EL 64. <https://hdl.handle.net/10419/339529>
  *Use:* the reduced GB network, substations and demand, solved as a linear optimal power flow.
- **Centre for Net Zero, 2026.** Powering the AI Economy. <https://www.centrefornetzero.org/papers/powering-ai>
  *Use:* data-centre flexibility in GB in 2030, built on PyPSA-GB
  (code: <https://github.com/centrefornetzero/pypsa-gb-data-centres>).
- **Mantel, N., 2026.** Beyond the Exogenous Mask: learning feedback, policy thresholds, and sensitive intervention
  points in energy system models. Research Square preprint. <https://doi.org/10.21203/rs.3.rs-10792382/v1>
  *Use:* demand, generation and capacity-factor profiles.
- **Malkawi, G. and Elsayed, A.A., 2026.** DSC-Grid: reproducibility software and derived data. Zenodo.
  <https://doi.org/10.5281/zenodo.22976151>
  *Use:* dispatch matrices and linear programmes extracted from PyPSA-GB.
- **Hodges, S., Bukhsh, W., Bell, K.R.W., Grothey, A., McKinnon, K.I.M. and Tian, S., 2025.** An AC/DC
  security-constrained optimal power flow for Great Britain HVDC integration. IEEE PES ISGT Europe 2025.
  <https://doi.org/10.1109/isgteurope64741.2025.11305365>
  *Use:* the reduced GB network as the test system for a security-constrained optimal power flow.
- **Dergunova, T. and Lyden, A., 2024.** Great Britain's hydrogen infrastructure development: investment
  priorities and locational flexibility. *Applied Energy*, 375, 124017.
  <https://doi.org/10.1016/j.apenergy.2024.124017>
  *Use:* hourly renewable output and other inputs for a GB hydrogen network model (PyPSA-GB-H2).
- **Desguers, T., Lyden, A. and Friedrich, D., 2024.** Integration of curtailed wind into flexible electrified
  heating networks with demand-side response and thermal storage: practicalities and need for market mechanisms.
  *Energy Conversion and Management*, 304, 118203. <https://doi.org/10.1016/j.enconman.2024.118203>
  *Use:* the GB power-system modelling behind its curtailed-wind analysis.
- **Lyden, A., Alene, S., Connor, P., Renaldi, R. and Watson, S., 2024.** Impact of locational pricing on the roll
  out of heat pumps in the UK. *Energy Policy*, 187, 114043. <https://doi.org/10.1016/j.enpol.2024.114043>
  *Use:* unified and locational wholesale prices for a future GB power system.
- **Lyden, A., Sun, W., Friedrich, D. and Harrison, G., 2023.** Electricity system security of supply in Scotland.
  ClimateXChange report for the Scottish Government. <https://doi.org/10.7488/era/3737>
  *Use:* FES scenarios and power flows for Scotland's security of supply.

## Keeping this list current

`scripts/utilities/find_publications.py` queries OpenAlex (works citing the PyPSA-GB paper, and works mentioning
"PyPSA-GB") and Semantic Scholar, and lists every hit not yet on this page. A work goes on the page once its text,
abstract or code shows that it uses the model.
