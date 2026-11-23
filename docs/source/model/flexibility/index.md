# Flexibility

Everything that moves electricity in time: storage that charges and discharges, the hydrogen system that turns
electricity into hydrogen and back, and consumers whose use can shift. All three are dispatched by the same optimisation
as generation, so they respond to prices and network limits within each solve window.

| Page | What it covers | PyPSA components | Rules |
|---|---|---|---|
| {doc}`storage` | grid and domestic batteries, pumped hydro, long-duration storage | `StorageUnit` | `add_storage_to_network` |
| {doc}`hydrogen` | electrolysers, hydrogen storage and demand, hydrogen turbines | `Bus`, `Link`, `Store`, `Load` | `add_hydrogen_system` |
| {doc}`consumer` | smart-charged and vehicle-to-grid EVs, heat pumps, heat networks, storage heaters, demand response, direct air capture | `StorageUnit` on the demand buses | `finalize_demand` (built with demand) |

How they differ:

- **Storage** is an asset in its own right: it charges and discharges against prices, and is settled as storage.
- **Hydrogen** links the power system to a hydrogen bus. Electrolysers are a flexible load, the store carries hydrogen
  between seasons, and the turbines turn it back into electricity. Hydrogen demand outside the power sector has to be
  met.
- **Consumer flexibility** is demand that can move: it is carved from the demand it shifts, has to meet that demand
  within each window, is held at its Stage 1 schedule in balancing, and counts as demand in every account.

Storage is added after thermal plant and marginal costs, and hydrogen after storage; consumer flexibility is added
with demand ({doc}`../index`).

```{toctree}
:maxdepth: 2

storage
hydrogen
consumer
```
