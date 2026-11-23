## Top-level keys

File: `config/defaults/base.yaml`. Keys below are relative to the top level.

Base defaults: the top-level keys. base.yaml and every file under base/ are the complete, neutral defaults every scenario starts from; a file's path is its key path (base/market/balancing.yaml holds market.balancing). The method, source and dispatch overlays, then the scenario's study entry, change only what differs (config/defaults/README.md).

| Key | Default | Description |
|---|---|---|
| `timestep_minutes` | `60` | Base defaults: the top-level keys. base.yaml and every file under base/ are the complete, neutral defaults every scenario starts from; a file's path is its key path (base/market/balancing.yaml holds market.balancing). The method, source and dispatch overlays, then the scenario's study entry, change only what differs (config/defaults/README.md). model timestep, minutes: 60 = hourly, 30 = half-hourly |
| `solve_period` | (mapping) | The full modelled year is solved unless a scenario sets enabled: true with its own start and end. |
| `solve_period.enabled` | `false` |  |
| `logging` | (mapping) |  |
| `logging.level` | `INFO` | DEBUG \| INFO \| WARNING \| ERROR \| CRITICAL |
| `logging.log_to_file` | `true` |  |
| `logging.log_directory` | `logs` |  |
