# Using AI assistants

PyPSA-GB has a large, layered configuration and a long workflow. An AI assistant can help you find your way around
it, write scenarios and read results. There are two kinds of help.

| | Where | Good for |
|---|---|---|
| **Ask AI** | The button on every page of this site | Questions about the model and the documentation: what a setting does, how a part of the model works, where to find something. |
| **A coding assistant** | Your own clone, in your editor or terminal (Claude Code, Codex, GitHub Copilot, Cursor and others) | Doing things: writing a scenario, running and debugging the workflow, reading a solved network. |

## Setting up a coding assistant

The repository has a guide written for assistants, `AGENTS.md` at its root: where things are, how the configuration
is layered, how to run the workflow, and the conventions to keep. Most assistants read `AGENTS.md` on their own.
Claude Code reads `CLAUDE.md`, which imports `AGENTS.md`. Nothing else needs setting up: open the repository in your
assistant, in the environment from {doc}`installation`.

## Tasks that work well

Start with something small, and say which scenario and which part of the model you mean. For example:

- **Write a scenario.** *"Add a scenario to `config/scenarios.yaml` for FES 2025 Holistic Transition 2035 on the
  clustered network, one week in January, weather year 2019. Check it with a dry run."*
- **Explain a setting.** *"Which file sets `market.balancing` for `HT35_market`, and what do the values mean?"* The
  assistant can answer from `python config/config_loader.py --explain HT35_market --path market.balancing`.
- **Find why a run failed.** *"The last run stopped at `solve_network`. Read its log in `logs/` and tell me why."*
- **Read a result.** *"Open `resources/network/Tutorial_2020_reduced_solved.nc` and summarise generation by
  technology and the most loaded lines."*
- **Compare two runs.** *"How do generation, imports and prices differ between `Tutorial_2020_reduced` and
  `Tutorial_2020_zonal`?"*
- **Extend a tutorial.** *"Change tutorial 1 to the week of 7 December 2020 and re-run it."* (see
  {doc}`../tutorials/index`)

## Ground rules

- **Dry run first.** Ask for `snakemake -n` before anything long, and read what it plans to run. An assistant can
  start builds and solves that take hours on the full ETYS network.
- **Keep runs short while trying things.** A one-week `solve_period` on the Reduced or Zonal network solves in
  minutes.
- **Check the numbers yourself.** Before you use a result, open the solved network and compare it with the
  validation pages ({doc}`../studies/validation_2020`). An assistant's summary can be wrong in ways that look
  right.
- **Review changes to shared files.** Editing a data file or script that many rules read rebuilds everything
  downstream of it ({doc}`running_the_workflow`). Ask the assistant to say what it changed and why.
