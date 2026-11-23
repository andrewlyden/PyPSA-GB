## `scripts/analysis/`

| Module | Summary |
|---|---|
| `scripts/analysis/analyze_solved_network.py` | Comprehensive analysis and visualization of solved PyPSA networks. |
| `scripts/analysis/build_bill_docs_values.py` | Generate the consumer-bill methodology chapter's run figures from the bill outputs. |
| `scripts/analysis/build_cp30_bill_notebook.py` | Generate notebooks/cp30_2030_consumer_bill_figures.ipynb (bill, provenance, ownership, cost recovery, acquisition). |
| `scripts/analysis/build_dispatch_pricing_explainer.py` | Build the standalone dispatch and wholesale-pricing explainer. |
| `scripts/analysis/build_fes2050_results_notebook.py` | Generate notebooks/fes_ht_2050_results.ipynb: the 2050 public-ownership results. |
| `scripts/analysis/build_landing_figures.py` | Draw the three maps on the documentation's landing page (docs/source/_static/landing/). |
| `scripts/analysis/build_ownership_explorer_data.py` | Assemble the data the public-ownership explorer artifact embeds (item K1 reporting). |
| `scripts/analysis/diagnose_adequacy_controls.py` | Post-solve controlled-adequacy diagnosis (no topology changes). |
| `scripts/analysis/diagnose_topology_adequacy.py` | Diagnose clustered FES topology and adequacy outcomes without changing capacity. |
| `scripts/analysis/docs_figures/common.py` | Shared style for the documentation figures (docs/source/_static/figures/, _static/landing/). |
| `scripts/analysis/docs_figures/dispatch.py` | Draw the figures of the dispatch chapter (docs/source/_static/figures/dispatch/). |
| `scripts/analysis/docs_figures/network.py` | Draw the figures of the network pages of the documentation (docs/source/_static/figures/network/). |
| `scripts/analysis/docs_figures/studies.py` | Figures for the study pages of the documentation (docs/source/studies/). |
| `scripts/analysis/docs_figures/supply.py` | Draw the figures of the documentation's demand and supply pages (docs/source/_static/figures/supply/). |
| `scripts/analysis/generate_analysis_notebook.py` | Generate interactive Jupyter notebooks for analyzing PyPSA-GB solved networks. |
| `scripts/analysis/public_ownership_explorer/build_explorer.py` | Inject the explorer data into the template and write the Public Ownership Bill Explorer page. |
| `scripts/analysis/public_ownership_explorer/extract_explorer_data.py` | Build the data file for the Public Ownership Bill Explorer from the public-ownership outputs. |
| `scripts/analysis/public_ownership_pathway_explorer/build_pathway_explorer.py` | Build the cumulative 2030-2050 public ownership page from the pathway summary. |
| `scripts/analysis/review_completed_validation_months_2020.py` | Review completed 2020 validation months against NESO, ELEXON, and ESPENI. |
| `scripts/analysis/validation_volume_cost.py` | Balancing volume and cost against outturn, month by month, for a historical validation family. |
