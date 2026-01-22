.. PyPSA-GB documentation master file

==============================================
PyPSA-GB: Great Britain Energy System Model
==============================================

.. raw:: html

   <p style="font-size: 1.2em; color: #555; margin-bottom: 1.5em;">
   Open-source energy system modelling for Great Britain's electricity network
   </p>

   <div style="display: flex; gap: 10px; margin-bottom: 2em; flex-wrap: wrap;">
   <a href="https://github.com/andrewlyden/PyPSA-GB" style="text-decoration: none;">
   <img src="https://img.shields.io/badge/GitHub-PyPSA--GB-blue?style=for-the-badge&logo=github" alt="GitHub">
   </a>
   <img src="https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
   <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License: MIT">
   <a href="https://pypsa-gb.readthedocs.io/en/latest/?badge=latest" style="text-decoration: none;">
   <img src="https://img.shields.io/badge/docs-latest-brightgreen?style=for-the-badge" alt="Documentation Status">
   </a>
   </div>

PyPSA-GB is a comprehensive model of the Great Britain electricity system built on `PyPSA <https://pypsa.org>`_ (Python for Power System Analysis). It combines **detailed network topology**, **real-world data sources**, and **future energy scenario projections** to enable research into Britain's energy transition.

----

.. grid:: 2
   :gutter: 4

   .. grid-item-card:: 🚀 Getting Started
      :link: getting_started/index
      :link-type: doc
      :class-card: sd-shadow-md

      **New to PyPSA-GB?** Install the model and run your first scenario with our step-by-step guide.

   .. grid-item-card:: 📖 User Guide
      :link: user_guide/index
      :link-type: doc
      :class-card: sd-shadow-md

      **Deep dive into configuration.** Learn how to customise scenarios, configure solvers, and analyse optimisation results.

   .. grid-item-card:: 📊 Data Reference
      :link: data_reference/index
      :link-type: doc
      :class-card: sd-shadow-md

      **Understand the data.** Comprehensive documentation of FES, DUKES, REPD, ETYS, and all input data sources.

   .. grid-item-card:: 🔬 Tutorials
      :link: tutorials/index
      :link-type: doc
      :class-card: sd-shadow-md

      **Learn by example.** Jupyter notebooks exploring historical and future scenario analysis.

----

Key Features
============

- **Multiple Network Models**: Full ETYS (2000+ buses), Reduced (32 buses), or Zonal (17 zones)
- **Historical & Future Scenarios**: Model years 2010-2024 (historical) or 2025-2050 (FES projections)
- **NESO Future Energy Scenarios**: Holistic Transition, Electric Engagement, and other pathways
- **High Resolution**: Half-hourly or hourly timesteps with full network constraints
- **Open Source**: MIT license, transparent assumptions, community contributions welcome


Quick Example
=============

Run a 2035 Holistic Transition scenario:

.. code-block:: bash

   # Activate environment
   conda activate pypsa-gb

   # Run the workflow
   snakemake resources/network/HT35_solved.nc -j 4


.. What Can You Model?
.. ===================

.. 1. **Wholesale Electricity Market**: Single-bus unit commitment dispatch
.. 2. **Balancing Mechanism**: Network-constrained optimal power flow
.. 3. **Capacity Planning**: Network and generator expansion studies
.. 4. **Renewable Integration**: Wind, solar, and storage with realistic profiles
.. 5. **Security of Supply**: LOLE/LOLP analysis for system adequacy


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started/index
   getting_started/installation
   getting_started/quickstart
   getting_started/first_scenario

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/index
   user_guide/workflow
   user_guide/configuration
   user_guide/scenarios
   user_guide/network_models
   user_guide/clustering

.. toctree::
   :maxdepth: 2
   :caption: Data Reference
   :hidden:

   data_reference/index
   data_reference/data_sources
   data_reference/network_data
   data_reference/maintenance

.. toctree::
   :maxdepth: 3
   :caption: Tutorials
   :hidden:

   tutorials/index
   tutorials/1-historical-baseload-2015
   tutorials/2-historical-renewables-2023
   tutorials/3-future-holistic-transition-2035
   tutorials/4-future-electric-engagement-2050
   tutorials/5-networks
   tutorials/6-demand
   tutorials/7-generators
   tutorials/8-marginal-costs
   tutorials/9-renewables
   tutorials/10-storage
   tutorials/11-interconnectors
   tutorials/12-hydrogen


.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index
   api/core_modules
   api/integration_modules

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   development/contributing
   development/architecture
   development/release_notes
   development/troubleshooting


Contributors
============

PyPSA-GB is developed at the Institute for Energy Systems, University of Edinburgh.

- **Lead Developer**: Dr Andrew Lyden
- **Contributors**: Dr Wei Sun, Dr Iain Struthers, Dr Seb Hudson, Dr Lukas Franken

This work was completed as part of the INTEGRATE project led by Prof Daniel Friedrich.

Citation
========

If you use PyPSA-GB in your research, please cite:

   **Lyden, A., Sun, W., Struthers, I., Franken, L., Hudson, S., Wang, Y. and Friedrich, D., 2024.**
   PyPSA-GB: An open-source model of Great Britain's power system for simulating future energy scenarios.
   *Energy Strategy Reviews*, 53, p.101375.


Papers Using PyPSA-GB
=====================

- **Dergunova, T. and Lyden, A., 2024.** Great Britain's hydrogen infrastructure development—Investment priorities and locational flexibility. *Applied Energy*, 375, p.124017.

- **Desguers, T., Lyden, A. and Friedrich, D., 2024.** Integration of curtailed wind into flexible electrified heating networks with demand-side response and thermal storage: Practicalities and need for market mechanisms. *Energy Conversion and Management*, 304, p.118203.

- **Lyden, A., Alene, S., Connor, P., Renaldi, R. and Watson, S., 2024.** Impact of locational pricing on the roll out of heat pumps in the UK. *Energy Policy*, 187, p.114043.

- **Lyden, A., Sun, W., Friedrich, D. and Harrison, G., 2023.** Electricity system security of supply in Scotland. Study for the Scottish Government via ClimateXChange.

License
=======

PyPSA-GB is released under the **MIT License**. See data reference for details on data licenses.

