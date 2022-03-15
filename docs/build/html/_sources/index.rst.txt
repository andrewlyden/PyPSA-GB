.. PyPSA-GB documentation master file, created by
   sphinx-quickstart on Wed Sep  8 11:14:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyPSA-GB: An open source model of the GB power system for future energy scenarios
=================================================================================

PyPSA-GB is an open dataset and power dispatch model of the GB transmission network using country-specific data over historical years and National Grid's Future Energy Scenarios for future years. 

Two aspects of the GB electricity market can be readily modelled: (i) the wholesale electricity market, by solving a single bus unit commitment optimisation problem to dispatch generators and storage, and (ii) the balancing mechanism, by solving a network constrained linear optimal power flow.

National Grid's Future Energy Scenarios (Steady Progression, System Transformation, Consumer Transformation, and Leading The Way) can be simulated for the years 2021 to 2050. This requires the choice of a baseline year for weather data and demand, and this can be chosen for 2010-2020. The historical years 2010-2020 can also be simulated. Simulations can be carried out in half-hourly or hourly timesteps.

Assumptions made in the model are transparent and can be modified directly through the appropriate data files. It is the intention of the developers that this model and dataset can provide a foundation for active development and improvement. The 'Issues' section on the GitHub page contains suggestions for improvements. Contributions are welcomed and encouraged.

**Developers**

PyPSA-GB is developed at the Institute for Energy Systems, University of Edinburgh. 

Contributors:

- Andrew Lyden
- Iain Struthers
- Seb Hudson
- Lukas Franken

**Licence**

PyPSA-GB is released under the open source MIT License. ESPENI and ERA5 datasets are used under a CC-BY-4.0 licence.

.. Documentation
.. =============

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   .. introduction
   installation
   .. quick_run

.. toctree::
   :maxdepth: 1
   :caption: Data and Functionality

   notebooks/1 - Network.ipynb
   notebooks/2 - Demand.ipynb
   notebooks/3 - Generator and Marginal Prices.ipynb
   notebooks/4 - Renewable Power and Storage.ipynb
   notebooks/5a - LOPF Historical.ipynb
   notebooks/5b - LOPF Future.ipynb
   notebooks/6 - Unit Commitment.ipynb
   .. notebooks/8 - Two Step Dispatch.ipynb
   notebooks/9 - Comparison To Historical Data.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Examples

   notebooks/7 - Network Expansion.ipynb
   seasonal_thermal_energy_storage
   .. high_temperature_thermal_energy_storage
   .. marine_energy

.. toctree::
   :maxdepth: 1
   :caption: References

   API_reference
   release_notes
   contributing
