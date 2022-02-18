.. PyPSA-GB documentation master file, created by
   sphinx-quickstart on Wed Sep  8 11:14:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyPSA-GB: An open source model of the GB power system for future energy scenarios
=================================================================================

Energy system models with high spatial and temporal resolution are required to analyse systems reliant on variable renewable generation. This paper presents PyPSA-GB which is an open dataset and power dispatch model of the GB transmission network using country-specific data over historical years and for future energy scenarios. Two aspects of the GB electricity market can be readily modelled: (i) the wholesale electricity market, by solving a single bus unit commitment optimisation problem to dispatch generators and storage, and (ii) the balancing mechanism, by solving a network constrained linear optimal power flow. The model is showcased through an analysis of network expansion for National Grid’s net zero future energy scenarios.

**Developers**

Institute for Energy Systems, University of Edinburgh. Andrew Lyden...

**Licence**

To do...

.. Documentation
.. =============

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   introduction
   installation
   quick_run

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
   notebooks/7 - Network Expansion.ipynb
   notebooks/8 - Two Step Dispatch.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Examples

   seasonal_thermal_energy_storage
   high_temperature_thermal_energy_storage
   marine_energy

.. toctree::
   :maxdepth: 1
   :caption: References

   API_reference
   release_notes
   contributing
   licencing
