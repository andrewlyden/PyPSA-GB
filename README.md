# PyPSA-GB

PyPSA-GB is a dataset and model of the Great Britain electricity system. It uses PyPSA (Python for Power Systems Analysis) to peform power dispatch and planning studies.

Energy system models with high spatial and temporal resolution are required to analyse systems reliant on variable renewable generation.  PyPSA-GB is an open dataset and power dispatch model of the GB transmission network using country-specific data over historical years and for future energy scenarios. Two aspects of the GB electricity market can be readily modelled: (i) the wholesale electricity market, by solving a single bus unit commitment optimisation problem to dispatch generators and storage, and (ii) the balancing mechanism, by solving a network constrained linear optimal power flow. 

See the notebooks in the PyPSA-GB folder for exploration of the data and functionality. See documentation for more details [here](pypsa-gb.readthedocs.io).

![PyPSA-GB Reduced Network Model](pics/voronoi_reduced_model.jpg)
