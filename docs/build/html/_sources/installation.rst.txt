############
Installation
############

Install Python
##############

The code for PyPSA-GB is written in Python and relies on a number of standard Python packages. Python has emerged as a popular programming language for energy system modellers. Installing Anaconda (https://www.anaconda.com/) is a highly recommended starting point for beginners looking to get Python running quickly on their computer. Anaconda includes several of the necessary scientific packages and the conda package can be used to manage virtual environments. The following installation instructions assume an Anaconda installation including, but alternative setup routes are possible. 

Clone repository using Git
##########################

Clone the repository from GitHub through a terminal using Git. Using a terminal navigate to a target folder where the repo will be cloned and run the following Git clone command (https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) for the PyPSA-GB repository. (Warning: this may take a long time and repository size is around 2GB)::

	git clone https://github.com/andrewlyden/PyPSA-GB.git

Modify .env file to own path
############################

Please modify the path to your own directory for the /PyPSA-GB folder in the .env file. This allows scripts to be imported in any directory.

Install Gurobi solver
#####################

PyPSA-GB solves optimisation problems and currently relies on Gurobi as an optimisation solver. An academic licence of Gurobi can be freely obtained, see (https://www.gurobi.com/downloads/end-user-license-agreement-academic/). An open-source solver will be integrated in a future version.

Create conda environment
########################

PyPSA-GB's depends on PyPSA to undertake the power flow calculations, and also has other dependencies which can be found in PyPSA-GB_requirements.yaml. This file can be used to create a virtual environment through the conda package, see (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

Navigate to the requirements directory and create a PyPSA-GB conda environment::

	conda env create -f PyPSA-GB_requirements.yml

Then to activate the new environment::

	conda activate PyPSA-GB

Run jupyter notebooks
#####################

This environment can be used to easily run the Jupyter notebooks by navigating to /notebooks directory and installing the environment as a kernel::

	python -m ipykernel install --user --name=PyPSA-GB

Then starting the jupyter notebook::

	jupyter notebook

Finally, make sure that the PyPSA-GB kernel is chosen (to change, open the notebook, click 'Kernel', 'Change kernel', and PyPSA-GB should be an option.)

Further dependencies
####################

There are scripts which are used to generate renewable power time-series and which utilise the Atlite package. Due to conflicting dependencies it is recommended that a separate virtual environment is used, and the requirements file, atlite_requirements.yml, can be used to create a conda virtual environment  (see https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

Checklist
#########

#. Anaconda installation
#. Clone repository via GitHub using Git
#. Modify .env file to own path
#. Install solver
#. Create and activate conda environment
#. Install conda environment as kernel for jupyter notebooks
#. Optionally create conda environment for using Atlite
#. Enjoy!
