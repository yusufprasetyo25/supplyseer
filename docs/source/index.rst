.. supplyseer documentation master file, created by
   sphinx-quickstart on Thu Nov 14 09:06:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SupplySeer's documentation!
====================================

SupplySeer is a Python library for computational supply chain analytics, combining stochastic processes, 
optimization, and machine learning approaches to model and analyze supply chain dynamics.
Currently still in pre-release alpha stage - this means it will undergo significant changes before an official 1.0 release.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials/index
   api/index
   examples/index

Features
--------
* Stochastic demand modeling using Geometric Brownian Motion
* Inventory optimization and control
* Supply chain network analysis
* Risk assessment and simulation
* Advanced visualization tools

Quick Install
------------
.. code-block:: bash

   git clone https://github.com/supplyseer-ai/supplyseer.git
   cd supplyseer

Quick Example
------------
.. code-block:: python

   from supplyseer.models import StochasticDemandProcess, DemandSimConfig

   # Configure demand simulation
   config = DemandSimConfig(
       initial_demand=600000,
       drift=0.05,
       volatility=0.1,
       decay_rate=0.05,
       time_horizon=2.0,
       n_paths=1000,
       n_steps=50,
       random_seed=42
   )

   # Create and run simulation
   sim = StochasticDemandProcess(config)
   results = sim.simulate()

   # Visualize results
   sim.plot_results()

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
