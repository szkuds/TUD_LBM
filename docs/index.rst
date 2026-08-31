.. tud_lbm documentation master file, created by
   sphinx-quickstart on Wed May  5 22:45:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tud_lbm's documentation!
==========================================================

**tud_lbm** is a JAX-accelerated lattice Boltzmann method framework developed
at Delft University of Technology.  It supports single-phase, multiphase,
wetting, hysteresis, and chemical-step simulations through a
configuration-driven workflow, with a registry-based operator system that
lets new physics be added by dropping in a file.

New here?  Start with the :doc:`quickstart`.

.. toctree::
  :maxdepth: 2
  :caption: Getting Started:

  quickstart

.. toctree::
  :maxdepth: 2
  :caption: Design & Architecture:

  architecture
  operators
  lattice
  adapters

.. toctree::
  :maxdepth: 2
  :caption: API Reference:

  autoapi/src/index

.. toctree::
  :maxdepth: 1
  :caption: Development:

  notes/README.dev
  notes/project_setup
  notes/delftblue_setup
  notes/performance

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
