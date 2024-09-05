Getting Started with VitalDSP
=============================

Welcome to VitalDSP, a comprehensive digital signal processing (DSP) library designed for healthcare and biomedical applications. This guide will walk you through the initial steps to get the library up and running and explore its core functionalities.

Installation
============

To install VitalDSP, you can use `pip`:

.. code-block:: bash

   pip install vital-DSP

Once installed, you can begin using the library to process and analyze healthcare signals such as ECG, PPG, and more.

Basic Usage
===========

After installation, you can import the core modules and start working with your signals. Here’s a basic example to get you started:

.. code-block:: python

   import vitalDSP as vidsp

   # Example of using the filtering module
   from vitalDSP.filtering.artifact_removal import ArtifactRemoval
   signal = np.array([1, 100, 3, 4, 5])
   ar = ArtifactRemoval(signal)
   clean_signal = ar.median_filter_removal(kernel_size=3)
   print(clean_signal)

Explore More with Jupyter Notebooks
===================================

For more detailed examples and practical tutorials, explore the Jupyter Notebooks provided with the library:

.. toctree::
   :maxdepth: 2
   :caption: Jupyter Notebooks:

   sample_notebooks

The notebooks will help you understand various features of VitalDSP, including preprocessing, filtering, transformations, and feature extraction. They provide hands-on examples to help you get the most out of this powerful DSP toolkit.

Support and Contributions
=========================

If you have any questions or need support, check out our GitHub page:

`VitalDSP GitHub <https://github.com/Oucru-Innovations/vital-DSP>`_

You can also contribute by reporting issues or submitting pull requests. The community is active and encourages collaboration to improve the library and address healthcare DSP challenges.

