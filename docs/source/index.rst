Trove documentation
=========================

.. role:: raw-html(raw)
   :format: html

.. |readme| replace:: :raw-html:`<a href="https://github.com/BatsResearch/trove" target="_blank">README</a>`
.. |training| replace:: :doc:`guides/training`
.. |inference| replace:: :doc:`guides/inference`
.. |examples| replace:: :raw-html:`<a href="https://github.com/BatsResearch/trove/tree/main/examples" target="_blank">Examples</a>`
.. |data| replace:: :doc:`guides/data`
.. |modeling| replace:: :doc:`guides/modeling`

Trove is a flexible toolkit for training and evaluating dense retrievers.
It aims to keep the codebase simple and hackable, while offering a clean, unified interface for quickly experimenting with new ideas.

Repo: :raw-html:`<a href="https://github.com/BatsResearch/trove" target="_blank">BatsResearch/trove</a>`

.. raw:: html

   </br>

    <p align="center"><img width=300 alt="Trove Logo" src="https://huggingface.co/datasets/BatsResearch/trove-lib-documentation-assets/resolve/main/logo/logo_no_background.svg"/></p>

Install Trove from PyPI:

.. code-block:: bash

    pip install ir-trove

To get the latest changes, install from source:

.. code-block:: bash

    pip install git+https://github.com/BatsResearch/trove

To get started with Trove, explore the following resources:


* |readme|: General overview of Trove
* |training|: Step-by-step guide for training dense retrievers with Trove
* |inference|: Step-by-step guide for evaluation, hard negative mining, and encoding
* |examples|: Collection of self-contained scripts for training and inference
* |data|: Detailed guide for loading, preprocessing, and managing datasets
* |modeling|: Insight into Trove’s modeling architecture and how to extend it with custom models, loss functions, and more

.. toctree::
    :maxdepth: 1
    :caption: Guides:
    :hidden:

    guides/training
    guides/inference
    guides/data
    guides/modeling

.. toctree::
    :maxdepth: 1
    :caption: Main API
    :hidden:

    api_ref/index
