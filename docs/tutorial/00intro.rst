Intro
=====

Installation
------------
Install from pip using

.. code-block:: console

   $ pip install qandle


**qandle** runs on Python 3.8 or later, on Windows, Linux, and macOS, however, compiling using :code:`torch.compile` is only supported on Unix-based systems with PyTorch 2.0 or later.


Manual Installation for Development
-----------------------------------

**qandle** depends on PyTorch, einops, :code:`qW_Map` and NetworkX. To install the required packages manually, run :code:`pip install -r requirements.txt`
To install **qandle** manually, clone the repository and run :code:`pip install .` from the root directory.
Run tests using 

.. code-block:: console

   $ pytest qandle --cov=qandle --cov-report=html -W error \
    -W ignore::PendingDeprecationWarning:semantic_version.base \
    -W ignore::DeprecationWarning \
    -W ignore::pytest_cov.plugin.CovDisabledWarning

