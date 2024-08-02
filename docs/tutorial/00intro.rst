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

**qandle** depends on :code:`PyTorch`, :code:`einops`, :code:`qW_Map` and :code:`NetworkX`.

To install **qandle** manually, clone the repository and install :code:`poetry` and :code:`poethepoet` using :code:`pip install poetry poethepoet`. 
Then, run :code:`poetry install` in the root directory of the repository.
Tests can be run using :code:`poe test`. The documentation can be built using :code:`poe doc`.