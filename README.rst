BenchBench Package
=========================================

Overview
--------
The ``benchbench`` package simplifies benchmark agreement testing for NLP models. Compare multiple models across various benchmarks and generate comprehensive agreement reports easily.

It also powers `BenchBench` (https://huggingface.co/spaces/ibm/benchbench), a benchmark for comparing benchmarks.  *(Note: Removed the extra asterisks)*

Contributing a New Benchmark
--------------------------

To contribute a new benchmark, create a pull request with a new CSV file in ``src/bat/assets/benchmarks``. The filename should reflect the data source and snapshot date (see existing files for examples).


Usage
-----

While much of ``benchbench``'s functionality is available via the interactive `BenchBench` app (https://huggingface.co/spaces/ibm/benchbench), for more advanced usage and customization, clone the repository:

.. code-block:: bash

   git clone git@github.com:IBM/benchbench.git

Install in the environment of your choice:

.. code-block:: bash

   cd benchbench
   pip install -e .

And check out the example in ``examples/newbench_example`` (or here: https://github.com/IBM/benchbench/blob/main/examples/newbench_example.py) *(Note: Use backticks for file path)*

Contributing
------------
Contributions to the ``benchbench`` package are welcome! Please submit your pull requests or issues through our GitHub repository.

License
-------

This package is released under the MIT License.