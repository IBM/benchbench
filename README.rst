
BenchBench Package
=========================================

Overview
--------
The ``benchbench`` package is designed to facilitate benchmark agreement testing for NLP models. It allows users to easily compare multiple models against various benchmarks and generate comprehensive reports on their agreement.

It also supports `BenchBench <https://huggingface.co/spaces/ibm/benchbench>`_ the benchmark to compare benchmarks.

Installation
------------
To install the ``benchbech`` package, you can use out github:

.. code-block:: bash

    git+https://github.com/ibm/benchbench


Contribute a new Benchmark
--------------------------

Contributing a new benchmark is as simple as creating PR with a new CSV file in `src/bat/assets/benchmarks`
Note the name of the file represents the source and snapshot date of the data.
See files there for examples on how to structure the CSV.


Usage Example
-------------
Below is a step-by-step example of how to use the ``benchbench`` package to perform agreement testing.



**Step 1: Configuration**

First, set up the configuration for the tests:

.. code-block:: python

    import pandas as pd
    from bat import Tester, Config, Benchmark, Reporter
    from bat.utils import get_holistic_benchmark
    
    cfg = Config(
        exp_to_run="example",
        n_models_taken_list=[0],
        model_select_strategy_list=["random"],
        n_exps=10
    )

**Step 2: Fetch Model Names**

Fetch the names of the reference models to be used for scoring:

.. code-block:: python

    tester = Tester(cfg=cfg)
    models_for_benchmark_scoring = tester.fetch_reference_models_names(
        reference_benchmark=get_holistic_benchmark(), n_models=20
    )
    print(models_for_benchmark_scoring)

**Step 3: Load and Prepare Benchmark**

Load a new benchmark and add an aggregate column:

.. code-block:: python

    newbench_name = "fakebench"
    newbench = Benchmark(
        pd.read_csv(f"src/bat/assets/{newbench_name}.csv"),
        data_source=newbench_name,
    )
    newbench.add_aggregate(new_col_name=f"{newbench_name}_mwr")

**Step 4: Agreement Testing**

Perform all-vs-all agreement testing on the new benchmark:

.. code-block:: python

    newbench_agreements = tester.all_vs_all_agreement_testing(newbench)
    reporter = Reporter()
    reporter.draw_agreements(newbench_agreements)

**Step 5: Extend and Clean Benchmark**

Extend the new benchmark with holistic data and clear repeated scenarios:

.. code-block:: python

    allbench = newbench.extend(get_holistic_benchmark())
    allbench.clear_repeated_scenarios(source_to_keep=newbench_name)

**Step 6: Comprehensive Agreement Testing**

Perform comprehensive agreement testing and visualize:

.. code-block:: python

    all_agreements = tester.all_vs_all_agreement_testing(allbench)
    reporter.draw_agreements(all_agreements)

Contributing
------------
Contributions to the ``BAT`` package are welcome! Please submit your pull requests or issues through our GitHub repository.

License
-------
This package is released under the MIT License.
