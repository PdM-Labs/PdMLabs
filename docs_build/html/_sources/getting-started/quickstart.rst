Quickstart
========================

Get up and running in a few minutes using the main dataset entry point:
``pdmlabs.utils.dataset.Dataset``.

This page shows:

1. Data preparation with one ``Dataset`` handler
2. How the same runner pattern works across flavors
3. Flavor-specific examples using only APIs that exist in this repository

Step 1: Install
---------------

.. code-block:: bash

    pip install .

Optional (for experiment tracking UI):

.. code-block:: bash

    pip install mlflow

Step 2: Data Preparation with ``utils.dataset.Dataset``
-------------------------------------------------------

.. code-block:: python

    import pandas as pd
    from pdmlabs.utils.dataset import Dataset

    # Minimal multi-source toy data
    n = 240
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="H").tolist()
            + pd.date_range("2024-01-01", periods=n, freq="H").tolist(),
            "source": ["asset_1"] * n + ["asset_2"] * n,
            "sensor_1": [0.2 + i * 0.001 for i in range(n)] + [0.3 + i * 0.0012 for i in range(n)],
            "sensor_2": [1.0] * n + [0.9] * n,
            # event columns used to form episodes
            "maintenance": [0] * (n - 1) + [1] + [0] * (n - 1) + [1],
            "failure": [0] * (n - 5) + [1, 0, 0, 0, 0] + [0] * (n - 10) + [1] + [0] * 9,
        }
    )

    dataset_handler = Dataset(
        data=df,
        datetime_column="timestamp",
        source_column="source",
        maintenance_column="maintenance",
        failure_column="failure",
        train_sources=0.6,
        val_sources=0.2,
        test_sources=0.2,
    )

From the same ``dataset_handler``, you can create all flavor-specific dataset dictionaries:

.. code-block:: python

    ds_semi, _ = dataset_handler.get_semi_dataset()
    ds_unsup, _ = dataset_handler.get_unsupervised_dataset()
    ds_cls, _ = dataset_handler.get_Classification_dataset()
    ds_rul, _ = dataset_handler.get_rul_dataset()
    ds_sa, _ = dataset_handler.get_SA_dataset()

Step 3: One Common ``run_experiment`` Pattern
---------------------------------------------

Important behavior in this framework:

- ``event_preferences`` are taken from the dataset flavor dictionary (created by ``Dataset.get_*_dataset()``)
- In ``run_experiment``, pass the method class (for example ``IsolationForest``), not an instantiated object

.. code-block:: python

    from pdmlabs.RunExperiment import run_experiment
    from pdmlabs.thresholding.constant import ConstantThresholder

    def run_one_flavor(
        dataset,
        method_class,
        method_name,
        method_param_space,
        experiment_cls,
        experiment_name,
        *,
        fit_size=100,
        profile_size=2,
        thresholder_cls=ConstantThresholder,
        optimization_param="AD1_AUC",
        maximize=True,
    ):
        return run_experiment(
            dataset=dataset,
            # Pass class, not object. The pipeline handles event_preferences from dataset.
            methods=[method_class],
            param_space_dict_per_method=[method_param_space],
            method_names=[method_name],
            experiments=[experiment_cls],
            experiment_names=[experiment_name],
            MAX_RUNS=4,
            MAX_JOBS=1,
            INITIAL_RANDOM=1,
            fit_size=fit_size,
            profile_size=profile_size,
            thresholder=thresholder_cls,
            mlflow_port=None,
            optimization_param=optimization_param,
            maximize=maximize,
        )

Step 4: Run Each Flavor
-----------------------

Semi-supervised anomaly detection (auto-profile)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pdmlabs.experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
    from pdmlabs.method.isolation_forest import IsolationForest
    from pdmlabs.utils import automatic_parameter_generation

    result_auto_profile = run_one_flavor(
        dataset=ds_semi,
        method_class=IsolationForest,
        method_name="IF",
        method_param_space=automatic_parameter_generation.online_technique("IF", 100),
        experiment_cls=AutoProfileSemiSupervisedPdMExperiment,
        experiment_name="AutoProfile Semi",
        fit_size=100,
        profile_size=2,
    )

Unsupervised anomaly detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pdmlabs.experiment.batch.unsupervised_experiment import UnsupervisedPdMExperiment
    from pdmlabs.method.isolation_forest_uns import IsolationForestUnsupervised

    result_unsupervised = run_one_flavor(
        dataset=ds_unsup,
        method_class=IsolationForestUnsupervised,
        method_name="IF",
        method_param_space=automatic_parameter_generation.unsupervised_technique("IF", 100),
        experiment_cls=UnsupervisedPdMExperiment,
        experiment_name="Unsupervised",
        fit_size=100,
        profile_size=2,
    )

Supervised classification
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pdmlabs.experiment.batch.supervised_experiment import SupervisedPdMExperiment
    from pdmlabs.method.xgboost import XGBoost

    result_classification = run_one_flavor(
        dataset=ds_cls,
        method_class=XGBoost,
        method_name="XGBOOST",
        method_param_space=automatic_parameter_generation.supervised_technique("XGBOOST", 100),
        experiment_cls=SupervisedPdMExperiment,
        experiment_name="Classification",
        fit_size=100,
        profile_size=2,
    )

Supervised RUL regression
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pdmlabs.experiment.batch.RUL_experiment import SupervisedRULPdMExperiment
    from pdmlabs.method.sklearn_regression_wraper import RandomForestRUL

    result_rul = run_one_flavor(
        dataset=ds_rul,
        method_class=RandomForestRUL,
        method_name="RandomForestRUL",
        method_param_space={
            "n_estimators": [50, 100],
            "max_depth": [5, None],
            "random_state": [42],
        },
        experiment_cls=SupervisedRULPdMExperiment,
        experiment_name="RUL",
        fit_size=100,
        profile_size=2,
    )

Supervised survival-analysis flavor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pdmlabs.experiment.batch.SA_experiment import Supervised_SA_PdMExperiment
    from pdmlabs.thresholding.SurvSuperVisedTH import SurvToRUL

    result_sa = run_one_flavor(
        dataset=ds_sa,
        method_class=XGBoost,
        method_name="XGBOOST",
        method_param_space=automatic_parameter_generation.supervised_technique("XGBOOST", 100),
        experiment_cls=Supervised_SA_PdMExperiment,
        experiment_name="Survival Analysis",
        fit_size=100,
        profile_size=2,
        thresholder_cls=SurvToRUL,
    )

Inspect Results
---------------

.. code-block:: python

    print("AutoProfile:", result_auto_profile)
    print("Unsupervised:", result_unsupervised)
    print("Classification:", result_classification)
    print("RUL:", result_rul)
    print("SA:", result_sa)

Each call returns a list with one item for this setup (one method x one experiment),
typically containing keys such as ``best_params``, ``best_objective``, and ``th``.

Step 5: Load and Use Stored MLflow Models
-----------------------------------------

PdMLabs automatically logs the best pipeline (including preprocessors, methods, postprocessors, and thresholders) as an MLflow ``pyfunc`` artifact for each experiment.

You can easily load this pipeline later to generate predictions on test or streaming data:

.. code-block:: python

    import mlflow

    # 1. Find your experiment and run ID
    experiment_name = "AutoProfile Semi IF" # Name format is typically "{experiment_name} {method_name}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    run_id = runs.iloc[0].run_id

    # 2. Load the complete pipeline
    loaded_pipeline = mlflow.pyfunc.load_model(f"runs:/{run_id}/best_pdm_pipeline")

    # 3. Make predictions
    # The pipeline handles data transformations and thresholds transparently
    target_data = ds_semi['target_data'][0]
    target_source = ds_semi['target_sources'][0]
    
    # Pre-process raw test data (e.g., dropping timestamps) before prediction if necessary
    if 'timestamp' in target_data.columns:
        target_data = target_data.drop(columns=['timestamp'])

    predictions = loaded_pipeline.predict({
        'target_data': target_data, 
        'source': target_source,
        'event_data': ds_semi['event_data'] # optional
    })

    print("Anomaly Scores:", predictions['scores'])
    print("Dynamic Thresholds:", predictions.get('dynamic_thresholds'))

Troubleshooting
---------------

**"ModuleNotFoundError: No module named 'pdmlabs'"**

.. code-block:: bash

    pip install .

**No experiments are running**

Check that your event columns create valid episodes and that each train/val/test split
contains at least one failure episode.

**MLflow issues**

Keep ``mlflow_port=None`` for first runs. Enable later only if needed.

See Also
--------

- :doc:`index`
- :doc:`../concepts/index`
- :doc:`../user-guide/index`
- :doc:`../api-reference`
