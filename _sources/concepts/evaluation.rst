Evaluation & Metrics
====================

Evaluating Predictive Maintenance (PdM) models requires moving beyond simple accuracy due to the inherently imbalanced nature of industrial data (failures are rare events) and the different nature of the tasks (classification, regression, survival analysis). PdMLabs utilizes a comprehensive set of metrics adapted to each modeling task, encapsulated in a modular evaluation architecture.

Evaluation Architecture
-----------------------

PdMLabs evaluates experiments using an extensible interface pattern. Internally, `PdMExperiment` uses an orchestrator method (`_run_evaluators()`) to compute and log metrics. 

Depending on the chosen experiment flavor, a built-in "Default Evaluator" is automatically assigned:

- **TSAD / Classification**: Uses `DefaultADEvaluator` to compute classification and anomaly detection metrics.
- **Remaining Useful Life (RUL)**: Uses `DefaultRULEvaluator`, which computes regression metrics and re-uses AD metrics to evaluate thresholded classifications.
- **Survival Analysis (SA)**: Uses `DefaultSurvEvaluator`, computing survival-specific metrics while also integrating AD/RUL metrics.

All evaluators inherit from the `EvaluatorInterface` located in `pdmlabs/evaluation/evaluator.py`.

Supported Metrics
-----------------

**Time-Series Anomaly Detection (TSAD) & Classification**

For anomaly detection, detecting early signs of failure before they occur is critical.

- **Recall (AD1/AD2/AD3)**: Fraction of true anomalies detected. AD1, AD2, and AD3 levels correspond to different perspectives on detection tolerance and episode-aware splitting around failure timestamps.
- **Precision**: Fraction of detected anomalies that actually precede faults in a specific predictive horizon (PH).
- **F1-Score (AD1/AD2/AD3)**: Harmonic mean of precision and recall.
- **AUC-PR (AD1/AD2/AD3)**: Area Under the Precision-Recall Curve.
- **VUS (Volume Under the Surface)**: Optional volume-based metrics for time-series anomaly detection.

**Remaining Useful Life (RUL) & Survival Analysis (SA) Cross-Evaluation**

A unique feature of PdMLabs is the cross-evaluation between RUL and Survival Analysis models. Inspired by the **TITEUF SYSTEM** (see `TITEUF on GitHub <https://github.com/agiannoul/TITEUF/tree/main>`_), PdMLabs calculates Survival Analysis metrics for deterministic RUL predictions, and vice versa (calculating regression metrics from survival probabilities). This provides a holistic view of a model's performance regardless of its foundational approach.

For RUL prediction, models are evaluated as regression tasks, while also computing survival metrics:

- **MAPE**: Mean Absolute Percentage Error.
- **MDAPE**: Median Absolute Percentage Error.
- **MSE / RMSE**: Mean Squared Error and Root Mean Squared Error.
- **MAE**: Mean Absolute Error.
- **R² Score**: Coefficient of determination.

Survival models output probabilities over time, evaluated with metrics that handle right-censored data, while also extracting expected RUL to compute standard regression errors:

- **IBS (Integrated Brier Score)**: Measures Brier calibration and discrimination ability of survival probabilities over time.
- **Max Brier Score**: Maximum Brier score observed across time points.
- **C-Index (Concordance Index)**: Measures how well the model predicts the ordering of survival times.
- **Mean AUC-ROC**: Mean Area Under the Receiver Operating Characteristic Curve over time.

Adding Custom Metrics
---------------------

PdMLabs is designed to be easily extensible. If you want to compute custom business metrics, cost-based metrics, or domain-specific logic, you can inject your own evaluators into the pipeline without modifying the core experiment code.

**Step 1: Implement the Interface**

Create a new class inheriting from `EvaluatorInterface` and implement the `evaluate()` method. You have full access to the experiment object, the resulting predictions, and labels through `kwargs`.

.. code-block:: python

    from pdmlabs.evaluation.evaluator import EvaluatorInterface
    import mlflow

    class MyCustomCostEvaluator(EvaluatorInterface):
        def evaluate(self, experiment, **kwargs) -> dict:
            # Extract necessary variables from kwargs
            result_scores = kwargs.get('result_scores')
            result_labels = kwargs.get('result_labels')
            results_isfailure = kwargs.get('results_isfailure')
            
            # Compute your custom metric
            # e.g., total_cost = calculate_maintenance_cost(result_scores, results_isfailure)
            total_cost = 42.0 
            
            my_metrics = {
                "business_cost": total_cost,
                "custom_roi": 1.5
            }
            
            # Optionally log metrics directly to MLflow
            mlflow.log_metrics(my_metrics)
            
            # Return metrics dict so it gets included in the experiment summary
            return my_metrics


**Step 2: Pass to the Experiment**

Pass an instance of your custom evaluator to the `run_experiment()` function using the `custom_evaluators` argument. 

.. code-block:: python

    from pdmlabs.RunExperiment import run_experiment
    
    my_evaluator = MyCustomCostEvaluator()

    best_params = run_experiment(
        dataset=dataset_handler,
        methods=methods,
        param_space_dict_per_method=param_spaces,
        method_names=['MyMethod'],
        experiments=[SupervisedPdMExperiment],
        experiment_names=['My Custom Eval Run'],
        custom_evaluators=[my_evaluator]  # Inject your evaluator here!
    )

PdMLabs will seamlessly run the built-in default evaluator to give you standard metrics, and then iterate through your `custom_evaluators` to augment the MLflow logs with your proprietary metrics.
