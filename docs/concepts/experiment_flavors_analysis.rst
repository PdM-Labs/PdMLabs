===========================================
Analysis of PdMLabs Experiment Flavors
===========================================

This document provides a detailed architectural breakdown of the different experiment flavors in PdMLabs, highlighting their unique execution flows, data dependencies, and structural differences.

1. AutoProfileSemiSupervisedPdMExperiment
-----------------------------------------
**Goal**: Adapt to normal behavior dynamically by profiling the initial period of each target scenario.

* **Data Dependencies**: Focuses heavily on ``target_data`` and event logs (specifically ``reset`` and ``failure`` events). The ``historic_data`` is essentially ignored because the profile is built on the fly.
* **Execution Flow**:
  
  * Iterates over each target scenario.
  * Slices the scenario based on ``reset_dates``.
  * For each slice, takes the first N timestamps (where N = ``profile_size`` determined by hyperparameter search) to act as the "normal profile".
  * Fits the ``preprocessor`` and ``method`` on this profile, then calls ``predict()`` on the remainder of the slice.

* **Structural Uniqueness**: The ``fit()`` step happens *inside* the target prediction loop and inside the reset-date slicing loop. It dynamically splits dataframes on the fly.

2. IncrementalSemiSupervisedPdMExperiment
-----------------------------------------
**Goal**: Simulate online/streaming execution by updating the model incrementally over rolling windows.

* **Data Dependencies**: ``target_data``, plus specific hyperparameter keys: ``initial_incremental_window_length``, ``incremental_window_length``, and ``incremental_slide``.
* **Execution Flow**:
  
  * Employs a row-by-row iteration (or chunk-by-chunk) using pandas ``iterrows()``.
  * Maintains a ``current_window_buffer`` (for fitting) and a ``current_slide_buffer`` (for predicting).
  * Conditionally calls ``method.fit()`` when buffers fill up, and periodically destroys/re-instantiates or updates the method/preprocessor based on the ``refit_new_method_object`` flag.

* **Structural Uniqueness**: This flavor has a highly complex state-machine loop simulating streaming data, requiring variables like ``executed_initial_fit`` and tracking indices dynamically. It fundamentally breaks the standard "fit all, then predict all" batch paradigm.

3. SemiSupervisedPdMExperiment
------------------------------
**Goal**: Train on clean historic data, predict on new target data.

* **Data Dependencies**: Requires both ``historic_data`` and ``target_data``, relying on the ``match_sources`` dictionary to map target sources to the correct trained historic model.
* **Execution Flow**:
  
  * Fits the ``preprocessor`` and ``method`` globally on all ``historic_data`` at the start.
  * Loops over ``target_data`` and calls ``predict()`` using the corresponding mapped source.
  * Breaks predictions into episodes based on failure/reset dates.

* **Structural Uniqueness**: While this is the most "standard" machine learning flow, its reliance on ``match_sources`` to map source A to source B during the prediction phase is unique compared to unsupervised or auto-profile flows.

4. UnsupervisedPdMExperiment
----------------------------
**Goal**: Detect anomalies without any labeled data or clean historical baselines.

* **Data Dependencies**: Purely relies on ``target_data``.
* **Execution Flow**:
  
  * Completely skips the ``fit()`` phase for the method.
  * Iterates over ``target_data`` and directly calls ``preprocessor.transform()`` and ``method.predict()``.

* **Structural Uniqueness**: The complete absence of the ``fit()`` step.

5. SupervisedPdMExperiment
--------------------------
**Goal**: Standard classification workflow using explicit anomaly labels.

* **Data Dependencies**: Hard dependency on ``anomaly_labels`` existing in the pipeline dataset, matching the length and shape of ``historic_data``.
* **Execution Flow**:
  
  * Similar to Semi-Supervised, but explicitly passes ``anomaly_labels`` to the ``fit()`` methods of the preprocessor, method, and postprocessor.

* **Structural Uniqueness**: The ``fit()`` signature requires the ``anomaly_labels`` argument. It also includes strict assertions to guarantee label array lengths match the dataframe lengths before starting the optimization objective.

6. SupervisedRULPdMExperiment
-----------------------------
**Goal**: Predict Remaining Useful Life (RUL) as a continuous regression task.

* **Data Dependencies**: Requires ``anomaly_labels`` (which in this context act as historic RUL labels) and ``target_labels`` (ground truth RUL for evaluation). Checks for ``is_failure`` flags to identify run-to-failure scenarios.
* **Execution Flow**:
  
  * Fits using labels (like Supervised).
  * During prediction, it extracts the predictions and **bypasses** the standard episode-splitting logic, grouping entire scenarios together.
  * Uses ``rul_evaluate`` for evaluation (which calculates regression metrics like MAE/RMSE) instead of classification/AUC metrics.

* **Structural Uniqueness**: Replaces the evaluation engine and bypasses episode splitting, tracking run-to-failure (``rtf``) metadata for plotting and metrics.

7. Supervised_SA_PdMExperiment
------------------------------
**Goal**: Frame predictive maintenance as a Survival Analysis problem.

* **Data Dependencies**: Same as RUL, requires target labels and run-to-failure metadata.
* **Execution Flow**:
  
  * Fits models with labels.
  * Generates survival predictions.
  * **Crucially**, it calls ``thresholder.fit()`` *after* generating all target predictions, essentially using the target set (or a validation set) to learn a mapping from survival scores to RUL predictions.
  * Uses ``surv_evaluate`` for computing concordance and other survival metrics.

* **Structural Uniqueness**: The thresholder is fitted *post-prediction* on the aggregated target scores, unlike anomaly detection where thresholding is often determined per-episode or inferred statically. Evaluates using a completely different mathematical domain (survival metrics).

Conclusion on Modularity
------------------------
Pushing the entire execution loop into ``PdMExperiment`` is practically impossible because:

1. **Loop Paradigms**: ``Incremental`` uses row-by-row streaming simulation, ``AutoProfile`` slices dataframes on the fly, while the rest process full batches.
2. **Fitting Timings**: ``Unsupervised`` never fits, ``Semi-Supervised`` fits globally at the start, ``AutoProfile`` fits inside the target loop, and ``Survival Analysis`` fits the thresholder at the very end.
3. **Data Signatures**: Supervised flavors require passing labels to ``fit()``; others do not.
4. **Evaluation Routines**: Flavors branch into standard AUC evaluation, RUL evaluation, or Survival evaluation, with different data formatting requirements.
