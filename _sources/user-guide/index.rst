📖 User Guide
=============

This guide focuses on practical usage of PdMLabs once you already understand the
core concepts.

Use this page to decide what to run, how to configure experiments, and how to
interpret outputs in a consistent way.


Who This Guide Is For
---------------------

This guide is intended for users who want to:

- run repeatable predictive maintenance experiments,
- compare different methods and experiment flavors fairly,
- tune configurations with clear constraints,
- log and analyze results with MLflow,
- extend the framework with custom components.


Recommended Workflow
--------------------

For most projects, this sequence works well:

1. Prepare your dataset dictionary (including events and source metadata).
2. Choose the experiment flavor based on data assumptions.
3. Select one or more methods and parameter spaces.
4. Start with default pre/post/threshold components.
5. Run a small search budget first, then scale up.
6. Inspect AD/RUL metrics and qualitative plots.
7. Iterate on parameter spaces and component choices.


Choosing The Right Experiment Flavor
------------------------------------

Use the following rule of thumb:

- ``AutoProfileSemiSupervisedPdMExperiment``
	Use when you want profile-based learning with event-aware resets.

- ``IncrementalSemiSupervisedPdMExperiment``
	Use when behavior changes over time and rolling re-training is important.

- ``SemiSupervisedPdMExperiment``
	Use when you have representative historical data and a stable deployment setting.

- ``UnsupervisedPdMExperiment``
	Use when fitting on historical normal data is not available or not desired.

- ``SupervisedPdMExperiment``
	Use for labeled classification-style workflows.

- ``SupervisedRULPdMExperiment``
	Use for remaining useful life prediction workflows.

- ``Supervised_SA_PdMExperiment``
	Use for survival-analysis-oriented time-to-event workflows.


Dataset Configuration Notes
---------------------------

Most runtime issues come from dataset contract mismatches.
Verify these early:

- ``target_data`` and ``target_sources`` have matching lengths.
- ``historic_data`` and ``historic_sources`` have matching lengths.
- ``dates`` points to a valid timestamp column (or pre-indexed datetime index).
- ``event_data`` has expected columns: date, type, source, description.
- ``event_preferences`` correctly map failures and resets.

For supervised flows, also verify:

- ``anomaly_labels`` length matches each historic scenario length.
- ``target_labels`` is provided where required (RUL / survival flows).


Parameter Search Strategy
-------------------------

PdMLabs uses Mango for optimization with optional constraints.

Practical guidance:

- Start with small ``MAX_RUNS`` to validate setup.
- Keep ``INITIAL_RANDOM`` > 0 for robust initialization.
- Use constraints to eliminate invalid combinations early.
- Increase budget only after reviewing first-run logs and plots.

When available, ``pdmlabs.utils.automatic_parameter_generation`` provides
good initial spaces for common methods.


Pipeline Component Choices
--------------------------

Start simple, then specialize:

- Preprocessor: begin with ``DefaultPreProcessor``.
- Postprocessor: begin with ``DefaultPostProcessor``.
- Thresholder: begin with ``ConstantThresholder``.

Then progressively test alternatives (smoothing, dynamic thresholding,
survival-to-RUL conversion) when baseline behavior is understood.


Interpreting Results
--------------------

PdMLabs reports metrics designed for predictive maintenance behavior, not only
generic binary classification quality.

Focus on:

- whether alarms occur in useful predictive-horizon windows,
- precision/recall trade-offs for operations,
- stability across sources and episodes,
- robustness across parameter configurations.

Use MLflow logs and plots to combine numeric and visual inspection.


Common Pitfalls
---------------

Watch for these frequent issues:

- date handling problems (wrong column, unsorted timestamps),
- source mapping mismatches (especially with ``match_sources``),
- label length mismatch in supervised workflows,
- unrealistic parameter ranges causing invalid runs,
- over-interpreting one metric without episode-level inspection.


Reproducibility Checklist
-------------------------

To keep experiments reproducible:

- set random seeds where possible,
- keep parameter spaces versioned,
- log all runs to MLflow,
- document dataset version and preprocessing assumptions,
- compare methods under the same dataset split and evaluation settings.


Implementing Custom Methods
---------------------------

Want to create your own anomaly detection method? The framework provides clear interfaces
for each experiment flavor:

- :doc:`implementing-methods/01_unsupervised_method` — Online detection without training
- :doc:`implementing-methods/02_semi_supervised_method` — Learn from clean profiles
- :doc:`implementing-methods/03_classification_method` — Binary classification approach
- :doc:`implementing-methods/04_rul_regression_method` — Predict remaining useful life
- :doc:`implementing-methods/05_survival_analysis_method` — Model time-to-failure with censoring

Start with :doc:`implementing-methods/index` for an overview.


Next Reading
------------

- :doc:`../concepts/index`
- :doc:`../getting-started/quickstart`
- :doc:`../examples/index`
- :doc:`../api-reference`
- :doc:`implementing-methods/index`


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Implementing Methods

   implementing-methods/index
   implementing-methods/01_unsupervised_method
   implementing-methods/02_semi_supervised_method
   implementing-methods/03_classification_method
   implementing-methods/04_rul_regression_method
   implementing-methods/05_survival_analysis_method