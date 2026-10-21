"""Experiment orchestration and execution for predictive maintenance anomaly detection.

This module provides the main entry point for running predictive maintenance experiments
across different learning paradigms (supervised, unsupervised, semi-supervised) with
automatic hyperparameter optimization.

Key Features:
    - Multi-method, multi-experiment orchestration
    - Hyperparameter optimization using MANGO (Bayesian optimization)
    - Pipeline construction (preprocessing → method → postprocessing → thresholding)
    - MLflow integration for experiment tracking
    - Support for 7 experiment types with different learning strategies
    - Constraint-based parameter validation for different experiment types
    - Parallel execution for hyperparameter search

Core Functions:
    run_experiment: Main entry point to execute predictive maintenance experiments
    get_method_type: Map experiment type to appropriate method interface
    is_port_in_use: Check port availability for MLflow tracking server
    run_mlflow_server: Start or verify MLflow tracking server

Example:
    >>> from pdmlabs.RunExperiment import run_experiment
    >>> from pdmlabs.utils.dataset import Dataset
    >>> dataset = Dataset(data, datetime_column='timestamp')
    >>> methods = [IsolationForest()]
    >>> param_spaces = [{'n_estimators': [100, 200]}]
    >>> results = run_experiment(
    ...     dataset=dataset.get_rul_dataset()[0],
    ...     methods=methods,
    ...     param_space_dict_per_method=param_spaces,
    ...     method_names=['IF'],
    ...     experiments=[UnsupervisedPdMExperiment],
    ...     experiment_names=['Baseline'],
    ...     MAX_RUNS=20
    ... )
"""

from pdmlabs.pipeline.pipeline import PdMPipeline
from pdmlabs.preprocessing.record_level.default import DefaultPreProcessor
from pdmlabs.postprocessing.default import DefaultPostProcessor
from pdmlabs.thresholding.constant import ConstantThresholder
from pdmlabs.thresholding.SurvSuperVisedTH import SurvToRUL
from pdmlabs.constraint_functions.constraint import auto_profile_max_wait_time_constraint
from pdmlabs.utils.utils import calculate_mango_parameters, calculate_optimizer_budget

from pdmlabs.experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
from pdmlabs.experiment.batch.incremental_semi_supervised_experiment import IncrementalSemiSupervisedPdMExperiment
from pdmlabs.experiment.batch.unsupervised_experiment import UnsupervisedPdMExperiment
from pdmlabs.experiment.batch.semi_supervised_experiment import SemiSupervisedPdMExperiment
from pdmlabs.experiment.batch.supervised_experiment import SupervisedPdMExperiment
from pdmlabs.experiment.batch.RUL_experiment import SupervisedRULPdMExperiment
from pdmlabs.experiment.batch.SA_experiment import Supervised_SA_PdMExperiment

from pdmlabs.method.semi_supervised_method import SemiSupervisedMethodInterface
from pdmlabs.method.unsupervised_method import UnsupervisedMethodInterface
from pdmlabs.method.supervised_method import SupervisedMethodInterface

from pdmlabs.constraint_functions.constraint import self_tuning_constraint_function, incremental_constraint_function, combine_constraint_functions, auto_profile_max_wait_time_constraint, incremental_max_wait_time_constraint
from pdmlabs.constraint_functions.constraint import sand_parameters_constraint_function, combine_constraint_functions, self_tuning_constraint_function, unsupervised_max_wait_time_constraint, unsupervised_distance_based
import math
import socket
import subprocess

from pdmlabs.utils.automatic_parameter_generation import profile_values


def _coerce_profile_sizes(value, name):
    """Normalise a profile-size argument to ``list[int]`` (or ``None``).

    A list is the documented form; a bare int is accepted and wrapped so existing
    callers keep working.

    Args:
        value: ``None``, an int, or a list of ints.
        name (str): Parameter name, used in the error message.

    Returns:
        list[int] | None: The normalised value.

    Raises:
        TypeError: If *value* is neither None, an int, nor a list of ints.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f'{name} must be an int or a list of ints, got bool')
    if isinstance(value, int):
        return [value]
    if isinstance(value, list) and value and all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        return list(value)
    raise TypeError(
        f'{name} must be a list of ints (e.g. [10, 20]) or a single int, '
        f'got {type(value).__name__}: {value!r}'
    )


def _infer_profile_sizes(dataset):
    """Derive profile-size candidates from the shortest target scenario.

    Uses the same horizon the project already documents for ``max_wait_time`` --
    a third of the shortest target scenario -- and hands it to
    :func:`~pdmlabs.utils.automatic_parameter_generation.profile_values`, which
    spreads 16 integer candidates over ``[max_wait/10, max_wait]``.

    Args:
        dataset (dict): The dataset dict passed to :func:`run_experiment`.

    Returns:
        list[int]: Candidate profile sizes, none exceeding ``max_wait``.

    Raises:
        ValueError: If the dataset carries no target data to infer from.
    """
    min_target_scenario_len = dataset.get('min_target_scenario_len')
    if not min_target_scenario_len:
        target_data = dataset.get('target_data') or []
        if not len(target_data):
            raise ValueError(
                'Cannot infer profile_size: the dataset has no target_data. '
                'Pass profile_size explicitly, e.g. profile_size=[10, 20].'
            )
        min_target_scenario_len = min(df.shape[0] for df in target_data)

    max_wait = math.ceil((1 / 3) * min_target_scenario_len)
    return profile_values(max_wait)


def get_method_type(experiment):
    """Map experiment class to corresponding method interface.
    
    Determines which method interface (supervised, unsupervised, or semi-supervised)
    is required for a given experiment type. This ensures the correct method base
    class is used during method instantiation.
    
    Parameters
    ----------
    experiment : type
        Experiment class (not instance). One of:
        - AutoProfileSemiSupervisedPdMExperiment
        - IncrementalSemiSupervisedPdMExperiment
        - SemiSupervisedPdMExperiment
        - UnsupervisedPdMExperiment
        - SupervisedPdMExperiment
        - SupervisedRULPdMExperiment
        - Supervised_SA_PdMExperiment
    
    Returns
    -------
    type
        Method interface class:
        - SemiSupervisedMethodInterface for semi-supervised experiments
        - UnsupervisedMethodInterface for unsupervised experiments
        - SupervisedMethodInterface for supervised/RUL/SA experiments
    
    Raises
    ------
    ValueError
        If experiment type is not recognized.
    
    Examples
    --------
    >>> from pdmlabs.RunExperiment import get_method_type
    >>> from pdmlabs.experiment.batch.unsupervised_experiment import UnsupervisedPdMExperiment
    >>> interface = get_method_type(UnsupervisedPdMExperiment)
    >>> print(interface.__name__)
    'UnsupervisedMethodInterface'
    """
    if experiment in [AutoProfileSemiSupervisedPdMExperiment, IncrementalSemiSupervisedPdMExperiment, SemiSupervisedPdMExperiment]:
        return SemiSupervisedMethodInterface
    elif experiment == UnsupervisedPdMExperiment:
        return UnsupervisedMethodInterface
    elif experiment == SupervisedPdMExperiment or experiment == SupervisedRULPdMExperiment:
        return SupervisedMethodInterface
    elif experiment == Supervised_SA_PdMExperiment:
        return SupervisedMethodInterface
    raise ValueError(f"Unknown experiment type: {experiment}.")

def is_port_in_use(host, port):
    """Check if a given port is in use on the specified host.
    
    Attempts a socket connection to verify if a port is listening.
    Useful for checking if a server (e.g., MLflow UI) is already running.
    
    Parameters
    ----------
    host : str
        Host IP address or hostname (e.g., '127.0.0.1', 'localhost', '0.0.0.0').
    port : int
        Port number to check (0-65535).
    
    Returns
    -------
    bool
        True if port is in use (connection succeeds), False if available.
    
    Examples
    --------
    >>> is_port_in_use('127.0.0.1', 5000)
    False  # Port 5000 is free
    
    >>> is_port_in_use('127.0.0.1', 8080)
    True   # Port 8080 is in use
    
    Notes
    -----
    - Quick check: returns result once connection is attempted
    - Safe: uses context manager to ensure socket is properly closed
    - Non-blocking: does not hang on connection refused
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def run_mlflow_server(mlflow_port):
    """Start or verify MLflow tracking server for experiment logging.
    
    Checks if MLflow UI server is already running on the specified port.
    If not, starts a new MLflow UI process listening on localhost.
    
    Parameters
    ----------
    mlflow_port : int
        Port number for MLflow UI server (e.g., 5000, 8080).
    
    Returns
    -------
    None
        Prints status messages but returns nothing.
    
    Notes
    -----
    - Host is hardcoded to 127.0.0.1 (localhost)
    - Starts MLflow in non-blocking mode (subprocess)
    - Assumes 'mlflow' command is available in system PATH
    - Multiple calls with same port: only first actually starts server
    - Server can be accessed at http://127.0.0.1:{mlflow_port}
    
    Examples
    --------
    >>> run_mlflow_server(5000)
    MLflow server started at http://127.0.0.1:5000.
    
    >>> run_mlflow_server(5000)  # Second call
    MLflow server is already running at http://127.0.0.1:5000.
    """
    host = "127.0.0.1"
    port = mlflow_port

    if is_port_in_use(host, port):
        print(f"MLflow server is already running at http://{host}:{port}.")
    else:
        print("Starting MLflow server...")
        # subprocess.Popen(["export","MLFLOW_TRACKING_URI=sqlite:///mlruns.db"])
        subprocess.Popen(
            ["mlflow", "ui", "--host", host, "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"MLflow server started at http://{host}:{port}.")


def run_experiment(dataset,methods, param_space_dict_per_method,method_names,experiments,
                   experiment_names,additional_parameters={},MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1,profile_size=None,
                   initial_profile_size=None,postprocessor=DefaultPostProcessor,preprocessor = DefaultPreProcessor,
                   thresholder=ConstantThresholder,mlflow_port=None,debug=True,optimization_param="AD1_AUC",maximize=True, custom_evaluators=None,
                   optimizer: str = 'mango', use_cache: bool = False,
                   log_best_pipeline: bool = True,
                   energy_tracking: str = None, energy_output_dir: str = './energy_runs',
                   energy_study_name: str = None, energy_backend_kwargs: dict = None,
                   energy_phase_label: str = 'SEARCH',
                   energy_extra_dims: dict = None, random_state: int = 42,
                   deterministic: bool = True, raw_preprocessor=None):
    """Execute predictive maintenance anomaly detection experiments with hyperparameter optimization.
    
    Orchestrates complete experiments: constructs pipelines, performs hyperparameter search,
    logs results to MLflow, and returns optimal parameters. Supports multiple methods and
    experiments with cross-product evaluation (each method × each experiment combination).
    
    Parameters
    ----------
    dataset : dict
        Dataset configuration dictionary (typically from Dataset.get_*_dataset() methods).
        Must contain:
        - 'target_data': List of test feature dataframes
        - 'target_sources': List of test source identifiers
        - Additional dataset metadata (see loadAnomalyDetectionDataset.py)
    
    methods : list
        List of instantiated method classes.
        Each method should inherit from MethodInterface.
        Order must correspond to param_space_dict_per_method and method_names.
    
    param_space_dict_per_method : list[dict]
        Hyperparameter search spaces for each method.
        Each dict maps parameter names to lists of candidate values.
        Example: [{'n_estimators': [50, 100], 'max_samples': [256, 512]}]
    
    method_names : list[str]
        Human-readable names for each method (for logging).
        Used in MLflow experiment naming and artifact paths.
    
    experiments : list[type]
        List of experiment class types (not instances) to execute.
        Supported: AutoProfileSemiSupervisedPdMExperiment, IncrementalSemiSupervisedPdMExperiment,
                   UnsupervisedPdMExperiment, SemiSupervisedPdMExperiment,
                   SupervisedPdMExperiment, SupervisedRULPdMExperiment, Supervised_SA_PdMExperiment
        Each method will be evaluated on all experiments (cross-product).
    
    experiment_names : list[str]
        Human-readable names for each experiment (for logging/identification).
    
    additional_parameters : dict, default={}
        Extra hyperparameters for pipeline components (preprocessing, postprocessing, thresholding).
        Key format: '{component}_{param_name}' (e.g., 'postprocessor_window_length', 'preprocessor_features'),
        where component is 'raw_preprocessor', 'preprocessor', 'method', 'postprocessor' or
        'thresholder'. A key reaches the component it starts with, so 'raw_preprocessor_n_bins'
        never reaches the preprocessor.
        Values should be lists of candidate values (for grid search).
    
    MAX_RUNS : int, default=1
        Maximum number of hyperparameter configurations to evaluate per method-experiment pair.
        Higher values allow more thorough exploration but increase computation.
    
    MAX_JOBS : int, default=1
        Number of parallel processes for hyperparameter search (via MANGO).
        Typically 1-8 depending on system CPU cores.
    
    INITIAL_RANDOM : int, default=1
        Number of initial random hyperparameter samples before Bayesian optimization.
        Provides diversity in the exploration phase.
    
    profile_size : list[int], optional
        Candidate sizes (in records) of every profile collected *after* the first
        one in a target source. AutoProfile re-profiles at each reset date; this
        is the size used for those subsequent profiles.
        A single int is accepted and wrapped into a one-element list.
        If None: inferred from the data (see Notes).
        Only used by AutoProfileSemiSupervisedPdMExperiment.
    
    initial_profile_size : list[int], optional
        Candidate sizes (in records) of the *first* profile collected in each
        target source, i.e. the slice from the start of the scenario to its first
        reset date.
        A single int is accepted and wrapped into a one-element list.
        If None: inherits the values of profile_size.
        Only used by AutoProfileSemiSupervisedPdMExperiment.
    
    use_cache : bool, default=False
        Reuse metrics from a previous FINISHED MLflow run whose parameters match,
        instead of re-evaluating that configuration. Matching spans every run in
        the experiment, including earlier invocations under the same experiment
        name, so results then depend on leftover MLflow history. Off by default.
    
    postprocessor : type, default=DefaultPostProcessor
        Post-processing class for score smoothing/normalization.
        Will be instantiated with appropriate parameters during pipeline construction.
        Options: DefaultPostProcessor, MovingAveragePostProcessor, MinMaxPostProcessor, etc.
    
    preprocessor : type, default=DefaultPreProcessor
        Pre-processing class for data preparation/transformation.
        Will be instantiated with appropriate parameters.
        Options: DefaultPreProcessor, FeatureSelector, MinMaxScaler, etc.
    
    raw_preprocessor : type, optional
        Raw-level class that turns each snapshot into one row of features, for datasets
        built from snapshots (``Dataset(data=folder)`` or
        ``Dataset(data={source: folder_or_list_of_dfs})``). It runs before
        ``preprocessor`` and is logged to MLflow as its own step, ``raw_preprocessor``.
        Pass the class, not an instance: every trial builds its own, with parameters
        tuned through ``additional_parameters`` keys ``'raw_preprocessor_<param>'``
        (e.g. ``'raw_preprocessor_n_bins': [10, 20]``).
        Options: StatisticalRawPreProcessor. If None on a snapshot dataset:
        DefaultRawPreProcessor (only accepts single-row snapshots). Raises TypeError
        for an instance, and ValueError on a dataset without snapshots.
    
    thresholder : type, default=ConstantThresholder
        Thresholding class to convert anomaly scores to binary labels.
        Will be instantiated with appropriate parameters.
        Options: ConstantThresholder, SurvToRUL, DynamicThresholder, etc.
    
    mlflow_port : int, optional
        Port number for MLflow tracking UI. If provided, starts/verifies MLflow server.
        If None: skips MLflow setup (no experiment logging).
    
    debug : bool, default=True
        If True: enables verbose logging and debug messages during experiment execution.
        If False: minimal logging, only results and warnings.
    
    optimization_param : str, default="AD1_AUC"
        Metric to optimize during hyperparameter search.
        Options: "AD1_AUC", "AD2_AUC", "avg_time_to_alarm", "false_alarm_rate", etc.
        See evaluation module for complete list of available metrics.
    
    maximize : bool, default=True
        If True: MANGO maximizes optimization_param.
        If False: MANGO minimizes optimization_param.
        Typically True for AUC, F1-score; False for error rate, false alarms.
    
    optimizer : str, default='mango'
        Hyperparameter optimization backend to use.
        Supported values:

        - ``'mango'``: Mango Bayesian optimizer (Gaussian Process surrogate, parallel via joblib).
        - ``'mango_random'``: Mango in pure random-search mode.
        - ``'smac'``: SMAC3 HyperparameterOptimizationFacade (parallel via native workers).
        - ``'gpyopt'``: GPyOpt Bayesian optimization with batch acquisition (parallel via joblib).
          Requires ``pip install pdmlabs[gpyopt]``.
        - ``'hyperopt'``: Hyperopt TPE (sequential only; emits a warning when MAX_JOBS > 1).
          Requires ``pip install pdmlabs[hyperopt]``.
        - ``'optuna'``: Optuna 5 TPE with multivariate mode and constant_liar strategy
          (parallel via joblib + JournalStorage).
          Requires ``pip install pdmlabs[optuna]``.

        The value is also logged as an MLflow parameter per run for traceability.

    log_best_pipeline : bool, default=True
        If True, each experiment ends by cloudpickling its winning pipeline into
        MLflow as a ``Best_Pipeline_Model`` run, reloadable with
        ``mlflow.pyfunc.load_model``. Set False on large sweeps: the artifact
        costs time per experiment and space in the tracking store, and nothing
        else in the run depends on it.
    
    energy_tracking : str, optional
        Energy and carbon measurement backend. ``None`` (default) disables
        tracking entirely with zero overhead and no behaviour change.

        - ``'codecarbon'``: CPU + GPU + RAM energy, kWh and gCO2eq, grid carbon
          intensity. Requires ``pip install pdmlabs[energy]``.
        - ``'rapl'``: raw Intel RAPL counters plus NVML's exact energy counter.
          Highest fidelity, Linux + Intel only.
        - ``'noop'``: wall-clock only.

        .. warning::
           On a kernel that restricts RAPL (the default since CVE-2020-8694)
           CodeCarbon falls back to a TDP x utilisation model **without
           raising**, which makes energy a near-deterministic function of
           runtime and would reduce a cross-optimizer study to "the faster
           backend won". Call :func:`pdmlabs.energy.preflight` first; it gates
           on the measurement source rather than on the look of the numbers.

        Per-trial energy is recorded only when ``MAX_JOBS == 1``. With more
        jobs, concurrent trials share machine-wide counters and cannot be
        attributed individually, so only phase-level energy is recorded.

    energy_output_dir : str, default='./energy_runs'
        Directory for the energy spool. Records are the study data and are never
        deleted automatically.

    energy_study_name : str, optional
        Groups records from several experiments into one study. Defaults to the
        experiment name.

    random_state : int, default=42
        Seed for Python, NumPy and (when present) torch RNGs. Applied to the
        experiment process, forwarded to the HPO backend's own proposal RNG,
        and re-derived per trial from that trial's parameters so it reaches
        worker processes at ``n_jobs > 1`` as well. **Vary it across repeats.**
        Left fixed, repeated runs of the same configuration reproduce the
        identical search, so the spread between them reflects only hardware
        noise -- not the run-to-run variance of the optimizer, which is usually
        the quantity of interest.

    deterministic : bool, default=True
        Disable the cuDNN autotuner and request deterministic torch algorithms
        so that a GPU run reproduces itself. Set False to regain the
        autotuner's speed on convolutional methods (CNN), at the cost of
        run-to-run variation on the same hardware.

    energy_extra_dims : dict, optional
        Extra key/value pairs stamped onto every energy record, e.g.
        ``{'repeat_idx': 2}``. Used by
        :func:`pdmlabs.energy.study.run_energy_study` to make repeats
        distinguishable in the exported table.

    energy_backend_kwargs : dict, optional
        Backend options, e.g. ``{'country_iso_code': 'GRC',
        'measure_power_secs': 1.0, 'gpu_ids': [0]}``.

    Returns
    -------
    list[dict]
        Best hyperparameters for each method-experiment combination (in execution order).
        Each dict maps parameter names to optimal values discovered by MANGO.
        Length = len(methods) × len(experiments)
    
    Pipeline Construction:
        For each method-experiment pair:
        1. Create PdMPipeline with: preprocessor → method → postprocessor → thresholder
        2. Configure AUC resolution=30 (granularity of ROC curve)
        3. Assign experiment type (Supervised/Unsupervised/SemiSupervised)
        4. Build search space: pipeline params + method params + additional params
    
    Constraint Functions:
        Different experiments apply parameter constraints:
        - AutoProfileSemiSupervisedPdMExperiment: max_wait_time constraints
        - IncrementalSemiSupervisedPdMExperiment: incremental + max_wait constraints
        - UnsupervisedPdMExperiment: SAND or distance-based constraints (KNN) or max_wait constraints
    
    Examples
    --------
    >>> from pdmlabs.RunExperiment import run_experiment
    >>> from pdmlabs.utils.automatic_parameter_generation import online_technique
    >>> from pdmlabs.method.unsupervised_method import IF
    
    >>> # Load dataset
    >>> dataset_obj = Dataset(data, 'timestamp', failure_column='failure')
    >>> train_data, val_data = dataset_obj.get_rul_dataset()
    
    >>> # Configure experiment
    >>> methods = [IF]
    >>> param_spaces = [online_technique('IF', maximum_profile=500)]
    >>> best_params = run_experiment(
    ...     dataset=train_data,
    ...     methods=methods,
    ...     param_space_dict_per_method=param_spaces,
    ...     method_names=['IF'],
    ...     experiments=[UnsupervisedPdMExperiment],
    ...     experiment_names=['Online'],
    ...     MAX_RUNS=20,
    ...     MAX_JOBS=4,
    ...     INITIAL_RANDOM=2,
    ...     mlflow_port=5000,
    ...     optimization_param='AD1_AUC'
    ... )
    >>> print(best_params[0])  # Best parameters for IF in Online experiment
    
    Notes
    -----
    - **Important**: initial_profile_size defaults to profile_size if not specified.
      When both are None, profile_size is inferred from the target data as
      ``max_wait = ceil(min_target_scenario_len / 3)``, then spread over 16 integer
      candidates in ``[max_wait // 10, max_wait]`` (see
      :func:`pdmlabs.utils.automatic_parameter_generation.profile_values`).
      initial_profile_size then inherits those values.
    - MANGO parameters (num, jobs, initial_random) calculated from space size and MAX_RUNS
    - Constraint functions prevent invalid hyperparameter combinations
    - MLflow artifacts saved to: ./artifacts/{experiment_name} artifacts/
    - All experiments execute in sequence (not parallel), but internal MANGO uses parallelization
    - Method-experiment cross-product: if 2 methods × 3 experiments = 6 total runs
    """


    profile_size = _coerce_profile_sizes(profile_size, 'profile_size')
    initial_profile_size = _coerce_profile_sizes(initial_profile_size, 'initial_profile_size')

    # Two independent rules, applied in order. Together they cover all four
    # combinations: both unset -> infer then inherit; only the initial one unset ->
    # inherit; only profile_size unset -> infer and keep the supplied initial one.
    if profile_size is None:
        profile_size = _infer_profile_sizes(dataset)
    if initial_profile_size is None:
        initial_profile_size = profile_size

    all_experiments_best_parameters = []
    for current_method, current_method_param_space, current_method_name in zip(methods, param_space_dict_per_method,
                                                                               method_names):


        if mlflow_port is not None:
            run_mlflow_server(mlflow_port)
        for experiment, experiment_name in zip(experiments, experiment_names):
            current_param_space_dict = {

            }


            # Survival Analysis turns survival curves into RUL times *through* the
            # thresholder. ConstantThresholder cannot do that: it returns
            # [threshold_value] * len(scores), which is [None, ...] by default, and the
            # SA evaluator then feeds those Nones to mean_squared_error and fails with
            # "Input contains NaN". Default SA to SurvToRUL; an explicit thresholder=
            # from the caller still wins.
            current_thresholder = thresholder
            if experiment == Supervised_SA_PdMExperiment and thresholder is ConstantThresholder:
                current_thresholder = SurvToRUL

            my_pipeline = PdMPipeline(
                steps={
                    'raw_preprocessor': raw_preprocessor,
                    'preprocessor': preprocessor,
                    'method': current_method,
                    'postprocessor': postprocessor,
                    'thresholder': current_thresholder,
                },
                dataset=dataset,
                auc_resolution=100,
                experiment_type=get_method_type(experiment)
            )
            # Only AutoProfile reads these. Injecting them everywhere would add
            # search dimensions the other flavors never consume, inflating the
            # optimizer budget for nothing.
            if experiment in [AutoProfileSemiSupervisedPdMExperiment]:
                current_param_space_dict['profile_size'] = profile_size
                current_param_space_dict['initial_profile_size'] = initial_profile_size

            for key, value in current_method_param_space.items():
                current_param_space_dict[f'method_{key}'] = value
            for key, value in additional_parameters.items():
                current_param_space_dict[key] = value

            budget = calculate_optimizer_budget(
                optimizer, current_param_space_dict, MAX_JOBS, INITIAL_RANDOM, MAX_RUNS
            )
            num, jobs, initial_random = budget['n_iterations'], budget['n_jobs'], budget['initial_random']
            constraint_function=None
            if experiment in [AutoProfileSemiSupervisedPdMExperiment]:
                constraint_function = auto_profile_max_wait_time_constraint(my_pipeline)
            elif experiment in [IncrementalSemiSupervisedPdMExperiment]:
                constraint_function = combine_constraint_functions(incremental_max_wait_time_constraint(my_pipeline),incremental_constraint_function)
            elif experiment in [UnsupervisedPdMExperiment]:
                constraint_function = sand_parameters_constraint_function(my_pipeline) if 'SAND' == current_method_name else combine_constraint_functions(unsupervised_distance_based, unsupervised_max_wait_time_constraint(my_pipeline)) if 'KNN' == current_method_name else unsupervised_max_wait_time_constraint(my_pipeline)

            my_experiment = experiment(
                experiment_name=experiment_name + ' ' + current_method_name,
                target_data=dataset['target_data'],
                target_sources=dataset['target_sources'],
                pipeline=my_pipeline,
                param_space=current_param_space_dict,
                num_iteration=num,
                n_jobs=jobs,
                initial_random=initial_random,
                artifacts='./artifacts/' + experiment_name + ' artifacts',
                constraint_function=constraint_function,
                debug=debug,
                optimization_param=optimization_param,
                maximize=maximize,
                custom_evaluators=custom_evaluators,
                optimizer=optimizer,
                use_cache=use_cache,
                log_best_pipeline=log_best_pipeline,
                energy_tracking=energy_tracking,
                energy_output_dir=energy_output_dir,
                energy_study_name=energy_study_name,
                energy_backend_kwargs=energy_backend_kwargs,
                energy_phase_label=energy_phase_label,
                energy_extra_dims=energy_extra_dims,
                random_state=random_state,
                deterministic=deterministic
            )


            best_params = my_experiment.execute()
            print(experiment_name)
            print(best_params)
            all_experiments_best_parameters.append(best_params)
    return all_experiments_best_parameters