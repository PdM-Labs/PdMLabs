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
from pdmlabs.constraint_functions.constraint import auto_profile_max_wait_time_constraint
from pdmlabs.utils.utils import calculate_mango_parameters

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
import socket
import subprocess


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
        subprocess.Popen(["mlflow", "ui", "--host", host, "--port", str(port)])
        print(f"MLflow server started at http://{host}:{port}.")


def run_experiment(dataset,methods, param_space_dict_per_method,method_names,experiments,
                   experiment_names,additional_parameters={},MAX_RUNS=1, MAX_JOBS=1, INITIAL_RANDOM=1,profile_size=2,
                   fit_size=None,postprocessor=DefaultPostProcessor,preprocessor = DefaultPreProcessor,
                   thresholder=ConstantThresholder,mlflow_port=None,debug=True,optimization_param="AD1_AUC",maximize=True, custom_evaluators=None):
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
        Key format: '{component}_{param_name}' (e.g., 'postprocessor_window_length', 'preprocessor_features').
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
    
    profile_size : int or list[int], default=2
        Historical buffer size (number of samples) used by online methods.
        If list: allows multiple buffer sizes to be tested.
        For online/streaming evaluation: how much historical data to keep.
    
    fit_size : int or list[int], optional
        Initial profile size ("warm-up" buffer) before evaluation begins.
        If None: defaults to profile_size.
        Used in AutoProfile and Incremental experiments for initialization.
    
    postprocessor : type, default=DefaultPostProcessor
        Post-processing class for score smoothing/normalization.
        Will be instantiated with appropriate parameters during pipeline construction.
        Options: DefaultPostProcessor, MovingAveragePostProcessor, MinMaxPostProcessor, etc.
    
    preprocessor : type, default=DefaultPreProcessor
        Pre-processing class for data preparation/transformation.
        Will be instantiated with appropriate parameters.
        Options: DefaultPreProcessor, FeatureSelector, MinMaxScaler, etc.
    
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
    - **Important**: fit_size defaults to profile_size if not specified
    - MANGO parameters (num, jobs, initial_random) calculated from space size and MAX_RUNS
    - Constraint functions prevent invalid hyperparameter combinations
    - MLflow artifacts saved to: ./artifacts/{experiment_name} artifacts/
    - All experiments execute in sequence (not parallel), but internal MANGO uses parallelization
    - Method-experiment cross-product: if 2 methods × 3 experiments = 6 total runs
    """


    if fit_size is None:
        fit_size=profile_size
    all_experiments_best_parameters = []
    for current_method, current_method_param_space, current_method_name in zip(methods, param_space_dict_per_method,
                                                                               method_names):


        if mlflow_port is not None:
            run_mlflow_server(mlflow_port)
        for experiment, experiment_name in zip(experiments, experiment_names):
            current_param_space_dict = {

            }


            my_pipeline = PdMPipeline(
                steps={
                    'preprocessor': preprocessor,
                    'method': current_method,
                    'postprocessor': postprocessor,
                    'thresholder': thresholder,
                },
                dataset=dataset,
                auc_resolution=30,
                experiment_type=get_method_type(experiment)
            )
            if isinstance(profile_size, list):
                current_param_space_dict['profile_size'] = profile_size
            else:
                current_param_space_dict['profile_size'] = [profile_size]
            if isinstance(fit_size, list):
                current_param_space_dict['init_profile_size'] = fit_size
            else:
                current_param_space_dict['init_profile_size'] = [fit_size]

            for key, value in current_method_param_space.items():
                current_param_space_dict[f'method_{key}'] = value
            for key, value in additional_parameters.items():
                current_param_space_dict[key] = value

            num, jobs, initial_random = calculate_mango_parameters(current_param_space_dict, MAX_JOBS, INITIAL_RANDOM,
                                                                   MAX_RUNS)
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
                custom_evaluators=custom_evaluators
            )


            best_params = my_experiment.execute()
            print(experiment_name)
            print(best_params)
            all_experiments_best_parameters.append(best_params)
    return all_experiments_best_parameters