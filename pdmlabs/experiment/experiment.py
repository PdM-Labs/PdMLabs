import contextlib
import logging
import os
import math
import abc

import random
import re
from typing import Callable
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import uuid

from pdmlabs.pipeline.pipeline import PdMPipeline, STEP_NAMES, split_step_params
from pdmlabs.evaluation.evaluation import AUCPR_new as pdm_evaluate
from pdmlabs.evaluation.evaluation import AUCPR_ranges_new as pdm_evaluate_ranges
from pdmlabs.utils.seeding import set_global_seed, seed_for_params, seeded_objective
from pdmlabs.utils.rul_transformations import hard_transform_survival, softmax_distance_survival_batch, \
    sigmoid_survival_batch

logging.basicConfig(level=logging.INFO)


def process_data(current_data, header, data_type) -> list[pd.DataFrame]:
    """Process and normalize data inputs to a standardized format.

    Converts various input formats (DataFrame, CSV file/directory, or list) into 
    a list of DataFrames for consistent handling throughout the framework.

    Args:
        current_data: The data to process. Can be:
            - pd.DataFrame: Single DataFrame, wrapped in a list
            - str: Path to single CSV file or directory containing CSV files
            - list: List of DataFrames
        header (str or int): Row number(s) to use as column names when reading CSV. 
            Passed directly to pd.read_csv. Use 'infer' for automatic detection.
        data_type (str): Name of the parameter (for error messages).

    Returns:
        list[pd.DataFrame]: List of DataFrames ready for processing.

    Raises:
        Exception: If input type is not supported or list contains non-DataFrame elements.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> result = process_data(df, 'infer', 'test_data')
        >>> print(type(result)); print(len(result))
        <class 'list'>
        1

        >>> result = process_data('/path/to/data.csv', 'infer', 'test_data')
        >>> print(type(result[0]))
        <class 'pandas.core.frame.DataFrame'>

        >>> result = process_data([df, df], 'infer', 'test_data')
        >>> print(len(result))
        2
    """
    if len(current_data) == 0:
        return current_data
    if isinstance(current_data, pd.DataFrame):
        result = [current_data]
    elif isinstance(current_data, str):
        # if it is a string check if it is a csv file or directory containing csv files
        if current_data.endswith('.csv'):
            result = [pd.read_csv(current_data, header=header)]
        elif Path(current_data).is_dir():
            result = []

            current_directory_files = os.listdir(current_data)
            current_csv_files = [file for file in current_directory_files if file.endswith('.csv')]
            for csv_file in current_csv_files:
                current_csv_file_path = os.path.join(current_data, csv_file)
                result.append(pd.read_csv(current_csv_file_path, header=header))
    elif isinstance(current_data, list):
        result = current_data
        # necessarily nested in order to avoid exception when looping on a variable that is not a list because python does not support short-circuit evaluation
        if not all(isinstance(item, pd.DataFrame) for item in current_data):
            raise Exception(f'Some element of the list parameter \'{data_type}\' has unsupported type')
    else:
        raise Exception(f'Not supported type {type(current_data)} for parameter \'{data_type}\'')

    return result


class PdMExperiment(abc.ABC):
    """Base abstract class for all predictive maintenance experiment flavors.

    This class orchestrates the automated execution of anomaly detection experiments
    using Bayesian optimization (via Mango) to search parameter spaces and MLflow
    for run tracking and reproducibility.

    An experiment combines a PdMPipeline (which defines the processing steps) with
    a parameter space to search over. It performs hyperparameter optimization by:

    1. Registering an MLflow experiment
    2. Running objective evaluations with different parameter combinations
    3. Training, predicting, and evaluating across train/test splits
    4. Returning the best found parameters and their performance metrics

    Concrete implementations (e.g., AutoProfileSemiSupervisedPdMExperiment, 
    SupervisedPdMExperiment) override the abstract execute() method to implement
    experiment-specific logic (e.g., semi-supervised, supervised, RUL prediction).

    Attributes:
        experiment_name (str): Name of the experiment (MLflow experiment identifier).
        pipeline (PdMPipeline): Pipeline defining dataset, preprocessing, method, 
            postprocessing, and thresholding steps.
        param_space (dict): Parameter space for Mango optimization.
            Keys are parameter names (e.g., 'method_alpha', 'preprocessor_scale'),
            values are parameter ranges.
        optimization_param (str): Metric to optimize ('AD1_AUC', 'AD2_AUC', 'AD3_AUC', etc).
        initial_random (int): Number of initial random exploration steps before Bayesian optimization.
        num_iteration (int): Total number of optimization iterations.
        n_jobs (int): Number of parallel jobs for optimization.
        random_state (int): Random seed for reproducibility.
        maximize (bool): Whether to maximize (True) or minimize (False) optimization_param.
        debug (bool): If True, generates debug plots and logs them to MLflow.
        event_data: Event mappings from the pipeline (failures, resets, sources).

    Raises:
        ValueError: If required dataset keys are missing (e.g., 'anomaly_labels' for supervised).
        IncompatibleMethodException: If the selected method is incompatible with the experiment flavor.
    """
    def __init__(self,
                 experiment_name: str,
                 pipeline: PdMPipeline,
                 param_space: dict,
                 constraint_function: Callable = None,
                 target_data: list[pd.DataFrame] = None,
                 # TODO str for directory with csv files for each scenario or single csv file of one scenario
                 target_sources: list[str] = None,
                 historic_data: list[pd.DataFrame] = [],
                 # TODO str for directory with csv files for each scenario or single csv file of one scenario
                 historic_sources: list[str] = [],
                 optimization_param: str = 'AD1_AUC',
                 initial_random: int = 2,
                 num_iteration: int = 20,
                 batch_size: int = 1,
                 n_jobs: int = 1,
                 random_state: int = 42,
                 random_n_tries: int = 3,
                 constraint_max_retries: int = 10,
                 historic_data_header: str = 'infer',
                 target_data_header: str = 'infer',
                 artifacts: str = 'artifacts',
                 debug: bool = False,
                 delay: float = None,  # in milliseconds
                 log_best_scores: bool = False,
                 log_best_pipeline: bool = True,
                 maximize: bool = True,
                 custom_evaluators: list = None,
                 optimizer: str = 'mango',
                 use_cache: bool = False,
                 energy_tracking: str = None,
                 energy_output_dir: str = './energy_runs',
                 energy_study_name: str = None,
                 energy_backend_kwargs: dict = None,
                 energy_phase_label: str = 'SEARCH',
                 energy_extra_dims: dict = None,
                 deterministic: bool = True
                 ):
        """Initialize a PdM experiment with dataset, pipeline, and optimization settings.

        Args:
            experiment_name (str): Human-readable name for this experiment (used in MLflow).
            pipeline (PdMPipeline): PdMPipeline instance defining the data and processing steps.
            param_space (dict): Parameter search space for Mango. Example: 
                {'method_alpha': [0.1, 1.0], 'preprocessor_scale': [True, False]}.
            constraint_function (callable, optional): Function checking parameter validity.
                Should return True if parameters are valid, False otherwise. Defaults to None.
            target_data (list[pd.DataFrame], optional): Test/target data. Extracted from pipeline 
                if not provided. Defaults to None.
            target_sources (list[str], optional): Source labels for target data. Extracted from 
                pipeline if not provided. Defaults to None.
            historic_data (list[pd.DataFrame], optional): Training/historic data. Extracted from 
                pipeline if not provided. Defaults to [].
            historic_sources (list[str], optional): Source labels for historic data. Extracted 
                from pipeline if not provided. Defaults to [].
            optimization_param (str): Metric to optimize. Must be one of 
                'AD1_AUC', 'AD2_AUC', 'AD3_AUC', 'AD1_f1', 'AD2_f1', 'AD3_f1', 
                'AD1_rcl', 'AD2_rcl', 'AD3_rcl', 'prc', or 'VUS_*'. Defaults to 'AD1_AUC'.
            initial_random (int): Number of initial random parameter samples before Bayesian 
                optimization kicks in. Defaults to 2.
            num_iteration (int): Total iterations for Mango optimization. Defaults to 20.
            batch_size (int): Batch size for optimization (currently unused). Defaults to 1.
            n_jobs (int): Number of parallel jobs/processes for optimization. Defaults to 1.
            random_state (int): Seed for random number generators (numpy, torch, Python).
                Propagated to the optimizer backend and re-applied inside every
                trial, so it reaches worker processes too. Defaults to 42.
            deterministic (bool): Disable the cuDNN autotuner and request
                deterministic torch algorithms, so a GPU run reproduces itself.
                Set False to regain the autotuner's speed. Defaults to True.
            random_n_tries (int): Number of random tries for constraint satisfaction 
                (unused). Defaults to 3.
            constraint_max_retries (int): Max retries for constraint-respecting sampling 
                (unused). Defaults to 10.
            historic_data_header (str): Row numbers for column names in historic CSV files. 
                Passed to pd.read_csv. Use 'infer' for automatic. Defaults to 'infer'.
            target_data_header (str): Row numbers for column names in target CSV files. 
                Passed to pd.read_csv. Use 'infer' for automatic. Defaults to 'infer'.
            artifacts (str): Directory to save MLflow model artifacts. Defaults to 'artifacts'.
            debug (bool): If True, generates debug plots (PR curves, anomaly scores, RUL) 
                and logs them to MLflow. Defaults to False.
            delay (float, optional): Delay in milliseconds between processing target sources 
                (for rate-limiting). Defaults to None.
            log_best_scores (bool): If True, logs best-run anomaly scores to MLflow artifacts. 
            maximize (bool): Whether to maximize (True) or minimize (False) optimization_param. 
            log_best_pipeline (bool): If True (default), ``execute()`` ends by
                logging the winning pipeline to MLflow as a ``Best_Pipeline_Model``
                run, which makes it reloadable with ``mlflow.pyfunc.load_model``.
                Set False to skip it: the pipeline is cloudpickled whole, so on a
                large hyperparameter sweep it costs noticeable time per experiment
                and a lot of space in the tracking store. Metrics, parameters and
                debug plots are unaffected either way.
            custom_evaluators (list, optional): List of EvaluatorInterface objects for custom metrics.
            optimizer (str, optional): HPO backend name. Defaults to 'mango'.
            use_cache (bool, optional): Reuse metrics from a matching previous FINISHED
                MLflow run instead of re-evaluating that configuration. Matching spans
                earlier invocations under the same experiment name. Defaults to False.
        Raises:
            Exception: If an unsupported data type is passed for historic/target data.

        Examples:
            >>> from pdmlabs.pipeline.pipeline import PdMPipeline
            >>> from pdmlabs.experiment.batch.semi_supervised_experiment import (
            ...     SemiSupervisedPdMExperiment
            ... )
            >>> pipeline = PdMPipeline(
            ...     dataset=dataset_dict,
            ...     method=MyMethod,
            ...     preprocessor=NoPreprocessor,
            ...     postprocessor=NoPostprocessor,
            ...     thresholder=StaticThreshold
            ... )
            >>> param_space = {'method_alpha': [0.1, 1.0]}
            >>> experiment = SemiSupervisedPdMExperiment(
            ...     experiment_name='my-experiment',
            ...     pipeline=pipeline,
            ...     param_space=param_space,
            ...     num_iteration=20,
            ...     n_jobs=4
            ... )
            >>> results = experiment.run_experiment()
        """
        self.experiment_name = experiment_name
        # TODO target and historic data and sources parameter should be removed, became default parameters for backwards compatibility
        self.historic_data = pipeline.dataset['historic_data']
        self.historic_sources = pipeline.dataset['historic_sources']
        self.target_data = pipeline.dataset['target_data']
        self.target_sources = pipeline.dataset['target_sources']
        self.pipeline = pipeline
        self.param_space = param_space
        self.optimization_param = optimization_param
        self.initial_random = initial_random
        self.num_iteration = num_iteration
        self.maximize = maximize
        # self.batch_size = batch_size currently commented out because of using only scheduler.parallel, more info on issue #97 on Mango - alternatives include using only scheduler.parallel or letting the user decide depending on his hardware
        self.n_jobs = n_jobs
        self.optimizer_name = optimizer
        self.use_cache = use_cache
        self.energy_tracking = energy_tracking
        self.energy_output_dir = energy_output_dir
        self.energy_study_name = energy_study_name
        self.energy_backend_kwargs = energy_backend_kwargs or {}
        self.energy_phase_label = energy_phase_label
        self.energy_extra_dims = energy_extra_dims or {}
        self._energy_backend = None
        self._energy_sink = None
        self._energy_setup()
        self.random_state = random_state
        self.deterministic = deterministic
        self.historic_data_header = historic_data_header
        self.target_data_header = target_data_header
        self.artifacts = artifacts

        self.debug = debug
        self.delay = delay
        self.custom_evaluators = custom_evaluators if custom_evaluators else []

        self.log_best_scores = log_best_scores
        self.log_best_pipeline = log_best_pipeline
        current_uuid = uuid.uuid4()
        self.lock_file_path = f'pdm_evaluation_framework_lock_file_{current_uuid}.lock'
        self.best_scores_info_dict_path = f'best_scores_info_{current_uuid}.pkl'

        self.event_data = self.pipeline.event_data
        self.constraint_function = constraint_function

        # TODO the next line is probably useless
        Path(self.artifacts).mkdir(parents=True, exist_ok=True)
        self.extra_metrics = {}
        # Set here rather than only inside optimization_objective: a second
        # execute() on the same object would otherwise re-log the first run's
        # pipeline whenever the second produces none.
        self.best_pipeline = None
        # process historic data
        self.historic_data = process_data(self.historic_data, historic_data_header, 'historic_data')

        # process target data
        self.target_data = process_data(self.target_data, target_data_header, 'target_data')

        self.experiment_id = None

        # One call seeds random / numpy / torch (CPU and every CUDA device) and
        # turns off the cuDNN autotuner. Trials reseed again from their own
        # parameters -- see _instrument_objective -- because at n_jobs > 1 they
        # run in worker processes that never see this one.
        set_global_seed(self.random_state, deterministic=self.deterministic)

        # current_dir = os.getcwd()
        # os.chdir("./src/pdmlabs/evaluation/RBPR_official")

        # Run make clean

        # if os.name == 'nt':
        #     powershell_command = "(Get-Content Makefile) -replace 'rm', 'del' | Out-File -encoding ASCII Makefile"
        #     subprocess.run(["powershell", "-Command", powershell_command])
        #     subprocess.run(["powershell", "-Command", "make", "clean"])
        #     subprocess.run(["powershell", "-Command", "make"])
        # else:
        #     subprocess.call(["make","-f","MakefileL", "clean"])
        #     subprocess.call(["make","-f","MakefileL"])
        #     # Move the evaluate executable to the parent directory
        #     subprocess.call(["mv", "evaluate", ".."])
        # Run make

        # Change back to the original directory
        # os.chdir(current_dir)

    # ------------------------------------------------------------------ #
    # Energy accounting
    #
    # Entirely opt-in: with ``energy_tracking=None`` (the default) every helper
    # below short-circuits, no backend is constructed and no record is written,
    # so existing callers are bit-for-bit unaffected.
    # ------------------------------------------------------------------ #

    def __getstate__(self):
        """Never let a live energy backend cross a process boundary.

        With ``n_jobs > 1`` the trial objective is a closure over ``self``, so
        the experiment is pickled into each loky/dask worker. A CodeCarbon
        backend owns a sampling thread and open file handles and is not
        picklable -- without this the whole run dies with
        ``PicklingError: Could not pickle the task to send it to the workers``.

        Dropping it is also exactly the behaviour the design wants: a worker
        that finds no backend records wall-clock only with
        ``attribution="phase_only"``, which is correct, because concurrent
        trials share machine-wide counters and cannot be attributed
        individually. The sink survives -- it holds only strings.
        """
        state = self.__dict__.copy()
        state['_energy_backend'] = None
        return state

    def _energy_setup(self):
        """Construct the energy backend and spool, if tracking is enabled."""
        if not self.energy_tracking or self.energy_tracking == 'noop':
            return
        try:
            from pdmlabs.energy import get_energy_backend
            from pdmlabs.energy.sink import EnergySink
            study = self.energy_study_name or self.experiment_name
            self._energy_sink = EnergySink.create(study, base_dir=self.energy_output_dir)
            self._energy_backend = get_energy_backend(
                self.energy_tracking, **self.energy_backend_kwargs)
        except Exception as exc:
            logging.warning('Energy tracking could not be initialised (%s). '
                            'The experiment will run WITHOUT energy records.', exc)
            self._energy_backend, self._energy_sink = None, None

    def _energy_dims(self):
        """Dimensions stamped on every energy record from this experiment."""
        dims = {
            'flavor': type(self).__name__,
            'experiment_name': self.experiment_name,
            'optimizer': self.optimizer_name,
            'n_jobs': self.n_jobs,
            'num_iteration': self.num_iteration,
            'optimization_param': self.optimization_param,
            'maximize': bool(self.maximize),
            'use_cache': bool(self.use_cache),
            # Seeding is a first-class dimension: repeats that do not vary it
            # measure hardware noise rather than search variance.
            'random_state': self.random_state,
        }
        dims.update(self.energy_extra_dims)
        # Joins a SEARCH phase to exactly its own trials, however many repeats
        # or re-runs share the study directory (set per _run_optimizer call).
        if getattr(self, '_energy_phase_id', None):
            dims['phase_id'] = self._energy_phase_id
        return dims

    def _energy_record_reading(self, kind, label, reading, **extra):
        if self._energy_sink is None or reading is None:
            return
        try:
            rec = dict(kind=kind, phase=label, duration_s=reading.duration_s,
                       cpu_j=reading.cpu_j, gpu_j=reading.gpu_j, ram_j=reading.ram_j,
                       total_j=reading.total_j, co2e_g=reading.co2e_g,
                       power_source_cpu=reading.power_source_cpu,
                       power_source_gpu=reading.power_source_gpu,
                       ram_source=reading.ram_source, backend=reading.backend,
                       quality=reading.quality, is_measured=reading.is_measured,
                       carbon_intensity_g_per_kwh=reading.carbon_intensity_g_per_kwh)
            rec.update(self._energy_dims())
            rec.update(extra)
            self._energy_sink.record(**rec)
        except Exception as exc:
            logging.warning('Could not spool the %s energy record: %s', label, exc)

    @contextlib.contextmanager
    def _energy_phase(self, label):
        """Measure one lifecycle phase and spool it.

        Registers the backend as this process's active one for the duration, so
        the trial decorator records against the *same* instance that has the
        phase open. That is what makes ``sum(trial energy) <= phase energy``
        hold, and hence what makes ``E_search - sum(E_trial)`` a meaningful
        measure of optimizer overhead rather than a difference of two unrelated
        integrals.
        """
        if self._energy_backend is None:
            yield None
            return
        from pdmlabs.energy import context as energy_context
        energy_context.set_active_backend(self._energy_backend)
        cell = []
        try:
            with self._energy_backend.phase(label) as cell:
                yield cell
        finally:
            energy_context.set_active_backend(None)
            if cell:
                self._energy_record_reading('phase', label, cell[0])

    def _energy_bind_run(self, parent_run):
        """Join this trial's energy record to its MLflow run, both directions.

        Never raises: it is the first statement of ``_finish_run`` and must not
        be able to break trial teardown.
        """
        if self._energy_sink is None:
            return
        try:
            from pdmlabs.energy import context as energy_context
            trial_id = energy_context.bind_mlflow_run(parent_run)
            if trial_id:
                mlflow.log_param('energy_trial_id', trial_id)
        except Exception as exc:
            logging.warning('Could not bind the energy record to the MLflow run: %s', exc)

    def _instrument_objective(self, objective_fn):
        """Wrap the trial objective with per-trial energy accounting.

        Applied at the single dispatch point in :meth:`_run_optimizer`, so one
        wrapper covers all six optimizer backends without any adapter knowing
        that energy exists.

        At ``n_jobs > 1`` trials run in loky/dask worker processes. Those
        workers never have an active backend registered (it is keyed by pid and
        set only in the main process), so each trial there degrades to
        wall-clock with ``attribution="phase_only"``. That is deliberate:
        concurrent trials share machine-wide counters, so attributing energy to
        one of them individually would inflate every number by roughly
        ``n_jobs`` while still looking entirely plausible. Phase-level energy
        stays valid at any ``n_jobs``.
        """
        if self._energy_sink is None:
            return objective_fn
        if self.n_jobs > 1:
            logging.warning(
                'Per-trial energy attribution is disabled because n_jobs=%d. '
                'Concurrent trials share machine-wide energy counters, so '
                'per-trial numbers would multiply-count. Phase-level energy is '
                'still measured. Use n_jobs=1 for per-trial energy.', self.n_jobs)
        from pdmlabs.energy.instrument import measured_trial
        return measured_trial(objective_fn, self._energy_sink,
                              phase=self.energy_phase_label, dims=self._energy_dims())

    @abc.abstractmethod
    def execute(self) -> dict:
        """Execute the parameter optimization loop and return results.

        This method must be implemented by subclasses to define experiment-specific
        logic (e.g., semi-supervised, supervised, RUL prediction). It typically:

        1. Uses Mango tuner to search the parameter space
        2. For each parameter combination:
           - Creates pipeline components (method, preprocessor, postprocessor, thresholder)
           - Fits on historic/training data
           - Predicts on target/test data
           - Evaluates using PdM metrics
        3. Returns the best parameters and their performance

        Returns:
            dict: Result dictionary containing:
                - 'best_params': dict of best found parameters
                - 'best_objective': best optimization metric value
                - 'th': best threshold value
                - Additional experiment-specific results (e.g., 'per_method' for batch flavors)

        Raises:
            NotImplementedError: This is an abstract method and must be overridden.

        Examples:
            See subclasses like SemiSupervisedPdMExperiment.execute() for concrete examples.
        """
        pass

    def _register_experiment(self) -> None:
        """Register or retrieve an MLflow experiment.

        Creates a new MLflow experiment with the specified experiment_name, or retrieves
        an existing one if it already exists. Sets self.experiment_id for use in MLflow runs.

        The experiment ID is used to group all runs (parameter combinations) for this 
        experiment together in MLflow's UI.

        Raises:
            mlflow.exceptions.MlflowException: If MLflow server is unavailable.

        Examples:
            >>> experiment._register_experiment()
            >>> print(experiment.experiment_id)  # ID of the MLflow experiment
            123456789
        """
        # if self.delay is not None:
        #     print(f'Cooldown for {self.delay} milliseconds')
        #     time.sleep(self.delay / 1000)
        try:
            self.experiment_id = mlflow.create_experiment(name=self.experiment_name)
        except Exception as e:
            logging.warning(
                f'Experiment with experiment name \'{self.experiment_name}\' already exists. Be careful if you are sure about including your run in this experiment.')
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id

    def _inner_plot(self, color, rangearay, datesofscores, minvalue, maxvalue, label):
        """Add a shaded region plot to the current matplotlib figure.

        Helper for plotting anomaly detection results with multiple regions (e.g.,
        predictive horizon, lead time). Fills an area between min/max where rangearay is True.

        Args:
            color (str): Matplotlib color name or hex code (e.g., 'red', '#FF0000').
            rangearay (array-like): Boolean array indicating where to shade.
            datesofscores (array-like): X-axis values (timestamps/indices).
            minvalue (float): Bottom boundary of shaded region.
            maxvalue (float): Top boundary of shaded region.
            label (str): Legend label for this shaded region.

        Examples:
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> datesofscores = np.arange(100)
            >>> rangearay = datesofscores > 50
            >>> experiment._inner_plot('red', rangearay, datesofscores, 0, 1, 'anomaly window')
            >>> plt.show()
        """
        plt.fill_between(datesofscores, minvalue, maxvalue, where=rangearay, color=color,
                         alpha=0.3, label=label)

    def _plot_SA(self, plot_dictionary) -> None:
        """Generate and log survival analysis (SA) debug plots to MLflow.

        Visualizes RUL (remaining useful life) predictions and labels for survival analysis
        experiments. Creates a 3x3 grid of subplots, logs groups of 9 to MLflow.

        Only executed if self.debug is True.

        Args:
            plot_dictionary (dict): Dictionary with keys as source names and values as dicts
                containing 'scores' and 'labels' arrays for visualization.

        Examples:
            >>> plot_dict = {
            ...     'bearing_1': {'scores': [[10, 20], ...], 'labels': [[5, 1], ...]},
            ...     'bearing_2': {'scores': [[15, 25], ...], 'labels': [[7, 0], ...]}
            ... }
            >>> experiment._plot_SA(plot_dict)
            # Logs scores_0_3.png, scores_3_6.png, etc. to MLflow
        """
        # plot_rul_dictionary[current_target_source]={"scores":processed_target_scores,"labels":current_labels,"thresholds":None,"index":current_dates}
        if self.debug:
            plt.figure(figsize=(20, 20))
            counter = 0
            size = 3
            globalcounter = 0
            namescount = -1

            # plot_rul_dictionary[current_target_source]={"scores":processed_target_scores,"labels":current_labels,"thresholds":None,"index":current_dates}
            for key in plot_dictionary.keys():
                counter += 1
                globalcounter += 1
                if globalcounter > 30:
                    break
                plt.subplot(size * 100 + 10 + counter)

                self._plot_SA_inner(plot_dictionary[key]["scores"], plot_dictionary[key]["labels"])
                if counter == size:
                    namescount += 1
                    mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                    plt.clf()
                    counter = 0
            if counter > 0:
                namescount += 1
                mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                plt.clf()
                counter = 0

    def _plot_SA_inner(self, scores, labels):
        """Plot utility for visualizing individual survival analysis trajectories.

        Draws lines for score sequences and scatter points at label locations,
        color-coded by failure/non-failure status.

        Args:
            scores (list): List of (values, indices) tuples for plotting.
            labels (list): List of (label_value, failure_flag) tuples where failure_flag 
                is 1 for failure, 0 for normal.
        """
        pivcounter = -1
        pivot = 10
        cmap = matplotlib.colormaps['coolwarm']
        # Pick two colors (not the edges, e.g., 0.25 and 0.75)
        color1 = cmap(0.15)
        color2 = cmap(0.75)
        for score_length, lab in zip(scores, labels):
            pivcounter += 1
            if pivcounter % pivot != 0:
                continue
            plt.plot(score_length[1], score_length[0], color=color1)

            closest_pos = np.argmin(np.abs(np.array(score_length[1]) - lab[0]))
            color = "red" if lab[1] == 1 else "black"
            plt.scatter(score_length[1][closest_pos], score_length[0][closest_pos], color=color, zorder=3)

    def plot_SA_of_RUL(self, plot_test_preds, result_labels, is_rtf):
        """Generate and log RUL survival analysis plots with predictions vs labels.

        For each test set, overlays predicted RUL trajectories against ground-truth labels.
        Color indicates predicted status (red for failure, black for normal).

        Args:
            plot_test_preds (list): List of prediction arrays, one per target source.
            result_labels (list): Corresponding ground-truth RUL label arrays.
            is_rtf (list): Run-to-failure flags, one per array (1=RTF scenario, 0=otherwise).

        Examples:
            >>> preds = [[[10, 20, 15], [100, 101, 102]], ...]
            >>> labels = [[[5, 4, 3], [98, 97, 96]], ...]
            >>> flags = [1, 0, ...]
            >>> experiment.plot_SA_of_RUL(preds, labels, flags)
        """
        if self.debug:
            plt.figure(figsize=(20, 20))
            counter = 0
            size = 3
            globalcounter = 0
            namescount = -1

            cmap = matplotlib.colormaps['coolwarm']
            # Pick two colors (not the edges, e.g., 0.25 and 0.75)
            color1 = cmap(0.15)
            color2 = cmap(0.75)
            # plot_rul_dictionary[current_target_source]={"scores":processed_target_scores,"labels":current_labels,"thresholds":None,"index":current_dates}
            for pred_set, lab_set, rtf in zip(plot_test_preds, result_labels, is_rtf):
                counter += 1
                globalcounter += 1
                if globalcounter > 30:
                    break
                plt.subplot(size * 100 + 10 + counter)
                self._plot_SA_inner(pred_set, [(lab, rtf) for lab in lab_set])
                if counter == size:
                    namescount += 1
                    mlflow.log_figure(plt.gcf(), f"SA_{namescount * 9}_{namescount * 9 + counter}.png")
                    plt.clf()
                    counter = 0
            if counter > 0:
                namescount += 1
                mlflow.log_figure(plt.gcf(), f"SA_{namescount * 9}_{namescount * 9 + counter}.png")
                plt.clf()
                counter = 0

    def _plot_RUL(self, plot_dictionary) -> None:
        """Generate and log RUL prediction plots showing predictions vs ground truth.

        Creates a grid of time-series plots comparing predicted vs actual RUL trajectories
        for run-to-failure (RTF) scenarios. Only plots scenarios where plot_dictionary[key]['rtf'] == 1.

        Only executed if self.debug is True.

        Args:
            plot_dictionary (dict): Dictionary with keys as source names and values containing:
                - 'index': timestamps/timesteps
                - 'scores': predicted RUL values
                - 'labels': ground-truth RUL values
                - 'rtf': flag indicating run-to-failure scenario

        Examples:
            >>> plot_dict = {
            ...     'bearing_1': {
            ...         'index': range(100),
            ...         'scores': [50, 49, 48, ...],
            ...         'labels': [52, 51, 50, ...],
            ...         'rtf': 1
            ...     }
            ... }
            >>> experiment._plot_RUL(plot_dict)
        """
        plt.figure(figsize=(20, 20))
        if self.debug:
            counter = 0
            namescount = -1
            # plot_rul_dictionary[current_target_source]={"scores":processed_target_scores,"labels":current_labels,"thresholds":None,"index":current_dates}
            for key in plot_dictionary.keys():
                if key in ["recall", "prc", "anomaly_ranges", "lead_ranges"]:
                    continue
                else:
                    if plot_dictionary[key]["rtf"] != 1:
                        continue
                    counter += 1
                    if namescount > 40:
                        break
                    plt.subplot(910 + counter)
                    plt.plot(plot_dictionary[key]["index"], plot_dictionary[key]["scores"], ".-", color="red",
                             label="RUL predictions")
                    plt.plot(plot_dictionary[key]["index"], plot_dictionary[key]["labels"], ".-", color="black",
                             label="RUL LABELS ")
                    if counter == 9:
                        namescount += 1
                        mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                        plt.clf()
                        counter = 0
            if counter > 0:
                namescount += 1
                mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                plt.clf()
                counter = 0

    def _plot_scores(self, plot_dictionary, best_metrics_dict) -> None:
        """Generate and log anomaly score visualizations with predictive horizon.

        Creates two sets of plots:
        1. Precision-Recall curve showing the best threshold
        2. Time-series plots of anomaly scores per source with:
           - Detected anomaly scores (black line)
           - Best threshold (blue line)
           - Failure dates (red vertical lines)
           - Predictive horizon window (red shaded region)
           - Lead time window (grey shaded region)

        Only executed if self.debug is True. Logs plots to MLflow.

        Args:
            plot_dictionary (dict): Dictionary keyed by source name with values containing:
                - 'scores': anomaly score array
                - 'failures': list of failure timestamps
                - 'thresholds': decision thresholds
                - 'index': timestamps/timesteps
                - 'recall': recall values for PR curve
                - 'prc': precision values for PR curve
                - 'anomaly_ranges': boolean array marking predictive horizon
                - 'lead_ranges': boolean array marking lead time window
            best_metrics_dict (dict): Best metrics from evaluation, must include 
                'threshold_auc' (best threshold value).

        Examples:
            >>> plot_dict = {
            ...     'bearing_1': {
            ...         'scores': np.array([0.1, 0.2, 0.9, 0.8, ...]),
            ...         'failures': [pd.Timestamp('2024-01-10')],
            ...         'thresholds': np.array([0.5, 0.5, ...]),
            ...         'index': [...],
            ...         'recall': [0, 0.2, 0.5, 0.8, 1],
            ...         'prc': [1, 0.9, 0.7, 0.5, 0.1]
            ...     },
            ...     'anomaly_ranges': [...],
            ...     'lead_ranges': [...]
            ... }
            >>> best_metrics = {'threshold_auc': 0.65, 'AD1_AUC': 0.92}
            >>> experiment._plot_scores(plot_dict, best_metrics)
        """
        tups = []
        for rec, prc in zip(plot_dictionary['recall'], plot_dictionary['prc']):
            tups.append((rec, prc))
        tups = sorted(tups, key=lambda x: (x[0], -x[1]))
        xaxisvalue = []
        yaxisvalue = []
        for tup in tups:
            xaxisvalue.append(tup[0])
            yaxisvalue.append(tup[1])
        plt.plot(xaxisvalue, yaxisvalue, "-o")
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), 'pr_curve.png')

        plt.clf()
        plt.figure(figsize=(20, 20))
        if self.debug:
            counter = 0
            namescount = -1
            prelimit = 0
            for key in plot_dictionary.keys():
                if key == "recall" or key == "prc" or key == "anomaly_ranges" or key == "lead_ranges":
                    continue
                counter += 1
                data_to_plot = plot_dictionary[key]
                current_range = plot_dictionary["anomaly_ranges"][prelimit:prelimit + len(data_to_plot["scores"])]
                current_range_lead = plot_dictionary["lead_ranges"][prelimit:prelimit + len(data_to_plot["scores"])]

                prelimit += len(data_to_plot["scores"])
                plt.subplot(910 + counter)
                # print()
                plt.plot(data_to_plot["index"], data_to_plot["scores"], ".-", color="black", label="anomaly score")
                plt.plot(data_to_plot["index"],
                         [best_metrics_dict["threshold_auc"] for i in range(len(data_to_plot["index"]))], ".-",
                         color="dodgerblue", label="best threshold")

                for date in data_to_plot["failures"]:
                    plt.axvline(date, color="red")

                # plot PH
                self._inner_plot("red", current_range, data_to_plot["index"], min(data_to_plot["scores"]),
                                 max(data_to_plot["scores"]), "predictive horizon")

                # plot lead
                self._inner_plot("grey", current_range_lead, data_to_plot["index"], min(data_to_plot["scores"]),
                                 max(data_to_plot["scores"]), "lead time")
                plt.legend(loc="center left")
                plt.title(f'Source label: {key}')

                if counter == 9:
                    namescount += 1
                    mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                    plt.clf()
                    counter = 0
            if counter > 0:
                namescount += 1
                mlflow.log_figure(plt.gcf(), f"scores_{namescount * 9}_{namescount * 9 + counter}.png")
                plt.clf()
                counter = 0

    def _build_raw_preprocessor(self, raw_params: dict):
        """Build the raw-level step of a trial, or return None when the dataset has no snapshots.

        Args:
            raw_params (dict): The trial's ``raw_preprocessor_*`` parameters, without the prefix.

        Raises:
            ValueError: If raw_preprocessor parameters are given for a dataset without snapshots.
        """
        if self.pipeline.raw_preprocessor is None:
            if raw_params:
                raise ValueError(
                    f'raw_preprocessor parameters {sorted(raw_params)} were given but the dataset '
                    'has no snapshots, so the pipeline has no raw_preprocessor step.'
                )
            return None
        return self.pipeline.raw_preprocessor(**raw_params)

    def _finish_run(self, parent_run, current_steps, params: dict = None) -> None:
        """Log pipeline components, parameters, and metrics to the current MLflow run.

        Performs cleanup and logging at the end of a single parameter combination evaluation:

        1. Logs fitted models to MLflow (sklearn, pytorch, etc. depending on component)
        2. Logs all pipeline component parameters
        3. Logs PdM-specific configuration (predictive horizon, lead time, beta, slide window)
        4. Logs evaluation parameters (min scenario lengths, max wait time, etc.)
        5. Calls destruct() on the method for cleanup

        Args:
            parent_run: Active MLflow Run object to log to.
            current_steps (dict): Dictionary of pipeline components for this run:
                - 'raw_preprocessor': Raw-level step, or None when the dataset has no snapshots
                - 'preprocessor': Fitted preprocessor
                - 'method': Fitted anomaly detection method
                - 'postprocessor': Fitted postprocessor
                - 'thresholder': Fitted thresholder
            params (dict, optional): The trial's search-space parameters. Experiment
                level entries that no pipeline step reports (profile_size,
                initial_profile_size) are logged from here so that
                :meth:`_check_cached_run` can compare against them later.

        Examples:
            >>> steps = {
            ...     'preprocessor': my_preprocessor,
            ...     'method': my_method,
            ...     'postprocessor': my_postprocessor,
            ...     'thresholder': my_thresholder
            ... }
            >>> experiment._finish_run(mlflow.active_run(), steps)
        """
        # Bind the energy record to this trial's MLflow run. This must be the
        # FIRST statement: the objective's ``return`` is dedented out of the
        # ``with mlflow.start_run(...)`` block in every flavor, so the trial
        # decorator sees ``active_run() is None`` and cannot make the join
        # itself. Binding ahead of log_model and friends -- any of which can
        # raise -- means the join key survives a partial teardown.
        self._energy_bind_run(parent_run)

        if 'many' in current_steps['method'].get_library():
            model_sources, models = current_steps['method'].get_all_models()
            for model_source, model in zip(model_sources, models):
                current_subpackage = getattr(mlflow, re.sub('many_', '', current_steps['method'].get_library()))
                current_submodule = current_subpackage.log_model
                # TODO do not use self.artifacts
                current_submodule(model, f'{self.artifacts}/{str(current_steps["method"])}_source_{model_source}')
        elif current_steps['method'].get_library() == 'no_save':
            pass
        else:
            # TODO we should check if there is a log_model functionality for the method we have in the current run
            # TODO do not use self.artifacts
            current_subpackage = getattr(mlflow, current_steps['method'].get_library())
            current_submodule = current_subpackage.log_model
            current_submodule(current_steps['method'], f'{self.artifacts}/{str(current_steps["method"])}')

        # log parameters for each step
        for step in self.pipeline.get_steps().keys():
            mlflow.log_params({
                f'{step}_{key}': str(value)[:499] for key, value in current_steps[step].get_params().items()
            })
            mlflow.log_param(step, str(current_steps[step]))

        # Experiment level params are not reported by any step's get_params(), so
        # log them here; without this the cache lookup can never match on them.
        for param_name in ('profile_size', 'initial_profile_size'):
            if params and param_name in params:
                mlflow.log_param(param_name, params[param_name])

        # log the optimizer used for this run
        mlflow.log_param('optimizer', self.optimizer_name)

        # Seeding, logged per run so a single trial can be replayed on its own.
        # 'random_state' is the experiment seed; 'trial_seed' is what the RNGs
        # were actually set to for this trial, derived from its parameters by
        # seeded_objective -- the two differ, and it is the latter that has to
        # be reproduced to rerun this row in isolation.
        mlflow.log_param('random_state', self.random_state)
        mlflow.log_param('deterministic', self.deterministic)
        if params:
            mlflow.log_param('trial_seed', seed_for_params(self.random_state, params))

        if "anomaly_ranges" in self.pipeline.dataset.keys():
            mlflow.log_param('anomaly_ranges', self.pipeline.dataset['anomaly_ranges'])
            if self.pipeline.dataset["anomaly_ranges"]:
                mlflow.log_params({
                    'predictive_horizon': self.pipeline.slide,
                    'beta': self.pipeline.beta,
                    'lead': self.pipeline.slide
                })
            else:
                mlflow.log_params({
                    'predictive_horizon': self.pipeline.predictive_horizon,
                    'beta': self.pipeline.beta,
                    'lead': self.pipeline.lead
                })
        else:
            mlflow.log_params({
                'predictive_horizon': self.pipeline.predictive_horizon,
                'beta': self.pipeline.beta,
                'lead': self.pipeline.lead
            })

        for paramm in ["slide", "auc_resolution", "min_historic_scenario_len", "min_target_scenario_len",
                       "max_wait_time"]:
            if paramm in self.pipeline.dataset and paramm is not None:
                mlflow.log_param(paramm, self.pipeline.dataset[paramm])
        # mlflow.log_params({
        #     'slide': self.pipeline.dataset['slide'],
        #     'auc_resolution': self.pipeline.auc_resolution,
        #     'min_historic_scenario_len': self.pipeline.dataset['min_historic_scenario_len'],
        #     'min_target_scenario_len': self.pipeline.dataset['min_target_scenario_len'],
        #     'max_wait_time': self.pipeline.dataset['max_wait_time']
        # })

        if 'reset_after_fail' in self.pipeline.dataset:
            mlflow.log_param('reset_after_fail', self.pipeline.dataset['reset_after_fail'])

        if 'setup_1_period' in self.pipeline.dataset:
            mlflow.log_param('setup_1_period', self.pipeline.dataset['setup_1_period'])

        current_steps['method'].destruct()

    def _finish_experiment(self, best_params: dict) -> dict:
        # Mango uses scikit learn and due to the autolog functionality it logs some runs to the default experiment, so we need to clear the default experiment to avoid confusion
        default_experiment_id = mlflow.get_experiment_by_name("Default").experiment_id

        runs = mlflow.search_runs(experiment_ids=default_experiment_id)

        for run in runs.iterrows():
            run_id = run[1]['run_id']
            mlflow.delete_run(run_id)

        if self.log_best_scores and os.path.exists(self.best_scores_info_dict_path):
            with open(self.best_scores_info_dict_path, 'rb') as file:
                best_scores_info_saved_dict = pickle.load(file)
                best_run_id = best_scores_info_saved_dict['best_run_id']
                pd.DataFrame(best_scores_info_saved_dict['best_scores']).to_csv(f'scores_{best_run_id}.csv',
                                                                                index=False, header=False)

                with mlflow.start_run(run_id=best_run_id, experiment_id=self.experiment_id):
                    mlflow.log_artifact(f'scores_{best_run_id}.csv')
                    os.remove(f'scores_{best_run_id}.csv')

            os.remove(self.best_scores_info_dict_path)

            os.remove(self.lock_file_path)

        return best_params

    def _run_evaluators(self, default_evaluator, **kwargs) -> dict:
        """
        Executes the default evaluator along with any user-provided custom evaluators.
        Aggregates the results and logs custom metrics to MLflow.
        """
        results = default_evaluator.evaluate(self, **kwargs)
        
        if self.custom_evaluators:
            for custom_eval in self.custom_evaluators:
                custom_results = custom_eval.evaluate(self, **kwargs)
                results.update(custom_results)
                mlflow.log_metrics({k: v for k, v in custom_results.items() if isinstance(v, (int, float, np.number))})
                
        return results

    def _check_cached_run(self, params: dict):
        """Look for a previously FINISHED MLflow run with identical parameters.

        Returns ``None`` immediately unless ``use_cache`` was enabled, because the
        search spans every FINISHED run in the experiment -- including those from
        earlier invocations of the same experiment name -- so with it on, re-running
        an experiment short-circuits most or all of its trials and the result then
        depends on leftover MLflow history.

        Args:
            params: The trial's parameters, as passed to the objective.

        Returns:
            ``(score, extras)`` when a cached run matches, where *extras* holds
            any recoverable ``th`` / ``th_to_rul`` from that run's logged metrics
            and may be empty; ``None`` when there is no match.
        """
        if not self.use_cache:
            return None

        current_params = params.copy()

        step_params = split_step_params(current_params)
        method_params = step_params['method']
        preprocessor_params = step_params['preprocessor']
        postprocessor_params = step_params['postprocessor']
        thresholder_params = step_params['thresholder']

        runs = mlflow.search_runs(self.experiment_id, filter_string='attributes.status = "FINISHED"')

        found_match, found_index, found_run = False, -1, None
        for index, current_run in runs.iterrows():
            found_match = True
            found_index = index
            found_run = current_run

            for param_name, param_value in current_params.items():
                if 'params.' + param_name not in current_run.index:
                    found_match = False
                    break

                if current_run.loc['params.' + param_name] != str(param_value):
                    found_match = False
                    break

            current_steps = {
                'method': self.pipeline.method(event_preferences=self.pipeline.event_preferences, **method_params),
                'preprocessor': self.pipeline.preprocessor(event_preferences=self.pipeline.event_preferences,
                                                           **preprocessor_params),
                'postprocessor': self.pipeline.postprocessor(event_preferences=self.pipeline.event_preferences,
                                                             **postprocessor_params),
                'thresholder': self.pipeline.thresholder(event_preferences=self.pipeline.event_preferences,
                                                         **thresholder_params),
                'raw_preprocessor': self._build_raw_preprocessor(step_params['raw_preprocessor']),
            }

            for step in self.pipeline.get_steps().keys():
                # Runs logged without this step (e.g. 'raw_preprocessor' of a snapshot dataset,
                # under an experiment name that also holds other datasets) cannot match.
                if 'params.' + step not in current_run.index \
                        or current_run.loc['params.' + step] != str(current_steps[step]):
                    found_match = False
                    break
            # Nor can runs logged with a step this pipeline does not have (a snapshot run's
            # raw_preprocessor, for a pipeline without one).
            for step in STEP_NAMES:
                if step not in self.pipeline.get_steps() and isinstance(current_run.get('params.' + step), str):
                    found_match = False

            predictive_horizon_to_check, beta_to_check, lead_to_check = -1, -1, -1
            if "anomaly_ranges" in self.pipeline.dataset.keys():
                if self.pipeline.dataset["anomaly_ranges"]:
                    predictive_horizon_to_check = self.pipeline.slide
                    beta_to_check = self.pipeline.beta
                    lead_to_check = self.pipeline.slide
                else:
                    predictive_horizon_to_check = self.pipeline.predictive_horizon
                    beta_to_check = self.pipeline.beta
                    lead_to_check = self.pipeline.lead
            else:
                predictive_horizon_to_check = self.pipeline.predictive_horizon
                beta_to_check = self.pipeline.beta
                lead_to_check = self.pipeline.lead

            if str(predictive_horizon_to_check) != current_run.loc['params.predictive_horizon'] \
                    or str(beta_to_check) != current_run.loc['params.beta'] \
                    or str(lead_to_check) != current_run.loc['params.lead']:
                found_match = False

            if found_match:
                break

        if found_match:
            # A cache hit does no work, so its energy must not join the per-trial
            # distribution -- it would drag the mean toward zero. The trial
            # returns before ``mlflow.start_run``, so this is the only place the
            # decorator can learn about it.
            if self._energy_sink is not None:
                try:
                    from pdmlabs.energy import context as energy_context
                    energy_context.mark_cached()
                except Exception:
                    pass
            logging.info(
                f'Found cached run with parameters: {current_params}, steps={[str(step) for step in current_steps.values()]}, predictive_horizon={predictive_horizon_to_check}, beta={beta_to_check} and lead={lead_to_check}. Skipping...')
            # The thresholds were logged as metrics on the cached run, so they can
            # be handed to the trial sink instead of being lost. Without this a
            # cached trial contributes a score to the optimizer but no 'th', and
            # if it wins there is nothing to reconcile against 'best_objective'.
            extras = {}
            for metric_name, extra_name in (('threshold_auc', 'th'),
                                            ('th_to_rul', 'th_to_rul')):
                column = 'metrics.' + metric_name
                if column in found_run.index:
                    value = found_run.loc[column]
                    if pd.notna(value):
                        extras[extra_name] = float(value)
            return found_run.loc['metrics.' + self.optimization_param], extras
        else:
            return None

    def _run_optimizer(self, param_space: dict, objective_fn, maximize: bool = True) -> dict:
        """Dispatch HPO to the configured optimizer backend.

        The *objective_fn* must be a raw callable with signature
        ``(**params) -> float`` (single config, single score).  Each adapter
        is responsible for wrapping it into the calling convention its backend
        expects (batch list for Mango, ``(config, seed)`` for SMAC, etc.).

        Performs the categorical guard check before any trial is submitted.
        Numeric (int/float) lists are sorted before dispatch so that backends
        that are order-sensitive (e.g. SMAC's OrdinalHyperparameter) always
        receive a well-ordered space. Boolean lists are intentionally excluded
        from sorting because ``isinstance(True, int)`` is ``True`` in Python and
        sorting ``[True, False]`` is semantically meaningless.

        Args:
            param_space: PdMLabs native param space dict.
            objective_fn: Raw ``(**params) -> float`` callable.
            maximize: If True maximise; otherwise minimise.

        Returns:
            dict with 'best_params', 'best_objective', 'params_tried',
            'objective_values' (unified across all backends).

        Raises:
            CategoricalSpaceNotSupportedException: If the space contains string
                values and the adapter has ``supports_categorical=False``.
            ValueError: If ``self.optimizer_name`` is not in OPTIMIZER_REGISTRY.
        """
        # Sort numeric lists to normalise potentially unordered user input.
        # Mixed int+float lists are also covered. rv_frozen, str lists, and
        # bool lists pass through unchanged.
        sorted_param_space = {}
        for _name, _values in param_space.items():
            if (
                isinstance(_values, list)
                and len(_values) > 0
                and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in _values
                )
            ):
                sorted_param_space[_name] = sorted(_values)
            else:
                sorted_param_space[_name] = _values

        from pdmlabs.optimization import get_optimizer
        adapter = get_optimizer(self.optimizer_name, random_state=self.random_state)
        adapter._check_categorical(sorted_param_space)

        # Reseed per trial, before the energy wrapper so the reseed is inside
        # the measured region and costs the same on every trial. The seed comes
        # from the trial's own parameters, so a configuration scores the same
        # whichever worker runs it and in whatever order -- the property that
        # makes the search reproducible rather than merely deterministic.
        objective_fn = seeded_objective(objective_fn, self.random_state,
                                        deterministic=self.deterministic)

        # Instrument here rather than in the adapters: this is the one place the
        # objective is handed to every backend, so a single wrapper covers all
        # six. It also sits inside ``minimize``'s ``negated`` closure, so the
        # sign flip applies to the score without disturbing energy accounting.
        self._energy_phase_id = uuid.uuid4().hex if self._energy_sink is not None else None
        objective_fn = self._instrument_objective(objective_fn)

        with self._energy_phase(self.energy_phase_label):
            if maximize:
                return adapter.maximize(
                    sorted_param_space,
                    objective_fn,
                    n_iterations=self.num_iteration,
                    n_jobs=self.n_jobs,
                    initial_random=self.initial_random,
                    constraint_fn=self.constraint_function,
                )
            else:
                return adapter.minimize(
                    sorted_param_space,
                    objective_fn,
                    n_iterations=self.num_iteration,
                    n_jobs=self.n_jobs,
                    initial_random=self.initial_random,
                    constraint_fn=self.constraint_function,
                )

    def _new_trial_sink(self):
        """Create the per-``execute()`` spool used to ship best trials home.

        Every backend that honours ``n_jobs > 1`` runs ``optimization_objective``
        in a separate process, so assignments it makes to ``self`` land on a copy
        that is discarded when the task ends. The returned sink is captured by the
        objective closure and travels to the workers with it.

        Returns:
            A fresh :class:`~pdmlabs.optimization.trial_sink.TrialSink`. The
            caller owns it and must call ``cleanup()``.
        """
        from pdmlabs.optimization.trial_sink import TrialSink
        return TrialSink.create(maximize=self.maximize)

    def _collect_best_trial(self, results: dict, sink):
        """Populate ``extra_metrics`` / ``best_pipeline`` from the trial spool.

        Never re-runs the objective. If nothing was spooled -- every trial hit the
        MLflow run cache, raised, or scored non-finite -- the thresholds are
        reported as ``None`` and no pipeline is logged.

        Args:
            results: The dict returned by :meth:`_run_optimizer`.
            sink: The :class:`TrialSink` passed to the objective closure.

        Returns:
            The recovered ``TrialRecord``, or ``None`` if nothing was spooled.
        """
        best = sink.best(prefer_params=results.get('best_params'))

        if best is None:
            self.extra_metrics['best'] = None
            self.extra_metrics['th'] = None
            self.extra_metrics['th_to_rul'] = None
            self.extra_metrics['best_pipeline_objective'] = None
            self.extra_metrics['best_params_used'] = None
            self.extra_metrics['best_pipeline_th'] = None
            self.best_pipeline = None
            logging.warning(
                'No trial was recovered (every run was cached, errored, or scored '
                'non-finite). "th" will be None and no best pipeline will be '
                'logged to MLflow.'
            )
            return None

        self.extra_metrics['best'] = best.score
        self.extra_metrics['th'] = best.th
        self.extra_metrics['th_to_rul'] = best.th_to_rul
        # These describe the pipeline actually logged, which is normally the
        # winning trial's but can be a fallback (see below).
        self.extra_metrics['best_pipeline_objective'] = best.pipeline_score
        self.extra_metrics['best_params_used'] = best.pipeline_params
        # The threshold to stamp on the logged model: the winner's in the normal
        # case, the fallback pipeline's own when the winner produced none.
        self.extra_metrics['best_pipeline_th'] = best.pipeline_th
        self.best_pipeline = best.pipeline

        if best.pipeline is None:
            logging.warning(
                'Best trial (%s=%r) was recovered but no pipeline object is '
                'available for any trial: %s. "th" is still valid; no model will '
                'be logged to MLflow.',
                self.optimization_param, best.score,
                best.pipeline_error or 'the winning trial came from the MLflow run cache',
            )
        elif not best.pipeline_is_winner:
            logging.warning(
                'The best trial (%s=%r, params %r) produced no pipeline object, so '
                'the model logged to MLflow is the best one that did: %s=%r with '
                'params %r. "th" belongs to the winning trial, not to that model.',
                self.optimization_param, best.score, best.params,
                self.optimization_param, best.pipeline_score, best.pipeline_params,
            )

        optimizer_best = results.get('best_objective')
        if optimizer_best is not None:
            optimizer_best = float(optimizer_best)
            # Not exact equality: SMAC round-trips ordinals through str()/float()
            # and dedups by config_id (averaging costs), GPyOpt snaps discrete
            # floats. A sink score that is *better* is legitimate for the same
            # reasons, so only a strictly worse one is worth reporting.
            agrees = (
                math.isclose(best.score, optimizer_best, rel_tol=1e-9, abs_tol=1e-12)
                or sink.is_better(best.score, optimizer_best)
            )
            if not agrees:
                logging.warning(
                    'Optimizer reports best %s=%r but the best materialised pipeline '
                    'scored %r. This happens when the optimizer\'s best trial was '
                    'served from the MLflow run cache, for which no pipeline object '
                    'exists. "th"/"best_pipeline" therefore describe the %r trial, '
                    'not "best_params".',
                    self.optimization_param, optimizer_best, best.score, best.score,
                )
            elif not best.matched_best_params:
                # Scores agree but the configurations do not: the optimizer's
                # chosen parameterization was never spooled (several configs
                # reached the same objective, or its trial came from the cache).
                logging.warning(
                    '"th"/"best_pipeline" come from configuration %r, which reached '
                    'the same %s=%r as "best_params" %r but is a different '
                    'configuration (%d trial(s) tied at that score). See '
                    '"best_pipeline_params" in the returned dict.',
                    best.params, self.optimization_param, best.score,
                    results.get('best_params'), best.tied_count,
                )
        return best