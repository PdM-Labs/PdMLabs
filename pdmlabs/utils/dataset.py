"""Dataset preparation and management for predictive maintenance tasks.

This module provides the Dataset class for handling time-series data preparation,
episode management, train/validation/test splitting, and generation of labeled
datasets for various learning paradigms (supervised, unsupervised, semi-supervised).

Key Features:
    - Automatic episode extraction from time-series data
    - Intelligent train/val/test splitting with failure-aware strategy
    - Support for multiple dataset formats (RUL, survival analysis, classification)
    - Event data integration and wildcard-based event preference handling
    - Configurable predictive horizon and sliding window parameters

Example:
    >>> import pandas as pd
    >>> from pdmlabs.utils.dataset import Dataset
    >>> data = pd.read_csv('sensor_data.csv')
    >>> dataset = Dataset(
    ...     data=data,
    ...     datetime_column='timestamp',
    ...     failure_column='is_failure',
    ...     source_column='equipment_id'
    ... )
    >>> train_data, test_data = dataset.get_rul_dataset()
"""

import math
import random
import warnings

import pandas as pd

#: Separator inserted between an original source name and its episode index.
EPISODE_SEPARATOR = '_ep'

#: Minimum number of episodes needed to populate train, validation and test.
MIN_EPISODES_FOR_SPLIT = 3


class Dataset:
    """
    A class to handle dataset preparation and processing for predictive maintenance tasks.
    This includes splitting data into episodes, calculating sliding windows, and preparing
    training, validation, and testing datasets.

    Parameters
    ----------
    data : pd.DataFrame, dict, str or os.PathLike
        The input data containing time-series information. It can also be a snapshot dataset,
        described by `snapshot_options`: the path of a folder in which every subfolder holding
        files is a source, named by its relative path (see
        :func:`pdmlabs.preprocessing.raw_level.snapshots.discover_sources`), or a dict mapping
        each source name to a folder (one file per snapshot) or to a list of DataFrames (one per
        snapshot). Each snapshot becomes one row at its recording time and is turned into
        features by the pipeline's raw-level step. If `event_df` is not provided but
        `maintenance_column` and/or `failure_column` are provided, they are expected to be binary
        indicator columns of `data` and are used to derive the episodes. If neither `event_df`,
        `event_indicator`, `maintenance_column` nor `failure_column` is provided, every source is
        assumed to be a single run-to-failure episode.
    datetime_column : str
        The name of the column representing datetime values.
    event_indicator : str, default=None
        The name of the column indicating event occurrence (binary), constant per source. If
        provided, it is used to mark the end of the single episode of each source
        (0: maintenance/reset or censored, 1: failure).
    maintenance_column : str or list, default=None
        Without `event_df`: the name of the binary column of `data` that flags maintenance/reset
        events. With `event_df`: the list of values of the `code` column of `event_df` that are
        considered maintenance (resetting) events.
    failure_column : str or list, default=None
        Without `event_df`: the name of the binary column of `data` that flags failure events.
        With `event_df`: the list of values of the `code` column of `event_df` that are considered
        failure events.
    event_df : pd.DataFrame, optional
        A DataFrame containing event data. When provided it must contain the columns
        `datetime_column`, `source_column` and `code`, and `failure_column` (and optionally
        `maintenance_column`) must be given as lists of `code` values. Codes that appear in both
        lists are treated as failures.
    source_column : str, default='source'
        The name of the column representing the source of the data.
    beta : int, default=1
        A parameter used for objective calculations.
    slide : int, optional
        The sliding window size. If None, it is calculated automatically.
    lead : str, default="0 seconds"
        The lead time for predictions.
    predictive_horizon : str, optional
        The predictive horizon for the dataset. If None, it is calculated automatically.
    train_sources : float or list, default=0.6
        The ratio (float) or list of source/episode names used for training. If a float, it
        represents the proportion of sources used for training.
    val_sources : float or list, default=0.2
        The ratio (float) or list of source/episode names used for validation.
    test_sources : float or list, default=0.2
        The ratio (float) or list of source/episode names used for testing.
    max_wait_time : int, optional
        Controls the maximum length of the profile parameter in OnlineFlavor and Sliding Window
        flavor (i.e. the maximum length of the data used to fit anomaly detectors). This is the
        time the user is willing to wait before detectors produce alarms. If None, it is set to
        2/3 of the minimum episode length.
    in_source_split : bool, default=False
        Whether to select train/val/test episodes from within each source (True) or to split at
        the source level (False). Splitting at the source level is the only way to guarantee that
        no source contributes to more than one split.
    keep_censored_tail : bool, default=True
        When episodes are derived from events, whether to keep the data recorded after the last
        event of a source as an extra censored (non run-to-failure) episode. Setting this to False
        discards that data, which is the behaviour of releases up to 0.0.2.
    random_state : int, optional
        When provided, sources/episodes are shuffled with this seed before being split. When None
        (the default) the split follows the order in which sources appear in `data`, which keeps
        the split reproducible without shuffling.
    DIVIDER : int, default=3600
        Number of seconds a single RUL unit corresponds to (3600 => RUL expressed in hours).
        Ignored when `data` already carries a ``RUL`` column and episodes are whole sources
        (the `event_indicator` strategy and the implicit run-to-failure fallback): that column
        is taken as given. When episodes are cut out of a source by events, a per-episode RUL is
        always recomputed, because a source-wide RUL would not restart at each episode.
    snapshot_options : dict, optional
        Only for snapshot datasets (`data` given as a folder or a dict). Exactly one timestamp
        option is required: ``timestamp_format`` (``strptime`` format of the file names, e.g.
        ``'%Y.%m.%d.%H.%M.%S'`` for IMS), ``datetime_column`` (a column inside each snapshot)
        or ``interval`` (artificial spacing such as ``'1min'``, with an optional ``start``).
        Optional, for folder sources only: ``read_kwargs`` (e.g.
        ``{'sep': '\\t', 'header': None}``) and ``reader``.
        ``read_kwargs`` applies to every file of every source, so selecting different
        columns per source needs a ``reader``. In IMS, for example, test 1 has two channels
        per bearing and tests 2 and 3 have one; this reader keeps one channel per bearing,
        numbered 0-3, in every test::

            def read_ims(path, **kwargs):
                cols = [0, 2, 4, 6] if '1st_test' in path else [0, 1, 2, 3]
                return pd.read_csv(path, sep='\\t', header=None,
                                   usecols=cols).set_axis(range(4), axis=1)

            snapshot_options = {'timestamp_format': '%Y.%m.%d.%H.%M.%S', 'reader': read_ims}

        Events must then come from `event_df`, or every source is treated as one run-to-failure
        episode. See
        :func:`pdmlabs.preprocessing.raw_level.snapshots.build_snapshot_frame`. The same options
        build the input of a saved model's ``predict`` from new snapshots, with
        :func:`pdmlabs.preprocessing.raw_level.snapshots.snapshot_frame`.

    Attributes
    ----------
    sources : list[str]
        Episode names, in episode order.
    episode_to_source : dict[str, str]
        Maps an episode name to the original source it was extracted from.
    rtf_dict : dict[str, int]
        Maps an episode name to 1 when the episode ends with a failure, 0 otherwise.
    sources_for_train, sources_for_val, sources_for_test : list[str]
        Episode names of each split, in the same order as `train_dfs`, `val_dfs` and `test_dfs`.
    train_dfs, val_dfs, test_dfs : list[pd.DataFrame]
        Episodes of each split.
    snapshot_column : str or None
        Column holding the snapshot references of a snapshot dataset, None otherwise. It is
        also published as ``'snapshot_column'`` in the dictionaries of the ``get_*_dataset``
        methods, which makes the pipeline add its raw-level step.
    """

    def __init__(self, data, datetime_column, event_indicator=None, maintenance_column=None,
                 failure_column=None, event_df=None, source_column='source',
                 beta=1, slide=None, lead="0 seconds", predictive_horizon=None,
                 train_sources=0.6, val_sources=0.2, test_sources=0.2, max_wait_time=None,
                 in_source_split=False, keep_censored_tail=True, random_state=None,
                 DIVIDER=3600, snapshot_options=None):

        # Dataset. `data` is copied so that the caller's frame is never modified in place.
        self.in_source_split = in_source_split
        self.datetime_column = datetime_column
        self.source_column = source_column
        self.random_state = random_state

        # A snapshot dataset becomes a placeholder frame with one row per snapshot, so that
        # episodes, splits, horizon and slide are derived from the snapshot timestamps exactly
        # as for any other dataset. The snapshots are featurized later, by the pipeline.
        self.snapshot_column = None
        if isinstance(data, str) or hasattr(data, '__fspath__'):
            # One folder for the whole dataset: every subfolder holding files is a source.
            from pdmlabs.preprocessing.raw_level.snapshots import discover_sources
            data = discover_sources(data)
        if isinstance(data, dict):
            from pdmlabs.preprocessing.raw_level.snapshots import (SNAPSHOT_COLUMN,
                                                                   build_snapshot_frame)
            if event_indicator is not None:
                raise ValueError(
                    "event_indicator names a column of data, which snapshot datasets do not "
                    "have. Describe the events with event_df, or omit it to treat every source "
                    "as one run-to-failure episode."
                )
            data = build_snapshot_frame(data, datetime_column, source_column,
                                        snapshot_options or {})
            self.snapshot_column = SNAPSHOT_COLUMN
        elif snapshot_options is not None:
            raise ValueError("snapshot_options only applies when data is a snapshot dataset "
                             "(a dict of sources or a folder).")

        data = data.copy()
        data[self.datetime_column] = pd.to_datetime(data[self.datetime_column])
        data[source_column] = data[source_column].astype(str)

        episodes, run_to_failure, episode_names, original_s_has_f, episode_to_source = \
            episodes_formulation(data, datetime_column, event_indicator, maintenance_column,
                                 failure_column, event_df, source_column, DIVIDER,
                                 keep_censored_tail)

        if len(episodes) == 0:
            raise ValueError(
                "No episode could be extracted from the given data. Check that the event "
                "definition (event_df/event_indicator/maintenance_column/failure_column) matches "
                "the data."
            )

        self.original_sources = data[source_column].unique().tolist()
        self.sources = episode_names
        self.episode_to_source = episode_to_source
        self.original_s_has_f = original_s_has_f

        self.train_sources = train_sources
        self.val_sources = val_sources
        self.test_sources = test_sources

        self.train_source_name = 'train'
        self.split_sources_to_train_test_val(episodes, run_to_failure)
        # Calculated in split_sources_to_train_test_val:
        # self.matches
        # self.rtf_dict
        # self.train_dfs / self.val_dfs / self.test_dfs
        # self.sources_for_train / self.sources_for_val / self.sources_for_test

        self.max_wait_time = max_wait_time
        if self.max_wait_time is None:
            self.max_wait_time = max(10, int(2 * min(ep.shape[0] for ep in episodes) / 3))

        self.rul_column = 'RUL'

        # Objective parameters
        self.beta = beta
        self.lead = lead
        # when predictive_horizon is not provided, derive it from the shortest failing episode
        if predictive_horizon is None:
            durations = [
                (ep.iloc[-1][self.datetime_column]
                 - ep.iloc[0][self.datetime_column]).total_seconds() / 3600.0
                for ep, rtf in zip(episodes, run_to_failure) if rtf == 1
            ]
            if not durations:
                raise ValueError(
                    "predictive_horizon cannot be derived automatically because no episode ends "
                    "with a failure event. Pass predictive_horizon explicitly."
                )
            # Ignore failing episodes that span no time at all (a failure logged on the very
            # first observation of a source): a single one of them would otherwise drive the
            # horizon to zero, which silently disables every predictive-horizon truncation.
            positive_durations = [duration for duration in durations if duration > 0]
            if not positive_durations:
                raise ValueError(
                    "predictive_horizon cannot be derived automatically because every failing "
                    "episode spans zero time. Pass predictive_horizon explicitly."
                )
            if len(positive_durations) < len(durations):
                warnings.warn(
                    f"{len(durations) - len(positive_durations)} failing episode(s) span zero "
                    "time and were ignored when deriving predictive_horizon.",
                    stacklevel=2,
                )
            self.predictive_horizon = f"{min(positive_durations) / 10.0} hours"
        else:
            self.predictive_horizon = predictive_horizon

        if slide is None:
            self.slide = self.slide_calculation(episodes, run_to_failure)
        else:
            self.slide = slide

    # ------------------------------------------------------------------ #
    # Horizon / slide helpers
    # ------------------------------------------------------------------ #
    def slide_calculation(self, episodes, run_to_failure):
        """Calculate optimal sliding window step size for dataset generation.

        Ensures that slide + predictive_horizon equals approximately 1/3 of the
        smallest failure episode. This balances training data size with prediction lead time.

        Parameters
        ----------
        episodes : list[pd.DataFrame]
            List of episode dataframes, each representing one run-to-failure sequence.
        run_to_failure : list[int]
            List indicating which episodes contain failures (1) or are healthy runs (0).

        Returns
        -------
        int
            Optimal sliding window step size. Minimum value is 1.

        Notes
        -----
        The sliding window step determines how many samples between consecutive windows.
        Larger steps = fewer training samples but faster processing.
        Smaller steps = more training samples but more computation.

        Formula: slide = (episode_length / 3) - predictive_horizon_length
        """
        min_duration = float('inf')
        min_episode = None
        for episode, is_rtf in zip(episodes, run_to_failure):
            if is_rtf == 1:
                duration = (episode.iloc[-1][self.datetime_column]
                            - episode.iloc[0][self.datetime_column]).total_seconds() / 3600.0
                if duration < min_duration:
                    min_duration = duration
                    min_episode = episode

        if min_episode is None:
            raise ValueError(
                "slide cannot be derived automatically because no episode ends with a failure "
                "event. Pass slide explicitly."
            )

        episode_length = min_episode.shape[0]
        last_time = min_episode.iloc[-1][self.datetime_column]
        horizon_start = 0
        for i in range(episode_length):
            current_time = min_episode.iloc[i][self.datetime_column]
            if current_time >= last_time - pd.Timedelta(self.predictive_horizon):
                horizon_start = i
                break
        horizon_length = episode_length - horizon_start
        slide = int(episode_length / 3) - horizon_length
        return max(slide, 1)

    # ------------------------------------------------------------------ #
    # Splitting
    # ------------------------------------------------------------------ #
    def split_sources_to_train_test_val(self, episodes, run_to_failure):
        """
        Splits the sources into training, validation, and testing datasets.

        Parameters
        ----------
        episodes : list
            A list of dataframes, where each dataframe corresponds to an episode.
        run_to_failure : list
            A list of integers indicating whether each episode is a run-to-failure (1) or not (0).

        Returns
        -------
        None
            The method updates the following attributes of the class:
            - self.train_dfs: Dataframes for training.
            - self.val_dfs: Dataframes for validation.
            - self.test_dfs: Dataframes for testing.
            - self.sources_for_train: Episodes used for training.
            - self.sources_for_val: Episodes used for validation.
            - self.sources_for_test: Episodes used for testing.
            - self.matches: A dictionary mapping validation/testing episodes to the training set.

        Notes
        -----
        `sources_for_*` is always kept in the same order as the corresponding `*_dfs` list, so the
        two can safely be zipped together.
        """
        if len(episodes) != len(run_to_failure) or len(episodes) != len(self.sources):
            raise ValueError(
                f"episodes ({len(episodes)}), run_to_failure ({len(run_to_failure)}) and the "
                f"known episode names ({len(self.sources)}) must all have the same length."
            )

        for i in range(len(episodes)):
            episodes[i][self.datetime_column] = pd.to_datetime(episodes[i][self.datetime_column])

        episode_names = self.sources
        self.rtf_dict = {name: int(rtf) for name, rtf in zip(episode_names, run_to_failure)}
        episode_by_name = {name: episode for name, episode in zip(episode_names, episodes)}

        ratios_given = [isinstance(value, float) or isinstance(value, int)
                        for value in (self.train_sources, self.val_sources, self.test_sources)]
        lists_given = [isinstance(value, (list, tuple, set))
                       for value in (self.train_sources, self.val_sources, self.test_sources)]

        if all(ratios_given):
            for_train, for_val, for_test = self._split_by_ratio(episode_names)
        elif all(lists_given):
            for_train, for_val, for_test = self._split_by_explicit_lists(episode_names)
        else:
            raise ValueError(
                "train_sources, val_sources and test_sources must either all be ratios (floats) "
                "or all be lists of source/episode names, got "
                f"{type(self.train_sources).__name__}, {type(self.val_sources).__name__} and "
                f"{type(self.test_sources).__name__}."
            )

        self._validate_split(for_train, for_val, for_test)

        # Keep the resolved episode names on the *_sources attributes so that they are consistent
        # across every code path (they used to hold original source names in one branch only).
        self.train_sources = list(for_train)
        self.val_sources = list(for_val)
        self.test_sources = list(for_test)

        self.train_dfs = [episode_by_name[name] for name in for_train]
        self.val_dfs = [episode_by_name[name] for name in for_val]
        self.test_dfs = [episode_by_name[name] for name in for_test]

        self.matches = {name: self.train_source_name for name in list(for_val) + list(for_test)}

        self.sources_for_train = list(for_train)
        self.sources_for_val = list(for_val)
        self.sources_for_test = list(for_test)

    def _split_by_ratio(self, episode_names):
        """Split the episodes according to the configured train/val/test ratios."""
        total = self.train_sources + self.val_sources + self.test_sources
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "When train_sources, val_sources and test_sources are passed as floats (ratios), "
                f"they must sum to 1, got {total}."
            )
        if min(self.train_sources, self.val_sources, self.test_sources) < 0:
            raise ValueError("train_sources, val_sources and test_sources ratios must be >= 0.")

        self.train_ratio = float(self.train_sources)
        self.val_ratio = float(self.val_sources)
        self.test_ratio = float(self.test_sources)

        episodes_per_source = {}
        for name in episode_names:
            episodes_per_source.setdefault(self.episode_to_source[name], []).append(name)

        # One generator for the whole split: a fresh random.Random(seed) per group would
        # apply the identical permutation to every group, which systematically sends the
        # same episode index of every source to the same split.
        rng = random.Random(self.random_state) if self.random_state is not None else None

        sources_with_failure = [source for source in self.original_sources
                                if self.original_s_has_f.get(source, False)]

        if not self.in_source_split and len(sources_with_failure) >= MIN_EPISODES_FOR_SPLIT:
            # Split at the original source level: a source never contributes to two splits.
            sources_without_failure = [source for source in self.original_sources
                                       if not self.original_s_has_f.get(source, False)]
            groups = [self._shuffled(sources_with_failure, rng),
                      self._shuffled(sources_without_failure, rng)]
            split_sources = self._allocate_groups(groups)
            split_sources = self._repair_split(
                split_sources,
                {source: int(self.original_s_has_f.get(source, False))
                 for source in self.original_sources},
                unit_name='source',
            )
            return tuple(
                [name for source in bucket for name in episodes_per_source.get(source, [])]
                for bucket in split_sources
            )

        # Either in_source_split is True or fewer than three original sources contain a failure:
        # fall back to splitting at the episode level, source by source.
        groups = [self._shuffled(episodes_per_source[source], rng)
                  for source in self.original_sources if source in episodes_per_source]
        split_episodes = self._allocate_groups(groups)
        split_episodes = self._repair_split(split_episodes, self.rtf_dict, unit_name='episode')
        return tuple(split_episodes)

    @staticmethod
    def _shuffled(units, rng):
        """Return `units` shuffled with `rng`, or untouched when `rng` is None."""
        units = list(units)
        if rng is not None:
            rng.shuffle(units)
        return units

    def _allocate_groups(self, groups):
        """Allocate every group of units across train/val/test honouring the configured ratios.

        Allocation is global rather than per group: each group is sized against the *cumulative*
        ideal implied by the ratios, so rounding losses in one group are repaid by the next. A
        per-group allocation would round every small group independently and, because each group
        would then have to be made non-empty on its own, would collapse the realised split towards
        equal thirds no matter what was requested. Non-emptiness of the three splits is a global
        property and is enforced once, by `_repair_split`.
        """
        ratios = [self.train_ratio, self.val_ratio, self.test_ratio]
        buckets = [[], [], []]
        totals = [0, 0, 0]
        assigned = 0

        for units in groups:
            count = len(units)
            if count == 0:
                continue
            assigned += count
            sizes = self._allocate_counts(count, ratios, totals, assigned)
            position = 0
            for index, size in enumerate(sizes):
                buckets[index].extend(units[position:position + size])
                totals[index] += size
                position += size
        return buckets

    @staticmethod
    def _allocate_counts(count, ratios, totals, assigned):
        """Size one group of `count` units against the cumulative ratio targets.

        `totals` holds what each split already received and `assigned` the number of units
        allocated once this group is placed, so ``ratios[k] * assigned - totals[k]`` is the number
        of units this group still owes split `k`. Whole units are handed out first and the
        leftover goes to the splits with the largest outstanding fraction, which is largest
        remainder rounding carried across groups instead of restarted for each one.

        Returns three non-negative sizes summing exactly to `count`.
        """
        needs = [ratios[index] * assigned - totals[index] for index in range(3)]
        sizes = [max(0, int(math.floor(need))) for need in needs]

        # Clamping negative needs to zero can push the total over `count`; give back from
        # whichever split is furthest above what it is owed.
        while sum(sizes) > count:
            index = max((k for k in range(3) if sizes[k] > 0),
                        key=lambda k: (sizes[k] - needs[k], k))
            sizes[index] -= 1

        remainder = count - sum(sizes)
        order = sorted(range(3), key=lambda index: (-(needs[index] - sizes[index]), index))
        for index in order[:remainder]:
            sizes[index] += 1
        return sizes

    @staticmethod
    def _most_deficient(totals, ratios):
        """Return the index of the bucket that is furthest below its target share."""
        assigned = sum(totals) + 1
        deficits = [ratios[index] * assigned - totals[index] for index in range(3)]
        return max(range(3), key=lambda index: (deficits[index], -index))

    def _repair_split(self, buckets, has_failure, unit_name):
        """Make sure every split is non-empty and contains at least one failing unit."""
        total_units = sum(len(bucket) for bucket in buckets)
        if total_units < MIN_EPISODES_FOR_SPLIT:
            raise ValueError(
                f"Cannot build a train/validation/test split from {total_units} {unit_name}(s): "
                f"at least {MIN_EPISODES_FOR_SPLIT} are required. Provide more sources, enable "
                "in_source_split so that episodes are split within each source, or pass explicit "
                "train_sources/val_sources/test_sources lists."
            )

        # (a) every split must hold at least one unit
        for index in range(3):
            if buckets[index]:
                continue
            donor = max(range(3), key=lambda other: len(buckets[other]))
            if len(buckets[donor]) < 2:
                raise ValueError(
                    f"Cannot build a train/validation/test split from {total_units} "
                    f"{unit_name}(s): at least {MIN_EPISODES_FOR_SPLIT} are required."
                )
            warnings.warn(
                f"The requested ratios left the {('train', 'validation', 'test')[index]} split "
                f"empty; one {unit_name} was moved into it because every split must hold at "
                "least one run-to-failure episode.",
                stacklevel=2,
            )
            buckets[index].append(buckets[donor].pop())

        # (b) every split must hold at least one run-to-failure unit
        failing_total = sum(1 for bucket in buckets for unit in bucket if has_failure.get(unit, 0))
        for index in range(3):
            if any(has_failure.get(unit, 0) for unit in buckets[index]):
                continue
            donor = None
            for other in sorted(range(3),
                                key=lambda k: -sum(1 for u in buckets[k] if has_failure.get(u, 0))):
                if sum(1 for unit in buckets[other] if has_failure.get(unit, 0)) >= 2:
                    donor = other
                    break
            if donor is None:
                raise ValueError(
                    f"At least one {unit_name} with a failure event must be present in each of the "
                    f"train, validation and test sets, but only {failing_total} of {total_units} "
                    f"{unit_name}(s) end with a failure. Provide more failing sources, enable "
                    "in_source_split so that episodes are split within each source, or pass "
                    "explicit train_sources/val_sources/test_sources lists."
                )
            failing_unit = next(unit for unit in buckets[donor] if has_failure.get(unit, 0))
            healthy_unit = next((unit for unit in buckets[index] if not has_failure.get(unit, 0)),
                                None)
            buckets[donor].remove(failing_unit)
            buckets[index].append(failing_unit)
            if healthy_unit is not None:
                buckets[index].remove(healthy_unit)
                buckets[donor].append(healthy_unit)
        return buckets

    def _split_by_explicit_lists(self, episode_names):
        """Resolve user supplied train/val/test lists into episode names.

        Entries may be either episode names (``T01_ep0``) or original source names (``T01``); a
        source name is expanded into all of its episodes.
        """
        known_episodes = set(episode_names)
        episodes_per_source = {}
        for name in episode_names:
            episodes_per_source.setdefault(self.episode_to_source[name], []).append(name)

        resolved = []
        for label, requested in (('train_sources', self.train_sources),
                                 ('val_sources', self.val_sources),
                                 ('test_sources', self.test_sources)):
            names = []
            for entry in requested:
                entry = str(entry)
                if entry in known_episodes:
                    if entry in episodes_per_source:
                        warnings.warn(
                            f"{label} entry '{entry}' names both an episode and a source; it is "
                            "read as the episode. Rename the source to remove the ambiguity.",
                            stacklevel=2,
                        )
                    names.append(entry)
                elif entry in episodes_per_source:
                    names.extend(episodes_per_source[entry])
                else:
                    raise ValueError(
                        f"{label} refers to '{entry}', which is neither a known source "
                        f"({sorted(episodes_per_source)}) nor a known episode "
                        f"({sorted(known_episodes)})."
                    )
            duplicates = {name for name in names if names.count(name) > 1}
            if duplicates:
                raise ValueError(f"{label} contains duplicate episodes: {sorted(duplicates)}.")
            resolved.append(names)
        return tuple(resolved)

    def _validate_split(self, for_train, for_val, for_test):
        """Check that the three splits are usable: non-empty, disjoint and failure covering."""
        named_splits = (('train', for_train), ('validation', for_val), ('test', for_test))

        for label, names in named_splits:
            if not names:
                raise ValueError(
                    f"The {label} split is empty. Adjust train_sources/val_sources/test_sources "
                    "or provide more sources."
                )

        for (first_label, first), (second_label, second) in ((named_splits[0], named_splits[1]),
                                                             (named_splits[0], named_splits[2]),
                                                             (named_splits[1], named_splits[2])):
            overlap = set(first) & set(second)
            if overlap:
                raise ValueError(
                    f"The {first_label} and {second_label} splits overlap on {sorted(overlap)}; "
                    "an episode cannot belong to two splits."
                )

        if not self.in_source_split:
            sources_per_split = [{self.episode_to_source[name] for name in names}
                                 for _, names in named_splits]
            for first in range(3):
                for second in range(first + 1, 3):
                    shared = sources_per_split[first] & sources_per_split[second]
                    if shared:
                        warnings.warn(
                            f"Sources {sorted(shared)} contribute episodes to both the "
                            f"{named_splits[first][0]} and {named_splits[second][0]} splits.",
                            stacklevel=2,
                        )

        assigned = set(for_train) | set(for_val) | set(for_test)
        unused = [name for name in self.sources if name not in assigned]
        if unused:
            warnings.warn(
                f"{len(unused)} episode(s) are not part of any split and will be ignored: "
                f"{unused}.",
                stacklevel=2,
            )

        for label, names in named_splits:
            if not any(self.rtf_dict[name] == 1 for name in names):
                raise ValueError(
                    f"At least one source/episode with a failure event must be present in each of "
                    f"train, val and test sets, but the {label} split ({names}) contains none."
                )

    def safe_splitting(self, source_units, at_least_one_failure_in_train=False):
        """Split a list of source/episode names into train, validation and test.

        Kept for backwards compatibility; the splitting logic now lives in `_allocate_groups`
        and `_repair_split`, which also guarantee failure coverage across the three splits.

        Parameters
        ----------
        source_units : list[str]
            Source or episode names to split.
        at_least_one_failure_in_train : bool, default=False
            When True, a failing unit is guaranteed to end up in the training split.

        Returns
        -------
        tuple[list, list, list]
            The train, validation and test name lists.
        """
        units = list(source_units)
        if not units:
            return [], [], []

        if not hasattr(self, 'train_ratio'):
            raise ValueError(
                "safe_splitting needs the train/val/test ratios, which are only recorded when "
                "the Dataset was built with float ratios. This Dataset was built from explicit "
                "source lists, so there are no ratios to apply."
            )
        ratios = [self.train_ratio, self.val_ratio, self.test_ratio]
        if len(units) >= MIN_EPISODES_FOR_SPLIT:
            # A standalone allocation: no carry-over, since this helper splits one list on its own.
            sizes = self._allocate_counts(len(units), ratios, [0, 0, 0], len(units))
        elif len(units) == 2:
            sizes = [1, 1, 0]
        else:
            sizes = [1, 0, 0]

        buckets = [[], [], []]
        position = 0
        for index, size in enumerate(sizes):
            buckets[index].extend(units[position:position + size])
            position += size

        if at_least_one_failure_in_train and buckets[0] and \
                not any(self.rtf_dict.get(unit, 0) for unit in buckets[0]):
            for bucket in buckets[1:]:
                failing = next((unit for unit in bucket if self.rtf_dict.get(unit, 0)), None)
                if failing is not None:
                    bucket.remove(failing)
                    bucket.append(buckets[0].pop())
                    buckets[0].append(failing)
                    break
        return buckets[0], buckets[1], buckets[2]

    # ------------------------------------------------------------------ #
    # Flavor specific dataset builders
    # ------------------------------------------------------------------ #
    def get_rul_dataset(self, keep_sources=None):
        """Generate RUL (Remaining Useful Life) prediction dataset.

        Creates training, validation, and testing datasets optimized for RUL regression tasks.
        Uses only run-to-failure episodes for training and generates RUL labels indicating
        time remaining until failure.

        Parameters
        ----------
        keep_sources : str, optional
            If provided, preserves this column (e.g., 'source') in the dataset
            for source tracking. Otherwise, removes source and RUL columns.

        Returns
        -------
        tuple[dict, dict]
            (dataset, test_dataset) - Two dictionaries containing:
            - 'match_sources': Source mapping for transfer learning
            - 'target_sources': Sources used for validation/testing
            - 'target_data': Feature data for val/test
            - 'target_labels': RUL values (time to failure) for val/test
            - 'is_failure': Whether each source had failures
            - 'historic_data': Training data (run-to-failure episodes only)
            - 'historic_sources': Source names for training data
            - 'anomaly_labels': RUL labels for training data
            - 'predictive_horizon': Time window before failure
            - 'slide': Sliding window step size
            - 'lead': Lead time for predictions
            - 'beta': Objective weighting parameter

        Examples
        --------
        >>> dataset_obj = Dataset(data, 'timestamp', failure_column='is_failure')
        >>> train_set, test_set = dataset_obj.get_rul_dataset()
        >>> # Access training RUL data
        >>> rul_labels = train_set['anomaly_labels'][0]
        """
        self._validate_keep_sources(keep_sources)
        failing_train_dfs = [df for df in self.train_dfs
                             if self.rtf_dict[df.iloc[0][self.source_column]] == 1]
        if not failing_train_dfs:
            raise ValueError(
                "get_rul_dataset needs at least one run-to-failure episode in the training split."
            )
        concatenated_train = pd.concat(failing_train_dfs, ignore_index=True)

        cols_to_drop = [self.source_column, self.rul_column]
        if keep_sources is not None and keep_sources in cols_to_drop:
            cols_to_drop.remove(keep_sources)

        event_data, event_preferences = self._empty_event_definition()

        dataset = {}
        dataset['match_sources'] = self.matches
        dataset['target_sources'] = [str(vid) for vid in self.sources_for_val]
        dataset['target_data'] = self._build_target_data(self.val_dfs, cols_to_drop, keep_sources)
        dataset['is_failure'] = [self.rtf_dict[str(vid)] for vid in self.sources_for_val]
        dataset['target_labels'] = [df[self.rul_column].values for df in self.val_dfs]

        # Read the labels before `keep_sources` can add a column of the same name.
        rul_labels = concatenated_train[self.rul_column].values
        if keep_sources is not None:
            concatenated_train[keep_sources] = list(concatenated_train[self.source_column])
        dataset['historic_data'] = [concatenated_train.drop(columns=cols_to_drop)]
        dataset['historic_sources'] = [self.train_source_name]
        dataset['anomaly_labels'] = [rul_labels]
        dataset["dates"] = self.datetime_column
        dataset["event_preferences"] = event_preferences
        dataset["event_data"] = event_data
        self._add_objective_parameters(dataset)

        # ############# test dataset ############## #

        test_dataset = {}
        test_dataset['match_sources'] = self.matches
        test_dataset['target_sources'] = [str(vid) for vid in self.sources_for_test]
        test_dataset['target_data'] = self._build_target_data(self.test_dfs, cols_to_drop,
                                                              keep_sources)
        test_dataset['target_labels'] = [df[self.rul_column].values for df in self.test_dfs]
        test_dataset['is_failure'] = [self.rtf_dict[str(vid)] for vid in self.sources_for_test]

        test_dataset['historic_data'] = [concatenated_train.drop(columns=cols_to_drop)]
        test_dataset['historic_sources'] = [self.train_source_name]
        test_dataset['anomaly_labels'] = [rul_labels]
        test_dataset["dates"] = self.datetime_column
        test_dataset["event_preferences"] = event_preferences
        test_dataset["event_data"] = event_data
        self._add_objective_parameters(test_dataset)

        return dataset, test_dataset

    def df_to_x_y_surv(self, df, indicator=None, event_column="event"):
        """Convert dataframe to survival analysis format (time, event) tuples.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing RUL and event columns.
        indicator : int, optional
            Event indicator value (0 or 1). If None, uses `event_column` from df.
        event_column : str, default="event"
            Name of the column holding the event indicator.

        Returns
        -------
        list[tuple]
            List of (rul, event) tuples for survival analysis models.
        """
        if indicator is None:
            y = [(rul, ev) for ev, rul in zip(df[event_column], df[self.rul_column])]
        else:
            y = [(rul, indicator) for rul in df[self.rul_column]]
        return y

    def get_SA_dataset(self, keep_sources=None):
        """Generate Survival Analysis dataset with reliability labels.

        Creates datasets for survival regression tasks where the goal is to predict
        survival probabilities or remaining time until events. Combines all training
        episodes and marks event indicators (failure/maintenance).

        Parameters
        ----------
        keep_sources : str, optional
            If provided, preserves this column for source tracking.

        Returns
        -------
        tuple[dict, dict]
            (dataset, test_dataset) - Dictionaries containing:
            - 'target_labels': Tuples of (RUL, event_indicator) for each sample
            - 'anomaly_labels': Tuples of (RUL, event_flag) for training
            - Other fields same as get_rul_dataset()

        Notes
        -----
        - Survival analysis labels are tuples (time, event) used by survival methods
        - Event indicator: 1 for failure, 0 for maintenance/reset
        - Combines event information from the rtf_dict (run-to-failure mapping)
        """
        self._validate_keep_sources(keep_sources)
        event_column = self._survival_event_column()
        train_dfs_with_events = []
        for df in self.train_dfs:
            df_with_event = df.copy()
            df_with_event[event_column] = self.rtf_dict[df.iloc[0][self.source_column]]
            train_dfs_with_events.append(df_with_event)
        concatenated_train = pd.concat(train_dfs_with_events, ignore_index=True)

        # TODO: investigate how to deal with the case of only run-to-failure episodes
        if concatenated_train[event_column].min() > 0:
            event_list = list(concatenated_train[event_column])
            event_list[0] = 0
            concatenated_train[event_column] = event_list

        cols_to_drop = [event_column, self.source_column, self.rul_column]
        if keep_sources is not None and keep_sources in cols_to_drop:
            cols_to_drop.remove(keep_sources)
        target_cols_to_drop = [self.source_column, self.rul_column]
        if keep_sources is not None and keep_sources in target_cols_to_drop:
            target_cols_to_drop.remove(keep_sources)

        event_data, event_preferences = self._empty_event_definition()

        dataset = {}
        dataset['match_sources'] = self.matches
        dataset['target_sources'] = [str(vid) for vid in self.sources_for_val]
        dataset['target_data'] = self._build_target_data(self.val_dfs, target_cols_to_drop,
                                                         keep_sources)
        dataset['target_labels'] = [
            self.df_to_x_y_surv(df, indicator=self.rtf_dict[df.iloc[0][self.source_column]])
            for df in self.val_dfs
        ]
        dataset['is_failure'] = [self.rtf_dict[str(vid)] for vid in self.sources_for_val]
        if keep_sources is not None:
            concatenated_train[keep_sources] = list(concatenated_train[self.source_column])
        dataset['historic_data'] = [concatenated_train.drop(columns=cols_to_drop)]
        dataset['historic_sources'] = [self.train_source_name]
        dataset['anomaly_labels'] = [self.df_to_x_y_surv(concatenated_train,
                                                         event_column=event_column)]
        dataset["dates"] = self.datetime_column
        dataset["event_preferences"] = event_preferences
        dataset["event_data"] = event_data
        self._add_objective_parameters(dataset)

        # ############# test dataset ############## #

        test_dataset = {}
        test_dataset['match_sources'] = self.matches
        test_dataset['target_sources'] = [str(vid) for vid in self.sources_for_test]
        test_dataset['target_data'] = self._build_target_data(self.test_dfs, target_cols_to_drop,
                                                              keep_sources)
        test_dataset['target_labels'] = [
            self.df_to_x_y_surv(df, indicator=self.rtf_dict[df.iloc[0][self.source_column]])
            for df in self.test_dfs
        ]
        test_dataset['is_failure'] = [self.rtf_dict[str(vid)] for vid in self.sources_for_test]
        test_dataset['historic_data'] = [concatenated_train.drop(columns=cols_to_drop)]
        test_dataset['historic_sources'] = [self.train_source_name]
        test_dataset['anomaly_labels'] = [
            self.df_to_x_y_surv(concatenated_train, event_column=event_column)]
        test_dataset["dates"] = self.datetime_column
        test_dataset["event_preferences"] = event_preferences
        test_dataset["event_data"] = event_data
        self._add_objective_parameters(test_dataset)

        return dataset, test_dataset

    def generate_binary_labels(self, sources, list_dfs):
        """Generate binary anomaly labels based on predictive horizon and lead time.

        Creates binary labels (0=normal, 1=anomaly) by identifying samples within
        the predictive horizon before failure events and considering lead time.

        Parameters
        ----------
        sources : list[str]
            Source identifiers corresponding to dataframes, in the same order as `list_dfs`.
        list_dfs : list[pd.DataFrame]
            List of episode dataframes to label.

        Returns
        -------
        tuple[list, list]
            (final_ranges, leadranges) - Lists of binary label arrays and lead time flags.

        Notes
        -----
        - Uses predictive_horizon and lead time from Dataset initialization
        - Any sample within lead range (before failure) is marked as 1
        - Helper uses _data_formulation and extract_anomaly_ranges from evaluation module
        """
        from pdmlabs.evaluation.evaluation import _data_formulation, extract_anomaly_ranges

        if len(sources) != len(list_dfs):
            raise ValueError(
                f"generate_binary_labels received {len(sources)} sources for {len(list_dfs)} "
                "dataframes; the two must be aligned."
            )

        datesofscores = [[dtt for dtt in pd.to_datetime(df[self.datetime_column])]
                         for df in list_dfs]

        PH = self.predictive_horizon
        lead = self.lead
        isfailure = [self.rtf_dict[str(vid)] for vid in sources]

        predictions, threshold, datesofscores, maintenances, isfailure, PHS_leads = \
            _data_formulation(datesofscores, datesofscores, datesofscores, isfailure, None, [],
                              PH, lead)

        anomalyranges, leadranges = extract_anomaly_ranges(maintenances, PHS_leads, isfailure,
                                                           datesofscores)
        final_ranges = []
        pos = 0
        for df in list_dfs:
            temp_copy = anomalyranges[pos:pos + df.shape[0]].copy()
            temp_lead_copy = leadranges[pos:pos + df.shape[0]].copy()
            for i in range(len(temp_copy)):
                if temp_lead_copy[i] != 0:
                    temp_copy[i] = 1
            final_ranges.append(temp_copy)
            pos += df.shape[0]
        return final_ranges, leadranges

    def get_events_from_df(self, df_list):
        """Build ``[date, type, source, description]`` rows for the end of each episode."""
        events = []
        for df in df_list:
            is_fail = self.rtf_dict[df.iloc[0][self.source_column]]
            if is_fail == 1:
                events.append([df[self.datetime_column].max(), "failure",
                               df.iloc[0][self.source_column], "failure"])
            else:
                events.append([df[self.datetime_column].max(), "reset",
                               df.iloc[0][self.source_column], "maintenance"])
        return events

    def get_Classification_dataset(self, keep_sources=None):
        """Generate a binary classification dataset.

        From train episodes without failures, the last predictive_horizon period is ignored to
        ensure healthy operation, based on the objective the user wants to optimize. Then binary
        labels are generated for all training data, labeling every record as 0, except those that
        lie within the predictive horizon before a failure event, which are labeled as 1.

        Parameters
        ----------
        keep_sources : str, optional
            If provided, preserves this column for source tracking.

        Returns
        -------
        tuple[dict, dict]
            (dataset, test_dataset) for validation and testing respectively.
        """
        self._validate_keep_sources(keep_sources)
        events = []
        events.extend(self.get_events_from_df(self.val_dfs))
        events.extend(self.get_events_from_df(self.test_dfs))
        event_data, event_preferences = self._episode_event_definition(events)

        clean_dfs = []
        clean_sources = []
        for name, df in zip(self.sources_for_train, self.train_dfs):
            is_fail = self.rtf_dict[df.iloc[0][self.source_column]]
            if is_fail == 0:
                new_df = df[df[self.datetime_column]
                            <= (df[self.datetime_column].iloc[-1]
                                - pd.Timedelta(self.predictive_horizon))]
            else:
                new_df = df
            if new_df.shape[0] > 0:
                clean_dfs.append(new_df)
                clean_sources.append(name)

        if not clean_dfs:
            raise ValueError(
                "Every training episode became empty after removing the last predictive_horizon "
                "period; use a shorter predictive_horizon or longer episodes."
            )

        historical_labels, leads = self.generate_binary_labels(clean_sources, clean_dfs)
        concatenated_train = pd.concat(clean_dfs, ignore_index=True)
        historical_labels = [label for sublist in historical_labels for label in sublist]

        cols_to_drop = [self.source_column]
        if "event" in concatenated_train.columns:
            cols_to_drop.append("event")
        if self.rul_column in concatenated_train.columns:
            cols_to_drop.append(self.rul_column)
        if keep_sources is not None and keep_sources in cols_to_drop:
            cols_to_drop.remove(keep_sources)

        dataset = {}
        dataset['match_sources'] = self.matches
        dataset['target_sources'] = [str(vid) for vid in self.sources_for_val]
        dataset['target_data'] = self._build_target_data(self.val_dfs, cols_to_drop, keep_sources)

        if keep_sources is not None:
            concatenated_train[keep_sources] = list(concatenated_train[self.source_column])
        dataset['historic_data'] = [concatenated_train.drop(columns=cols_to_drop)]
        dataset['historic_sources'] = [self.train_source_name]
        dataset['anomaly_labels'] = [historical_labels]
        dataset["dates"] = self.datetime_column
        dataset["event_preferences"] = event_preferences
        dataset["event_data"] = event_data
        self._add_objective_parameters(dataset)

        # ############# test dataset ############## #

        test_dataset = {}
        test_dataset['match_sources'] = self.matches
        test_dataset['target_sources'] = [str(vid) for vid in self.sources_for_test]
        test_dataset['target_data'] = self._build_target_data(self.test_dfs, cols_to_drop,
                                                              keep_sources)
        test_dataset['historic_data'] = [concatenated_train.drop(columns=cols_to_drop)]
        test_dataset['historic_sources'] = [self.train_source_name]
        test_dataset['anomaly_labels'] = [historical_labels]
        test_dataset["dates"] = self.datetime_column
        test_dataset["event_preferences"] = event_preferences
        test_dataset["event_data"] = event_data
        self._add_objective_parameters(test_dataset)

        return dataset, test_dataset

    def get_semi_dataset(self):
        """Generate a semi-supervised dataset.

        From the train episodes the last predictive_horizon period is removed so that only
        healthy operation remains, based on the objective the user wants to optimize. The result
        is used as unlabeled historical data to fit a semi-supervised anomaly detector.

        Returns
        -------
        tuple[dict, dict]
            (dataset, test_dataset) for validation and testing respectively.
        """
        events = []
        events.extend(self.get_events_from_df(self.val_dfs))
        events.extend(self.get_events_from_df(self.test_dfs))
        event_data, event_preferences = self._episode_event_definition(events)

        clean_dfs = []
        for df in self.train_dfs:
            # Always extract healthy data by removing the anomalous tail
            new_df = df[df[self.datetime_column]
                        <= (df[self.datetime_column].iloc[-1]
                            - pd.Timedelta(self.predictive_horizon))]
            if new_df.shape[0] > 0:
                clean_dfs.append(new_df)

        if not clean_dfs:
            raise ValueError(
                "Every training episode became empty after removing the last predictive_horizon "
                "period; use a shorter predictive_horizon or longer episodes."
            )

        concatenated_train = pd.concat(clean_dfs, ignore_index=True)

        cols_to_drop = [self.source_column]
        if "event" in concatenated_train.columns:
            cols_to_drop.append("event")
        if self.rul_column in concatenated_train.columns:
            cols_to_drop.append(self.rul_column)

        dataset = {}
        dataset['match_sources'] = self.matches
        dataset['target_sources'] = [str(vid) for vid in self.sources_for_val]
        dataset['target_data'] = self._build_target_data(self.val_dfs, cols_to_drop, None)
        dataset['historic_data'] = [concatenated_train.drop(columns=cols_to_drop)]
        dataset['historic_sources'] = [self.train_source_name]
        dataset["dates"] = self.datetime_column
        dataset["event_preferences"] = event_preferences
        dataset["event_data"] = event_data
        self._add_objective_parameters(dataset)

        # ############# test dataset ############## #

        test_dataset = {}
        test_dataset['match_sources'] = self.matches
        test_dataset['target_sources'] = [str(vid) for vid in self.sources_for_test]
        test_dataset['target_data'] = self._build_target_data(self.test_dfs, cols_to_drop, None)
        test_dataset['historic_data'] = [concatenated_train.drop(columns=cols_to_drop)]
        test_dataset['historic_sources'] = [self.train_source_name]
        test_dataset["dates"] = self.datetime_column
        test_dataset["event_preferences"] = event_preferences
        test_dataset["event_data"] = event_data
        self._add_objective_parameters(test_dataset)

        return dataset, test_dataset

    def get_unsupervised_dataset(self):
        """Generate an unsupervised dataset.

        No historical data is produced: the train and validation episodes are exposed as targets
        of the validation dictionary and the test episodes as targets of the test dictionary, so
        that an unsupervised detector is scored directly on the data it consumes.

        Returns
        -------
        tuple[dict, dict]
            (dataset, test_dataset) for validation and testing respectively.
        """
        events = []
        events.extend(self.get_events_from_df(self.train_dfs))
        events.extend(self.get_events_from_df(self.val_dfs))
        events.extend(self.get_events_from_df(self.test_dfs))
        event_data, event_preferences = self._episode_event_definition(events)

        train_val = list(self.train_dfs) + list(self.val_dfs)
        train_val_sources = list(self.sources_for_train) + list(self.sources_for_val)

        cols_to_drop = [self.source_column]
        if "event" in train_val[0].columns:
            cols_to_drop.append("event")
        if self.rul_column in train_val[0].columns:
            cols_to_drop.append(self.rul_column)

        # No `match_sources`: this flavor has no historic data, so mapping its targets onto a
        # non-existent 'train' source would be wrong and would suppress the identity fallback
        # that the batch experiments apply when the key is absent.
        dataset = {}
        dataset['target_sources'] = [str(vid) for vid in train_val_sources]
        dataset["max_wait_time"] = self.max_wait_time
        dataset['target_data'] = self._build_target_data(train_val, cols_to_drop, None)
        dataset['historic_data'] = []
        dataset['historic_sources'] = []
        dataset["dates"] = self.datetime_column
        dataset["event_preferences"] = event_preferences
        dataset["event_data"] = event_data
        dataset['predictive_horizon'] = self.predictive_horizon
        dataset['slide'] = self.slide
        dataset['lead'] = self.lead
        dataset['beta'] = self.beta
        self._add_snapshot_column(dataset)

        # ############# test dataset ############## #

        test_dataset = {}
        test_dataset["max_wait_time"] = self.max_wait_time
        test_dataset['target_sources'] = [str(vid) for vid in self.sources_for_test]
        test_dataset['target_data'] = self._build_target_data(self.test_dfs, cols_to_drop, None)
        test_dataset['historic_data'] = []
        test_dataset['historic_sources'] = []
        test_dataset["dates"] = self.datetime_column
        test_dataset["event_preferences"] = event_preferences
        test_dataset["event_data"] = event_data
        test_dataset['predictive_horizon'] = self.predictive_horizon
        test_dataset['slide'] = self.slide
        test_dataset['lead'] = self.lead
        test_dataset['beta'] = self.beta
        self._add_snapshot_column(test_dataset)

        return dataset, test_dataset

    # ------------------------------------------------------------------ #
    # Small shared helpers
    # ------------------------------------------------------------------ #
    def _survival_event_column(self):
        """Pick a name for the synthetic survival event column that no feature already uses."""
        name = "event"
        existing = set(self.train_dfs[0].columns)
        while name in existing:
            name = f"_{name}"
        return name

    def _validate_keep_sources(self, keep_sources):
        """Reject a `keep_sources` name that would overwrite labels or a real feature column."""
        if keep_sources is None:
            return
        if keep_sources == self.rul_column:
            raise ValueError(
                f"keep_sources cannot be '{self.rul_column}': the source names would overwrite "
                "the RUL labels."
            )
        if keep_sources == self.source_column:
            return
        existing = set(self.train_dfs[0].columns) - {self.source_column, self.rul_column}
        if keep_sources in existing:
            raise ValueError(
                f"keep_sources='{keep_sources}' collides with an existing feature column; the "
                "source names would overwrite it. Choose a name that is not already in the data."
            )

    def _build_target_data(self, dfs, cols_to_drop, keep_sources):
        """Drop the bookkeeping columns from every target episode."""
        target_data = []
        for df in dfs:
            columns = [column for column in cols_to_drop if column in df.columns]
            tdf = df.drop(columns=columns).reset_index(drop=True).copy()
            if keep_sources is not None:
                tdf[keep_sources] = df[self.source_column].reset_index(drop=True)
            target_data.append(tdf)
        return target_data

    def _add_objective_parameters(self, dataset):
        """Attach the shared objective parameters to a flavor dictionary."""
        dataset['predictive_horizon'] = self.predictive_horizon
        dataset['slide'] = self.slide
        dataset['lead'] = self.lead
        dataset['beta'] = self.beta
        dataset['max_wait_time'] = self.max_wait_time
        self._add_snapshot_column(dataset)

    def _add_snapshot_column(self, dataset):
        """Mark a flavor dictionary of a snapshot dataset, so the pipeline featurizes it."""
        if self.snapshot_column is not None:
            dataset['snapshot_column'] = self.snapshot_column

    @staticmethod
    def _empty_event_definition():
        """Return an empty event frame plus empty event preferences."""
        event_data = pd.DataFrame(columns=["date", "type", "source", "description"])
        event_preferences = {'failure': [], 'reset': []}
        return event_data, event_preferences

    @staticmethod
    def _episode_event_definition(events):
        """Return the event frame and preferences describing the end of each episode."""
        from pdmlabs.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple

        event_data = pd.DataFrame(events, columns=["date", "type", "source", "description"])
        event_preferences: EventPreferences = {
            'failure': [
                EventPreferencesTuple(description='*', type='failure', source='*',
                                      target_sources='=')
            ],
            'reset': [
                EventPreferencesTuple(description='*', type='failure', source='*',
                                      target_sources='='),
                EventPreferencesTuple(description='*', type='reset', source='*',
                                      target_sources='=')
            ]
        }
        return event_data, event_preferences


def episodes_formulation(data, datetime_column, event_indicator=None, maintenance_list=None,
                         failure_list=None, event_df=None, source_column='source', DIVIDER=3600,
                         keep_censored_tail=True):
    """Split `data` into episodes and mark which of them end with a failure.

    Four mutually exclusive strategies are supported, in this order of precedence:

    1. `event_df` is given: episodes are delimited by the events of `event_df` whose `code`
       belongs to `failure_list` (run-to-failure) or `maintenance_list` (reset).
    2. `event_indicator` is given: every source forms a single episode whose outcome is read
       from that (constant per source) binary column.
    3. `maintenance_list`/`failure_list` name binary indicator columns of `data`: episodes are
       delimited by the rows where those columns equal 1.
    4. Nothing is given: every source is assumed to be a single run-to-failure episode.

    Returns
    -------
    tuple
        (episodes, run_to_failure, episode_sources, original_s_has_f, episode_to_source)
    """
    if event_df is not None:
        if datetime_column not in event_df.columns or source_column not in event_df.columns \
                or "code" not in event_df.columns:
            raise ValueError(
                f"event_df must contain the columns '{datetime_column}', '{source_column}' and "
                f"'code', got {list(event_df.columns)}."
            )
        if failure_list is None and maintenance_list is None:
            raise ValueError(
                "When event_df is provided, failure_column and/or maintenance_column must list "
                "the event codes that end an episode."
            )
        if datetime_column not in data.columns or source_column not in data.columns:
            raise ValueError(
                f"data must contain the columns '{datetime_column}' and '{source_column}'."
            )

        # A bare string would be iterated character-wise into a set of letters.
        if isinstance(failure_list, str):
            failure_list = [failure_list]
        if isinstance(maintenance_list, str):
            maintenance_list = [maintenance_list]
        failure_codes = set(failure_list or [])
        maintenance_codes = set(maintenance_list or []).difference(failure_codes)
        maintenance_col = "maintenance_event"
        failure_col = "failure_event"

        event_df = event_df.copy()
        event_df[maintenance_col] = [1 if code in maintenance_codes else 0
                                     for code in event_df["code"].values]
        event_df[failure_col] = [1 if code in failure_codes else 0
                                 for code in event_df["code"].values]
        event_df[datetime_column] = pd.to_datetime(event_df[datetime_column])
        event_df[source_column] = event_df[source_column].astype(str)
        event_data = event_df[[datetime_column, source_column, maintenance_col, failure_col]].copy()

        # `episodes_formulation` is a public entry point, so normalise both sides here rather
        # than relying on the caller: an int/str or float/str mismatch between the two frames
        # would match no event at all and quietly turn every source into a censored episode.
        data = data.copy()
        data[source_column] = data[source_column].astype(str)
        data[datetime_column] = pd.to_datetime(data[datetime_column])

        data_is_aware = isinstance(data[datetime_column].dtype, pd.DatetimeTZDtype)
        events_are_aware = isinstance(event_data[datetime_column].dtype, pd.DatetimeTZDtype)
        if data_is_aware != events_are_aware:
            raise ValueError(
                f"The '{datetime_column}' column is timezone-aware in "
                f"{'data' if data_is_aware else 'event_df'} but timezone-naive in "
                f"{'event_df' if data_is_aware else 'data'}; make both sides consistent "
                "(for example with .dt.tz_localize(None))."
            )

        recognised_events = event_data[(event_data[failure_col] == 1)
                                       | (event_data[maintenance_col] == 1)]
        if recognised_events.shape[0] and not set(recognised_events[source_column]) \
                & set(data[source_column]):
            warnings.warn(
                "No event source matches any source of data "
                f"(events: {sorted(set(recognised_events[source_column]))[:5]}, "
                f"data: {sorted(set(data[source_column]))[:5]}); every source will become a "
                "single censored episode.",
                stacklevel=2,
            )

        all_sources = []
        all_episodes = []
        all_run_to_failure = []
        original_s_has_f = {}
        episode_to_source = {}
        for source in data[source_column].unique():
            df_source = data[data[source_column] == source].copy()

            episodes, rtfs, new_sources = data_split_by_event(
                df_source, event_data[event_data[source_column] == source].copy(),
                datetime_column, failure_col, maintenance_col, source_column, DIVIDER,
                keep_censored_tail,
            )
            all_episodes.extend(episodes)
            all_run_to_failure.extend(rtfs)
            # A source without any recognised event yields at most one censored episode, so
            # `rtfs` can legitimately be empty here.
            original_s_has_f[source] = max(rtfs) == 1 if rtfs else False
            all_sources.extend(new_sources)
            for new_source in new_sources:
                episode_to_source[new_source] = source

            if not episodes:
                warnings.warn(
                    f"Source '{source}' produced no episode: none of its events matched the "
                    "configured failure/maintenance codes and no data remained.",
                    stacklevel=2,
                )

        return all_episodes, all_run_to_failure, all_sources, original_s_has_f, episode_to_source

    if event_indicator is not None:
        # group by source and read the (constant per source) event indicator
        all_episodes = []
        all_run_to_failure = []
        all_sources = []
        original_s_has_f = {}
        episode_to_source = {}
        for source, group_df in data.groupby(source_column):
            group_df = group_df.sort_values(by=datetime_column).reset_index(drop=True)
            unique_values = set(group_df[event_indicator].unique())
            if not unique_values.issubset({0, 1}):
                raise ValueError(
                    f"event_indicator column must be binary (0 and 1) for each source. Source "
                    f"{source} has values {sorted(unique_values)}."
                )
            if len(unique_values) > 1:
                raise ValueError(
                    f"event_indicator column must be constant per source, but source {source} "
                    f"has values {sorted(unique_values)}. If '{event_indicator}' flags the row on "
                    f"which an event occurred, pass it as failure_column instead so that the "
                    f"source is split into episodes."
                )
            if "RUL" not in group_df.columns:
                maxdate = group_df[datetime_column].max()
                group_df["RUL"] = [(maxdate - dtime).total_seconds() / DIVIDER
                                   for dtime in group_df[datetime_column]]

            is_failure = int(group_df.iloc[0][event_indicator])
            all_run_to_failure.append(is_failure)
            original_s_has_f[source] = is_failure == 1
            all_episodes.append(group_df.drop(columns=[event_indicator]))
            all_sources.append(source)
            episode_to_source[source] = source
        return all_episodes, all_run_to_failure, all_sources, original_s_has_f, episode_to_source

    if _is_indicator_column(maintenance_list, data) or _is_indicator_column(failure_list, data):
        return _episodes_from_indicator_columns(data, datetime_column, maintenance_list,
                                                failure_list, source_column, DIVIDER,
                                                keep_censored_tail)

    if maintenance_list is not None or failure_list is not None:
        raise ValueError(
            f"maintenance_column={maintenance_list!r} / failure_column={failure_list!r} are "
            "neither columns of data nor lists of event codes accompanied by event_df."
        )

    warnings.warn(
        "No event column was found in data and no event_df was given; every source is treated as "
        "a single run-to-failure episode.",
        stacklevel=2,
    )
    all_episodes = []
    all_run_to_failure = []
    original_s_has_f = {}
    all_sources = []
    episode_to_source = {}
    for source, group_df in data.groupby(source_column):
        group_df = group_df.sort_values(by=datetime_column).reset_index(drop=True)
        if "RUL" not in group_df.columns:
            maxdate = group_df[datetime_column].max()
            group_df["RUL"] = [(maxdate - dtime).total_seconds() / DIVIDER
                               for dtime in group_df[datetime_column]]
        all_run_to_failure.append(1)
        original_s_has_f[source] = True
        all_episodes.append(group_df)
        all_sources.append(source)
        episode_to_source[source] = source
    return all_episodes, all_run_to_failure, all_sources, original_s_has_f, episode_to_source


def _is_indicator_column(candidate, data):
    """True when `candidate` names a column of `data`."""
    return isinstance(candidate, str) and candidate in data.columns


def _episodes_from_indicator_columns(data, datetime_column, maintenance_column, failure_column,
                                     source_column, DIVIDER, keep_censored_tail):
    """Derive episodes from binary maintenance/failure indicator columns living inside `data`."""
    indicator_columns = [column for column in (maintenance_column, failure_column)
                         if _is_indicator_column(column, data)]
    for column in (maintenance_column, failure_column):
        if column is not None and column not in indicator_columns:
            raise ValueError(f"Column '{column}' is not present in data.")

    # Bare ``== 1`` comparisons would silently treat '1', 2 or NaN as "no event", so the whole
    # source would become one censored episode and the user would be told, much later, that no
    # episode ends with a failure.
    for column in indicator_columns:
        values = set(pd.unique(data[column].dropna()))
        invalid = {value for value in values if value not in (0, 1, True, False)}
        if invalid or data[column].isna().any():
            raise ValueError(
                f"Column '{column}' must be a binary 0/1 indicator, but it contains "
                f"{sorted(invalid, key=repr) if invalid else []}"
                f"{' and missing values' if data[column].isna().any() else ''}."
            )

    all_episodes = []
    all_run_to_failure = []
    all_sources = []
    original_s_has_f = {}
    episode_to_source = {}

    for source, group_df in data.groupby(source_column):
        group_df = group_df.sort_values(by=datetime_column).reset_index(drop=True)

        boundaries = []
        for position in range(group_df.shape[0]):
            if failure_column is not None and group_df.iloc[position][failure_column] == 1:
                boundaries.append((position, 1))
            elif maintenance_column is not None \
                    and group_df.iloc[position][maintenance_column] == 1:
                boundaries.append((position, 0))

        segments = []
        start = 0
        for position, is_failure in boundaries:
            if position < start:
                continue
            segments.append((start, position, is_failure))
            start = position + 1
        if keep_censored_tail and start < group_df.shape[0]:
            segments.append((start, group_df.shape[0] - 1, 0))

        counter = 0
        for first, last, is_failure in segments:
            episode = group_df.iloc[first:last + 1].copy()
            if episode.shape[0] == 0:
                continue
            episode = episode.drop(columns=indicator_columns).reset_index(drop=True)
            maxdate = episode[datetime_column].max()
            episode["RUL"] = [(maxdate - dtime).total_seconds() / DIVIDER
                              for dtime in episode[datetime_column]]
            episode_name = f"{source}{EPISODE_SEPARATOR}{counter}"
            episode[source_column] = episode_name
            all_episodes.append(episode)
            all_run_to_failure.append(int(is_failure))
            all_sources.append(episode_name)
            episode_to_source[episode_name] = source
            counter += 1

        original_s_has_f[source] = any(is_failure for _, _, is_failure in segments)

    return all_episodes, all_run_to_failure, all_sources, original_s_has_f, episode_to_source


def data_split_by_event(df_source, event_source, datetime_column, failure_column,
                        maintenance_column, source_column='source', DIVIDER=3600,
                        keep_censored_tail=True):
    """Split one source into episodes delimited by its failure/maintenance events.

    Every episode spans the interval ``(previous recognised event, current event]`` and is marked
    run-to-failure (1) when it ends on a failure event and reset (0) when it ends on a
    maintenance event. The first episode also includes the very first observation of the source,
    and the data recorded after the last event is emitted as a censored episode when
    `keep_censored_tail` is True.

    Returns
    -------
    tuple
        (episodes, run_to_failure, episode_names)
    """
    df_source = df_source.sort_values(by=datetime_column).reset_index(drop=True)

    episodes = []
    rtfs = []
    new_sources = []
    counter = 0
    source_name = df_source.iloc[0][source_column] if df_source.shape[0] else None

    # Collapse the events into one boundary per timestamp before slicing. Two events logged at
    # the same instant would otherwise produce an empty second episode whose label is discarded,
    # making the outcome depend on the row order of `event_df`; a failure always wins over a
    # maintenance event at the same instant, mirroring the code-level rule in
    # `episodes_formulation`.
    boundaries = {}
    dropped_undated = 0
    for _, event_row in event_source.iterrows():
        if event_row[failure_column] == 1:
            found = 1
        elif event_row[maintenance_column] == 1:
            found = 0
        else:
            # Events that match neither list do not delimit an episode and must not move the
            # start of the next one either, otherwise the data in between is silently dropped.
            continue

        event_time = event_row[datetime_column]
        if pd.isna(event_time):
            # An undated event cannot delimit anything, and letting it through would make the
            # censored tail compare against NaT and silently vanish.
            dropped_undated += 1
            continue
        boundaries[event_time] = max(boundaries.get(event_time, 0), found)

    if dropped_undated:
        warnings.warn(
            f"Ignored {dropped_undated} event(s) of source '{source_name}' because their "
            f"'{datetime_column}' value is missing.",
            stacklevel=2,
        )

    previous_event_time = None

    for event_time, found in sorted(boundaries.items()):
        if previous_event_time is None:
            mask = df_source[datetime_column] <= event_time
        else:
            mask = ((df_source[datetime_column] > previous_event_time)
                    & (df_source[datetime_column] <= event_time))
        previous_event_time = event_time

        episode = df_source[mask].copy()
        if episode.shape[0] == 0:
            continue

        episode = episode.reset_index(drop=True)
        maxdate = episode[datetime_column].max()
        episode["RUL"] = [(maxdate - dtime).total_seconds() / DIVIDER
                          for dtime in episode[datetime_column]]
        episode_name = f"{source_name}{EPISODE_SEPARATOR}{counter}"
        episode[source_column] = episode_name

        episodes.append(episode)
        rtfs.append(found)
        new_sources.append(episode_name)
        counter += 1

    if keep_censored_tail:
        if previous_event_time is None:
            tail = df_source.copy()
        else:
            tail = df_source[df_source[datetime_column] > previous_event_time].copy()
        if tail.shape[0] > 0:
            tail = tail.reset_index(drop=True)
            maxdate = tail[datetime_column].max()
            tail["RUL"] = [(maxdate - dtime).total_seconds() / DIVIDER
                           for dtime in tail[datetime_column]]
            episode_name = f"{source_name}{EPISODE_SEPARATOR}{counter}"
            tail[source_column] = episode_name
            episodes.append(tail)
            rtfs.append(0)
            new_sources.append(episode_name)

    return episodes, rtfs, new_sources
