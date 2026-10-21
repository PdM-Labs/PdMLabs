"""Snapshot datasets: references to snapshots and their conversion to feature rows.

A snapshot dataset is a folder per source (one file per snapshot), a list of DataFrames
per source, or one folder whose subfolders are the sources (:func:`discover_sources`).

The Dataset class does not read the snapshots: it builds a placeholder frame with one row
per snapshot (timestamp, source and a :class:`SnapshotRef`), so that episodes, splits,
predictive horizon and slide are derived as for any dataset. The raw-level step of the
pipeline reads the snapshots later and replaces each reference by one row of features
(:func:`featurize`).

Features are cached in memory per snapshot and raw-level configuration, so within a Python
session each snapshot is read once per configuration.
"""

import hashlib
import math
import numbers
import os
import re
import warnings
from datetime import date, datetime, timedelta

import cloudpickle
import numpy as np
import pandas as pd

#: Column of the placeholder frame that holds the snapshot references.
SNAPSHOT_COLUMN = '__snapshot__'

#: Keys accepted in ``Dataset(snapshot_options=...)``.
SNAPSHOT_OPTIONS = ('timestamp_format', 'datetime_column', 'interval', 'start',
                    'read_kwargs', 'reader')

#: Start of the artificial timeline when ``interval`` is used without ``start``.
DEFAULT_START = '2000-01-01 00:00:00'

_feature_cache = {}
_reader_wrappers = {}
_loaded_readers = {}


class _ReaderByValue:
    """Pickle a reader by value, so that trial workers (MAX_JOBS > 1) get it too.

    cloudpickle also carries a reader defined in the user's script. One wrapper is shared by
    all references to a reader, so the reader is pickled once per transfer.
    """

    def __init__(self, reader):
        self.reader = reader

    def __reduce__(self):
        return _load_reader, (cloudpickle.dumps(self.reader),)


def _load_reader(pickled):
    # One object per distinct reader, so that the in-memory cache, which keys on the reader,
    # keeps its entries across the trials a worker runs.
    if pickled not in _loaded_readers:
        _loaded_readers[pickled] = cloudpickle.loads(pickled)
    return _loaded_readers[pickled]


class SnapshotRef:
    """A reference to one snapshot: a file on disk or an in-memory DataFrame.

    ``key`` identifies the snapshot and its content (file path, size, modification time and
    read options, or source, position and content hash), so a changed file never gets its
    features from the cache. ``label`` names the snapshot in error messages: the file path,
    or the DataFrame's position and source.
    """

    def __init__(self, key, path=None, frame=None, read_kwargs=None, reader=None, time_column=None,
                 label=None):
        self.key = key
        self.label = label or (f"'{path}'" if path is not None else key)
        self.path = path
        self.frame = frame
        self.read_kwargs = dict(read_kwargs or {})
        self.reader = reader
        self.time_column = time_column

    def load(self) -> pd.DataFrame:
        """Return the snapshot's samples."""
        if self.frame is not None:
            return self.frame
        reader = self.reader if self.reader is not None else pd.read_csv
        return reader(self.path, **self.read_kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.reader is not None:
            state['reader'] = _reader_wrappers.setdefault(self.reader, _ReaderByValue(self.reader))
        return state

    def __repr__(self):
        return f'SnapshotRef({self.key!r})'


def build_snapshot_frame(data: dict, datetime_column: str, source_column: str,
                         options: dict) -> pd.DataFrame:
    """Build the placeholder frame of a snapshot dataset.

    Args:
        data (dict): Source name -> folder path (one file per snapshot) or list of
            DataFrames (one per snapshot).
        datetime_column (str): Name of the timestamp column of the frame.
        source_column (str): Name of the source column of the frame.
        options (dict): The ``snapshot_options`` of :class:`pdmlabs.utils.dataset.Dataset`.
            Folder files are ordered by their timestamps, parsed from the file names with
            ``timestamp_format`` or read from ``datetime_column``; with ``interval`` they are
            ordered by the case-insensitive natural order of their names (``2.csv`` before
            ``10.csv``), the exact name breaking ties (``01.csv`` before ``1.csv``), so the
            order does not depend on the file system. Lists of DataFrames keep the order
            given. With ``datetime_column`` a snapshot's time is the column's first value;
            with ``interval`` (a positive time span with a unit, e.g. ``'1min'``) snapshot ``i`` is
            stamped ``start + i * interval`` (``start`` defaults to ``DEFAULT_START``).
            ``reader(path, **read_kwargs)`` returns one snapshot (``pd.read_csv`` by
            default); ``reader`` and ``read_kwargs`` only apply to folder sources.

    Returns:
        pd.DataFrame: One row per snapshot with columns ``datetime_column``,
            ``source_column`` and ``SNAPSHOT_COLUMN``, in source order then time order.

    Raises:
        ValueError: On unknown options, a missing or ambiguous timestamp option, an
            ``interval`` that is not a positive time span or a ``start`` that is not a date,
            options that do not apply to a source, a snapshot file that cannot be read (with
            ``datetime_column`` every file is read for its time), a missing timestamp, date
            text that is not in one format, timestamps in different time zones, or
            timestamps that are not strictly increasing within a source (two snapshots with
            the same timestamp, or DataFrames given out of time order).
        TypeError: If a source is neither a folder path nor a list of DataFrames, or a
            ``reader`` does not return a DataFrame.
    """
    unknown = set(options) - set(SNAPSHOT_OPTIONS)
    if unknown:
        raise ValueError(f'Unknown snapshot_options {sorted(unknown)}; expected a subset of '
                         f'{list(SNAPSHOT_OPTIONS)}.')
    timestamp_options = [name for name in ('timestamp_format', 'datetime_column', 'interval')
                         if options.get(name) is not None]
    if len(timestamp_options) != 1:
        raise ValueError(
            'snapshot_options must give exactly one way to timestamp the snapshots: '
            "'timestamp_format' (parse the file names), 'datetime_column' (a column inside "
            "each snapshot) or 'interval' (artificial spacing), got "
            f'{timestamp_options or "none"}.'
        )
    if options.get('start') is not None and options.get('interval') is None:
        raise ValueError("snapshot_options 'start' is only used together with 'interval'.")
    if options.get('interval') is not None:
        if _is_bare_number(options['interval']):
            # pandas would read it as nanoseconds, so the snapshots would be (almost) simultaneous.
            raise ValueError("snapshot_options['interval'] needs a unit, e.g. '60s' or '1min'; got "
                             f"{options['interval']!r}, which pandas would read as nanoseconds.")
        try:
            interval = pd.Timedelta(options['interval'])
        except (TypeError, ValueError):
            interval = pd.NaT
        if pd.isna(interval) or interval <= pd.Timedelta(0):
            raise ValueError("snapshot_options['interval'] must be a positive time span such as '1min', "
                             f"got {options['interval']!r}.")
    if options.get('start') is not None:
        if _is_bare_number(options['start'], strings=False):
            raise ValueError("snapshot_options['start'] must be a date such as '2000-01-01 00:00:00'; got "
                             f"{options['start']!r}, which pandas would read as nanoseconds after 1970.")
        try:
            start = pd.Timestamp(options['start'])
        except (TypeError, ValueError):
            start = pd.NaT
        if pd.isna(start):
            raise ValueError("snapshot_options['start'] must be a date such as '2000-01-01 00:00:00', "
                             f"got {options['start']!r}.")

    frames = []
    for source, snapshots in data.items():
        if isinstance(snapshots, (str, os.PathLike)):
            refs, timestamps = _folder_snapshots(str(source), os.fspath(snapshots), options)
        elif isinstance(snapshots, (list, tuple)):
            refs, timestamps = _frame_snapshots(str(source), list(snapshots), options)
        else:
            raise TypeError(
                f"Snapshots of source '{source}' must be a folder path or a list of DataFrames, "
                f'got {type(snapshots).__name__}.'
            )
        if len(refs) == 0:
            raise ValueError(f"Source '{source}' has no snapshots.")
        timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps))
        _check_increasing(timestamps, refs, source)
        frames.append(pd.DataFrame({
            datetime_column: timestamps,
            source_column: str(source),
            SNAPSHOT_COLUMN: refs,
        }))
    return pd.concat(frames, ignore_index=True)


def discover_sources(root) -> dict:
    """Find the sources of a snapshot dataset stored under a single folder.

    Every folder under ``root`` that directly holds files is a source, named by its relative
    path (e.g. ``'1st_test/1st_test'`` in IMS); a ``root`` that only holds files is one source.
    Hidden files are skipped, and files directly in ``root`` (e.g. a readme) are ignored with
    a warning.

    Args:
        root (str | os.PathLike): The dataset's folder.

    Returns:
        dict: Source name -> folder, in case-insensitive natural order of the names, the
            exact name breaking ties.

    Raises:
        ValueError: If ``root`` is a file, does not exist or holds no files.
    """
    root = os.fspath(root)
    if os.path.isfile(root):
        raise ValueError(f"'{root}' is a file. Pass a DataFrame, a dict of sources (folders or lists of "
                         'DataFrames), or a folder that holds the snapshot files of every source.')
    if not os.path.isdir(root):
        raise ValueError(f"Snapshot dataset folder '{root}' does not exist.")

    sources, root_files = {}, []
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = sorted(name for name in subdirectories if not name.startswith('.'))
        files = [name for name in files if not name.startswith('.')]
        if not files:
            continue
        if os.path.normpath(directory) == os.path.normpath(root):
            root_files = files
        else:
            sources[os.path.relpath(directory, root).replace(os.sep, '/')] = directory
    if not sources:
        if not root_files:
            raise ValueError(f"No snapshot files found under '{root}'.")
        return {os.path.basename(os.path.normpath(root)): root}
    if root_files:
        warnings.warn(f"Ignored {len(root_files)} file(s) directly in '{root}' (e.g. {sorted(root_files)[:3]}): "
                      'the snapshot files are in its subfolders, one source per subfolder.')
    return dict(sorted(sources.items(), key=lambda item: (_natural_key(item[0]), item[0])))


def snapshot_frame(snapshots, **snapshot_options) -> pd.DataFrame:
    """Build the ``target_data`` of a saved pipeline's ``predict`` from the snapshots of one source.

    The result has the layout the pipelines are trained on: the snapshot timestamps as index
    and a single ``SNAPSHOT_COLUMN`` of references, which the pipeline's raw-level step
    turns into features. The snapshots are only read then.

    Args:
        snapshots (str | os.PathLike | list[pd.DataFrame]): A folder with one file per
            snapshot, or a list of DataFrames (one per snapshot).
        **snapshot_options: The ``snapshot_options`` of :class:`pdmlabs.utils.dataset.Dataset`
            (see :func:`build_snapshot_frame`), so the options of the training Dataset can be
            passed as they are; for a list of DataFrames, leave out ``read_kwargs`` and
            ``reader``, which only apply to folders.

    Returns:
        pd.DataFrame: One row per snapshot, in time order.

    Examples:
        With ``read_ims`` as in the ``snapshot_options`` example of
        :class:`pdmlabs.utils.dataset.Dataset`:

        >>> frame = snapshot_frame('/path/to/IMS/3rd_test/4th_test/txt',
        ...                        timestamp_format='%Y.%m.%d.%H.%M.%S', reader=read_ims)
        >>> model.predict({'target_data': frame, 'source': '3rd_test/4th_test/txt'})
    """
    name = os.fspath(snapshots) if isinstance(snapshots, (str, os.PathLike)) else 'snapshots'
    frame = build_snapshot_frame({name: snapshots}, 'timestamp', 'source', snapshot_options)
    result = frame[[SNAPSHOT_COLUMN]]
    result.index = pd.DatetimeIndex(frame['timestamp']).rename(None)
    return result


def _folder_snapshots(source, folder, options):
    if os.path.isfile(folder):
        raise ValueError(f"The path of source '{source}' is a file ('{folder}'); expected a folder holding "
                         'one file per snapshot.')
    if not os.path.isdir(folder):
        raise ValueError(f"Snapshot folder '{folder}' of source '{source}' does not exist.")
    names = [name for name in os.listdir(folder)
             if not name.startswith('.') and os.path.isfile(os.path.join(folder, name))]
    timestamp_format = options.get('timestamp_format')
    if timestamp_format is not None:
        parsed = sorted((_parse_file_name(name, timestamp_format), name) for name in names)
        names = [name for _, name in parsed]
    else:
        # The exact name breaks ties (1.csv and 01.csv, a.csv and A.csv), which would
        # otherwise keep the file system's listing order.
        names = sorted(names, key=lambda name: (_natural_key(name), name))

    refs = []
    read_key = _read_key(options)
    for name in names:
        path = os.path.abspath(os.path.join(folder, name))
        stat = os.stat(path)
        refs.append(SnapshotRef(
            key=f'file:{path}:{stat.st_size}:{stat.st_mtime_ns}:{read_key}',
            path=path,
            read_kwargs=options.get('read_kwargs'),
            reader=options.get('reader'),
            time_column=options.get('datetime_column'),
        ))

    if timestamp_format is not None:
        timestamps = [timestamp for timestamp, _ in parsed]
    elif options.get('datetime_column') is not None:
        # The names say nothing about the order here, so the files are ordered by the
        # timestamps they hold; equal ones are rejected by _check_increasing.
        values = [_first_value(_load(ref), options['datetime_column'], source) for ref in refs]
        timestamps = _parse_first_values(values, options['datetime_column'], source)
        stamped = sorted(zip(timestamps, refs), key=lambda pair: pair[0])
        timestamps = [timestamp for timestamp, _ in stamped]
        refs = [ref for _, ref in stamped]
    else:
        timestamps = _interval_timestamps(len(refs), options)
    return refs, timestamps


def _frame_snapshots(source, snapshots, options):
    if not all(isinstance(snapshot, pd.DataFrame) for snapshot in snapshots):
        raise TypeError(f"Every snapshot of source '{source}' must be a pandas DataFrame.")
    if options.get('timestamp_format') is not None:
        raise ValueError(
            "snapshot_options 'timestamp_format' parses file names and only applies to folder "
            f"sources; source '{source}' is a list of DataFrames. Use 'datetime_column' or "
            "'interval' instead."
        )
    ignored = [name for name in ('read_kwargs', 'reader') if options.get(name)]
    if ignored:
        raise ValueError(
            f"snapshot_options {ignored} only apply to folder sources; source '{source}' is a list "
            "of DataFrames. Select the channels of the DataFrames before passing them.")

    refs = []
    time_column = options.get('datetime_column')
    for index, snapshot in enumerate(snapshots):
        content = hashlib.sha1(pd.util.hash_pandas_object(snapshot, index=True).to_numpy().tobytes())
        content.update(repr(list(snapshot.columns)).encode())
        refs.append(SnapshotRef(
            key=f'frame:{source}#{index}:{content.hexdigest()}:{time_column!r}',
            frame=snapshot,
            time_column=time_column,
            label=f"#{index} (DataFrame) of source '{source}'",
        ))

    if options.get('datetime_column') is not None:
        values = [_first_value(snapshot, options['datetime_column'], source) for snapshot in snapshots]
        timestamps = _parse_first_values(values, options['datetime_column'], source)
    else:
        timestamps = _interval_timestamps(len(refs), options)
    return refs, timestamps


def _read_key(options):
    """Describe how snapshot files are read, for the snapshot keys of the feature cache.

    The description holds the reader's name, ``read_kwargs`` and the time column that is
    removed, so that a file read differently never shares a cached result.
    """
    reader = options.get('reader')
    name = 'pandas.read_csv' if reader is None else \
        f"{getattr(reader, '__module__', '')}.{getattr(reader, '__qualname__', type(reader).__qualname__)}"
    read_kwargs = sorted((options.get('read_kwargs') or {}).items())
    return f'{name}:{read_kwargs!r}:{options.get("datetime_column")!r}'


def _parse_file_name(name, timestamp_format):
    candidates = [name, os.path.splitext(name)[0]]
    for candidate in candidates:
        try:
            return datetime.strptime(candidate, timestamp_format)
        except ValueError:
            pass
    raise ValueError(f"Snapshot file name '{name}' does not match timestamp_format "
                     f"'{timestamp_format}'.")


def _natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def _first_value(snapshot, column, source):
    if column not in snapshot.columns:
        raise ValueError(f"A snapshot of source '{source}' has no '{column}' column "
                         "(snapshot_options['datetime_column']).")
    if pd.api.types.is_numeric_dtype(snapshot[column]):
        raise ValueError(
            f"The '{column}' column of the snapshots of source '{source}' holds numbers, whose "
            "unit (seconds, milliseconds, ...) is ambiguous. Convert it to datetimes first, e.g. "
            "with pd.to_datetime(values, unit='s') in snapshot_options['reader'] or before "
            'passing the DataFrames.'
        )
    return snapshot[column].iloc[0] if len(snapshot) else None


def _parse_first_values(values, column, source):
    """Turn the first ``column`` value of each snapshot of a source into its timestamp."""
    timestamps = []
    for value in values:
        timestamp = pd.to_datetime(value)
        if pd.isna(timestamp):
            raise ValueError(f"A snapshot of source '{source}' has no timestamp: the first value of its "
                             f"'{column}' column (snapshot_options['datetime_column']) is missing.")
        timestamps.append(timestamp)
    _check_one_time_zone(timestamps, source)
    _check_one_date_format(values, timestamps, column, source)
    return timestamps


def _check_one_date_format(values, timestamps, column, source):
    """Reject date text that is not read the same way all together as one value at a time.

    Each value is parsed on its own, so '01/02/2020' (month first) and '13/02/2020' (day first)
    would both be accepted, and sorting by them would silently mix up the snapshots.
    """
    if not all(isinstance(value, str) for value in values):
        return
    try:
        pd.to_datetime(pd.Series(values), format='ISO8601')
        return
    except (TypeError, ValueError):
        pass
    try:
        together = pd.to_datetime(pd.Series(values))
        consistent = all(a == b for a, b in zip(together, timestamps))
    except (TypeError, ValueError):
        consistent = False
    if not consistent:
        raise ValueError(
            f"The '{column}' values of the snapshots of source '{source}' are not all in one date "
            'format (e.g. month first in some and day first in others), so their order would be a '
            "guess. Use one unambiguous format, e.g. 'YYYY-MM-DD HH:MM:SS', or convert them in "
            "snapshot_options['reader']."
        )


def _check_one_time_zone(timestamps, source):
    zones = {str(timestamp.tz) for timestamp in timestamps}
    if len(zones) > 1:
        raise ValueError(
            f"The snapshot timestamps of source '{source}' are in different time zones "
            f"({sorted(zones)}; 'None' means no time zone). Convert them to one time zone, e.g. "
            "in snapshot_options['reader'] or before passing the DataFrames."
        )


def _check_increasing(timestamps, refs, source):
    if timestamps.has_duplicates:
        timestamp = timestamps[timestamps.duplicated()][0]
        first, second = [refs[position].label
                         for position in (timestamps == timestamp).nonzero()[0][:2]]
        raise ValueError(f'Snapshots {first} and {second} have the same timestamp {timestamp}; '
                         f"every snapshot of source '{source}' needs its own time.")
    if not timestamps.is_monotonic_increasing:
        # Folder snapshots are ordered by time, so only DataFrames given out of order get here.
        position = next(position for position in range(1, len(timestamps))
                        if timestamps[position] < timestamps[position - 1])
        raise ValueError(
            f"The snapshot timestamps of source '{source}' must increase in the order the "
            f'DataFrames are given, but {refs[position].label} ({timestamps[position]}) '
            f'comes after {refs[position - 1].label} ({timestamps[position - 1]}).'
        )


def _is_bare_number(value, strings=True):
    """Whether ``value`` is a number without a unit: 60, 60.0, a numpy number and, with
    ``strings``, '60'. Time objects (timedelta, datetime, numpy's) are not."""
    if isinstance(value, (timedelta, date, np.timedelta64, np.datetime64)):
        return False
    if isinstance(value, numbers.Number):
        return True
    if strings and isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return False
        return True
    return False


def _interval_timestamps(count, options):
    start = pd.Timestamp(DEFAULT_START if options.get('start') is None else options['start'])
    return [start + index * pd.Timedelta(options['interval']) for index in range(count)]


def featurize(frame: pd.DataFrame, raw_step) -> pd.DataFrame:
    """Replace the snapshot references of a frame by one row of features each.

    Rows are matched by position, so the rows, order and index of ``frame`` are kept, and
    the feature columns follow its other columns. A frame without snapshot references is
    returned unchanged.

    Args:
        frame (pd.DataFrame): Rows of a snapshot dataset.
        raw_step (RawLevelPreProcessorInterface): The feature extractor.

    Returns:
        pd.DataFrame: The featurized frame.

    Raises:
        ValueError: If a snapshot cannot be read, ``raw_step`` fails on it or returns
            missing or infinite features, the snapshots do not all yield the same
            features, or a feature name clashes with another column of ``frame``. The
            message names the snapshot involved.
        TypeError: If a ``reader`` does not return a DataFrame, or ``raw_step.extract()``
            does not return a dict of numbers.
    """
    if SNAPSHOT_COLUMN not in frame.columns:
        return frame

    token = _cache_token(raw_step)
    refs = list(frame[SNAPSHOT_COLUMN])
    rows = [_features_of(ref, raw_step, token) for ref in refs]
    columns = list(rows[0]) if rows else []
    if rows and not columns:
        advice = ("Set how the snapshot files are read with snapshot_options['read_kwargs'], e.g. "
                  "{'sep': '\\t', 'header': None} for tab-separated files without a header."
                  if refs[0].path is not None else
                  'Convert the channels of the DataFrames to numbers before passing them.')
        raise ValueError(
            f'{raw_step} returned no features for snapshot {refs[0].label}, e.g. because it has no '
            f'numeric channels. {advice}'
        )
    for ref, row in zip(refs, rows):
        if list(row) != columns:
            only_first = [name for name in columns if name not in row]
            only_this = [name for name in row if name not in columns]
            detail = (f'only in the first: {only_first[:5]}, only in the second: {only_this[:5]}'
                      if only_first or only_this else 'the same features in another order')
            raise ValueError(
                f'{raw_step} returned {len(columns)} features for snapshot {refs[0].label} but '
                f'{len(row)} for snapshot {ref.label} ({detail}). Every snapshot must have the same '
                f'channels, in the same order: {channel_advice([refs[0], ref])}.'
            )

    rest = frame.drop(columns=[SNAPSHOT_COLUMN])
    clashes = set(rest.columns) & set(columns)
    if clashes:
        raise ValueError(f'Feature columns {sorted(map(str, clashes))} clash with existing '
                         'columns of the data.')
    features = pd.DataFrame(rows, columns=columns)
    result = pd.concat([rest.reset_index(drop=True), features], axis=1)
    result.index = frame.index
    return result


def channel_advice(refs) -> str:
    """Say how to give every snapshot the same channels, for the kind of snapshots in ``refs``.

    Args:
        refs (Iterable[SnapshotRef]): The snapshots that disagree.

    Returns:
        str: Advice for snapshot files (a ``reader``) or for DataFrames.
    """
    if any(ref.path is not None for ref in refs):
        # Not read_kwargs: it applies to every file alike, so it cannot pick different columns
        # where sources store their channels differently (e.g. IMS test 1 has two per bearing).
        return ("select them with snapshot_options['reader'], a function reader(path, **read_kwargs) "
                'that reads one file and returns the same channels in the same order for every source')
    return 'select the channels of the DataFrames before passing them'


def _cache_token(raw_step):
    """Identify a raw-level configuration by its class name and get_params()."""
    cls = type(raw_step)
    return f'{cls.__module__}.{cls.__qualname__}{sorted(raw_step.get_params().items())!r}'


def _load(ref):
    """Read a snapshot, naming it in any error (an empty file, a failing reader, ...)."""
    try:
        snapshot = ref.load()
    except Exception as error:
        raise ValueError(f'Could not read snapshot {ref.label}: {type(error).__name__}: {error}') from error
    if not isinstance(snapshot, pd.DataFrame):
        raise TypeError(f"snapshot_options['reader'] must return a pandas DataFrame, but returned "
                        f'{type(snapshot).__name__} for snapshot {ref.label}.')
    return snapshot


def _features_of(ref, raw_step, token):
    # The class and the reader also count as objects: one defined again under the same name
    # (a notebook cell run again, a reader made by a factory) gets its own entries.
    cache_key = (ref.key, ref.reader, type(raw_step), token)
    if cache_key in _feature_cache:
        return _feature_cache[cache_key]

    snapshot = _load(ref)
    try:
        if ref.time_column is not None:
            # The timestamp is not a measurement; as a feature it would tell the method how far
            # into the run a snapshot was recorded.
            snapshot = snapshot.drop(columns=[ref.time_column])
        features = raw_step.extract(snapshot)
    except Exception as error:
        raise ValueError(f'{raw_step} could not compute the features of snapshot {ref.label}: '
                         f'{type(error).__name__}: {error}') from error
    if not isinstance(features, dict):
        raise TypeError(f'{raw_step}.extract() must return a dict, got {type(features).__name__}.')
    # Checked before caching, so that a fixed file is read again. A non-finite feature would
    # otherwise only fail inside the method, without naming the snapshot.
    non_finite = []
    for name, value in features.items():
        # None and pd.NA are missing values too, not values of the wrong type.
        if pd.api.types.is_scalar(value) and pd.isna(value):
            non_finite.append(name)
            continue
        # Text such as '1.5' is not a number either, even though float() would accept it.
        if not isinstance(value, numbers.Real):
            raise TypeError(f'{raw_step}.extract() must return numbers, but feature {name!r} of snapshot '
                            f'{ref.label} is {value!r}.')
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            non_finite.append(name)
    if non_finite:
        advice = ('check the file for missing or invalid samples (e.g. empty cells, or text read as '
                  "NaN), or fill them in snapshot_options['reader']"
                  if ref.path is not None else 'fill or drop them in the DataFrame before passing it')
        raise ValueError(f'{raw_step} returned non-finite features {non_finite[:5]} for snapshot '
                         f'{ref.label}: {advice}.')
    _feature_cache[cache_key] = features
    return features
