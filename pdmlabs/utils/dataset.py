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

import random

import pandas as pd


class Dataset:
    """
    A class to handle dataset preparation and processing for predictive maintenance tasks.
    This includes splitting data into episodes, calculating sliding windows, and preparing
    training, validation, and testing datasets.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing time-series information. If `event_df` is not provided but `maintenance_column` and `failure_column` are provided
        it except that `maintenance_column` and `failure_column` are included in data, and use them to derive on the episodes.
        If `maintenance_column` and `failure_column` and `event_df` are not provided it assumes every source as a single run-to-failure episode.
    datetime_column : str
        The name of the column representing datetime values.
    event_indicator : str, default=None
        The name of the column indicating event occurrence (binary). If provided, it is used to derive episode ending (0: maintenance/reset or 1: failure).
    maintenance_column : str, default=None
        The name of the column indicating maintenance events.
    failure_column : str, default=None
        The name of the column indicating failure events.
    event_df : pd.DataFrame, optional
        A DataFrame containing event data. If provided, `data`, `datetime_column`, and `failure_column` must also be given.
        The `event_df` must contain the columns `datetime_column`, `source_column`, `maintenance_column`, and `failure_column`.
        Events are generated based on the following:
        - Failure event: `failure_column=1`
        - Maintenance (resetting) event: `maintenance_column=1`
    source_column : str, default='source'
        The name of the column representing the source of the data.
    beta : int, default=1
        A parameter used for objective calculations.
    slide : int, optional
        The sliding window size. If None, it is calculated automatically.
    lead : str, default="2 seconds"
        The lead time for predictions.
    predictive_horizon : str, optional
        The predictive horizon for the dataset. If None, it is calculated automatically.
    train_sources : float or list, default=0.6
        The ratio (float) or list of source names used for training. If a float, it represents the proportion of sources used for training.
    val_sources : float or list, default=0.2
        The ratio (float) or list of source names used for validation. If a float, it represents the proportion of sources used for validation.
    test_sources : float or list, default=0.2
        The ratio (float) or list of source names used for testing. If a float, it represents the proportion of sources used for testing.
    max_wait_time : int, control the maximym length of profile parameter in OnlineFlavor and Sliding Window flavor (i.e. the maximum length of the
        data to fit anomaly detectors). This is the time that the user is willing to wait before detectors produce alarms. If None, it is set to
        2/3 of the minimum episode length.
    in_source_split : bool, default=False Whether to select train/val/test sources from within each source (True) or from the overall sources (False).
    """
    def __init__(self,data,datetime_column,event_indicator=None,maintenance_column=None,failure_column=None,
                 event_df=None,source_column='source',
                 beta=1, slide=None, lead="0 seconds",predictive_horizon=None,
                 train_sources=0.6,val_sources=0.2,test_sources=0.2,max_wait_time=None,in_source_split=False,DIVIDER=3600):

        # Dataset
        self.in_source_split=in_source_split
        self.datetime_column = datetime_column
        data[self.datetime_column]=pd.to_datetime(data[self.datetime_column])
        data[source_column]=data[source_column].astype(str)
        episodes, run_to_failure,_,original_s_has_f = episodes_formulation(data, datetime_column,event_indicator, maintenance_column,
                                                                failure_column,event_df, source_column,DIVIDER)
        self.original_sources=data[source_column].unique().tolist()
        self.sources = [ep.iloc[0][source_column] for ep in episodes]
        self.original_s_has_f =original_s_has_f


        self.source_column = source_column

        self.train_sources = train_sources
        self.val_sources = val_sources
        self.test_sources = test_sources

        self.train_source_name='train'
        self.split_sources_to_train_test_val(episodes,run_to_failure)
        # Calcuylated in split_sources_to_train_test_val:
        # self.matches = matches
        # self.rtf_dict
        # self.train_dfs = train_dfs
        # self.val_dfs = val_dfs
        # self.test_dfs = test_dfs
        #
        # self.sources_for_train = for_train
        # self.sources_for_val = for_val
        # self.sources_for_test = for_test
        self.max_wait_time = max_wait_time
        if self.max_wait_time is None:
            self.max_wait_time = max(10,int(2*min([ep.shape[0] for ep in episodes])/3))


        self.rul_column = 'RUL'


        # Objective parameters
        self.beta = beta
        self.lead = lead
        # when lead and predictive_horizon are not provided, set them to default values
        if predictive_horizon is None:
            durations=[(ep.iloc[-1][self.datetime_column]-ep.iloc[0][self.datetime_column]).total_seconds()/3600.0 for ep,rtf in zip(episodes,run_to_failure) if rtf==1]
            self.predictive_horizon = f"{min(durations)/10.0} hours"
        else:
            self.predictive_horizon = predictive_horizon
        if slide is None:
            self.slide = self.slide_calculation(episodes, run_to_failure)
        else:
            self.slide = slide

    def slide_calculation(self,episodes, run_to_failure):
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
        minlen = float('inf')
        minep = None
        for ep, rtfi in zip(episodes, run_to_failure):
            if rtfi == 1:
                dur = ( ep.iloc[-1][self.datetime_column]-ep.iloc[0][self.datetime_column]).total_seconds() / 3600.0
                if dur < minlen:
                    minlen = dur
                    minep = ep
        length_min_ep = minep.shape[0]
        lastime = minep.iloc[-1][self.datetime_column]
        pos = 0
        for i in range(length_min_ep):
            ctime = minep.iloc[i][self.datetime_column]
            if ctime >= lastime - pd.Timedelta(self.predictive_horizon):
                pos = i
                break
        ph_length = length_min_ep - pos
        slide=int(length_min_ep / 3) - ph_length
        return max(slide,1)

    def split_sources_to_train_test_val(self,episodes,ran_to_failure):
        """
        Splits the sources into training, validation, and testing datasets.

        Parameters
        ----------
        episodes : list
            A list of dataframes, where each dataframe corresponds to an episode.
        ran_to_failure : list
            A list of integers indicating whether each episode is a run-to-failure (1) or not (0).

        Returns
        -------
        None
            The method updates the following attributes of the class:
            - self.train_dfs: Dataframes for training.
            - self.val_dfs: Dataframes for validation.
            - self.test_dfs: Dataframes for testing.
            - self.sources_for_train: Sources used for training.
            - self.sources_for_val: Sources used for validation.
            - self.sources_for_test: Sources used for testing.
            - self.matches: A dictionary mapping training sources to validation and testing sources.
        """
        for i in range(len(episodes)):
            episodes[i][self.datetime_column]=pd.to_datetime(episodes[i][self.datetime_column])
        unvid=self.sources
        self.rtf_dict={source: rtf for source, rtf in zip(unvid, ran_to_failure)}

        for_train = []
        for_val = []
        for_test = []
        if isinstance(self.train_sources, float) and isinstance(self.val_sources, float) and isinstance(self.test_sources, float):
            if self.train_sources + self.val_sources + self.test_sources != 1.0:
                raise ValueError("When train_sources, val_sources, test_sources  are pass as floats (ratio), they must sum 1.")

            for_train = []
            # if there are at least three ORIGINAL SOURCES that contain failure
            if self.in_source_split==False and len([1 for key in self.original_sources if  self.original_s_has_f[key]])>=3:
                random.seed(42)
                or_sources_with_failure = [sui for sui in self.original_sources if self.original_s_has_f[sui]]
                or_sources_without_failure = list(set(self.original_sources) - set(or_sources_with_failure))

                train_source, val_source, test_source = self.safe_splitting(or_sources_with_failure)
                c_train_source, c_val_source, c_test_source = self.safe_splitting(or_sources_without_failure)

                self.train_sources = train_source + c_train_source
                self.val_sources = val_source + c_val_source
                self.test_sources = test_source + c_test_source


                for source in unvid:
                    if source.split("_ep")[0] in self.val_sources:
                        for_val.append(source)
                    elif source.split("_ep")[0] in self.test_sources:
                        for_test.append(source)
                    elif source.split("_ep")[0] in self.train_sources:
                        for_train.append(source)
            # Either in_source_split is True or there are less than three original sources with failure
            # look at episode level spliting
            else:
                self.train_sources = []
                self.val_sources = []
                self.test_sources = []
                at_least_one_failure_in_train=True
                for orginal_source in self.original_sources:
                    source_episodes = [ep for ep in episodes if
                                       ep[self.source_column].iloc[0].startswith(orginal_source)]
                    train_source, val_source, test_source = self.safe_splitting(source_episodes,at_least_one_failure_in_train)
                    at_least_one_failure_in_train=False
                    self.train_sources.extend(train_source)
                    self.val_sources.extend(val_source)
                    self.test_sources.extend(test_source)
                    for_train.extend(train_source)
                    for_val.extend(val_source)
                    for_test.extend(test_source)
        else:
            for_train.extend(self.train_sources)
            for_val.extend(self.val_sources)
            for_test.extend(self.test_sources)


        if self.train_sources is None or self.val_sources is None or self.test_sources is None:
            raise ValueError(
                "Either provide train_sources,val_sources and test_sources as ratio or as list of sources.")
        else:

            traif = max([self.rtf_dict[source] for source in for_train])
            testf = max([self.rtf_dict[source] for source in for_test])
            valf = max([self.rtf_dict[source] for source in for_val])
            if traif+testf+valf<3:
                raise ValueError("At least one source/episode with failure event must be present in each of train, val and test sets.")


            train_dfs = [episodes[i] for i, ep in enumerate(episodes) if ep[self.source_column].iloc[0] in for_train]
            val_dfs = [episodes[i] for i, ep in enumerate(episodes) if ep[self.source_column].iloc[0] in for_val]
            test_dfs = [episodes[i] for i, ep in enumerate(episodes) if ep[self.source_column].iloc[0] in for_test]

            matches = {}
            for source in for_val + for_test:
                matches[source] = self.train_source_name
        self.matches=matches

        self.train_dfs=train_dfs
        self.val_dfs=val_dfs
        self.test_dfs=test_dfs

        self.sources_for_train=for_train
        self.sources_for_val = for_val
        self.sources_for_test=for_test

    def safe_splitting(self,source_episodes,at_least_one_failure_in_train=False):
        train_source = []
        val_source = []
        test_source = []
        train_source_count = 0
        val_source_count = 0
        test_source_count = 0
        if len(source_episodes) >= 3:
            val_source_count = int(self.val_sources * len(source_episodes))
            test_source_count = int(self.test_sources * len(source_episodes))
            train_source_count = len(source_episodes) - val_source_count - test_source_count
            if test_source_count == 0:
                test_source_count = 1
                train_source_count -= 1
            if val_source_count == 0:
                val_source_count = 1
                train_source_count -= 1
        elif len(source_episodes) == 2:
            test_source_count = 1
            val_source_count = 1
        else:
            test_source_count = 1
        fail_in_train=False
        for i, source in enumerate(source_episodes):
            if i < train_source_count and (at_least_one_failure_in_train==False or fail_in_train):
                if self.rtf_dict.get(source,0)==1:
                    fail_in_train=True
                train_source.append(source)
            elif i < train_source_count + val_source_count:
                val_source.append(source)
            else:
                test_source.append(source)
        return train_source, val_source, test_source
    def get_rul_dataset(self,keep_sources=None):
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
        concatinated_train = pd.concat([df for df in self.train_dfs if self.rtf_dict[df.iloc[0][self.source_column]]==1], ignore_index=True)
        
        
        cols_to_drop = [self.source_column, self.rul_column]

        if keep_sources is not None and keep_sources in cols_to_drop:
            cols_to_drop.remove(keep_sources)

        dataset = {}
        dataset['match_sources'] = self.matches
        dataset['target_sources'] = [str(vid) for vid in self.sources_for_val]

        target_data=[]
        for df in self.val_dfs:
            tdf=df.copy()
            tdf=tdf.drop(columns=cols_to_drop).reset_index(drop=True).copy()
            if keep_sources is not None:
                tdf[keep_sources]=df[self.source_column]
            target_data.append(tdf)
        dataset['target_data'] = target_data
        dataset['is_failure'] = [self.rtf_dict[str(vid)] for vid in self.sources_for_val]
        dataset['target_labels'] = [df[self.rul_column].values for df in self.val_dfs]


        if keep_sources is not None:
            concatinated_train[keep_sources]=[s for s in concatinated_train[self.source_column]]
        dataset['historic_data'] = [concatinated_train.drop(columns=cols_to_drop)]
        dataset['historic_sources'] = [self.train_source_name]
        dataset['anomaly_labels'] = [concatinated_train[self.rul_column].values]
        dataset["dates"] = self.datetime_column

        from pdmlabs.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple

        event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

        event_preferences: EventPreferences = {
            'failure': [],
            'reset': []
        }
        dataset["event_preferences"] = event_preferences
        dataset["event_data"] = event_data
        dataset['predictive_horizon'] = self.predictive_horizon
        dataset['slide'] = self.slide
        dataset['lead'] = self.lead
        dataset['beta'] = self.beta
        dataset['max_wait_time'] = self.max_wait_time

        ############## test dataset ###############

        test_dataset = {}
        test_dataset['match_sources'] = self.matches
        test_dataset['target_sources'] = [str(vid) for vid in self.sources_for_test]

        target_data = []
        for df in self.test_dfs:
            tdf = df.copy()
            tdf= tdf.drop(columns=cols_to_drop).reset_index(drop=True).copy()
            if keep_sources is not None:
                tdf[keep_sources] = df[self.source_column]
            target_data.append(tdf)
        test_dataset['target_data'] = target_data
        test_dataset['target_labels'] = [df[self.rul_column].values for df in self.test_dfs]
        test_dataset['is_failure'] = [self.rtf_dict[str(vid)] for vid in self.sources_for_test]


        if keep_sources is not None:
            concatinated_train[keep_sources] = [s for s in concatinated_train[self.source_column]]
        test_dataset['historic_data'] = [concatinated_train.drop(columns=cols_to_drop)]
        test_dataset['historic_sources'] = [self.train_source_name]
        test_dataset['anomaly_labels'] = [concatinated_train[self.rul_column].values]

        test_dataset["dates"] = self.datetime_column

        test_dataset["event_preferences"] = event_preferences
        test_dataset["event_data"] = event_data
        test_dataset['predictive_horizon'] = self.predictive_horizon
        test_dataset['slide'] = self.slide
        test_dataset['lead'] = self.lead
        test_dataset['beta'] = self.beta
        test_dataset['max_wait_time'] = self.max_wait_time

        return dataset, test_dataset

    def df_to_x_y_surv(self,df,indicator=None):
        """Convert dataframe to survival analysis format (time, event) tuples.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing RUL and event columns.
        indicator : int, optional
            Event indicator value (0 or 1). If None, uses 'event' column from df.

        Returns
        -------
        list[tuple]
            List of (rul, event) tuples for survival analysis models.
        """
        if indicator is None:
            y = [(rul, ev) for ev, rul in zip(df["event"], df[self.rul_column])]
        else:
            y = [(rul, indicator) for rul in df[self.rul_column]]
        return y

    def get_SA_dataset(self,keep_sources=None):
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
        train_dfs_with_events=[]
        for df in self.train_dfs:
            df_with_event=df.copy()
            df_with_event["event"]=self.rtf_dict[df.iloc[0][self.source_column]]
            train_dfs_with_events.append(df_with_event)
        concatinated_train = pd.concat(train_dfs_with_events, ignore_index=True)

        #TO-DO: investigate how to deal with case of only run to failure episodes
        if concatinated_train["event"].min()>0:
            event_list=[ev for ev in concatinated_train["event"]]
            event_list[0]=0
            concatinated_train["event"]=event_list


        cols_to_drop = ["event", self.source_column, self.rul_column]
        if keep_sources is not None and keep_sources in cols_to_drop:
            cols_to_drop.remove(keep_sources)
        dataset = {}
        dataset['match_sources'] = self.matches
        dataset['target_sources'] = [str(vid) for vid in self.sources_for_val]

        target_data = []
        for df in self.val_dfs:
            tdf = df.copy()
            tdf=tdf.drop(columns=[self.source_column, self.rul_column]).reset_index(drop=True).copy()
            if keep_sources is not None:
                tdf[keep_sources] = df[self.source_column]
            target_data.append(tdf)
        dataset['target_data'] = target_data
        dataset['target_labels'] = [self.df_to_x_y_surv(df,indicator=self.rtf_dict[df.iloc[0][self.source_column]]) for df in self.val_dfs]
        dataset['is_failure'] = [self.rtf_dict[str(vid)] for vid in self.sources_for_val]
        if keep_sources is not None:
            concatinated_train[keep_sources] = [s for s in concatinated_train[self.source_column]]
        dataset['historic_data'] = [concatinated_train.drop(columns=cols_to_drop)]
        dataset['historic_sources'] = [self.train_source_name]
        dataset['anomaly_labels'] = [self.df_to_x_y_surv(concatinated_train)]

        dataset["dates"] = self.datetime_column

        from pdmlabs.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple

        event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

        event_preferences: EventPreferences = {
            'failure': [],
            'reset': []
        }
        dataset["event_preferences"] = event_preferences
        dataset["event_data"] = event_data
        dataset['predictive_horizon'] = self.predictive_horizon
        dataset['slide'] = self.slide
        dataset['lead'] = self.lead
        dataset['beta'] = self.beta
        dataset['max_wait_time'] = self.max_wait_time

        ############## test dataset ###############

        test_dataset = {}
        test_dataset['match_sources'] = self.matches
        test_dataset['target_sources'] = [str(vid) for vid in self.sources_for_test]

        target_data = []
        for df in self.test_dfs:
            tdf = df.copy()
            tdf = tdf.drop(columns=[self.source_column, self.rul_column]).reset_index(drop=True).copy()
            if keep_sources is not None:
                tdf[keep_sources] = df[self.source_column]
            target_data.append(tdf)
        test_dataset['target_data'] = target_data
        test_dataset['target_labels'] = [self.df_to_x_y_surv(df, indicator=self.rtf_dict[df.iloc[0][self.source_column]]) for
                                    df in self.test_dfs]
        test_dataset['is_failure'] = [self.rtf_dict[str(vid)] for vid in self.sources_for_test]
        if keep_sources is not None:
            concatinated_train[keep_sources] = [s for s in concatinated_train[self.source_column]]
        test_dataset['historic_data'] = [concatinated_train.drop(columns=cols_to_drop)]
        test_dataset['historic_sources'] = [self.train_source_name]
        test_dataset['anomaly_labels'] = [self.df_to_x_y_surv(concatinated_train)]

        test_dataset["dates"] = self.datetime_column

        test_dataset["event_preferences"] = event_preferences
        test_dataset["event_data"] = event_data
        test_dataset['predictive_horizon'] = self.predictive_horizon
        test_dataset['slide'] = self.slide
        test_dataset['lead'] = self.lead
        test_dataset['beta'] = self.beta
        test_dataset['max_wait_time'] = self.max_wait_time
        return dataset, test_dataset



    def generate_binary_labels(self,sources,list_dfs):
        """Generate binary anomaly labels based on predictive horizon and lead time.

        Creates binary labels (0=normal, 1=anomaly) by identifying samples within
        the predictive horizon before failure events and considering lead time.

        Parameters
        ----------
        sources : list[str]
            Source identifiers corresponding to dataframes.
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
        def to_span(timestamps_list, n):
            timestamps = timestamps_list[0]
            if isinstance(n, int):
                if len(timestamps) < n:
                    n = len(timestamps) - 2
                last_n = timestamps[-n:]
                time_diff = max(last_n) - min(last_n)
                hours = time_diff.total_seconds() / 3600
                return f"{int(hours)} hours"
            return n

        datesofscores = [[dtt for dtt in pd.to_datetime(df[self.datetime_column])] for df in list_dfs]

        PH = self.predictive_horizon
        lead = self.lead
        isfailure = [self.rtf_dict[str(vid)] for vid in sources]

        predictions, threshold, datesofscores, maintenances, isfailure, PHS_leads = _data_formulation(datesofscores,
                                                                                                      datesofscores,
                                                                                                      datesofscores,
                                                                                                      isfailure,
                                                                                                      None,
                                                                                                      [], PH, lead)

        anomalyranges, leadranges = extract_anomaly_ranges(maintenances, PHS_leads, isfailure, datesofscores)
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

    def get_events_from_df(self,df_list):
        events = []
        for df in df_list:
            is_fail = self.rtf_dict[df.iloc[0][self.source_column]]
            if is_fail == 1:
                events.append(
                    [df[self.datetime_column].max(), "failure", df.iloc[0][self.source_column],
                     "failure"])
            else:
                events.append(
                    [df[self.datetime_column].max(), "reset", df.iloc[0][self.source_column],
                     "maintenance"])
        return events

    def get_Classification_dataset(self,keep_sources=None):
        """
        From train episodes without failures, we ignore the last predictive_horizon period to ensure healthy operation,
        based on the objective the user wants to optimize. Then generate binary labels for all training data, labeling
        every record as 0, except those that lie within the predictive horizon before a failure event, which are labeled
         as 1.


        Returns
        -------

        """
        events = []
        # events.extend(self.get_events_from_df(self.train_dfs))
        events.extend(self.get_events_from_df(self.val_dfs))
        events.extend(self.get_events_from_df(self.test_dfs))

        from pdmlabs.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple

        event_data = pd.DataFrame(events,columns=["date", "type", "source", "description"])

        event_preferences: EventPreferences = {
            'failure': [EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')],
            'reset': [EventPreferencesTuple(description='*', type='failure', source='*', target_sources='='),
                      EventPreferencesTuple(description='*', type='reset', source='*', target_sources='=')]
        }

        clean_dfs = []
        for df in self.train_dfs:
            is_fail = self.rtf_dict[df.iloc[0][self.source_column]]
            if is_fail == 0:
                new_df=df[df[self.datetime_column]<= (df[self.datetime_column].iloc[-1]-pd.Timedelta(self.predictive_horizon))]
                clean_dfs.append(new_df)
            else:
                clean_dfs.append(df)


        historical_labels,leads=self.generate_binary_labels(self.sources_for_train,clean_dfs)


        concatinated_train = pd.concat(clean_dfs, ignore_index=True)

        historical_labels=[label for sublist in historical_labels for label in sublist]
        cols_to_drop = [self.source_column]
        if "event" in concatinated_train.columns:
            cols_to_drop.append("event")
        if self.rul_column in concatinated_train.columns:
            cols_to_drop.append(self.rul_column)
        if keep_sources is not None and keep_sources in cols_to_drop:
            cols_to_drop.remove(keep_sources)

        inner_sources_for_val= [df[self.source_column].iloc[0] for df in self.val_dfs]

        dataset={}
        dataset['match_sources'] = self.matches
        dataset['target_sources'] = [str(vid) for vid in inner_sources_for_val]

        target_data = []
        for df in self.val_dfs:
            tdf = df.copy()
            tdf = tdf.drop(columns=cols_to_drop).reset_index(drop=True).copy()
            if keep_sources is not None:
                tdf[keep_sources] = df[self.source_column]
            target_data.append(tdf)

        dataset['target_data'] = target_data

        if keep_sources is not None:
            concatinated_train[keep_sources] = [s for s in concatinated_train[self.source_column]]
        dataset['historic_data'] = [concatinated_train.drop(columns=cols_to_drop)]
        dataset['historic_sources'] = [self.train_source_name]
        dataset['anomaly_labels'] = [historical_labels]

        dataset["dates"] = self.datetime_column
        dataset["event_preferences"] = event_preferences
        dataset["event_data"] = event_data
        dataset['predictive_horizon'] = self.predictive_horizon
        dataset['slide'] = self.slide
        dataset['lead'] = self.lead
        dataset['beta'] = self.beta
        dataset['max_wait_time'] = self.max_wait_time

        ############## test dataset ###############
        inner_sources_for_test = [df[self.source_column].iloc[0] for df in self.test_dfs]

        test_dataset = {}
        test_dataset['match_sources'] = self.matches
        test_dataset['target_sources'] = [str(vid) for vid in inner_sources_for_test]

        target_data = []
        for df in self.test_dfs:
            tdf = df.copy()
            tdf = tdf.drop(columns=[self.source_column, self.rul_column]).reset_index(drop=True).copy()
            if keep_sources is not None:
                tdf[keep_sources] = df[self.source_column]
            target_data.append(tdf)
        test_dataset['target_data'] = target_data


        if keep_sources is not None:
            concatinated_train[keep_sources] = [s for s in concatinated_train[self.source_column]]
        test_dataset['historic_data'] = [concatinated_train.drop(columns=cols_to_drop)]
        test_dataset['historic_sources'] = [self.train_source_name]
        test_dataset['anomaly_labels'] = [historical_labels]

        test_dataset["dates"] = self.datetime_column
        test_dataset["event_preferences"] = event_preferences
        test_dataset["event_data"] = event_data
        test_dataset['predictive_horizon'] = self.predictive_horizon
        test_dataset['slide'] = self.slide
        test_dataset['lead'] = self.lead
        test_dataset['beta'] = self.beta
        test_dataset['max_wait_time'] = self.max_wait_time

        return dataset, test_dataset

    def get_semi_dataset(self):
        """
        From train episodes we only keep those without failures, and we ignore the last predictive_horizon period to
        ensure healthy operation, based on the objective the user wants to optimize. These are used as historical data
        without labels, to train Semi Supervised anomaly detector.

        Returns
        -------

        """
        events = []
        # events.extend(self.get_events_from_df(self.train_dfs))
        events.extend(self.get_events_from_df(self.val_dfs))
        events.extend(self.get_events_from_df(self.test_dfs))

        from pdmlabs.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple

        event_data = pd.DataFrame(events, columns=["date", "type", "source", "description"])

        event_preferences: EventPreferences = {
            'failure': [EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')],
            'reset': [EventPreferencesTuple(description='*', type='failure', source='*', target_sources='='),
                      EventPreferencesTuple(description='*', type='reset', source='*', target_sources='=')]
        }

        clean_dfs = []
        for df in self.train_dfs:
            is_fail = self.rtf_dict[df.iloc[0][self.source_column]]
            # Always extract healthy data by removing the anomalous tail
            new_df = df[df[self.datetime_column] <= (
                        df[self.datetime_column].iloc[-1] - pd.Timedelta(self.predictive_horizon))]
            if new_df.shape[0] > 0:
                clean_dfs.append(new_df)

        concatinated_train = pd.concat(clean_dfs, ignore_index=True)

        cols_to_drop = [self.source_column]
        if "event" in self.val_dfs[0].columns:
            cols_to_drop.append("event")
        if self.rul_column in self.val_dfs[0].columns:
            cols_to_drop.append(self.rul_column)

        dataset = {}
        dataset['match_sources'] = self.matches
        inner_sources_for_val = [df[self.source_column].iloc[0] for df in self.val_dfs]
        dataset['target_sources'] = [str(vid) for vid in inner_sources_for_val]

        dataset['target_data'] = [df.drop(columns=cols_to_drop).reset_index(drop=True).copy() for df in
                                  self.val_dfs]
        dataset['historic_data'] = [concatinated_train.drop(columns=cols_to_drop)]
        dataset['historic_sources'] = [self.train_source_name]

        dataset["dates"] = self.datetime_column
        dataset["event_preferences"] = event_preferences
        dataset["event_data"] = event_data
        dataset['predictive_horizon'] = self.predictive_horizon
        dataset['slide'] = self.slide
        dataset['lead'] = self.lead
        dataset['beta'] = self.beta
        dataset['max_wait_time'] = self.max_wait_time

        ############## test dataset ###############
        test_dataset = {}
        test_dataset['match_sources'] = self.matches
        inner_sources_for_test = [df[self.source_column].iloc[0] for df in self.test_dfs]
        test_dataset['target_sources'] = [str(vid) for vid in inner_sources_for_test]

        test_dataset['target_data'] = [df.drop(columns=cols_to_drop).reset_index(drop=True).copy() for df in
                                       self.test_dfs]
        test_dataset['historic_data'] = [concatinated_train.drop(columns=cols_to_drop)]
        test_dataset['historic_sources'] = [self.train_source_name]

        test_dataset["dates"] = self.datetime_column
        test_dataset["event_preferences"] = event_preferences
        test_dataset["event_data"] = event_data
        test_dataset['predictive_horizon'] = self.predictive_horizon
        test_dataset['slide'] = self.slide
        test_dataset['lead'] = self.lead
        test_dataset['beta'] = self.beta
        test_dataset['max_wait_time'] = self.max_wait_time

        return dataset, test_dataset
    def get_unsupervised_dataset(self):
        """
        From train episodes we only keep those without failures, and we ignore the last predictive_horizon period to
        ensure healthy operation, based on the objective the user wants to optimize. These are used as historical data
        without labels, to train Semi Supervised anomaly detector.

        Returns
        -------

        """
        events = []
        events.extend(self.get_events_from_df(self.train_dfs))
        events.extend(self.get_events_from_df(self.val_dfs))
        events.extend(self.get_events_from_df(self.test_dfs))

        from pdmlabs.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple

        event_data = pd.DataFrame(events, columns=["date", "type", "source", "description"])

        event_preferences: EventPreferences = {
            'failure': [EventPreferencesTuple(description='*', type='failure', source='*', target_sources='=')],
            'reset': [EventPreferencesTuple(description='*', type='failure', source='*', target_sources='='),
                      EventPreferencesTuple(description='*', type='reset', source='*', target_sources='=')]
        }

        train_val = []
        train_val.extend(self.train_dfs)
        train_val.extend(self.val_dfs)

        cols_to_drop = [self.source_column]
        if "event" in train_val[0].columns:
            cols_to_drop.append("event")
        if self.rul_column in train_val[0].columns:
            cols_to_drop.append(self.rul_column)

        dataset = {}
        inner_sources_for_val = [df[self.source_column].iloc[0] for df in train_val]
        dataset['target_sources'] = [str(vid) for vid in inner_sources_for_val]
        dataset["max_wait_time"]=self.max_wait_time
        dataset['target_data'] = [df.drop(columns=cols_to_drop).reset_index(drop=True).copy() for df in train_val]
        dataset['historic_data'] = []
        dataset['historic_sources'] = []

        dataset["dates"] = self.datetime_column
        dataset["event_preferences"] = event_preferences
        dataset["event_data"] = event_data
        dataset['predictive_horizon'] = self.predictive_horizon
        dataset['slide'] = self.slide
        dataset['lead'] = self.lead
        dataset['beta'] = self.beta

        ############## test dataset ###############
        test_dataset = {}

        test_dataset["max_wait_time"] = self.max_wait_time
        inner_sources_for_test = [df[self.source_column].iloc[0] for df in self.test_dfs]
        test_dataset['target_sources'] = [str(vid) for vid in inner_sources_for_test]

        test_dataset['target_data'] = [df.drop(columns=cols_to_drop).reset_index(drop=True).copy() for df in self.test_dfs]
        test_dataset['historic_data'] = [ ]
        test_dataset['historic_sources'] = []

        test_dataset["dates"] = self.datetime_column
        test_dataset["event_preferences"] = event_preferences
        test_dataset["event_data"] = event_data
        test_dataset['predictive_horizon'] = self.predictive_horizon
        test_dataset['slide'] = self.slide
        test_dataset['lead'] = self.lead
        test_dataset['beta'] = self.beta

        return dataset, test_dataset


def episodes_formulation(data,datetime_column,event_indicator=None,maintenance_list=None,failure_list=None,event_df=None,source_column='source',DIVIDER=3600):

    if event_df is not None:
        # source
        if datetime_column not in event_df.columns or source_column not in event_df.columns or "code" not in event_df.columns:
            raise ValueError("datetime_column and source_column must be present in event data.")
        maintenance_list=set(maintenance_list).difference(failure_list)
        maintenance_col="maintenance_event"
        failure_col="failure_event"
        event_df[maintenance_col]=[1 if code in maintenance_list else 0 for code in event_df["code"].values]
        event_df[failure_col]=[1 if code in failure_list else 0 for code in event_df["code"].values]
        event_df[datetime_column]=pd.to_datetime(event_df[datetime_column])
        event_data = event_df[[datetime_column, source_column, maintenance_col, failure_col]].copy()
        if datetime_column not in event_df.columns or source_column not in event_df.columns:
            raise ValueError("datetime_column and source_column must be present in data.")

        all_sources = []
        all_episodes = []
        all_run_to_failure = []
        original_s_has_f = {}
        for source in data[source_column].unique():
            df_source = data[data[source_column] == source].copy()

            episodes, rtfs, new_sources = data_split_by_event(df_source,
                                                              event_data[event_data[source_column] == source].copy(),
                                                              datetime_column, failure_col, maintenance_col,
                                                              source_column)
            all_episodes.extend(episodes)
            all_run_to_failure.extend(rtfs)
            original_s_has_f[source] = max(rtfs) == 1 or original_s_has_f.get(source, False)
            all_sources.extend(new_sources)

        return all_episodes, all_run_to_failure, all_sources, original_s_has_f


    elif event_indicator is not None:
        #group by source and indicate the event:
        all_episodes = []
        all_run_to_failure = []
        all_sources = []
        original_s_has_f={}
        for source,group_df in data.groupby(source_column):
            group_df=group_df.sort_values(by=datetime_column).reset_index(drop=True)
            if len(group_df[event_indicator].unique())>2:
                raise ValueError(f"event_indicator column must be binary (0 and 1) for each source. Source {source} has values {group_df[event_indicator].unique()}")
            if "RUL" not in group_df.columns:
                maxdate = group_df[datetime_column].max()
                group_df["RUL"] = [(maxdate - dtime).total_seconds() / DIVIDER for dtime in
                                   group_df[datetime_column]]

            all_run_to_failure.append(group_df.iloc[0][event_indicator])
            original_s_has_f[source]=group_df.iloc[0][event_indicator]==1  or original_s_has_f.get(source,False)
            all_episodes.append(group_df.drop(columns=[event_indicator]))
            all_sources.append(f"{source}_ep0")
        return all_episodes, all_run_to_failure,all_sources,original_s_has_f
    # check if maintenance_column and failure_column are in data
    else:
        print("Warning: event column is not in data and not eventDf was given, we consider each source as run_to_failure.")
        all_episodes = []
        all_run_to_failure = []
        original_s_has_f = {}
        all_sources = []
        for source, group_df in data.groupby(source_column):
            group_df = group_df.sort_values(by=datetime_column).reset_index(drop=True)
            if "RUL" not in group_df.columns:
                maxdate=group_df[datetime_column].max()
                group_df["RUL"]=[(maxdate - dtime).total_seconds()/DIVIDER for dtime in group_df[datetime_column]]
            all_run_to_failure.append(1)
            original_s_has_f[source] = True or original_s_has_f.get(source, False)
            all_episodes.append(group_df)
            all_sources.append(f"{source}_ep0")
        return all_episodes, all_run_to_failure, all_sources, original_s_has_f


    # # check if datetime_column, source_column are in data
    # if datetime_column not in data.columns or source_column not in data.columns:
    #     raise ValueError("datetime_column and source_column must be present in data.")
    #
    # all_sources=[]
    # all_episodes = []
    # all_run_to_failure = []
    # original_s_has_f = {}
    # for source in data[source_column].unique():
    #     df_source=data[data[source_column]==source].copy()
    #
    #     episodes, rtfs,new_sources = data_split_by_event(df_source,event_data[event_data[source_column]==source],
    #                                                      datetime_column,failure_column,maintenance_column,source_column,DIVIDER)
    #     all_episodes.extend(episodes)
    #     all_run_to_failure.extend(rtfs)
    #     original_s_has_f[source] = max(rtfs)==1 or original_s_has_f.get(source, False)
    #     all_sources.extend(new_sources)
    #
    # return all_episodes, all_run_to_failure,all_sources,original_s_has_f




def data_split_by_event(df_source,event_source,datetime_column,failure_column,maintenance_column,source_column='source',DIVIDER=3600):
    df_source.sort_values(by=datetime_column, inplace=True)
    df_source.reset_index(drop=True, inplace=True)
    event_source.sort_values(by=datetime_column, inplace=True)
    event_source.reset_index(drop=True, inplace=True)
    episodes = []
    rtfs = []
    new_sources=[]
    counter=0
    for idx, event_row in event_source.iterrows():
        event_time = event_row[datetime_column]
        found = None
        if event_row[failure_column] == 1:
            # Failure event
            found=1
        elif event_row[maintenance_column] == 1:
            found = 0
        if found is not None:
            if idx == 0:
                start_time = df_source[datetime_column].min()
            else:
                prev_event_time = event_source.loc[idx - 1, datetime_column]
                start_time = prev_event_time
            end_time = event_time
            episode = df_source[(df_source[datetime_column] > start_time) & (df_source[datetime_column] <= end_time)].copy()
            episode["RUL"]= [(episode[datetime_column].max() - dtime).total_seconds()/DIVIDER for dtime in episode[datetime_column]]
            episode[source_column]=f"{df_source.iloc[0][source_column]}_ep{counter}"

            if episode.shape[0] == 0:
                continue
                
            episodes.append(episode)
            rtfs.append(found)
            new_sources.append(f"{df_source.iloc[0][source_column]}_ep{counter}")
            counter += 1
    return episodes, rtfs, new_sources














