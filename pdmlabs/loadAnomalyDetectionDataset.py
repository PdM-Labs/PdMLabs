import sys
import os
import pickle
import pandas as pd

from pdmlabs.evaluation.evaluation import _data_formulation, extract_anomaly_ranges
from pdmlabs.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple
from pdmlabs.utils.utils import expand_event_preferences


def generate_labels(dataset):
    if isinstance(dataset['event_preferences']['failure'], list):
        if len(dataset['event_preferences']['failure']) == 0:
            run_to_failure_scenarios = True
        else:
            run_to_failure_scenarios = False
    elif dataset['event_preferences']['failure'] is None:
        run_to_failure_scenarios = True
    else:
        run_to_failure_scenarios = False

    # Save initial historic_data and historic_sources before any modification
    initial_historic_data = dataset["historic_data"].copy()
    initial_historic_sources = dataset["historic_sources"].copy()

    def to_span(timestamps_list, n):
       timestamps=timestamps_list[0]
       if isinstance(n, int):
            if len(timestamps)<n:
               n= len(timestamps)-2
            last_n= timestamps[-n:]
            time_diff = max(last_n) - min(last_n)
            hours = time_diff.total_seconds() / 3600
            return f"{int(hours)} hours"

       return n

    if not run_to_failure_scenarios:
        splitted_historic_data = []
        splitted_historic_sources = []
        for historic_df, historic_source in zip(dataset["historic_data"], dataset["historic_sources"]):
            current_dates = pd.to_datetime(historic_df[dataset["dates"]])
            # print(historic_source)
            
            # extract failure dates for current source
            current_failure_dates = []
            expanded_event_preferences = expand_event_preferences(event_data=dataset['event_data'], event_preferences=dataset['event_preferences']) 
            for current_preference in expanded_event_preferences['failure']:
                matched_rows = dataset['event_data'].loc[(dataset['event_data']['type'] == current_preference.type) & (dataset['event_data']['source'] == current_preference.source) & (dataset['event_data']['description'] == current_preference.description)]
                for row_index, row in matched_rows.iterrows():
                    if current_preference.target_sources == '=' and str(row.source) == str(historic_source.split('_')[1]):
                        current_failure_dates.append(row['date'])
                    elif historic_source.split('_')[1] in current_preference.target_sources:
                        current_failure_dates.append(row['date'])
                    elif current_preference.target_sources == '*':
                        current_failure_dates.append(row['date'])

            current_failure_dates = sorted(list(set(current_failure_dates)))

            # Split historic_df at each failure date
            prev_idx = 0
            splits = []
            for i, fail_date in enumerate(current_failure_dates):
                # Only consider failure dates within the range of the dataframe
                mask = (historic_df[dataset["dates"]] > current_failure_dates[i-1]) if i > 0 else (historic_df[dataset["dates"]] >= historic_df[dataset["dates"]].min())
                mask = mask & (historic_df[dataset["dates"]] <= fail_date)
                split_df = historic_df[mask].copy()
                if not split_df.empty:
                    splits.append((split_df, f"{historic_source}_{i}"))
            
            
            # Add the part after the last failure (if any)
            if current_failure_dates:
                mask = historic_df[dataset["dates"]] > current_failure_dates[-1]
                split_df = historic_df[mask].copy()
                if not split_df.empty:
                    splits.append((split_df, f"{historic_source}_{len(current_failure_dates)}"))
            else:
                assert False

            for split_df, split_source in splits:
                splitted_historic_data.append(split_df.reset_index(drop=True))
                splitted_historic_sources.append(split_source)

        # Replace original historic_data and historic_sources
        dataset["historic_data"] = splitted_historic_data
        dataset["historic_sources"] = splitted_historic_sources

    datesofscores = [
        [
            dtt for dtt in pd.to_datetime(df[dataset["dates"]])
        ] for df in dataset["historic_data"]
    ]

    datesofscores = datesofscores

    PH = to_span(datesofscores,dataset["predictive_horizon"])
    lead = to_span(datesofscores,dataset["lead"])

    if "historical_isfailure" not in dataset:
        dataset["historical_isfailure"] = [1 for _ in dataset["historic_data"]]
    isfailure = dataset["historical_isfailure"]

    predictions, threshold, datesofscores, maintenances, isfailure, PHS_leads = _data_formulation(datesofscores,
                                                                                                  datesofscores,
                                                                                                  datesofscores,
                                                                                                  isfailure,
                                                                                                  None,
                                                                                                  [], PH, lead)

    anomalyranges, leadranges = extract_anomaly_ranges(maintenances, PHS_leads, isfailure, datesofscores)
    final_ranges=[]
    pos=0

    if not run_to_failure_scenarios:
        merged_ranges = {}
        merged_lead_ranges = {}
        merged_lengths = {}
        merged_sources = {}
        for idx, df in enumerate(dataset["historic_data"]):
            orig_source = dataset["historic_sources"][idx].rsplit('_', 1)[0]  # e.g. train_1_0 -> train_1
            temp_copy=anomalyranges[pos:pos+df.shape[0]].copy()
            temp_lead_copy=leadranges[pos:pos+df.shape[0]].copy()

            for i in range(len(temp_copy)):
                if temp_lead_copy[i]!=0:
                    temp_copy[i]=1

            if orig_source not in merged_ranges:
                merged_ranges[orig_source] = []
                merged_lead_ranges[orig_source] = []
                merged_lengths[orig_source] = []
                merged_sources[orig_source] = []

            merged_ranges[orig_source].extend(temp_copy)
            merged_lead_ranges[orig_source].extend(temp_lead_copy)
            merged_lengths[orig_source].append(df.shape[0])
            merged_sources[orig_source].append(idx)
            pos+=df.shape[0]

        # Now, restore the merged order as in the original input
        final_ranges = [merged_ranges[src] for src in sorted(merged_ranges.keys(), key=lambda s: dataset["historic_sources"].index(s+"_0") if s+"_0" in dataset["historic_sources"] else 0)]
        final_lead_ranges = [merged_lead_ranges[src] for src in sorted(merged_lead_ranges.keys(), key=lambda s: dataset["historic_sources"].index(s+"_0") if s+"_0" in dataset["historic_sources"] else 0)]
    else:
        for df in dataset["historic_data"]:
            temp_copy=anomalyranges[pos:pos+df.shape[0]].copy()
            temp_lead_copy=leadranges[pos:pos+df.shape[0]].copy()
            for i in range(len(temp_copy)):
                if temp_lead_copy[i]!=0:
                    temp_copy[i]=1
            final_ranges.append(temp_copy)
            pos+=df.shape[0]
        final_lead_ranges = [leadranges[pos-df.shape[0]:pos] for df in dataset["historic_data"]]

    # At the end, restore the initial historic_data and historic_sources
    dataset["historic_data"] = initial_historic_data
    dataset["historic_sources"] = initial_historic_sources

    return final_ranges, final_lead_ranges


def generate_RUL_labels(dataset,cut_off=0,percentage=True):

    def to_span(timestamps_list, n):
       timestamps=timestamps_list[0]
       if isinstance(n, int):
            if len(timestamps)<n:
               n= len(timestamps)-2
            last_n= timestamps[-n:]
            time_diff = max(last_n) - min(last_n)
            hours = time_diff.total_seconds() / 3600
            return f"{int(hours)} hours"
       return n


    datesofscores = [[dtt for dtt in pd.to_datetime(df[dataset["dates"]])] for df in dataset["historic_data"]]

    PH = to_span(datesofscores,dataset["predictive_horizon"])
    lead = to_span(datesofscores,dataset["lead"])
    dataset["predictive_horizon"] = PH
    dataset["slide"] = lead

    if "historical_isfailure" not in dataset:
        dataset["historical_isfailure"] = [1 for _ in dataset["historic_data"]]
    else:
        dataset["historic_data"] = [df for df,isf in zip(dataset["historic_data"],dataset["historical_isfailure"]) if isf > 0]
        dataset["historic_sources"] = [sc for sc,isf in zip(dataset["historic_sources"],dataset["historical_isfailure"]) if isf > 0]
        dataset["historical_isfailure"] = [isf for isf in dataset["historical_isfailure"] if isf > 0]

    rul_labels=[]
    for df in dataset["historic_data"]:
        if cut_off<1 and cut_off>0:
           min_size= int(df.shape[0]*cut_off)
        elif cut_off>0:
            min_size=cut_off
        else:
            min_size=0

        count_to_failure=[df.shape[0]-i if df.shape[0]-i<min_size else min_size for i in range(df.shape[0])]
        if percentage:
            count_to_failure = [i / max(count_to_failure) for i in count_to_failure]
        rul_labels.append(count_to_failure)
    dataset["anomaly_labels"] = rul_labels

    if "isfailure" not in dataset:
        dataset["isfailure"] = [1 for _ in dataset["target_data"]]
    else:
        dataset["target_data"] = [df for df, isf in zip(dataset["target_data"], dataset["isfailure"]) if
                                    isf > 0]
        dataset["target_sources"] = [sc for sc, isf in
                                       zip(dataset["target_sources"], dataset["isfailure"]) if isf > 0]
        dataset["isfailure"] = [isf for isf in dataset["isfailure"] if isf > 0]

    rul_labels_target = []
    for df in dataset["historic_data"]:
        if cut_off < 1 and cut_off > 0:
            min_size = int(df.shape[0] * cut_off)
        elif cut_off > 0:
            min_size = cut_off
        else:
            min_size = 0

        count_to_failure = [df.shape[0] - i if df.shape[0] - i < min_size else min_size for i in range(df.shape[0])]
        if percentage:
            count_to_failure = [i / max(count_to_failure) for i in count_to_failure]
        rul_labels_target.append(count_to_failure)
    dataset["target_labels"] = rul_labels_target


    return rul_labels,rul_labels_target




def load_pickle(name,generate_labels=False):
    with open(f'./DataFolder/{name}', 'rb') as handle:
        dataset = pickle.load(handle)

        if "beta" not in dataset.keys():
            dataset["beta"] = 1
        if len(dataset["historic_sources"])==0:
            dataset["min_historic_scenario_len"] = sys.maxsize
        else:
            dataset["min_historic_scenario_len"] = min(df.shape[0] for df in dataset["historic_data"])
        dataset["min_target_scenario_len"] = min(df.shape[0] for df in dataset["target_data"])
        if "max_wait_time" not in dataset:
            dataset["max_wait_time"] = max(dataset["min_target_scenario_len"] // 10, 10)
        elif dataset["max_wait_time"] is None:
            dataset["max_wait_time"] = max(dataset["min_target_scenario_len"]//10,10)

        if dataset["predictive_horizon"] is None:
            dataset["predictive_horizon"] = max(dataset["min_target_scenario_len"]//10,2)

        if generate_labels:
            if "anomaly_ranges" not in dataset.keys():
                anomaly_ranges,leads=generate_labels(dataset)
                dataset["anomaly_labels"] = anomaly_ranges
            elif not dataset["anomaly_ranges"]:
                anomaly_ranges,leads=generate_labels(dataset)
                dataset["anomaly_labels"]= anomaly_ranges
        return dataset

def load_run_to_failure(column_of_timestamp,list_df,predictive_horizon,sources_names=None,max_wait_time=0.1,lead="1 seconds",slide=10):
    """
    Used for Run to failure datasets
    Parameters
    ----------
    list_df: list of dataframes with data (each one ending in falure)
    predictive_horizon : the predictive horizon before the failure, to consider valid alarms (e.g. '10 days','5 minutes')
    max_wait_time : maximum value for the length of the initial data to consider for training in online flavor and sliding. Either portion (<1) or the length e.g. 0.1, 100.
    lead : the lead time before the failure, to ignore alarms too close to failures (e.g. '10 days','5 minutes')
    slide : the slide parameter for VUS
    sources_names: the name of each df (default none), if set to value need to be same length as list_df.
    Returns dataset dictionary
    -------

    """

    if sources_names is None:
        sources_names = [i for i,_ in enumerate(list_df)]
    elif len(sources_names) < len(list_df):
        sources_names=sources_names+[ f"{sources_names[-1]}_{i}" for i in range(len(list_df)-len(sources_names))]
    historic_data = []
    historic_sources = []

    target_data = []
    target_sources = []

    for df, source_n in zip(list_df,sources_names):
        target_data.append(df)
        target_sources.append(source_n)

    event_data = pd.DataFrame(columns=["date", "type", "source", "description"])

    event_preferences: EventPreferences = {
        'failure': [],
        'reset': []
    }

    dataset={}
    dataset["dates"]=column_of_timestamp
    dataset["event_preferences"]=event_preferences
    dataset["event_data"]=event_data
    dataset["target_data"]=target_data
    dataset["target_sources"]=target_sources
    dataset["historic_data"]=historic_data
    dataset["historic_sources"]=historic_sources
    dataset["predictive_horizon"]=predictive_horizon
    dataset["slide"]=slide
    dataset["lead"]=lead
    dataset["beta"]=1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)
    if max_wait_time>1:
        dataset["max_wait_time"] = int(max_wait_time)
    else:
        dataset["max_wait_time"] = int(max_wait_time*min([td.shape[0] for td in target_data]))

    return dataset


def load_dataset_single_source_dataframe(dfor,labels_col,index_col,source_name):
    df=dfor.copy()
    if index_col is None:
        df.index = pd.to_datetime(df.index)
    else:
        df.index = pd.to_datetime(df[index_col])
        df=df.drop(index_col,axis=1)
    df.sort_index(inplace=True)
    labels=[1 if lb>=1 else 0 for lb in df[labels_col].values]
    df = df.drop(labels_col, axis=1)
    datalabels = [dt for dt in df.index]
    df['date'] = [dt for dt in df.index]
    event_data=create_preferencies(labels, datalabels, source_name)

    return event_data,df,source_name,labels


def create_preferencies(labels,datalabels,source_name):
    date = []
    source = []
    description = []
    type = []

    for i in range(len(labels) - 1):
        if labels[i] == 1 and labels[i + 1] == 0:
            date.append(datalabels[i])
            source.append(source_name)
            description.append("anomaly")
            type.append("anomaly")

    event_data = pd.DataFrame(
        {"date": date, "type": type, "source": source, "description": description})
    event_data = event_data.sort_values(by='date')

    return event_data


def load_dataset_from_dataframe(df,labels_col,index_col=None,source_column=None,reset_after_anomaly=False):
    """
    
    :param df: data (along with timestamps and labels
    :param labels_col: column with label
    :param index_col: column of dates to index the timeseries
    :param source_column: column that indicates different sources (that should be separated)
    :param reset_after_anomaly: weather we should trigger rest event after anomaly (e.g. to fit again in some experiemnts)
    :return: 
    """
    target_data=[]
    target_sources=[]
    datalabels = []
    event_data=None
    if source_column is None:
        event_data,dfdata,source_name,datalabels_s=load_dataset_single_source_dataframe(df, labels_col, index_col, "s")
        target_data.append(dfdata)
        target_sources.append(source_name)
        datalabels.append(datalabels_s)
    else:
       
        for source_name in df[source_column].unique():
            dfor=df[df[source_column]==source_name]
            event_data_s,dfdata,source_name,datalabels_s=load_dataset_single_source_dataframe(dfor, labels_col, index_col, source_name)
            target_data.append(dfdata)
            target_sources.append(source_name)
            datalabels.append(datalabels_s)
            if event_data is None:
                event_data=event_data_s
            else:
                event_data=pd.concat([event_data,event_data_s])
    event_data = event_data.sort_values(by='date')
    if reset_after_anomaly:
        event_preferences: EventPreferences = {
            'failure': [
                EventPreferencesTuple(description='*', type='anomaly', source='*', target_sources='=')
            ],
            'reset': [
                EventPreferencesTuple(description='*', type='anomaly', source='*', target_sources='='),
            ]
        }
    else:
        event_preferences: EventPreferences = {
            'failure': [
                EventPreferencesTuple(description='*', type='anomaly', source='*', target_sources='=')
            ],
            'reset': []
        }

    dataset = {}
    flatten_labels=[x for xs in datalabels for x in xs]
    dataset["max_wait_time"] = 200
    dataset["anomaly_ranges"] = True
    dataset["label_dates"] = datalabels
    dataset["dates"] = "date"
    dataset["event_preferences"] = event_preferences
    dataset["event_data"] = event_data
    dataset["target_data"] = target_data
    dataset["target_sources"] = target_sources
    dataset["historic_data"] = []
    dataset["historic_sources"] = []
    dataset["predictive_horizon"] = datalabels # there are many episodes of length 1,2,3 ...
    dataset["slide"] = 3
    dataset["lead"] = [[0 for z in eplb] for eplb in datalabels]
    dataset["beta"] = 1
    dataset["min_historic_scenario_len"] = sys.maxsize
    dataset["min_target_scenario_len"] = min(df.shape[0] for df in target_data)

    return dataset