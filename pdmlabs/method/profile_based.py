import pandas as pd
import mlflow
import numpy as np

import pdmlabs.utils.distances as distances_utils
from pdmlabs.method.semi_supervised_method import SemiSupervisedMethodInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class ProfileBased(SemiSupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences, profile_size=5, distance_metric="euclidean", *args, **kwargs):
        super().__init__(event_preferences=event_preferences,)

        self.event_preferences=event_preferences

        self.profile_size=profile_size
        self.distance_metric=distance_metric

        self.buffer_profiles=[]
        self.buffer_sources=[]

        # mlflow.sklearn.autolog()
        self.profiles=[]
        self.sources=[]
        self.maxinner=[]


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for source, df in zip(historic_sources,historic_data):
            if source in self.sources:
                pos= self.sources.index(source)
                self.profiles[pos]=df.copy()
            else:
                self.sources.append(source)
                self.profiles.append(df.copy())
                max_inner_distance = distances_utils.calculate_distance_many_to_many(df.copy(),
                                                                                     df.copy(),
                                                                                     self.distance_metric)

                maxdistProfile = []
                for ar in max_inner_distance:
                    maxdistProfile.append(max(ar))

                max_inner_distance = max(maxdistProfile)
                if max_inner_distance ==0:
                    max_inner_distance=1
                self.maxinner.append(max_inner_distance)

    # TODO: Handle event data.
    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        all_scores=[]

        event_data=event_data.sort_values(by=['date'])
        event_data=event_data.reset_index(drop=True)
        event_pos=0
        for ind, data in target_data.iterrows():
            # while event_pos<len(event_data.index) and event_data.iloc[event_pos]["date"]<ind:
            #     if event_data.iloc[event_pos]["source"]==source:
            #         self.predict_one(event_data.iloc[event_pos],source,is_event=True)
            #     event_pos+=1
            score=self.predict_one(data,source,False)
            all_scores.append(score)
        return all_scores
    

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        if is_event:
            return self.handle_event(new_sample,source)
        else:
            if source in self.sources:
                pos_profile=self.sources.index(source)
                dists=distances_utils.calculate_distance_many_to_one(self.profiles[pos_profile],new_sample.values,self.distance_metric)
                max_inner_distance=self.maxinner[pos_profile]
                
                score=min(dists) / max_inner_distance
                if score is None:
                    score=0
                return score
            else: # NO Profile for that source, so create one as data arrive
                if source in self.buffer_sources:
                    pos_profile = self.buffer_sources.index(source)
                    self.buffer_profiles[pos_profile].append(new_sample.values)
                    if len(self.buffer_profiles[pos_profile])>=self.profile_size:
                        profile_data_frame=pd.DataFrame(np.array(self.buffer_profiles[pos_profile]))
                        self.profiles.append(profile_data_frame)
                        self.sources.append(source)

                        max_inner_distance = distances_utils.calculate_distance_many_to_many(profile_data_frame,
                                                                                             profile_data_frame,
                                                                                             self.distance_metric)

                        maxdistProfile = []
                        for ar in max_inner_distance:
                            maxdistProfile.append(max(ar))

                        max_inner_distance = max(maxdistProfile)

                        self.maxinner.append(max_inner_distance)


                        self.buffer_sources.remove(source)
                        self.buffer_profiles[pos_profile]=None
                        self.buffer_profiles.remove(None)
                else:  # NO buffer, entirely new source
                    self.buffer_sources.append(source)
                    self.buffer_profiles.append([new_sample.values])
        return 0


    def handle_event(self, new_sample, source):
        if self.checkReset(new_sample):
            self.reset(source)
        return 0


    def reset(self, source):
        if source in self.sources:
            pos_profile = self.sources.index(source)
            self.sources.remove(source)
            self.profiles[pos_profile] = None
            self.profiles.remove(None)
        if source in self.buffer_sources:
            pos_profile = self.buffer_sources.index(source)
            self.buffer_sources.remove(source)
            self.buffer_profiles[pos_profile] = None
            self.buffer_profiles.remove(None)


    def checkReset(self, event):
        for event_pref in self.event_preferences["reset"]:
            if event_pref.description == "*" or event_pref.description == event["description"]:
                if event_pref.type == "*" or event_pref.type == event["type"]:
                    if event_pref.source == "*" or event_pref.source == event["source"]:
                        return True
        return False


    def get_library(self) -> str:
        return 'no_save'


    def get_params(self) -> dict:
        params = {
            "profile_size": self.profile_size,
            "distance_metric": self.distance_metric
        }
        
        return params

    
    def __str__(self) -> str:
        return 'ProfileBased'


    def get_all_models(self):
        pass