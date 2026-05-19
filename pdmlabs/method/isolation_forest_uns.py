from sklearn.ensemble import IsolationForest as isolation_forest
import pandas as pd
import mlflow

from pdmlabs.method.unsupervised_method import UnsupervisedMethodInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class IsolationForestUnsupervised(UnsupervisedMethodInterface):
    def __init__(self, 
                 event_preferences: EventPreferences,
                 window=40,
                 slide=0.5,
                 policy="or",
                 *args, 
                 **kwargs
    ):
        super().__init__(event_preferences=event_preferences)
        self.slide = int(slide*window)
        self.window = window
        self.policy = policy
        
        self.initial_args = args
        self.initial_kwargs = kwargs
        
        self.clf_class = isolation_forest
        self.model_per_source = {}


    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        scores = self._inner_batch_predict(target_data)
        return scores


    def _inner_batch_predict(self,dfr:pd.DataFrame):
        tempmodel = self.clf_class(*(self.initial_args), **(self.initial_kwargs))
        final_score = {}
        pos = 0
        df=dfr.values
        for pos in range(self.window, len(df), self.slide):
            currentdf = df[max(pos - self.window, 0):pos]
            ids = [kati for kati in range(max(pos - self.window, 0), pos)]
            tempmodel.fit(currentdf)
            scores = -tempmodel.score_samples(currentdf)
            final_score = self.combinescores(final_score, ids,scores)
        if pos < len(df):
            pos = len(df)
            currentdf = df[max(pos - self.window, 0):pos]
            tempmodel.fit(currentdf)
            scores = -tempmodel.score_samples(currentdf)
            ids = [kati for kati in range(max(pos - self.window, 0), pos)]
            final_score = self.combinescores(final_score, ids,scores)
        scores_to_return = []
        for ind in range(0, len(df)):
            scores_to_return.append(final_score[ind])
        return scores_to_return


    def combinescores(self, final_score, ids,scores):
        for sc, ind in zip(scores, ids):
            if ind in final_score.keys():
                if self.policy == "or":
                    final_score[ind] = max(final_score[ind], sc)
                elif self.policy == "and":
                    final_score[ind] = min(final_score[ind], sc)
                elif self.policy == "first":
                    final_score[ind] = final_score[ind]
                elif self.policy == "last":
                    final_score[ind] = sc
                else:
                    final_score[ind] = sc
            else:
                final_score[ind] = sc
        return final_score


    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0
    

    def get_library(self) -> str:
        return 'no_save'
    
    
    def __str__(self) -> str:
        return 'IsolationForestUnsupervised'


    def get_params(self) -> dict:
        return {
            **(self.clf_class(*(self.initial_args), **(self.initial_kwargs)).get_params()),
            'window': self.window,
            'slide': self.slide,
            'policy': self.policy
        }


    def get_all_models(self):
        pass