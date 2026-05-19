import pandas as pd
import mlflow.pyfunc

class BasePdMPipeline(mlflow.pyfunc.PythonModel):
    """
    Base class for a Unified PdMLabs Pipeline.
    Encapsulates the preprocessor, method, postprocessor, and thresholder.
    Inherits from mlflow.pyfunc.PythonModel for seamless MLflow tracking and serving.
    """
    def __init__(self, preprocessor, method, postprocessor, thresholder):
        self.preprocessor = preprocessor
        self.method = method
        self.postprocessor = postprocessor
        self.thresholder = thresholder
        
        self.fitted_sources = []
        self.global_threshold = None

    def set_global_threshold(self, th: float):
        """Injects the globally optimal threshold found by the experiment's evaluation metric."""
        self.global_threshold = th

    def predict(self, context=None, model_input=None):
        """
        MLflow compliant inference signature.
        model_input can be a dictionary containing target_data, source, and event_data,
        or a single DataFrame (which defaults to the first fitted source).
        """
        if isinstance(model_input, dict):
            target_data = model_input.get('target_data')
            source = model_input.get('source', None)
            event_data = model_input.get('event_data', pd.DataFrame())
        else:
            if model_input is None:
                target_data = context.get('target_data')
                source = context.get('source', None)
                event_data = context.get('event_data', pd.DataFrame())
            else:
                target_data = model_input
                source = None
                event_data = pd.DataFrame()
            
        fallback_sources = []
        if hasattr(self, 'method') and hasattr(self.method, 'model_per_source') and isinstance(self.method.model_per_source, dict):
            fallback_sources = list(self.method.model_per_source.keys())

        if source is None or (hasattr(self, 'fitted_sources') and source not in self.fitted_sources):
            if hasattr(self, 'fitted_sources') and len(self.fitted_sources) > 0:
                source = self.fitted_sources[0]
            elif len(fallback_sources) > 0:
                source = fallback_sources[0]
            else:
                source = "default_source"
                
        # Execute the unified inference chain
        current_data = self.preprocessor.transform(target_data, source, event_data)
        scores = self.method.predict(current_data, source, event_data)
        processed_scores = self.postprocessor.transform(scores, source, event_data)
        thresholds = self.thresholder.infer_threshold(processed_scores, source, event_data, target_data.index)
        
        result = {
            'scores': processed_scores,
            'dynamic_thresholds': thresholds
        }
        
        if self.global_threshold is not None:
            result['global_best_threshold'] = self.global_threshold
            result['anomaly_labels'] = [1 if s > self.global_threshold else 0 for s in processed_scores]
            
        return result


    def predict_scores_only(self, target_data, source, event_data):
        """Executes preprocessor, method, and postprocessor but skips the thresholder."""
        fallback_sources = []
        if hasattr(self, 'method') and hasattr(self.method, 'model_per_source') and isinstance(self.method.model_per_source, dict):
            fallback_sources = list(self.method.model_per_source.keys())

        if source is None or (hasattr(self, 'fitted_sources') and source not in self.fitted_sources):
            if hasattr(self, 'fitted_sources') and len(self.fitted_sources) > 0:
                source = self.fitted_sources[0]
            elif len(fallback_sources) > 0:
                source = fallback_sources[0]
            else:
                source = "default_source"
                
        current_data = self.preprocessor.transform(target_data, source, event_data)
        scores = self.method.predict(current_data, source, event_data)
        processed_scores = self.postprocessor.transform(scores, source, event_data)
        return processed_scores

class SemiSupervisedPdMPipeline(BasePdMPipeline):
    """Pipeline for Semi-Supervised Flavors."""
    def fit(self, historic_data, historic_sources, event_data):
        self.fitted_sources = historic_sources
        
        self.preprocessor.fit(historic_data, historic_sources, event_data)
        
        transformed_historic = []
        for df, src in zip(historic_data, historic_sources):
            transformed_historic.append(self.preprocessor.transform(df, src, event_data))
            
        self.method.fit(transformed_historic, historic_sources, event_data)
        
        # Postprocessor is typically not fitted in semi-supervised, but we keep the interface available
        # if the specific postprocessor implements it.
        if hasattr(self.postprocessor, 'fit'):
            try:
                self.postprocessor.fit(transformed_historic, historic_sources, event_data)
            except Exception:
                pass
                
        return self


class SupervisedPdMPipeline(BasePdMPipeline):
    """Pipeline for Supervised Flavors."""
    def fit(self, historic_data, historic_sources, event_data, anomaly_ranges):
        self.fitted_sources = historic_sources
        
        self.preprocessor.fit(historic_data, historic_sources, event_data, anomaly_ranges)
        
        transformed_historic = []
        for df, src in zip(historic_data, historic_sources):
            transformed_historic.append(self.preprocessor.transform(df, src, event_data))
            
        self.method.fit(transformed_historic, historic_sources, event_data, anomaly_ranges)
        self.postprocessor.fit(transformed_historic, historic_sources, event_data, anomaly_ranges)
        return self


class UnsupervisedPdMPipeline(BasePdMPipeline):
    """Pipeline for Unsupervised Flavors."""
    def fit(self):
        # Unsupervised methods don't technically require historic data fitting.
        # This function marks the pipeline as ready and matches the API structure.
        return self


class RULPdMPipeline(SupervisedPdMPipeline):
    """Pipeline for Remaining Useful Life (RUL) Flavors."""
    def predict(self, context=None, model_input=None):
        """Override to return RUL specific predictions bypassing binary thresholds."""
        if isinstance(model_input, dict):
            target_data = model_input.get('target_data')
            source = model_input.get('source', None)
            event_data = model_input.get('event_data', pd.DataFrame())
        else:
            if model_input is None:
                target_data = context.get('target_data')
                source = context.get('source', None)
                event_data = context.get('event_data', pd.DataFrame())
            else:
                target_data = model_input
                source = None
                event_data = pd.DataFrame()
            
        fallback_sources = []
        if hasattr(self, 'method') and hasattr(self.method, 'model_per_source') and isinstance(self.method.model_per_source, dict):
            fallback_sources = list(self.method.model_per_source.keys())

        if source is None or (hasattr(self, 'fitted_sources') and source not in self.fitted_sources):
            if hasattr(self, 'fitted_sources') and len(self.fitted_sources) > 0:
                source = self.fitted_sources[0]
            elif len(fallback_sources) > 0:
                source = fallback_sources[0]
            else:
                source = "default_source"
                
        current_data = self.preprocessor.transform(target_data, source, event_data)
        scores = self.method.predict(current_data, source, event_data)
        processed_scores = self.postprocessor.transform(scores, source, event_data)
        
        # RUL generally just returns the scores (predictions) directly
        return {'rul_predictions': processed_scores}


class SAPdMPipeline(SupervisedPdMPipeline):
    """Pipeline for Survival Analysis (SA) Flavors."""
    def fit_thresholder(self, result_scores, target_sources, event_data, result_labels):
        """Allows the experiment to fit the SA thresholder on the validation set scores."""
        if hasattr(self.thresholder, 'fit'):
            self.thresholder.fit(result_scores, target_sources, event_data, result_labels)
        return self

    def predict(self, context=None, model_input=None):
        """Override to return SA specific predictions."""
        if isinstance(model_input, dict):
            target_data = model_input.get('target_data')
            source = model_input.get('source', None)
            event_data = model_input.get('event_data', pd.DataFrame())
        else:
            if model_input is None:
                target_data = context.get('target_data')
                source = context.get('source', None)
                event_data = context.get('event_data', pd.DataFrame())
            else:
                target_data = model_input
                source = None
                event_data = pd.DataFrame()
            
        fallback_sources = []
        if hasattr(self, 'method') and hasattr(self.method, 'model_per_source') and isinstance(self.method.model_per_source, dict):
            fallback_sources = list(self.method.model_per_source.keys())

        if source is None or (hasattr(self, 'fitted_sources') and source not in self.fitted_sources):
            if hasattr(self, 'fitted_sources') and len(self.fitted_sources) > 0:
                source = self.fitted_sources[0]
            elif len(fallback_sources) > 0:
                source = fallback_sources[0]
            else:
                source = "default_source"
                
        current_data = self.preprocessor.transform(target_data, source, event_data)
        scores = self.method.predict(current_data, source, event_data)
        processed_scores = self.postprocessor.transform(scores, source, event_data)
        
        # The SA thresholder returns the actual RUL survival mappings based on the scores
        dates = target_data.index
        thresholds = self.thresholder.infer_threshold(processed_scores, source, event_data, dates)
        
        return {'survival_curves': processed_scores, 'rul_predictions': thresholds}
