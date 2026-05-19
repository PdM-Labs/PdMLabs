import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from pdmlabs.method.supervised_method import SupervisedMethodInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class BaseSklearnRUL(SupervisedMethodInterface):
    """Base class for sklearn regressors following the RUL interface."""

    MODEL_CLASS = "sklearn"  # to be defined in subclass

    def __init__(self, event_preferences: EventPreferences,save_model=False, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.model_per_source = {}
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.save_model=save_model

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str],
            event_data: pd.DataFrame, anomaly_ranges: list[list]) -> None:
        for current_historic_data, current_historic_source, labels in zip(
                historic_data, historic_sources, anomaly_ranges):
            model = self.MODEL_CLASS(*self.initial_args, **self.initial_kwargs)
            model.fit(current_historic_data, labels)
            self.model_per_source[current_historic_source] = model
            if self.save_model:
                import pickle
                with open(f"{self.MODEL_CLASS.__name__.upper()}_{current_historic_source}.pkl", "wb") as f:
                    pickle.dump(self.model_per_source[current_historic_source], f)
    def predict(self, target_data: pd.DataFrame, source: str,
                event_data: pd.DataFrame) -> list[float]:
        if source not in self.model_per_source:
            raise ValueError(f"No model found for source '{source}'")
        preds = self.model_per_source[source].predict(target_data)
        return preds.tolist()

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        if source not in self.model_per_source:
            raise ValueError(f"No model found for source '{source}'")
        return float(self.model_per_source[source].predict([new_sample.to_numpy()])[0])

    def get_params(self) -> dict:
        model = self.MODEL_CLASS(*self.initial_args, **self.initial_kwargs)
        return model.get_params()

    def get_library(self) -> str:
        return 'no_save'

    def __str__(self) -> str:
        return self.MODEL_CLASS.__name__.upper()

    def get_all_models(self):
        pass


# --- Linear models ---
class LinearRegressionRUL(BaseSklearnRUL):
    MODEL_CLASS = LinearRegression


class RidgeRUL(BaseSklearnRUL):
    MODEL_CLASS = Ridge


class LassoRUL(BaseSklearnRUL):
    MODEL_CLASS = Lasso


class ElasticNetRUL(BaseSklearnRUL):
    MODEL_CLASS = ElasticNet


class BayesianRidgeRUL(BaseSklearnRUL):
    MODEL_CLASS = BayesianRidge


# --- Support Vector Regression ---
class SVRRUL(BaseSklearnRUL):
    MODEL_CLASS = SVR


# --- Tree-based models ---
class DecisionTreeRUL(BaseSklearnRUL):
    MODEL_CLASS = DecisionTreeRegressor


class RandomForestRUL(BaseSklearnRUL):
    MODEL_CLASS = RandomForestRegressor


class GradientBoostingRUL(BaseSklearnRUL):
    MODEL_CLASS = GradientBoostingRegressor


class AdaBoostRUL(BaseSklearnRUL):
    MODEL_CLASS = AdaBoostRegressor


# --- Previously added ones ---
class KNNRUL(BaseSklearnRUL):
    MODEL_CLASS = KNeighborsRegressor


class KernelRidgeRUL(BaseSklearnRUL):
    MODEL_CLASS = KernelRidge


class MLPRUL(BaseSklearnRUL):
    MODEL_CLASS = MLPRegressor


class GPRRUL(BaseSklearnRUL):
    MODEL_CLASS = GaussianProcessRegressor



# Assuming your wrappers are defined as:
# KNNRUL, KernelRidgeRUL, MLPRUL, GPRRUL

# def get_extended_regressor_configs():
#     """
#     Returns a list of tuples:
#     (method_class, method_name, param_space_dict_per_method)
#     including many sklearn regressors for RUL prediction.
#     """
#     configs = []
#
#     # --- Linear Models ---
#     # configs.append((LinearRegressionRUL, "LinearRegression", [{}]))
#     # configs.append((RidgeRUL, "Ridge", [{"alpha": [0.1, 1.0, 10.0]}]))
#     # configs.append((LassoRUL, "Lasso", [{"alpha": [0.001, 0.01, 0.1, 1.0]}]))
#     configs.append((ElasticNetRUL, "ElasticNet", [{"alpha": [0.001, 0.01, 0.1],
#                                                    "l1_ratio": [0.2, 0.5, 0.8],
#                                                    "max_iter": [1000, 2000,3000],
#                                                    "random_state": [42],
#                                                    "method_selection": ["cyclic", "random"]
#                                                    }]))
#
#     configs.append((RandomForestRUL, "RandomForest", [{"n_estimators": [100, 300],
#                                                        "max_depth": [5, 10, None],
#                                                        "n_jobs": [8],
#                                                        "min_samples_leaf": [1, 5, 10],
#                                                        "random_state": [42],
#                                                        "min_samples_split": [2,5,10],
#                                                        "max_features": ["sqrt", "log2"],
#                                                        }]))
#
#
#     # configs.append((BayesianRidgeRUL, "BayesianRidge", [{"alpha_1": [1e-6, 1e-5], "lambda_1": [1e-6, 1e-5]}]))
#
#
#     # --- Tree-based Models ---
#     # configs.append((DecisionTreeRUL, "DecisionTree", [{"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}]))
#     # configs.append((GradientBoostingRUL, "GradientBoosting", [{"n_estimators": [100, 300], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5]}]))
#     # configs.append((AdaBoostRUL, "AdaBoost", [{"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]}]))
#
#     # --- SVR ---
#     configs.append((SVRRUL, "SVR", [{"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "epsilon": [0.01, 0.1, 0.2]}]))
#
#
#     # --- Others already defined ---
#     # configs.append((KNNRUL, "sKNN", [{"n_neighbors": [3, 5, 10], "weights": ["uniform", "distance"], "n_jobs": [-1]}]))
#     # configs.append((KernelRidgeRUL, "sKernelRidge", [{"alpha": [0.1, 1.0], "kernel": ["linear", "rbf"], "gamma": [None, 0.1, 1.0]}]))
#     configs.append((MLPRUL, "sMLP", [{"hidden_layer_sizes": [(64,), (128, 64)], "activation": ["relu"], "learning_rate_init": [0.001, 0.0005], "max_iter": [500]}]))
#     configs.append((GPRRUL, "sGaussianProcess", [{"alpha": [1e-10, 1e-5, 1e-3], "normalize_y": [True, False]}]))
#
#     return configs