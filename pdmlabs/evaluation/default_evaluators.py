import os
import pickle
import pandas as pd
import mlflow
import locket

from pdmlabs.evaluation.evaluator import EvaluatorInterface
from pdmlabs.evaluation.evaluation import AUCPR_new as pdm_evaluate
from pdmlabs.evaluation.evaluation import AUCPR_ranges_new as pdm_evaluate_ranges

class BaseADEvaluator(EvaluatorInterface):
    """
    Base class for AD-style metrics. Computes standard PdM metrics like AD1_AUC, VUS, etc.
    """
    def __init__(self, debug=False):
        self.debug = debug

    def _compute_core_metrics(self, experiment, result_scores, result_dates, results_isfailure, plot_dictionary):
        pipeline = experiment.pipeline
        
        if "anomaly_ranges" in pipeline.dataset.keys():
            if pipeline.dataset["anomaly_ranges"]:
                allresults, results_vus, anomaly_ranges, lead_ranges = pdm_evaluate_ranges(
                    result_scores,
                    anomalyranges=pipeline.dataset["predictive_horizon"],
                    leadranges=pipeline.dataset["lead"],
                    beta=pipeline.beta,
                    resolution=pipeline.auc_resolution,
                    slidingWindow_vus=pipeline.slide
                )
            else:
                allresults, results_vus, anomaly_ranges, lead_ranges = pdm_evaluate(
                    result_scores,
                    datesofscores=result_dates,
                    isfailure=results_isfailure,
                    PH=pipeline.predictive_horizon,
                    lead=pipeline.lead,
                    beta=pipeline.beta,
                    resolution=pipeline.auc_resolution,
                    slidingWindow_vus=pipeline.slide
                )
        else:
            allresults, results_vus, anomaly_ranges, lead_ranges = pdm_evaluate(
                result_scores,
                datesofscores=result_dates,
                isfailure=results_isfailure,
                PH=pipeline.predictive_horizon,
                lead=pipeline.lead,
                beta=pipeline.beta,
                resolution=pipeline.auc_resolution,
                slidingWindow_vus=pipeline.slide
            )

        recalls = []
        precisions = []
        for row in allresults:
            recalls.append(row[3])
            precisions.append(row[6])
            
        plot_dictionary["recall"] = recalls
        plot_dictionary["prc"] = precisions
        plot_dictionary["anomaly_ranges"] = anomaly_ranges
        plot_dictionary["lead_ranges"] = lead_ranges

        all_results_appended_with_vus = []
        results_vus_keys = list(results_vus.keys())
        for row in allresults:
            result_to_append = list(row)
            result_to_append.extend([results_vus[key] for key in results_vus_keys])
            all_results_appended_with_vus.append(result_to_append)

        param_name_to_index_dict = {
            'AD1_rcl': 3, 'AD2_rcl': 4, 'AD3_rcl': 5, 'prc': 6,
            'AD1_f1': 0, 'AD2_f1': 1, 'AD3_f1': 2,
            'AD1_AUC': 8, 'AD2_AUC': 9, 'AD3_AUC': 10,
            'threshold_auc': 7,
        }

        for index, key in enumerate(results_vus_keys):
            param_name_to_index_dict['VUS_' + key] = index + 11

        return all_results_appended_with_vus, results_vus_keys, param_name_to_index_dict

    def _find_best_metrics(self, all_results_appended_with_vus, results_vus_keys, param_name_to_index_dict, optimization_metric):
        best_dict = {
            'AD1_rcl': -1, 'AD2_rcl': -1, 'AD3_rcl': -1, 'prc': -1,
            'AD1_f1': -1, 'AD2_f1': -1, 'AD3_f1': -1,
            'AD1_AUC': -1, 'AD2_AUC': -1, 'AD3_AUC': -1, 'threshold_auc': -1,
        }
        for metric in results_vus_keys:
            best_dict[f"VUS_{metric}"] = -1

        metric_index_to_choose_best_from = param_name_to_index_dict.get(optimization_metric, param_name_to_index_dict['AD1_f1'])
        
        best_row_index = -1
        for current_row_index, row in enumerate(all_results_appended_with_vus):
            if row[metric_index_to_choose_best_from] > best_dict.get(optimization_metric, best_dict['AD1_f1']):
                best_dict = {}
                for key, index in param_name_to_index_dict.items():
                    best_dict[key] = row[index]
                best_row_index = current_row_index

        best_dict_to_log = {key: value for key, value in best_dict.items()}
        return best_dict_to_log, best_row_index

    def _log_threshold_csv(self, all_results_appended_with_vus, results_vus_keys, best_row_index):
        if best_row_index >= 0 and best_row_index < len(all_results_appended_with_vus):
            all_results_appended_with_vus.pop(best_row_index)

        current_run = mlflow.active_run()
        if current_run is not None:
            columns = ['AD1_f1', 'AD2_f1', 'AD3_f1', 'AD1_rcl', 'AD2_rcl', 'AD3_rcl', 'prc', 'threshold_auc', 'AD1_AUC', 'AD2_AUC', 'AD3_AUC'] + [f'VUS_{key}' for key in results_vus_keys]
            metrics_for_other_thresholds_df = pd.DataFrame(all_results_appended_with_vus, columns=columns)

            csv_path = f'./metrics_for_other_thresholds_{current_run.info.run_id}.csv'
            metrics_for_other_thresholds_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)
            os.remove(csv_path)


class DefaultADEvaluator(BaseADEvaluator):
    def evaluate(self, experiment, **kwargs) -> dict:
        result_scores = kwargs.get('result_scores')
        result_dates = kwargs.get('result_dates')
        results_isfailure = kwargs.get('results_isfailure')
        plot_dictionary = kwargs.get('plot_dictionary')

        valid_optimization_params = ['AD1_AUC', 'AD2_AUC', 'AD3_AUC', 'AD1_f1', 'AD2_f1', 'AD3_f1', 'AD1_rcl', 'AD2_rcl', 'AD3_rcl', 'prc']
        
        # If the parameter is not a standard AD param, AD metrics might not be able to optimize on it properly.
        # But we still run the AD evaluation and return the AD1_f1-based metrics if so.
        opt_param = experiment.optimization_param if experiment.optimization_param in valid_optimization_params else 'AD1_f1'

        all_results_appended, vus_keys, param_dict = self._compute_core_metrics(
            experiment, result_scores, result_dates, results_isfailure, plot_dictionary
        )
        
        # For AD1_AUC optimization we pick the best AD1_f1 row according to original logic
        search_metric = opt_param if opt_param != 'AD1_AUC' else 'AD1_f1'
        
        best_dict_to_log, best_row_index = self._find_best_metrics(
            all_results_appended, vus_keys, param_dict, search_metric
        )

        # In original code, MLflow logging happens here
        mlflow.log_metrics(best_dict_to_log)

        # Log best scores info dict pickle
        current_run = mlflow.active_run()
        if experiment.log_best_scores and current_run:
            with locket.lock_file(experiment.lock_file_path):
                if not os.path.exists(experiment.best_scores_info_dict_path):
                    best_scores_info_dict_to_write_on_disk = {
                        'best_scores': result_scores,
                        'best_optimization_value': best_dict_to_log[experiment.optimization_param] if experiment.optimization_param in best_dict_to_log else best_dict_to_log['AD1_f1'],
                        'best_run_id': current_run.info.run_id
                    }
                    with open(experiment.best_scores_info_dict_path, 'wb') as file:
                        pickle.dump(best_scores_info_dict_to_write_on_disk, file)
                else:
                    best_scores_info_dict_to_write_on_disk = None
                    with open(experiment.best_scores_info_dict_path, 'rb') as file:
                        previously_saved_dict = pickle.load(file)
                        current_opt_val = best_dict_to_log[experiment.optimization_param] if experiment.optimization_param in best_dict_to_log else best_dict_to_log['AD1_f1']
                        if current_opt_val > previously_saved_dict['best_optimization_value']:
                            best_scores_info_dict_to_write_on_disk = {
                                'best_scores': result_scores,
                                'best_optimization_value': current_opt_val,
                                'best_run_id': current_run.info.run_id
                            }
                    if best_scores_info_dict_to_write_on_disk is not None:
                        with open(experiment.best_scores_info_dict_path, 'wb') as file:
                            pickle.dump(best_scores_info_dict_to_write_on_disk, file)

        self._log_threshold_csv(all_results_appended, vus_keys, best_row_index)

        return best_dict_to_log


class DefaultClassificationEvaluator(BaseADEvaluator):
    def evaluate(self, experiment, **kwargs) -> dict:
        result_scores = kwargs.get('result_scores')
        result_dates = kwargs.get('result_dates')
        results_isfailure = kwargs.get('results_isfailure')
        plot_dictionary = kwargs.get('plot_dictionary')

        all_results_appended, vus_keys, param_dict = self._compute_core_metrics(
            experiment, result_scores, result_dates, results_isfailure, plot_dictionary
        )
        
        # Classification metrics specifically target AD1_f1 to find the best row
        best_dict_to_log, best_row_index = self._find_best_metrics(
            all_results_appended, vus_keys, param_dict, 'AD1_f1'
        )

        self._log_threshold_csv(all_results_appended, vus_keys, best_row_index)

        return best_dict_to_log


import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sksurv.metrics import brier_score, integrated_brier_score, cumulative_dynamic_auc, concordance_index_censored

from pdmlabs.evaluation.metrics_utils import (
    root_mean_squared_error, mean_absolute_percentage_error, mdape, mape_mdape_bins, IBR_bins
)
from pdmlabs.utils.rul_transformations import hard_transform_survival, sigmoid_survival_batch

class DefaultSurvEvaluator(EvaluatorInterface):
    def __init__(self, debug=False):
        self.debug = debug

    def evaluate(self, experiment, **kwargs) -> dict:
        results_rul = kwargs.get('results_rul')
        result_scores = kwargs.get('result_scores')
        result_dates = kwargs.get('result_dates')
        result_labels = kwargs.get('result_labels')
        train_labels = kwargs.get('train_labels')
        plot_dictionary = kwargs.get('plot_dictionary')
        rtfs = kwargs.get('rtfs')

        flatten_preds = []
        flatten_labels = []
        is_rtf = []
        for predds, rtff in zip(results_rul, rtfs):
            flatten_preds.extend(predds)
            is_rtf.extend([rtff for i in range(len(predds))])
        for labss in result_labels:
            flatten_labels.extend([labb[0] for labb in labss])

        train_flatten_labels = []
        for labss in train_labels:
            train_flatten_labels.extend([labb[0] for labb in labss])

        flatten_for_rul = [flat_pred for flat_pred, isrtf in zip(flatten_preds, is_rtf) if isrtf == 1]
        labels_for_rul = [flat_pred for flat_pred, isrtf in zip(flatten_labels, is_rtf) if isrtf == 1]

        best_dict = {}
        if labels_for_rul and flatten_for_rul:
            best_dict['mse'] = mean_squared_error(labels_for_rul, flatten_for_rul)
            best_dict['r2'] = r2_score(labels_for_rul, flatten_for_rul)
            best_dict['mae'] = mean_absolute_error(labels_for_rul, flatten_for_rul)
            best_dict['rmse'] = root_mean_squared_error(labels_for_rul, flatten_for_rul)
            best_dict['mape'] = mean_absolute_percentage_error([l + 1 for l in labels_for_rul], [p + 1 for p in flatten_for_rul])
            best_dict['mdape'] = mdape([l + 1 for l in labels_for_rul], [p + 1 for p in flatten_for_rul])
            mape_bins, mdape_bins = mape_mdape_bins(flatten_for_rul, labels_for_rul, n=10)
        else:
            mape_bins, mdape_bins = None, None

        test_preds = []
        for pred in result_scores:
            test_preds.extend([inpred[0] for inpred in pred])

        times = np.unique([ty for ty in train_flatten_labels])
        times.sort()
        test_y = [(rtf, ty) for ty, rtf in zip(flatten_labels, is_rtf)]

        eval_survs = self._surv_eval(test_y, test_preds, times=times, train_y=None)
        if mape_bins is not None:
            eval_survs['mape_bins'] = mape_bins
            eval_survs['mdape_bins'] = mdape_bins

        if self.debug and hasattr(experiment, '_plot_SA'):
            experiment._plot_SA(plot_dictionary)

        for key in eval_survs:
            if key in ['brier_scores', 'roc_auc_list', 'c_index_list', "mape_bins", "mdape_bins", "IBR_bins"]:
                mlflow.log_table(eval_survs[key], f"survival_{key}.json")
            else:
                best_dict[key] = eval_survs[key]

        max_rul = max([max(predds) for predds in results_rul])

        # Delegate AD metric evaluation
        ad_evaluator = DefaultClassificationEvaluator()
        transformed_scores = [[max_rul - rulpred for rulpred in episode_scores] for episode_scores in results_rul]
        
        ad_kwargs = {
            'result_scores': transformed_scores,
            'result_dates': result_dates,
            'results_isfailure': rtfs,
            'plot_dictionary': plot_dictionary
        }
        best_dict_to_log = ad_evaluator.evaluate(experiment, **ad_kwargs)

        for key in best_dict_to_log:
            best_dict[key] = best_dict_to_log[key]

        mlflow.log_metrics({k: v for k, v in best_dict.items() if isinstance(v, (int, float, np.number))})
        
        return best_dict

    def _surv_eval(self, test_y, test_preds, times=None, train_y=None):
        test_preds = np.array(test_preds)
        if train_y is None:
            train_y = np.array(test_y, dtype=[('event', 'bool'), ('time', 'float')])
            new_test_y = []
            new_test_preds = []
            for ty, preds_i in zip(test_y, test_preds):
                new_test_y.append(ty)
                new_test_preds.append(preds_i)
            test_y = np.array(new_test_y, dtype=[('event', 'bool'), ('time', 'float')])
            test_preds = np.array(new_test_preds)

        if times is None:
            times = np.unique([ty[1] for ty in train_y])
            times.sort()

        maxtt = max([ty[1] for ty in test_y])
        mintt = min([ty[1] for ty in test_y])
        pos = 0
        for i, t in enumerate(times):
            if t >= maxtt:
                break
            pos = i
        pos_pre = 0
        for i, t in enumerate(times):
            if t >= mintt:
                pos_pre = i
                break

        times = times[pos_pre:pos]
        test_preds = test_preds[:, pos_pre:pos]

        b_times, b_score = brier_score(train_y, test_y, test_preds, times)
        integrated_brier_score_value = integrated_brier_score(train_y, test_y, test_preds, times)
        roc_pt, mean_roc = cumulative_dynamic_auc(train_y, test_y, -test_preds, times)

        evals = {}
        target_times = np.linspace(times.min(), times.max(), 20)
        positions = [np.argmin(np.abs(times - t)) for t in target_times]
        cis = []
        events = [ty[0] for ty in test_y]
        times_for_c = [ty[1] for ty in test_y]

        for i in positions:
            res = concordance_index_censored(events, times_for_c, [1 - tpred for tpred in test_preds[:, i]])
            cis.append((times_for_c[i], res[0]))

        summaris = [np.sum(test_preds[i, :]) for i in range(len(test_preds))]
        maxsum = max(summaris)
        sumres = concordance_index_censored(events, times_for_c, [maxsum - summm for summm in summaris])
        cis.append((-1, sumres[0]))

        evals['brier_scores'] = {"time": [bt for bt in b_times], "Brier": [br for br in b_score]}
        evals['roc_auc_list'] = {"time": [bt for bt in times], "ROC": [br for br in roc_pt]}
        evals['c_index_list'] = {"time": [ci[0] for ci in cis], "C-Index": [ci[1] for ci in cis]}
        evals['c_index_mean'] = np.mean([ci[1] for ci in cis])
        evals['c_index'] = np.max([ci[1] for ci in cis])
        evals['IBS'] = integrated_brier_score_value
        evals['Max_brier'] = np.max(b_score)
        evals['mean_roc'] = mean_roc

        return evals


class DefaultRULEvaluator(EvaluatorInterface):
    def __init__(self, debug=False):
        self.debug = debug

    def evaluate(self, experiment, **kwargs) -> dict:
        result_scores = kwargs.get('result_scores')
        result_dates = kwargs.get('result_dates')
        result_labels = kwargs.get('result_labels')
        plot_dictionary = kwargs.get('plot_dictionary')
        rtfs = kwargs.get('rtfs')

        flatten_preds = []
        flatten_labels = []
        is_rtf = []
        for predds, rtff in zip(result_scores, rtfs):
            flatten_preds.extend(predds)
            is_rtf.extend([rtff for i in range(len(predds))])
        for labss in result_labels:
            flatten_labels.extend(labss)

        flatten_for_rul = [flat_pred for flat_pred, isrtf in zip(flatten_preds, is_rtf) if isrtf == 1]
        labels_for_rul = [flat_pred for flat_pred, isrtf in zip(flatten_labels, is_rtf) if isrtf == 1]

        best_dict = {}
        if labels_for_rul and flatten_for_rul:
            best_dict['mse'] = mean_squared_error(labels_for_rul, flatten_for_rul)
            best_dict['r2'] = r2_score(labels_for_rul, flatten_for_rul)
            best_dict['mae'] = mean_absolute_error(labels_for_rul, flatten_for_rul)
            best_dict['rmse'] = root_mean_squared_error(labels_for_rul, flatten_for_rul)
            best_dict['mape'] = mean_absolute_percentage_error([l + 1 for l in labels_for_rul], [p + 1 for p in flatten_for_rul])
            best_dict['mdape'] = mdape([l + 1 for l in labels_for_rul], [p + 1 for p in flatten_for_rul])
            mape_bins, mdape_bins = mape_mdape_bins(flatten_for_rul, labels_for_rul, n=10)
        else:
            mape_bins, mdape_bins = None, None

        times = np.unique([ty for ty in flatten_labels])
        times.sort()
        test_y = [(rtf, ty) for ty, rtf in zip(flatten_labels, is_rtf)]

        eval_survs, max_rul = self._surv_eval_for_rul(
            experiment, test_y, flatten_preds, result_scores, result_labels, is_failure=rtfs, times=times, train_y=None
        )
        if mape_bins is not None:
            eval_survs['mape_bins'] = mape_bins
            eval_survs['mdape_bins'] = mdape_bins

        for key in eval_survs:
            if key in ['brier_scores', 'roc_auc_list', 'c_index_list', "mape_bins", "mdape_bins", "IBR_bins"]:
                mlflow.log_table(eval_survs[key], f"survival_{key}.json")
            else:
                best_dict[key] = eval_survs[key]

        # Delegate AD metric evaluation
        ad_evaluator = DefaultClassificationEvaluator()
        transformed_scores = [[max_rul - rulpred for rulpred in episode_scores] for episode_scores in result_scores]
        
        ad_kwargs = {
            'result_scores': transformed_scores,
            'result_dates': result_dates,
            'results_isfailure': rtfs,
            'plot_dictionary': plot_dictionary
        }
        best_dict_to_log = ad_evaluator.evaluate(experiment, **ad_kwargs)
        
        for key in best_dict_to_log:
            best_dict[key] = best_dict_to_log[key]

        mlflow.log_metrics({k: v for k, v in best_dict.items() if isinstance(v, (int, float, np.number))})

        return best_dict

    def _surv_eval_for_rul(self, experiment, test_y, flatten_preds, result_scores, result_labels, is_failure, times=None, train_y=None):
        new_test_y = []
        new_flatten_preds = []
        for ty, preds_i in zip(test_y, flatten_preds):
            new_test_y.append(ty)
            new_flatten_preds.append(preds_i)
        test_y = np.array(new_test_y, dtype=[('event', 'bool'), ('time', 'float')])
        flatten_preds = np.array(new_flatten_preds)

        if train_y is None:
            train_y = np.array(test_y, dtype=[('event', 'bool'), ('time', 'float')])

        if times is None:
            times = np.unique([ty[1] for ty in train_y])
            times.sort()

        maxtt = max([ty[1] for ty in test_y])
        mintt = min([ty[1] for ty in test_y])
        pos = 0
        for i, t in enumerate(times):
            if t >= maxtt:
                break
            pos = i
        pos_pre = 0
        for i, t in enumerate(times):
            if t >= mintt:
                pos_pre = i
                break

        times = times[pos_pre:pos]

        evals = {}
        events = [ty[0] for ty in test_y]
        times_for_c = [ty[1] for ty in test_y]
        max_rul = max(flatten_preds)
        res = concordance_index_censored(events, times_for_c, [max_rul - pred_rul for pred_rul in flatten_preds])
        evals['c_index'] = res[0]

        test_preds_hard = hard_transform_survival(times, flatten_preds)

        if self.debug and hasattr(experiment, 'plot_SA_of_RUL'):
            plot_test_preds = []
            for pred_set in result_scores:
                plot_test_preds.append([[predss, times] for predss in sigmoid_survival_batch(times, pred_set, tau=10)])
            experiment.plot_SA_of_RUL(plot_test_preds, result_labels, is_failure)

        b_times, b_score = brier_score(train_y, test_y, test_preds_hard, times)
        evals['brier_scores'] = {"time": [bt for bt in b_times], "Brier": [br for br in b_score]}
        evals['Max_brier_HM'] = np.max(b_score)

        roc_pt, mean_roc = cumulative_dynamic_auc(train_y, test_y, -test_preds_hard, times)
        evals['roc_auc_list'] = {"time": [bt for bt in times], "ROC": [br for br in roc_pt]}
        evals['mean_roc'] = mean_roc

        integrated_brier_score_value = integrated_brier_score(train_y, test_y, test_preds_hard, times)
        evals['IBS_HM'] = integrated_brier_score_value

        test_preds_sig = sigmoid_survival_batch(times, flatten_preds, tau=10)
        integrated_brier_score_value = integrated_brier_score(train_y, test_y, test_preds_sig, times)
        evals['IBS'] = integrated_brier_score_value
        b_times, b_score = brier_score(train_y, test_y, test_preds_sig, times)
        evals['Max_brier'] = np.max(b_score)

        return evals, max_rul
