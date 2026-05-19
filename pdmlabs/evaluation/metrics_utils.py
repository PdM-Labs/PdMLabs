import math
import numpy as np

def root_mean_squared_error(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def mdape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ape = np.abs((y_true - y_pred) / y_true)
    return np.median(ape)

def from_time_to_bins(labels, n=10):
    bin_size = math.ceil(len(labels) / n)
    sorted_labels = sorted(labels)
    bins = []
    for i in range(n):
        bins.append(sorted_labels[min((i + 1) * bin_size - 1, len(sorted_labels) - 1)])
    return bins

def mape_mdape_bins(preds, labels, n=10):
    bins = from_time_to_bins(labels, n)
    bin_dict = {}
    for b in bins:
        bin_dict[b] = {'preds': [], 'labels': []}
    for pred, label in zip(preds, labels):
        for b in bins:
            if label <= b:
                bin_dict[b]['preds'].append(pred)
                bin_dict[b]['labels'].append(label)
                break
    mape_per_bin = []
    mdape_per_bin = []
    for b in bins:
        if len(bin_dict[b]['labels']) == 0:
            continue
        current_mape = mean_absolute_percentage_error([l + 1 for l in bin_dict[b]['labels']],
                                                      [p + 1 for p in bin_dict[b]['preds']])
        current_mdape = mdape([l + 1 for l in bin_dict[b]['labels']],
                                   [p + 1 for p in bin_dict[b]['preds']])
        mape_per_bin.append((b, current_mape))
        mdape_per_bin.append((b, current_mdape))
    mape_per_bin = {"time": [bt[0] for bt in mape_per_bin], "MAPE": [br[1] for br in mape_per_bin]}
    mdape_per_bin = {"time": [bt[0] for bt in mdape_per_bin], "MdAPE": [br[1] for br in mdape_per_bin]}
    return mape_per_bin, mdape_per_bin

def IBR_bins(train_y, test_y, test_preds, times, n=10):
    from sksurv.metrics import integrated_brier_score
    bins = from_time_to_bins([ty[1] for ty in test_y], n)
    bin_dict = {}
    for b in bins:
        bin_dict[b] = {'train_y': [], 'test_y': [], 'test_preds': []}
    for ty, preds_i in zip(test_y, test_preds):
        pre_bin = 0
        for b in bins:
            if ty[1] <= b:
                bin_dict[b]['test_y'].append(ty)
                bin_dict[b]['test_preds'].append(
                    [pred for ts, pred in zip(times, preds_i) if ts < b and ts > pre_bin])
                break
            pre_bin = b
    for ty in train_y:
        for b in bins:
            if ty[1] <= b:
                bin_dict[b]['train_y'].append(ty)
                break
    ibr_per_bin = []
    pre_bin = 0
    for b in bins:
        if len(bin_dict[b]['test_y']) == 0:
            continue
        current_ibr = integrated_brier_score(
            np.array(bin_dict[b]['train_y'], dtype=[('event', 'bool'), ('time', 'float')]),
            np.array(bin_dict[b]['test_y'], dtype=[('event', 'bool'), ('time', 'float')]),
            np.array(bin_dict[b]['test_preds']),
            [tt for tt in times if tt < b and tt > pre_bin])
        pre_bin = b
        ibr_per_bin.append((b, current_ibr))
    ibr_per_bin = {"time": [bt[0] for bt in ibr_per_bin], "IBR": [br[1] for br in ibr_per_bin]}
    return ibr_per_bin
