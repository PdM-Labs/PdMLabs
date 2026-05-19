import collections
import traceback

import numpy as np
import sklearn
# from prts import ts_recall
import pandas as pd
import matplotlib.pyplot as plt
import math
import subprocess
import os
import uuid

from sklearn.preprocessing import MinMaxScaler


# from statsmodels.graphics.tukeyplot import results
from tqdm import tqdm
from pdmlabs.evaluation.vus.metrics import get_metrics

def calculate_ts_recall(anomalyranges, predictionsforrecall):
    generated_uuid = uuid.uuid4()
    anomaly_ranges_file_name = f'anomaly_ranges{generated_uuid}.real'
    anomaly_ranges_df = pd.DataFrame(anomalyranges, columns=['Numbers'])
    anomaly_ranges_df.to_csv(anomaly_ranges_file_name, index=False, header=False)

    prediction_file_name = f'anomaly_ranges{generated_uuid}.pred'
    predictions_df = pd.DataFrame(list(map(lambda x: 1 if x else 0, predictionsforrecall)), columns=['Numbers'])
    predictions_df.to_csv(prediction_file_name, index=False, header=False)
    from pdmlabs.evaluation import anomaly_evaluator

    res = anomaly_evaluator.evaluate(
        anomaly_ranges_file_name,
        prediction_file_name,
        beta=1.0,
        alpha_r=1.0,
        cardinality="one",
        bias_p="flat",
        bias_r="flat"
    )
    AD1 = res['Recall']
    res = anomaly_evaluator.evaluate(
        anomaly_ranges_file_name,
        prediction_file_name,
        beta=1.0,
        alpha_r=0,
        cardinality="one",
        bias_p="flat",
        bias_r="flat"
    )
    AD2 = res['Recall']
    res = anomaly_evaluator.evaluate(
        anomaly_ranges_file_name,
        prediction_file_name,
        beta=1.0,
        alpha_r=1.0,
        cardinality="one",
        bias_p="back",
        bias_r="back"
    )
    AD3 = res['Recall']

    os.remove(anomaly_ranges_file_name)
    os.remove(prediction_file_name)

    return AD1, AD2, AD3



def myeval(predictions, threshold, datesofscores=[], maintenances=None, isfailure=[], PH="100", lead="20",
           plotThem=True, ignoredates=[], beta=1):
    """
    This method is used to perform PdM evaluation of Run-to-Failures examples.

    Parameters
    ----------

    predictions: Either a flatted list of all predictions from all episodes or
                  list with a list of prediction for each of episodes
    datesofscores: Either a flatted list of all indexes (timestamps) from all episodes or
                   list with a list of  indexes (timestamps) for each of episodes
                   If it is empty list then aritificially indexes are gereated
    threshold: can be either a list of thresholds (equal size to all predictions), a list with size equal to number of episodes, a single number.
    maintenances: is used in case the predictions are passed as flatten array (default None)
      list of ints which indicate the time of maintenance (the position in predictions where a new episode begins) or the end of the episode.
    isfailure: a binary array which is used in case we want to pass episode which end with no failure, and thus don't contribute
      to recall calculation. For example isfailure=[1,1,0,1] indicates that the third episode end with no failure, while the others end with a failure.
      default value is empty list which indicate that all episodes end with failure.
    PH: is the predictive horizon used for recall, can be set in time domain using one from accepted time spans:
      ["days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"] using a space after the number e.g. PH="8 hours"
      in case of single number then a counter related predictive horizon is used (e.g. PH="100" indicates that the last 100 values
      are used for predictive horizon
    lead: represent the lead time (the time to ignore in last part of episode when we want to test the predict capabilities of algorithm)
      same rules as the PH are applied.
    ignoredates: a list with tuples in form (begin,end) which indicate periods to be ignored in the calculation of recall and precision
      begin and end values must be same type with datesofscores instances (pd.datetime or int)
    beta is used to calculate fbeta score deafult beta=1.

    """

    # format data to be in appropriate form for the evaluation, and check if conditions are true
    predictions, threshold, datesofscores, maintenances, isfailure = formulatedataForEvaluation(predictions, threshold,
                                                                                                datesofscores,
                                                                                                isfailure, maintenances)

    # calculate PH and lead
    numbertime, timestyle, numbertimelead, timestylelead = calculatePHandLead(PH, lead)

    anomalyranges = [0 for i in range(maintenances[-1])]

    prelimit = 0
    totaltp, totalfp, totaltn, totalfn = 0, 0, 0, 0
    arraytoplotall, toplotbordersall = [], []
    counter = -1
    thtoreturn = []
    episodescounts = len(maintenances)
    axes = None
    if plotThem:
        fig, axes = plt.subplots(nrows=math.ceil(episodescounts / 4), ncols=min(4, len(maintenances)), figsize=(28, 16))
        if math.ceil(episodescounts / 4) == 1:
            emptylist = []
            emptylist.append(axes)
            axes = emptylist
        if min(4, len(maintenances)) == 1:
            emptylist = []
            emptylist.append(axes)
            axes = emptylist
    predictionsforrecall = []
    countaxes = 0

    for maint in maintenances:

        counter += 1
        if isfailure[counter] == 1:

            episodePred = predictions[prelimit:maint]
            episodethreshold = threshold[prelimit:maint]

            episodealarms = [v > th for v, th in zip(episodePred, episodethreshold)]
            predictionsforrecall.extend([v > th for v, th in zip(episodePred, episodethreshold)])
            tempdatesofscores = datesofscores[prelimit:maint]

            tp, fp, borderph, border1 = episodeMyPresicionTPFP(episodealarms, tempdatesofscores,
                                                               PredictiveHorizon=numbertime, leadtime=numbertimelead,
                                                               timestylelead=timestylelead, timestyle=timestyle,
                                                               ignoredates=ignoredates)
            if counter > 0:
                for i in range(borderph + maintenances[counter - 1], border1 + maintenances[counter - 1]):
                    anomalyranges[i] = 1
            else:
                for i in range(borderph, border1):
                    anomalyranges[i] = 1
            totaltp += tp
            totalfp += fp
            prelimit = maint

            if plotThem:
                countaxes = plotforevalurion(timestyle, ignoredates, tempdatesofscores, episodethreshold, episodePred,
                                             countaxes,
                                             axes, borderph, border1)
        else:

            episodePred = predictions[prelimit:maint]
            episodethreshold = threshold[prelimit:maint]

            predictionsforrecall.extend([v > th for v, th in zip(episodePred, episodethreshold)])

            tempdatesofscores = datesofscores[prelimit:maint]
            if timestyle != "":
                for score, th, datevalue in zip(episodePred, episodethreshold, tempdatesofscores):
                    if ignore(datevalue, ignoredates):
                        if score > th:
                            totalfp += 1
            else:
                for score, th, datevalue in zip(episodePred, episodethreshold, tempdatesofscores):
                    if ingnorecounter(datevalue, ignoredates):
                        if score > th:
                            totalfp += 1
            if plotThem:
                countaxes = plotforevaluationNonFailure(timestyle, ignoredates, tempdatesofscores, episodethreshold,
                                                        episodePred, countaxes,
                                                        axes)

            prelimit = maint
        # print(f"Counter {counter} predslen: {len(predictionsforrecall)}, anomalyrangeslen:{len(anomalyranges)}" )

    ### Calculate AD levels
    if sum(predictionsforrecall) == 0:
        AD1 = 0
        AD2 = 0
        AD3 = 0
    else:
        if sum(predictionsforrecall) == 0:
            AD1 = 0
            AD2 = 0
            AD3 = 0
        else:
            AD1, AD2, AD3 = calculate_ts_recall(anomalyranges, predictionsforrecall)

            AD3 = AD3 * AD2

    ### Calculate Precision
    Precision = 0
    if totaltp + totalfp != 0:
        Precision = totaltp / (totaltp + totalfp)
    recall = [AD1, AD2, AD3]

    ### F ad scores
    f1 = []
    for rec in recall:
        if Precision + rec == 0:
            f1.append(0)
        else:
            F = ((1 + beta ** 2) * Precision * rec) / (beta ** 2 * Precision + rec)
            f1.append(F)
    return recall, Precision, f1, axes, anomalyranges


def Episodes_AUCPR(dictresults, ids=None, plotThem=False, PH="14 days", lead="1 hours", beta=2, phmapping=None):
    if phmapping is not None:

        PH = [(tup[0], tup[1]) for tup in phmapping]
        lead = [(tup[0], tup[2]) for tup in phmapping]
        if len([tup3 for tup3 in phmapping if tup3[0] == "non failure"]) < 1:
            PH.append(("non failure", "0"))
            lead.append(("non failure", "0"))
    Results = []
    if ids is None:
        ids = dictresults.keys()
    for keyd in ids:
        if isinstance(PH, str):
            AUCPR(dictresults[keyd]["episodescores"], datesofscores=dictresults[keyd]["episodesdates"], PH=PH,
                  lead=lead, isfailure=dictresults[keyd]["isfailure"],
                  plotThem=plotThem, beta=beta)

        else:
            AUCPR(dictresults[keyd]["episodescores"], Failuretype=dictresults[keyd]["failuretypes"],
                  datesofscores=dictresults[keyd]["episodesdates"],
                  PH=PH,
                  lead=lead,
                  isfailure=dictresults[keyd]["isfailure"],
                  plotThem=plotThem, beta=beta)

        if plotThem:
            plt.show()
    return Results


def AUCPR(predictions, Failuretype=None, datesofscores=[], maintenances=None, isfailure=[], PH="100", lead="20",
          plotThem=True, ignoredates=[], beta=1, resolution=100, slidingWindow_vus=0):
    predtemp = []
    if isinstance(predictions[0], collections.abc.Sequence):
        for predcs in predictions:
            predtemp.extend(predcs)
        flatened_scores = predtemp.copy()
        predtemp = list(set(predtemp))
    else:
        flatened_scores = predtemp.copy()
        predtemp = list(set(predictions))
    predtemp.sort()
    unique = list(set(predtemp))
    unique.sort()
    resolution = min(resolution, max(1, len(unique)))
    step = int(len(unique) / resolution)

    anomalyranges_for_vus = None

    tups_R_P1 = []
    tups_R_P2 = []
    tups_R_P3 = []
    allresults = []

    for i in range(resolution + 2):
        examined_th = unique[min(i * step, len(unique) - 1)]
        threshold = [examined_th for predcsss in predictions]
        if i == 0:
            # plt.subplot(121)
            ppplothem = False
        else:
            ppplothem = False

        if Failuretype is None:
            recall, Precision, f1, axes, anomalyranges = myeval(predictions, threshold, datesofscores=datesofscores,
                                                                maintenances=maintenances, isfailure=isfailure, PH=PH,
                                                                lead=lead, plotThem=ppplothem, ignoredates=ignoredates,
                                                                beta=beta)
            anomalyranges_for_vus = anomalyranges
            if Precision < 0.0000000000000001 and recall[0] < 0.0000000000000000001:
                continue
        else:
            recall, Precision, f1, axes, anomalyranges = myeval_multiPH(predictions,
                                                                        Failuretype,
                                                                        threshold,
                                                                        datesofscores=datesofscores,
                                                                        PH=PH,
                                                                        lead=lead,
                                                                        isfailure=isfailure,
                                                                        plotThem=ppplothem, beta=beta)
            anomalyranges_for_vus = anomalyranges
            # if Precision<0.0000000000000001 and recall[0]<0.0000000000000000001:
            #    continue
        tups_R_P1.append((recall[0], Precision))
        tups_R_P2.append((recall[1], Precision))
        tups_R_P3.append((recall[2], Precision))
        # All results
        allresults.append([f1[0], f1[1], f1[2], recall[0], recall[1], recall[2], Precision, examined_th])
    # allresults.append([0,0,0,1,1,1,0,min(unique)])
    allresults.append([0, 0, 0, 0, 0, 0, 1, max(unique)])
    allresultsforbestthreshold = allresults.copy()
    allresultsforbestthreshold.sort(key=lambda tup: tup[0], reverse=False)
    best_th = allresultsforbestthreshold[-1][-1]
    tups_R_P1 = sorted(tups_R_P1, key=lambda x: (x[0], -x[1]))
    # tups_R_P1.sort(key=lambda tup: tup[0], reverse=False)  # sorts in place
    tups_R_P2 = sorted(tups_R_P2, key=lambda x: (x[0], -x[1]))
    # tups_R_P2.sort(key=lambda tup: tup[0], reverse=False)  # sorts in place
    tups_R_P3 = sorted(tups_R_P3, key=lambda x: (x[0], -x[1]))
    # tups_R_P3.sort(key=lambda tup: tup[0], reverse=False)  # sorts in place

    recalls1 = [0] + [tup[0] for tup in tups_R_P1]
    recalls2 = [0] + [tup[0] for tup in tups_R_P2]
    recalls3 = [0] + [tup[0] for tup in tups_R_P3]

    press1 = [1] + [tup[1] for tup in tups_R_P1]
    press2 = [1] + [tup[1] for tup in tups_R_P2]
    press3 = [1] + [tup[1] for tup in tups_R_P3]

    # plt.subplot(121)
    # figtoplot=plt.figure(figsize=(28, 16))
    # ax = figtoplot.add_subplot()
    # ax.plot(recalls1,press1,"-o")
    # plt.plot(recalls2,press2)
    # plt.plot(press3,recalls3)

    if len(recalls1) == 1 or len(press1) == 1:
        AUC1 = 0.0
    else:
        AUC1 = sklearn.metrics.auc(recalls1, press1)

    if len(recalls2) == 1 or len(press2) == 1:
        AUC2 = 0.0
    else:
        AUC2 = sklearn.metrics.auc(recalls2, press2)

    if len(recalls3) == 1 or len(press3) == 1:
        AUC3 = 0.0
    else:
        AUC3 = sklearn.metrics.auc(recalls3, press3)

    for i in range(len(allresults)):
        allresults[i].append(AUC1)
        allresults[i].append(AUC2)
        allresults[i].append(AUC3)

    #### VUS RESULTS
    flatened_scores = np.array(flatened_scores)
    anomalyranges_for_vus = np.array(anomalyranges_for_vus)
    scaler = MinMaxScaler(feature_range=(0, 1))
    score = scaler.fit_transform(flatened_scores.reshape(-1, 1)).ravel()
    # results = get_metrics(score, anomalyranges_for_vus, best_threshold_examined=scaler.transform(np.array([[best_th]])).ravel()[0], slidingWindow=slidingWindow_vus)  # default metric='all'
    results = {}
    return allresults, results


def integrate(x, y):
    sm = 0
    for i in range(1, len(x)):
        h = x[i] - x[i - 1]
        sm += h * (y[i - 1] + y[i]) / 2

    return sm


def plotforevaluationNonFailure(timestyle, ignoredates, tempdatesofscores, episodethreshold, episodePred, countaxes,
                                axes):
    if timestyle == "":
        for i in range(len(ignoredates)):
            if ignoredates[i][1] > tempdatesofscores[0] and ignoredates[i][1] < tempdatesofscores[-1]:
                pos1 = -1
                pos2 = -1
                for q in range(len(tempdatesofscores)):
                    if tempdatesofscores[q] > ignoredates[i][0]:
                        pos1 = q
                        break
                for q in range(pos1, len(tempdatesofscores)):
                    if tempdatesofscores[q] > ignoredates[i][1]:
                        pos2 = q
                        break
                # print(
                #     f"in episode {ignoredates[0].tz_localize(None)}-{ignoredates[-1].tz_localize(None)} we have {ignoredates[pos1]}-{ignoredates[pos2]} ")
                # ignoreperiod.extend(tempdatesofscores[pos1:pos2])
                axes[countaxes // 4][countaxes % 4].fill_between(
                    tempdatesofscores[pos1:pos2], max(max(episodethreshold), max(episodePred)),
                    min(episodePred),
                    color="grey",
                    alpha=0.3,
                    label="ignore")
    else:
        for i in range(len(ignoredates)):
            if ignoredates[i][1] > tempdatesofscores[0].tz_localize(None) and ignoredates[i][1] < \
                    tempdatesofscores[-1].tz_localize(None):
                pos1 = -1
                pos2 = -1
                for q in range(len(tempdatesofscores)):
                    if tempdatesofscores[q].tz_localize(None) > ignoredates[i][0]:
                        pos1 = q
                        break
                for q in range(pos1, len(tempdatesofscores)):
                    if tempdatesofscores[q].tz_localize(None) > ignoredates[i][1]:
                        pos2 = q
                        break
                # print(
                #     f"in episode {ignoredates[0].tz_localize(None)}-{ignoredates[-1].tz_localize(None)} we have {ignoredates[pos1]}-{ignoredates[pos2]} ")
                # ignoreperiod.extend(tempdatesofscores[pos1:pos2])
                axes[countaxes // 4][countaxes % 4].fill_between(
                    tempdatesofscores[pos1:pos2], max(max(episodethreshold), max(episodePred)),
                    min(episodePred),
                    color="grey",
                    alpha=0.3,
                    label="ignore")
    # ============================================================

    axes[countaxes // 4][countaxes % 4].plot(tempdatesofscores, episodePred, color="green", label="pb n")
    axes[countaxes // 4][countaxes % 4].plot(tempdatesofscores, episodethreshold, color="k", linestyle="--", label="th")
    # axes[countaxes//4][countaxes%4].legend()
    countaxes += 1
    return countaxes


def plotforevalurion(timestyle, ignoredates, tempdatesofscores, episodethreshold, episodePred, countaxes, axes,
                     borderph, border1):
    if timestyle == "":
        for i in range(len(ignoredates)):
            if ignoredates[i][1] > tempdatesofscores[0] and ignoredates[i][1] < tempdatesofscores[-1]:
                pos1 = -1
                pos2 = -1
                for q in range(len(tempdatesofscores)):
                    if tempdatesofscores[q] > ignoredates[i][0]:
                        pos1 = q
                        break
                for q in range(pos1, len(tempdatesofscores)):
                    if tempdatesofscores[q] > ignoredates[i][1]:
                        pos2 = q
                        break
                # print(
                #     f"in episode {ignoredates[0].tz_localize(None)}-{ignoredates[-1].tz_localize(None)} we have {ignoredates[pos1]}-{ignoredates[pos2]} ")
                # ignoreperiod.extend(tempdatesofscores[pos1:pos2])
                axes[countaxes // 4][countaxes % 4].fill_between(
                    tempdatesofscores[pos1:pos2], max(max(episodethreshold), max(episodePred)),
                    min(episodePred),
                    color="grey",
                    alpha=0.3,
                    label="ignore")
    else:
        for i in range(len(ignoredates)):
            if ignoredates[i][1] > tempdatesofscores[0].tz_localize(None) and ignoredates[i][1] < \
                    tempdatesofscores[-1].tz_localize(None):
                pos1 = -1
                pos2 = -1
                for q in range(len(tempdatesofscores)):
                    if tempdatesofscores[q].tz_localize(None) > ignoredates[i][0]:
                        pos1 = q
                        break
                for q in range(pos1, len(tempdatesofscores)):
                    if tempdatesofscores[q].tz_localize(None) > ignoredates[i][1]:
                        pos2 = q
                        break
                # print(
                #     f"in episode {ignoredates[0].tz_localize(None)}-{ignoredates[-1].tz_localize(None)} we have {ignoredates[pos1]}-{ignoredates[pos2]} ")
                # ignoreperiod.extend(tempdatesofscores[pos1:pos2])
                axes[countaxes // 4][countaxes % 4].fill_between(
                    tempdatesofscores[pos1:pos2], max(max(episodethreshold), max(episodePred)),
                    min(episodePred),
                    color="grey",
                    alpha=0.3,
                    label="ignore")

    # ============================================================

    axes[countaxes // 4][countaxes % 4].plot(tempdatesofscores, episodePred, label="pb")
    axes[countaxes // 4][countaxes % 4].fill_between([tempdatesofscores[i] for i in range(borderph, border1)],
                                                     max(max(episodethreshold), max(episodePred)),
                                                     min(episodePred), where=[1 for i in range(borderph, border1)],
                                                     color="red",
                                                     alpha=0.3,
                                                     label="PH")
    axes[countaxes // 4][countaxes % 4].fill_between([tempdatesofscores[i] for i in range(border1, len(episodePred))],
                                                     max(max(episodethreshold), max(episodePred)),
                                                     min(episodePred),
                                                     where=[1 for i in range(border1, len(episodePred))],
                                                     color="grey",
                                                     alpha=0.3,
                                                     label="ignore")

    axes[countaxes // 4][countaxes % 4].plot(tempdatesofscores, episodethreshold, color="k", linestyle="--", label="th")

    # axes[countaxes//4][countaxes%4].legend()

    countaxes += 1
    return countaxes


def episodeMyPresicionTPFP(episodealarms, tempdatesofscores, PredictiveHorizon, leadtime, timestyle, timestylelead,
                           ignoredates):
    border2 = len(episodealarms)
    totaltp = 0
    totalfp = 0

    arraytoplot = []
    toplotborders = []

    border2date = tempdatesofscores[border2 - 1]

    border1 = border2 - 1
    if timestyle != "":
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - pd.Timedelta(leadtime, timestylelead):
                border1 = i - 1
                if border1 == -1:
                    border1 = 0
                break
        borderph = border1 - 1
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - pd.Timedelta(PredictiveHorizon, timestyle):
                borderph = i - 1
                if borderph == -1:
                    borderph = 0
                break
        # print(f"len :{border2} , border 1: {border1}, borderph: {borderph}")
        positivepred = episodealarms[borderph:border1]
        negativepred = episodealarms[:borderph]
        negativepredDates = tempdatesofscores[:borderph]
        for value in positivepred:
            if value:
                totaltp += 1
        for value, valuedate in zip(negativepred, negativepredDates):
            if ignore(valuedate, tupleIngoredaes=ignoredates):
                if value:
                    totalfp += 1
        return totaltp, totalfp, borderph, border1

    else:
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - leadtime:
                border1 = i - 1
                break
        borderph = border1 - 1
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - PredictiveHorizon:
                borderph = i - 1
                break

        # print(f"len :{border2} , border 1: {border1}, borderph: {borderph}")
        positivepred = episodealarms[borderph:border1]
        negativepred = episodealarms[:borderph]
        negativepredDates = tempdatesofscores[:borderph]
        for value in positivepred:
            if value:
                totaltp += 1
        for value, valuedate in zip(negativepred, negativepredDates):
            if ingnorecounter(valuedate, tupleIngoredaes=ignoredates):
                if value:
                    totalfp += 1
        return totaltp, totalfp, borderph, border1


def formulatedataForEvaluation(predictions, threshold, datesofscores, isfailure, maintenances):
    artificialindexes = []
    thresholdtempperepisode = []
    if isinstance(predictions[0], collections.abc.Sequence):
        temppreds = []
        maintenances = []
        if isinstance(threshold, collections.abc.Sequence) and len(threshold) == len(predictions):
            for episodepreds, thepisode in zip(predictions, threshold):
                if isinstance(thepisode, collections.abc.Sequence):
                    thresholdtempperepisode.extend(thepisode)
                else:
                    thresholdtempperepisode.extend([thepisode for i in range(len(episodepreds))])
                temppreds.extend([pre for pre in episodepreds])
                artificialindexes.extend([i + len(artificialindexes) for i in range(len(episodepreds))])
                maintenances.append(len(temppreds))
        else:
            for episodepreds in predictions:
                temppreds.extend([pre for pre in episodepreds])
                artificialindexes.extend([i + len(artificialindexes) for i in range(len(episodepreds))])
                maintenances.append(len(temppreds))
        predictions = temppreds

    if len(datesofscores) == 0:
        datesofscores = artificialindexes
    elif isinstance(datesofscores[0], collections.abc.Sequence):
        temppreds = []
        for episodeindexesss in datesofscores:
            temppreds.extend(episodeindexesss)
        datesofscores = temppreds
    if maintenances is None:
        assert False, "When you pass a flatten array for predictions, maintenances must be assigned to cutoffs time/indexes"
    if maintenances[-1] != len(predictions):
        assert False, "The maintenance indexes are not alligned with predictions length (last index of predictions should be the last element of maintenances)"
    if len(predictions) != len(datesofscores):
        assert False, f"Inconsistency in the size of scores (predictions) and dates-indexes {len(predictions)} != {len(datesofscores)}"

    if len(isfailure) == 0:
        isfailure = [1 for m in maintenances]

    if isinstance(threshold, collections.abc.Sequence) and len(threshold) == len(maintenances):
        threshold = thresholdtempperepisode
    elif isinstance(threshold, collections.abc.Sequence) == False:
        temp = [threshold for i in predictions]
        threshold = temp

    assert len(predictions) == len(
        threshold), f"Inconsistency in the size of scores (predictions {len(predictions)}) and thresholds {len(threshold)}"

    return predictions, threshold, datesofscores, maintenances, isfailure


def calculatePHandLead(PH, lead):
    if len(str(PH).split(" ")) < 2:
        numbertime = float(str(PH).split(" ")[0])
        timestyle = ""
    else:
        scale = PH.split(" ")[1]
        acceptedvalues = ["", "days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"]
        if scale in acceptedvalues:
            numbertime = float(PH.split(" ")[0])

            timestyle = scale
        else:
            assert False, f"PH parameter must be in form \"number timescale\" e.g. \"8 hours\", where posible values for timescale are {acceptedvalues}"

    if len(str(lead).split(" ")) < 2:
        if timestyle != "":
            assert False, f"When PH passed with time style (e.g.minutes, days, ...) , lead must have time style as well."
        numbertimelead = float(str(lead).split(" ")[0])
        timestylelead = ""
    else:
        if timestyle == "":
            assert False, f"When PH passed without time style (e.g.minutes, days, ...) , lead must not have time style as well."
        scale = lead.split(" ")[1]
        acceptedvalues = ["", "days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"]
        if scale in acceptedvalues:
            numbertimelead = float(lead.split(" ")[0])

            timestylelead = scale
        else:
            assert False, f"lead parameter must be in form \"number timescale\" e.g. \"8 hours\", where posible values for timescale are {acceptedvalues}"

    return numbertime, timestyle, numbertimelead, timestylelead


def ignore(valuedate, tupleIngoredaes):
    for tup in tupleIngoredaes:
        if valuedate.tz_localize(None) > tup[0] and valuedate.tz_localize(None) < tup[1]:
            return False
    return True


def ingnorecounter(valuedate, tupleIngoredaes):
    for tup in tupleIngoredaes:
        if valuedate > tup[0] and valuedate < tup[1]:
            return False
    return True



def myeval_multiPH(predictions, Failuretype, threshold, datesofscores=[], maintenances=None, isfailure=[],
                   PH=[("type 1", "100")], lead=[("type 1", "10")], plotThem=True, ignoredates=[], beta=1):
    """
    This method is used to perform PdM evaluation of Run-to-Failures examples.

    Parameters
    ----------

    predictions: Either a flatted list of all predictions from all episodes or
                  list with a list of prediction for each of episodes
    datesofscores: Either a flatted list of all indexes (timestamps) from all episodes or
                   list with a list of  indexes (timestamps) for each of episodes
                   If it is empty list then aritificially indexes are gereated
    threshold: can be either a list of thresholds (equal size to all predictions), a list with size equal to number of episodes, a single number.
    maintenances: is used in case the predictions are passed as flatten array (default None)
      list of ints which indicate the time of maintenance (the position in predictions where a new episode begins) or the end of the episode.
    isfailure: a binary array which is used in case we want to pass episode which end with no failure, and thus don't contribute
      to recall calculation. For example isfailure=[1,1,0,1] indicates that the third episode end with no failure, while the others end with a failure.
      default value is empty list which indicate that all episodes end with failure.
    PH: is the predictive horizon used for recall, can be set in time domain using one from accepted time spans:
      ["days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"] using a space after the number e.g. PH="8 hours"
      in case of single number then a counter related predictive horizon is used (e.g. PH="100" indicates that the last 100 values
      are used for predictive horizon
    lead: represent the lead time (the time to ignore in last part of episode when we want to test the predict capabilities of algorithm)
      same rules as the PH are applied.
    ignoredates: a list with tuples in form (begin,end) which indicate periods to be ignored in the calculation of recall and precision
      begin and end values must be same type with datesofscores instances (pd.datetime or int)
    beta is used to calculate fbeta score deafult beta=1.
    """

    # format data to be in appropriate form for the evaluation, and check if conditions are true
    predictions, threshold, datesofscores, maintenances, isfailure = formulatedataForEvaluation(predictions, threshold,
                                                                                                datesofscores,
                                                                                                isfailure, maintenances)
    if len(maintenances) != len(Failuretype):
        assert False, "when using eval_multiPH the type of failure/maintenance, for each maintenance is required"
    if isinstance(PH, collections.abc.Sequence) == False:
        assert False, "when using eval_multiPH PH and lead parameter must be a list of tuples of form, (\"type name\",\"PH value\")"
    uniqueCodes = list(set(Failuretype))
    phcodes = [tupp[0] for tupp in PH]
    for cod in Failuretype:
        if cod not in phcodes:
            assert False, f"You must provide the ph for all different types in Failuretype, there are no info for {cod} in PH tuples"
    leadcodes = [tupp[0] for tupp in lead]
    for cod in Failuretype:
        if cod not in leadcodes:
            assert False, f"You must provide the lead for all different types in Failuretype, there are no info for {cod} in lead tuples"

    # calculate PH and lead
    PHS_leads = []
    for failuretype in Failuretype:
        posph = phcodes.index(failuretype)
        poslead = leadcodes.index(failuretype)
        tuplead = lead[poslead]
        tupPH = PH[posph]
        numbertime, timestyle, numbertimelead, timestylelead = calculatePHandLead(tupPH[1], tuplead[1])
        PHS_leads.append((failuretype, numbertime, timestyle, numbertimelead, timestylelead))

    anomalyranges = [0 for i in range(maintenances[-1])]

    prelimit = 0
    totaltp, totalfp, totaltn, totalfn = 0, 0, 0, 0
    arraytoplotall, toplotbordersall = [], []
    counter = -1
    thtoreturn = []
    episodescounts = len(maintenances)
    axes = None
    if plotThem:
        fig, axes = plt.subplots(nrows=math.ceil(episodescounts / 4), ncols=min(4, len(maintenances)), figsize=(28, 16))
        if math.ceil(episodescounts / 4) == 1:
            emptylist = []
            emptylist.append(axes)
            axes = emptylist
        if min(4, len(maintenances)) == 1:
            emptylist = []
            emptylist.append(axes)
            axes = emptylist
    predictionsforrecall = []
    countaxes = 0

    for maint, tupPHLEAD in zip(maintenances, PHS_leads):

        counter += 1
        if isfailure[counter] == 1:

            episodePred = predictions[prelimit:maint]
            episodethreshold = threshold[prelimit:maint]

            episodealarms = [v > th for v, th in zip(episodePred, episodethreshold)]
            predictionsforrecall.extend([v > th for v, th in zip(episodePred, episodethreshold)])
            tempdatesofscores = datesofscores[prelimit:maint]

            tp, fp, borderph, border1 = episodeMyPresicionTPFP(episodealarms, tempdatesofscores,
                                                               PredictiveHorizon=tupPHLEAD[1], leadtime=tupPHLEAD[3],
                                                               timestylelead=tupPHLEAD[4], timestyle=tupPHLEAD[2],
                                                               ignoredates=ignoredates)
            if counter > 0:
                for i in range(borderph + maintenances[counter - 1], border1 + maintenances[counter - 1]):
                    anomalyranges[i] = 1
            else:
                for i in range(borderph, border1):
                    anomalyranges[i] = 1
            totaltp += tp
            totalfp += fp
            prelimit = maint

            if plotThem:
                countaxes = plotforevalurion(timestyle, ignoredates, tempdatesofscores, episodethreshold, episodePred,
                                             countaxes,
                                             axes, borderph, border1)
        else:

            episodePred = predictions[prelimit:maint]
            episodethreshold = threshold[prelimit:maint]

            predictionsforrecall.extend([v > th for v, th in zip(episodePred, episodethreshold)])

            tempdatesofscores = datesofscores[prelimit:maint]
            if timestyle != "":
                for score, th, datevalue in zip(episodePred, episodethreshold, tempdatesofscores):
                    if ignore(datevalue, ignoredates):
                        if score > th:
                            totalfp += 1
            else:
                for score, th, datevalue in zip(episodePred, episodethreshold, tempdatesofscores):
                    if ingnorecounter(datevalue, ignoredates):
                        if score > th:
                            totalfp += 1
            if plotThem:
                countaxes = plotforevaluationNonFailure(timestyle, ignoredates, tempdatesofscores, episodethreshold,
                                                        episodePred, countaxes,
                                                        axes)

            prelimit = maint
        # print(f"Counter {counter} predslen: {len(predictionsforrecall)}, anomalyrangeslen:{len(anomalyranges)}" )

    ### Calculate AD levels
    if sum(predictionsforrecall) == 0:
        AD1 = 0
        AD2 = 0
        AD3 = 0
    else:
        AD1, AD2, AD3 = calculate_ts_recall(anomalyranges, predictionsforrecall)

        AD3 = AD3 * AD2

    ### Calculate Precision
    Precision = 0
    if totaltp + totalfp != 0:
        Precision = totaltp / (totaltp + totalfp)
    recall = [AD1, AD2, AD3]

    ### F ad scores
    f1 = []
    for rec in recall:
        if Precision + rec == 0:
            f1.append(0)
        else:
            F = ((1 + beta ** 2) * Precision * rec) / (beta ** 2 * Precision + rec)
            f1.append(F)
    return recall, Precision, f1, axes, anomalyranges


def breakIntoEpisodes(alarms, failuredates, thresholds=None, dates=None):
    isfailure = []
    episodes = []
    episodesthreshold = []
    episodesdates = []
    # dates=[pd.to_datetime(datedd) for datedd in dates]
    # failuredates=[pd.to_datetime(datedd) for datedd in failuredates]

    failuredates = [fdate for fdate in failuredates if fdate > dates[0]]

    # no failures
    if len(failuredates) == 0 or len(dates) == 0:
        if len(alarms) > 0:
            isfailure.append(0)
            episodes.append(alarms)
            if thresholds is not None:
                episodesthreshold.append(thresholds)
            episodesdates.append(dates)
        return isfailure, episodes, episodesdates, episodesthreshold

    failuredates = [fdate for fdate in failuredates if fdate > dates[0]]

    counter = 0
    for fdate in failuredates:
        for i in range(counter, len(dates)):
            if dates[i] > fdate:
                if len(alarms[counter:i]) > 0:
                    isfailure.append(1)
                    episodes.append(alarms[counter:i])
                    if thresholds is not None:
                        episodesthreshold.append(thresholds[counter:i])
                    episodesdates.append(dates[counter:i])
                counter = i
                break
    if  dates[-1] <= failuredates[-1]:
        isfailure.append(1)
        episodes.append(alarms[counter:])
        if thresholds is not None:
            episodesthreshold.append(thresholds[counter:])
        episodesdates.append(dates[counter:])
    elif counter < len(alarms):
        if len(alarms[counter:]) > 0:
            isfailure.append(0)
            episodes.append(alarms[counter:])
            if thresholds is not None:
                episodesthreshold.append(thresholds[counter:])
            episodesdates.append(dates[counter:])
    return isfailure, episodes, episodesdates, episodesthreshold


def breakIntoEpisodesWithCodes(alarms, failuredates, failurecodes, thresholds, dates):
    isfailure = []
    failuretype = []
    episodes = []
    episodesthreshold = []
    episodesdates = []
    # dates=[pd.to_datetime(datedd) for datedd in dates]
    # failuredates=[pd.to_datetime(datedd) for datedd in failuredates]
    failuredates = [fdate for fdate in failuredates if fdate > dates[0]]

    # no failures
    if len(failuredates) == 0 or len(dates) == 0:
        if len(alarms) > 0:
            isfailure.append(0)
            failuretype.append("non failure")
            episodes.append(alarms)
            episodesthreshold.append(thresholds)
            episodesdates.append(dates)
        return isfailure, episodes, episodesdates, episodesthreshold, failuretype

    counter = 0
    for fdate, ftype in zip(failuredates, failurecodes):
        for i in range(counter, len(dates)):
            if dates[i] > fdate:
                if len(alarms[counter:i]) > 0:
                    isfailure.append(1)
                    failuretype.append(ftype)
                    episodes.append(alarms[counter:i])
                    episodesthreshold.append(thresholds[counter:i])
                    episodesdates.append(dates[counter:i])
                counter = i
                break
    if dates[-1] < failuredates[-1]:
        isfailure.append(1)
        failuretype.append(failurecodes[-1])
        episodes.append(alarms[counter:])
        episodesthreshold.append(thresholds[counter:])
        episodesdates.append(dates[counter:])
    elif counter < len(alarms):
        if len(alarms[counter:]) > 0:
            isfailure.append(0)
            failuretype.append("non failure")
            episodes.append(alarms[counter:])
            episodesthreshold.append(thresholds[counter:])
            episodesdates.append(dates[counter:])
    return isfailure, episodes, episodesdates, episodesthreshold, failuretype



def Gather_Episodes(score, timescore, thresholds, failures=[], failurecodes=[], dictresults={}, id="id",
                    description=""):
    """
    Parameters
    ----------
    score: the produced score of a technique
    timescore: the correspoiding timestamps (pandas.datetime) for the produced score
    thresholds: a list with a threshold value for each score value.
    failures: dates of failures
    failurecodes: the type of failure
    dictresults: the dictionary we want to store the parameters for evaluation.
    id: the name under whihch the parameters will be stored
    description : a string for description of experiment
    """
    if len(failurecodes) > 0:
        isfailure, episodescores, episodesdates, episodesthresholds, failuretypes = breakIntoEpisodesWithCodes(
            score,
            failures, failurecodes,
            thresholds,
            timescore)
    else:
        isfailure, episodescores, episodesdates, episodesthresholds = breakIntoEpisodes(
            score,
            failures,
            thresholds,
            timescore)
        failuretypes = []

    if id not in dictresults.keys():
        dictresults[id] = {"isfailure": isfailure, "episodescores": episodescores,
                           "episodesdates": episodesdates, "episodesthresholds": episodesthresholds,
                           "failuretypes": failuretypes, "description": description}
    else:
        dictresults[id]["isfailure"].extend(isfailure)
        dictresults[id]["episodescores"].extend(episodescores)
        dictresults[id]["episodesdates"].extend(episodesdates)
        dictresults[id]["episodesthresholds"].extend(episodesthresholds)
        dictresults[id]["failuretypes"].extend(failuretypes)

    return dictresults, id



def evaluate_episodes(dictresults, ids=None, plotThem=False, PH="14 days", lead="1 hours", beta=2, phmapping=None):
    """
    Parameters
    ----------
    dictresults: is a dictionary produced from Gather_Episodes, containing a dictionary for each evaluation with keys:
          isfailure, episodescores,episodesdates,episodesthresholds,failuretypes,failuretypes
    ids: The ids  in form of list of the dictionary which is going to be evaluated
    PH: is the predictive horizon used for recall, can be set in time domain using one from accepted time spans:
      ["days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"] using a space after the number e.g. PH="8 hours"
      in case of single number then a counter related predictive horizon is used (e.g. PH="100" indicates that the last 100 values
      are used for predictive horizon
      It is possible to accept differenc PH of different failure codes.
    lead: similar to PH for lead time.
    beta: the beta value to calculate f-beta score
    plotThem: True in case we want to plot episodes.
    """

    if phmapping is not None:

        PH = [(tup[0], tup[1]) for tup in phmapping]
        lead = [(tup[0], tup[2]) for tup in phmapping]
        if len([tup3 for tup3 in phmapping if tup3[0] == "non failure"]) < 1:
            PH.append(("non failure", "0"))
            lead.append(("non failure", "0"))
    Results = []
    if ids is None:
        ids = dictresults.keys()
    for keyd in ids:
        if isinstance(PH, str):
            recall, Precision, fbeta, axes = myeval(dictresults[keyd]["episodescores"],
                                                    dictresults[keyd]["episodesthresholds"],
                                                    datesofscores=dictresults[keyd]["episodesdates"], PH=PH,
                                                    lead=lead, isfailure=dictresults[keyd]["isfailure"],
                                                    plotThem=plotThem, beta=beta)
            if plotThem:
                print(f"=======================================================")
                print(f" RESULTS FOR {keyd}:")
                print(f" description: {dictresults[keyd]['description']}:")
                print(f"F{beta}: AD1 {fbeta[0]},AD2 {fbeta[1]},AD3 {fbeta[2]}")
                print(f"Recall: AD1 {recall[0]},AD2 {recall[1]},AD3 {recall[2]}")
                print(f"Precission: {Precision}")
            resultdict = {f"F{beta}_AD1": fbeta[0], f"F{beta}_AD2": fbeta[1], f"F{beta}_AD3": fbeta[2],
                          "AD1": recall[0], "AD2": recall[1], "AD3": recall[2],
                          "Precission": Precision}
            Results.append(resultdict)
        else:
            recall, Precision, fbeta, axes = myeval_multiPH(dictresults[keyd]["episodescores"],
                                                            dictresults[keyd]["failuretypes"],
                                                            dictresults[keyd]["episodesthresholds"],
                                                            datesofscores=dictresults[keyd]["episodesdates"],
                                                            PH=PH,
                                                            lead=lead,
                                                            isfailure=dictresults[keyd]["isfailure"],
                                                            plotThem=plotThem, beta=beta)
            if plotThem:
                print(f"=======================================================")
                print(f" RESULTS FOR {keyd}:")
                print(f" description: {dictresults[keyd]['description']}:")
                print(f"F{beta}: AD1 {fbeta[0]},AD2 {fbeta[1]},AD3 {fbeta[2]}")
                print(f"Recall: AD1 {recall[0]},AD2 {recall[1]},AD3 {recall[2]}")
                print(f"Precission: {Precision}")
            resultdict = {f"F{beta}_AD1": fbeta[0], f"F{beta}_AD2": fbeta[1], f"F{beta}_AD3": fbeta[2],
                          "AD1": recall[0], "AD2": recall[1], "AD3": recall[2],
                          "Precission": Precision}
            Results.append(resultdict)
        if plotThem:
            plt.show()
    return Results


def extract_anomaly_ranges(maintenances, PHS_leads, isfailure, datesofscores):
    anomalyranges = [0 for i in range(maintenances[-1])]
    leadranges = [0 for i in range(maintenances[-1])]
    prelimit = 0
    counter = -1

    for maint, tupPHLEAD in zip(maintenances, PHS_leads):

        counter += 1
        if isfailure[counter] == 1:
            tempdatesofscores = datesofscores[prelimit:maint]
            borderph, border1, border_episode = Episode_Borders(tempdatesofscores, PredictiveHorizon=tupPHLEAD[1],
                                                                leadtime=tupPHLEAD[3],
                                                                timestylelead=tupPHLEAD[4], timestyle=tupPHLEAD[2])
            if counter > 0:
                for i in range(borderph + prelimit, border1 + prelimit):
                    anomalyranges[i] = 1
            else:
                for i in range(borderph, border1):
                    anomalyranges[i] = 1

            if counter > 0:
                for i in range(border1 + prelimit, border_episode + prelimit):
                    leadranges[i] = 1
            else:
                for i in range(border1, border_episode):
                    leadranges[i] = 1

        prelimit = maint

    return anomalyranges, leadranges


def Episode_Borders(tempdatesofscores, PredictiveHorizon, leadtime, timestyle, timestylelead):
    border2 = len(tempdatesofscores)
    border2date = tempdatesofscores[border2 - 1]
    border1 = border2 - 1
    if timestyle != "":
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - pd.Timedelta(leadtime, timestylelead):
                border1 = i - 1
                if border1 == -1:
                    border1 = 0
                break
        borderph = border1 - 1
        for i in range(len(tempdatesofscores)):
            if tempdatesofscores[i] > border2date - pd.Timedelta(PredictiveHorizon, timestyle):
                borderph = i - 1
                if borderph == -1:
                    borderph = 0
                break
        # print(f"len :{border2} , border 1: {border1}, borderph: {borderph}")
        return borderph, border1, border2
    else:
        for i in range(len(tempdatesofscores)):
            if i > border2 - leadtime:
                border1 = i - 1
                break
        borderph = border1 - 1
        for i in range(len(tempdatesofscores)):
            if i > border2 - PredictiveHorizon:
                borderph = i - 1
                break

        # print(f"len :{border2} , border 1: {border1}, borderph: {borderph}")
        return borderph, border1, border2


def _data_formulation(predictions, threshold, datesofscores, isfailure, maintenances, Failuretype, PH, lead):
    # format data to be in appropriate form for the evaluation, and check if conditions are true
    predictions, threshold, datesofscores, maintenances, isfailure = formulatedataForEvaluation(predictions, threshold,
                                                                                                datesofscores,
                                                                                                isfailure, maintenances)

    if Failuretype is None or Failuretype == [] or type(PH) is str:
        PH = [("type_all", PH)]
        lead = [("type_all", lead)]
        Failuretype = ["type_all" for i in maintenances]

    if len(maintenances) != len(Failuretype):
        assert False, "when using eval_multiPH the type of failure/maintenance, for each maintenance is required"
    if isinstance(PH, collections.abc.Sequence) == False:
        assert False, "when using eval_multiPH PH and lead parameter must be a list of tuples of form, (\"type name\",\"PH value\")"
    uniqueCodes = list(set(Failuretype))
    phcodes = [tupp[0] for tupp in PH]
    for cod in Failuretype:
        if cod not in phcodes:
            assert False, f"You must provide the ph for all different types in Failuretype, there are no info for {cod} in PH tuples"
    leadcodes = [tupp[0] for tupp in lead]
    for cod in Failuretype:
        if cod not in leadcodes:
            assert False, f"You must provide the lead for all different types in Failuretype, there are no info for {cod} in lead tuples"

    # calculate PH and lead
    PHS_leads = []
    for failuretype in Failuretype:
        posph = phcodes.index(failuretype)
        poslead = leadcodes.index(failuretype)
        tuplead = lead[poslead]
        tupPH = PH[posph]
        numbertime, timestyle, numbertimelead, timestylelead = calculatePHandLead(tupPH[1], tuplead[1])
        PHS_leads.append((failuretype, numbertime, timestyle, numbertimelead, timestylelead))
    return predictions, threshold, datesofscores, maintenances, isfailure, PHS_leads



def pdm_eval_multi_PH(predictions, threshold, datesofscores=[], maintenances=None, isfailure=[],
                      PH=[("type 1", "100")], lead=[("type 1", "10")], Failuretype=[], plotThem=True, ignoredates=[],
                      beta=1):
    """
    This method is used to perform PdM evaluation of Run-to-Failures examples.

    Parameters
    ----------
    predictions: Either a flatted list of all predictions from all episodes or
                  list with a list of prediction for each of episodes
    datesofscores: Either a flatted list of all indexes (timestamps) from all episodes or
                   list with a list of  indexes (timestamps) for each of episodes
                   If it is empty list then aritificially indexes are gereated
    threshold: can be either a list of thresholds (equal size to all predictions), a list with size equal to number of episodes, a single number.
    maintenances: is used in case the predictions are passed as flatten array (default None)
      list of ints which indicate the time of maintenance (the position in predictions where a new episode begins) or the end of the episode.
    isfailure: a binary array which is used in case we want to pass episode which end with no failure, and thus don't contribute
      to recall calculation. For example isfailure=[1,1,0,1] indicates that the third episode end with no failure, while the others end with a failure.
      default value is empty list which indicate that all episodes end with failure.
    PH: is the predictive horizon used for recall, can be set in time domain using one from accepted time spans:
      ["days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"] using a space after the number e.g. PH="8 hours"
      in case of single number then a counter related predictive horizon is used (e.g. PH="100" indicates that the last 100 values
      are used for predictive horizon
    lead: represent the lead time (the time to ignore in last part of episode when we want to test the predict capabilities of algorithm)
      same rules as the PH are applied.
    ignoredates: a list with tuples in form (begin,end) which indicate periods to be ignored in the calculation of recall and precision
      begin and end values must be same type with datesofscores instances (pd.datetime or int)
    beta is used to calculate fbeta score deafult beta=1.
    """
    predictions, threshold, datesofscores, maintenances, isfailure, PHS_leads = _data_formulation(predictions,
                                                                                                  threshold,
                                                                                                  datesofscores,
                                                                                                  isfailure,
                                                                                                  maintenances,
                                                                                                  Failuretype, PH, lead)

    anomalyranges, leadranges = extract_anomaly_ranges(maintenances, PHS_leads, isfailure, datesofscores)
    ignore_range = _ingore_dates_range_(datesofscores, ignoredates)

    recall, Precision, f1, anomalyranges = calculate_AD_levels(anomalyranges, leadranges, predictions, ignore_range,
                                                               threshold, beta)
    return recall, Precision, f1


def _ingore_dates_range_(datesofscores, ignoredates):
    ignore_range = [0 for i in range(len(datesofscores))]
    if len(ignoredates) == 0:
        return ignore_range
    for i, date in enumerate(datesofscores):
        if ignore(date, ignoredates) == False:
            ignore_range[i] = 1
    return ignore_range


def calculate_AD_levels(anomalyranges, leadranges, predictions, ignore_range, threshold, beta):
    totaltp = len([1 for an, pr, th, ld, ig in zip(anomalyranges, predictions, threshold, leadranges, ignore_range) if
                   an == 1 and pr > th and ld == 0 and ig == 0])
    totalfp = len([1 for an, pr, th, ld, ig in zip(anomalyranges, predictions, threshold, leadranges, ignore_range) if
                   an == 0 and pr > th and ld == 0 and ig == 0])
    predictionsinner = [1 if pr > th else 0 for pr, th in zip(predictions, threshold)]
    ### Calculate AD levels
    if sum(predictionsinner) == 0:
        AD1 = 0
        AD2 = 0
        AD3 = 0
    else:
        AD1, AD2, AD3 = calculate_ts_recall(anomalyranges, predictionsinner)

        AD3 = AD3 * AD2

    ### Calculate Precision
    Precision = 0
    if totaltp + totalfp != 0:
        Precision = totaltp / (totaltp + totalfp)
    recall = [AD1, AD2, AD3]

    ### F ad scores
    f1 = []
    for rec in recall:
        if Precision + rec == 0:
            f1.append(0)
        else:
            F = ((1 + beta ** 2) * Precision * rec) / (beta ** 2 * Precision + rec)
            f1.append(F)
    return recall, Precision, f1


def AUCPR_new(predictions, Failuretype=None, datesofscores=[], maintenances=None, isfailure=[], PH="100", lead="20",
              plotThem=True, ignoredates=[], beta=1, resolution=100, slidingWindow_vus=0):
    predtemp = []
    if isinstance(predictions[0], collections.abc.Sequence):
        for predcs in predictions:
            predtemp.extend(predcs)
        flatened_scores = predtemp.copy()
        predtemp = list(set(predtemp))
    else:
        predtemp.extend(predictions)
        flatened_scores = predtemp.copy()
        predtemp = list(set(predictions))
    predtemp.sort()
    unique = list(set(predtemp))
    unique.sort()
    resolution = min(resolution, max(1, len(unique)))
    step = int(len(unique) / resolution)

    threshold = [0 for qqq in predictions]

    predictions, threshold, datesofscores, maintenances, isfailure, PHS_leads = _data_formulation(predictions,
                                                                                                  threshold,
                                                                                                  datesofscores,
                                                                                                  isfailure,
                                                                                                  maintenances,
                                                                                                  Failuretype, PH, lead)

    anomalyranges, leadranges = extract_anomaly_ranges(maintenances, PHS_leads, isfailure, datesofscores)
    ignore_range = _ingore_dates_range_(datesofscores, ignoredates)

    tups_R_P1 = []
    tups_R_P2 = []
    tups_R_P3 = []
    allresults = []

    for i in range(resolution + 2):
        examined_th = unique[min(i * step, len(unique) - 1)]
        threshold = [examined_th for predcsss in predictions]

        recall, Precision, f1 = calculate_AD_levels(anomalyranges, leadranges, predictions, ignore_range,
                                                    threshold, beta)
        if Precision < 0.0000000000000001 and recall[0] < 0.0000000000000000001:
            continue

        tups_R_P1.append((recall[0], Precision))
        tups_R_P2.append((recall[1], Precision))
        tups_R_P3.append((recall[2], Precision))
        # All results
        allresults.append([f1[0], f1[1], f1[2], recall[0], recall[1], recall[2], Precision, examined_th])
    # allresults.append([0,0,0,1,1,1,0,min(unique)])
    if len(tups_R_P1) > 0:
        allresults.append([0, 0, 0, 0, 0, 0, max([ttttup[1] for ttttup in tups_R_P1]), max(unique)])
        allresultsforbestthreshold = allresults.copy()
        allresultsforbestthreshold.sort(key=lambda tup: tup[0], reverse=False)
        best_th = allresultsforbestthreshold[-1][-1]
        tups_R_P1 = sorted(tups_R_P1, key=lambda x: (x[0], -x[1]))
        # tups_R_P1.sort(key=lambda tup: tup[0], reverse=False)  # sorts in place
        tups_R_P2 = sorted(tups_R_P2, key=lambda x: (x[0], -x[1]))
        # tups_R_P2.sort(key=lambda tup: tup[0], reverse=False)  # sorts in place
        tups_R_P3 = sorted(tups_R_P3, key=lambda x: (x[0], -x[1]))
        # tups_R_P3.sort(key=lambda tup: tup[0], reverse=False)  # sorts in place

        recalls1 = [0] + [tup[0] for tup in tups_R_P1]
        recalls2 = [0] + [tup[0] for tup in tups_R_P2]
        recalls3 = [0] + [tup[0] for tup in tups_R_P3]

        press1 = [max([ttttup[1] for ttttup in tups_R_P1])] + [tup[1] for tup in tups_R_P1]
        press2 = [max([ttttup[1] for ttttup in tups_R_P1])] + [tup[1] for tup in tups_R_P2]
        press3 = [max([ttttup[1] for ttttup in tups_R_P1])] + [tup[1] for tup in tups_R_P3]

        # plt.subplot(121)
        # figtoplot=plt.figure(figsize=(28, 16))
        # ax = figtoplot.add_subplot()
        # ax.plot(recalls1,press1,"-o")
        # plt.plot(recalls2,press2)
        # plt.plot(press3,recalls3)

        if len(recalls1) == 1 or len(press1) == 1:
            AUC1 = 0.0
        else:
            AUC1 = sklearn.metrics.auc(recalls1, press1)

        if len(recalls2) == 1 or len(press2) == 1:
            AUC2 = 0.0
        else:
            AUC2 = sklearn.metrics.auc(recalls2, press2)

        if len(recalls3) == 1 or len(press3) == 1:
            AUC3 = 0.0
        else:
            AUC3 = sklearn.metrics.auc(recalls3, press3)
    else:
        allresults.append([0, 0, 0, 0, 0, 0, 0, 0])
        AUC1, AUC2, AUC3 = 0, 0, 0
        best_th = 0.5

    for i in range(len(allresults)):
        allresults[i].append(AUC1)
        allresults[i].append(AUC2)
        allresults[i].append(AUC3)

    #### VUS RESULTS
    flatened_scores = np.array(flatened_scores)
    anomalyranges_for_vus = np.array(anomalyranges)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # check if flatened_scores contains infinity or a value too large for dtype('float64') - infinity bug
    max_float64 = np.finfo(np.float64).max

    mask = np.isinf(flatened_scores) | (np.abs(flatened_scores) > max_float64)

    flatened_scores[mask] = max_float64

    try:
        score = scaler.fit_transform(flatened_scores.reshape(-1, 1)).ravel()

        results = get_metrics(score, anomalyranges_for_vus,
                       best_threshold_examined=scaler.transform(np.array([[best_th]])).ravel()[0],
                       slidingWindow=slidingWindow_vus)  # default metric='all'
        # results={}
    except Exception as e:
        with open('exception_log.txt', 'w') as f:
            f.write("An exception occurred:\n")
            f.write(traceback.format_exc())

        np.save('exception_scores.npy', flatened_scores)

        print(e)
        vus_metrics_keys = [
            'AUC_ROC',
            'AUC_PR',
            'Precision',
            'Recall',
            'F',
            'Precision_at_k',
            'Rprecision',
            'Rrecall',
            'RF',
            'R_AUC_ROC',
            'R_AUC_PR',
            'VUS_ROC',
            'VUS_PR',
            'Affiliation_Precision',
            'Affiliation_Recall'
        ]

        results = {}

        for key in vus_metrics_keys:
            results[key] = 0

    return allresults, results, anomalyranges, leadranges


def AUCPR_ranges_new(predictions, anomalyranges, leadranges, beta=1, resolution=100, slidingWindow_vus=0):
    predtemp = []
    if isinstance(predictions[0], collections.abc.Sequence):
        for predcs in predictions:
            predtemp.extend(predcs)
        flatened_scores = predtemp.copy()
        predtemp = list(set(predtemp))
    else:
        predtemp.extend(predictions)
        flatened_scores = predtemp.copy()
        predtemp = list(set(predictions))
    predtemp.sort()
    unique = list(set(predtemp))
    unique.sort()
    resolution = min(resolution, max(1, len(unique)))
    step = int(len(unique) / resolution)

    rangetemp = []
    if isinstance(anomalyranges[0], collections.abc.Sequence):
        for rangess in anomalyranges:
            rangetemp.extend(rangess)
        flatened_anomalyranges = rangetemp.copy()
        anomalyranges = flatened_anomalyranges
    rangetemp = []
    if isinstance(leadranges[0], collections.abc.Sequence):
        for rangess in leadranges:
            rangetemp.extend(rangess)
        flatened_leadranges = rangetemp.copy()
        leadranges = flatened_leadranges

    ignore_range = [0 for an in anomalyranges]

    tups_R_P1 = []
    tups_R_P2 = []
    tups_R_P3 = []
    allresults = []
    maxf1 = 0
    maxth = unique[0]
    for i in range(resolution + 2):
        examined_th = unique[min(i * step, len(unique) - 1)]
        threshold = [examined_th for predcsss in flatened_scores]

        recall, Precision, f1 = calculate_AD_levels(anomalyranges, leadranges, flatened_scores, ignore_range,
                                                    threshold, beta)
        if f1[0] > maxf1:
            maxth = examined_th
            maxf1 = f1[0]
        if Precision < 0.0000000000000001 and recall[0] < 0.0000000000000000001:
            continue

        tups_R_P1.append((recall[0], Precision))
        tups_R_P2.append((recall[1], Precision))
        tups_R_P3.append((recall[2], Precision))
        # All results
        allresults.append([f1[0], f1[1], f1[2], recall[0], recall[1], recall[2], Precision, examined_th])
    # allresults.append([0,0,0,1,1,1,0,min(unique)])
    if len(tups_R_P1) > 0:
        allresults.append([0, 0, 0, 0, 0, 0, max([ttttup[1] for ttttup in tups_R_P1]), max(unique)])
        # allresults.append([0, 0, 0, 0, 0, 0, 1, max(unique)])
        allresultsforbestthreshold = allresults.copy()
        allresultsforbestthreshold.sort(key=lambda tup: tup[0], reverse=False)
        best_th = maxth
        tups_R_P1 = sorted(tups_R_P1, key=lambda x: (x[0], -x[1]))
        # tups_R_P1.sort(key=lambda tup: tup[0], reverse=False)  # sorts in place
        tups_R_P2 = sorted(tups_R_P2, key=lambda x: (x[0], -x[1]))
        # tups_R_P2.sort(key=lambda tup: tup[0], reverse=False)  # sorts in place
        tups_R_P3 = sorted(tups_R_P3, key=lambda x: (x[0], -x[1]))
        # tups_R_P3.sort(key=lambda tup: tup[0], reverse=False)  # sorts in place

        recalls1 = [0] + [tup[0] for tup in tups_R_P1]
        recalls2 = [0] + [tup[0] for tup in tups_R_P2]
        recalls3 = [0] + [tup[0] for tup in tups_R_P3]

        press1 = [max([ttttup[1] for ttttup in tups_R_P1])] + [tup[1] for tup in tups_R_P1]
        press2 = [max([ttttup[1] for ttttup in tups_R_P1])] + [tup[1] for tup in tups_R_P2]
        press3 = [max([ttttup[1] for ttttup in tups_R_P1])] + [tup[1] for tup in tups_R_P3]

        # plt.subplot(121)
        # figtoplot=plt.figure(figsize=(28, 16))
        # ax = figtoplot.add_subplot()
        # ax.plot(recalls1,press1,"-o")
        # plt.plot(recalls2,press2)
        # plt.plot(press3,recalls3)

        if len(recalls1) == 1 or len(press1) == 1:
            AUC1 = 0.0
        else:
            AUC1 = sklearn.metrics.auc(recalls1, press1)

        if len(recalls2) == 1 or len(press2) == 1:
            AUC2 = 0.0
        else:
            AUC2 = sklearn.metrics.auc(recalls2, press2)

        if len(recalls3) == 1 or len(press3) == 1:
            AUC3 = 0.0
        else:
            AUC3 = sklearn.metrics.auc(recalls3, press3)
    else:
        allresults.append([0, 0, 0, 0, 0, 0, 0, 0])
        AUC1, AUC2, AUC3 = 0, 0, 0
        best_th = 0.5

    for i in range(len(allresults)):
        allresults[i].append(AUC1)
        allresults[i].append(AUC2)
        allresults[i].append(AUC3)

    #### VUS RESULTS
    # TODO infinity bug
    import datetime
    print("before VUS Current date and time: ", datetime.datetime.now())
    flatened_scores = np.array(flatened_scores)
    anomalyranges_for_vus = np.array(anomalyranges)
    scaler = MinMaxScaler(feature_range=(0, 1))
    score = scaler.fit_transform(flatened_scores.reshape(-1, 1)).ravel()
    results = get_metrics(score, anomalyranges_for_vus,
                         best_threshold_examined=scaler.transform(np.array([[best_th]])).ravel()[0],
                         slidingWindow=slidingWindow_vus)  # default metric='all'
    # results={}
    print("after VUS Current date and time: ", datetime.datetime.now())
    return allresults, results, anomalyranges, leadranges
