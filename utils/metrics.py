import numpy as np
from utils.AUC import point_wise_AUC, Range_AUC
from utils.affiliation.generics import convert_vector_to_events
from utils.affiliation.metrics import pr_from_events
from utils.eval_utils import *
from utils.vus.metrics import get_range_vus_roc
from utils.tools import score_to_label

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def calculate_r2_pearson_robust(x, y):
    # check shape
    # print(f'y.shape: {y.shape}')
    if x.shape != y.shape:
        raise ValueError("The shape of x and y must be the same!")

    assert x.ndim >= 3, 'Error in x.shape...'

    flattened_x = x.reshape(-1, x.shape[-2], x.shape[-1])
    flattened_y = y.reshape(-1, y.shape[-2], y.shape[-1])

    # there was a bug here!!!!
    mean_x = np.mean(flattened_x, axis=1, keepdims=True)
    mean_y = np.mean(flattened_y, axis=1, keepdims=True)

    # correlation coefficient
    numerator = np.sum((flattened_x - mean_x) * (flattened_y - mean_y), axis=1)
    denominator = np.sqrt(np.sum((flattened_x - mean_x) ** 2, axis=1) * np.sum((flattened_y - mean_y) ** 2, axis=1))
    # avoid potential zero-divide error
    denominator[denominator < 1e-5] = np.nan
    pearson_correlation = numerator / denominator

    # R2
    total_variance = np.sum((flattened_y - mean_y) ** 2, axis=1)
    residuals = flattened_y - flattened_x
    residual_sum_of_squares = np.sum(residuals ** 2, axis=1)
    # avoid potential zero-divide error
    total_variance[total_variance < 1e-5] = np.nan
    r2 = 1 - (residual_sum_of_squares / total_variance)

    # mean
    mean_pearson = np.nanmean(pearson_correlation)
    mean_r2 = np.nanmean(r2)

    return mean_r2, mean_pearson


def calculate_mase(y_pred, y_true, y_naive=None):
    """
    MASE（Mean Absolute Scaled Error）。
    y_true: (b, l, n)
    y_pred: (b, l, n)
    y_naive: (b, l, n)
    """

    assert y_pred.ndim >= 3, 'Error in y_pred.shape...'

    if y_naive is None:
        y_naive = y_true

    if y_true.shape != y_pred.shape or y_true.shape != y_naive.shape:
        raise ValueError("The input shape must be the same.")

    # reshape to [batch, pred_len, channel]
    y_true_flat = y_true.reshape(-1, y_true.shape[-2], y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-2], y_pred.shape[-1])
    y_naive_flat = y_naive.reshape(-1, y_naive.shape[-2], y_naive.shape[-1])

    # MAE: [batch, channel]
    mae_model = np.mean(np.abs(y_true_flat - y_pred_flat), axis=1)

    # naive MAE [batch, channel]
    mae_naive = np.mean(np.abs(y_true_flat[:, 1:, :] - y_naive_flat[:, :-1, :]), axis=1)

    # avoid potential error
    mae_naive[mae_naive < 1e-5] = np.nan

    # MASE
    mase = np.nanmean(mae_model / mae_naive)

    return mase


def metric(pred, true):
    if np.any(np.isnan(true)):
        mask = ~np.isnan(true)
        pred = pred[mask]
        true = true[mask]

    mae = MAE(pred, true)
    mse = MSE(pred, true)

    # print(f'pred.shape[-2]: {pred.shape[-2]}')

    if pred.ndim > 1 and true.ndim > 1:
        r2, pear = calculate_r2_pearson_robust(pred, true)
        mase = calculate_mase(pred, true)
    else:
        r2, pear, mase = 0, 0, 0

    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe, r2, pear, mase

def calculate_affiliation(pred_label, true_label):
    events_pred = convert_vector_to_events(pred_label)
    events_gt = convert_vector_to_events(true_label)
    Trange = (0, len(true_label))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    affiliation_f1 = 2 * affiliation['precision'] * affiliation['recall'] / (affiliation['precision'] + affiliation['recall'] )
    return affiliation['precision'], affiliation['recall'], affiliation_f1

def calculate_ori_prf1(pred_label, true_label):
    _, precision_ori, recall_ori, f1_score_ori, f05_score_ori = get_accuracy_precision_recall_f1(pred_label, true_label)
    return precision_ori, recall_ori, f1_score_ori, f05_score_ori

def calculate_pa_prf1(pred_label, true_label):
    true_events = get_events(true_label)
    _, _, _, precision_pa, recall_pa, f1_score_pa = get_point_adjust_f1_score(pred_label, true_label, true_events)
    return precision_pa, recall_pa, f1_score_pa

def calculate_point_range_auc(pred_label, true_label):
    point_auc = point_wise_AUC(pred_label, true_label)
    range_auc = Range_AUC(pred_label, true_label)
    return point_auc, range_auc

def calculate_VUS(pred_label, true_label):
    results = get_range_vus_roc(true_label, pred_label, 100)
    return results["R_AUC_ROC"], results["R_AUC_PR"], results["VUS_ROC"], results["VUS_PR"]

def calculate_roc(pred_score, true_label):
    if len(np.unique(true_label)) < 2:
        return 0, 0, 0.5
    fpr, tpr, _ = roc_curve(true_label, pred_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def calculate_prc(pred_score, true_label):
    if len(np.unique(true_label)) < 2:
        return 0, 0, np.mean(true_label) if true_label.size > 0 else 0.5
    precision, recall, _ = precision_recall_curve(true_label, pred_score, pos_label=1)
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc

def combine_all_evaluation_score(pred_label, true_label, pred_score):
    affiliation_precision, affiliation_recall, affiliation_f1 = calculate_affiliation(pred_label, true_label)
    precision_ori, recall_ori, f1_score_ori, f05_score_ori = calculate_ori_prf1(pred_label, true_label)
    precision_pa, recall_pa, f1_score_pa = calculate_pa_prf1(pred_label, true_label)
    point_auc, range_auc = calculate_point_range_auc(pred_label, true_label)
    R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR = calculate_VUS(pred_label, true_label)

    fpr, tpr, roc_auc = calculate_roc(pred_score, true_label)
    _, _, pr_auc = calculate_prc(pred_score, true_label)

    return {
                'recall_ori': recall_ori,
                'precision_ori': precision_ori,
                "f1_score_ori": f1_score_ori,
                "f05_score_ori": f05_score_ori,

                "precision_pa": precision_pa,
                "recall_pa": recall_pa,
                "f_score_pa": f1_score_pa,

                "point_auc": point_auc,
                "range_auc": range_auc,
                "Affiliation precision": affiliation_precision,
                "Affiliation recall":  affiliation_recall,
                "Affiliation f1": affiliation_f1, 

                "R_AUC_ROC": R_AUC_ROC,
                "R_AUC_PR": R_AUC_PR,
                "VUS_ROC": VUS_ROC,
                "VUS_PR": VUS_PR,

                'ROC': roc_auc,
                'PRC': pr_auc
    }






