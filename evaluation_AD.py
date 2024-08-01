import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd


def top_K_precision_recall(pred_top_k_gnd, K, total_anomaly_num):
    """
    1 --> anomaly
    0 --> normal
    """
    top_K_anomaly_num = sum(list(pred_top_k_gnd)[:K])
    top_K_pred_num = int(K)  # because the top K nodes are considered as predcited anomaly

    Precision_K = float(top_K_anomaly_num / top_K_pred_num)
    Recall_K = float(top_K_anomaly_num / total_anomaly_num)

    return Precision_K, Recall_K


def szhou_AUC_AUPR(pred_anomaly_score, gnds):
    """
    pred_anomaly_score: numpy array;
    gnds: numpy array; 0->normal; 1->anomaly
    """
    roc_auc = roc_auc_score(gnds, pred_anomaly_score)
    roc_pr_area = average_precision_score(gnds, pred_anomaly_score)
    return roc_auc, roc_pr_area


def szhou_AD_metric(pred_anomaly_score, gnds):
    """
    pred_anomaly_score: numpy array;
    gnds: numpy array; 0->normal; 1->anomaly
    """
    roc_auc = roc_auc_score(gnds, pred_anomaly_score)
    roc_pr_area = average_precision_score(gnds, pred_anomaly_score)

    # precision@K
    sorted_index = np.argsort(-pred_anomaly_score, axis=0)  # high error node ranked top
    top_ranking_gnd = gnds[sorted_index]  # get corresponding label for ranking list

    # top K metric
    test_outliers_num = (np.where(gnds==1)[0]).shape[0]
    all_results = []
    pre_k = []
    rec_k = []
    for K in [100, 200, 300, 400,500,600]:
        precision_k, recall_k = top_K_precision_recall(top_ranking_gnd, K, test_outliers_num)
        pre_k.append(precision_k)
        rec_k.append(recall_k)
    pre_k.extend(rec_k)
    all_results.extend(pre_k)

    # save results
    all_results.append(roc_auc)
    all_results.append(roc_pr_area)

    print('Precision_100', 'Precision_200', 'Precision_300', 'Precision_400', 'Precision_500', 'Precision_600',
          'Recall_100', 'Recall_200', 'Recall_300', 'Recall_400', 'Recall_500', 'Recall_600',
          'AUC', 'AUPR')
    print(all_results)
    return all_results