import pickle
import copy
import random as rd
import numpy as np
import torch
import scipy.sparse as sp
import scipy
from scipy.io import loadmat
import copy as cp
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
from collections import defaultdict
from sklearn.model_selection import train_test_split


"""
    Utility functions to handle data and evaluate model.
"""
# load data

def read_data(root_path, data_source):
    data = scipy.io.loadmat(root_path+"{}.mat".format(data_source))
    gnds = data["gnd"]
    attributes_sprse = sp.csr_matrix(data["Attributes"])
    network = sp.lil_matrix(data["Network"])
    adj = network.todense()
    adj_norm = preprocess_adj(network)
    attributes = attributes_sprse.todense()

    reshaped_gnd = gnds.reshape(attributes.shape[0], -1)
    """
    [[0]
    [0]
    ...
    [0]
    [0]]
    """
    # print(reshaped_gnd.shape)
    # print('reshaped_gnd[0:20]', reshaped_gnd[0:10])

    return adj, adj_norm, attributes, reshaped_gnd

def purify_labels(all_label, anomaly_list, norm_list):
    #pure_all_label = all_label.clone()
    all_label[anomaly_list] = 1
    all_label[norm_list] = 0
    return all_label

# base detctor to prepare
def process_test_data(dataset, model, node_features, adj, ground_truth, paces, rate_ano_sel, epsilon=0.02, iterations=100):
    
    model.eval()
    final_pred_score, nodes_embeddings = model(node_features, adj)
    anomaly_score = final_pred_score.flatten()  
    y_true = ground_truth.flatten() # nosiy ones still

    # Apply epsilon greedy to find the best pace
    best_pace, candi_auc, noisy_candi = epsilon_greedy(dataset, anomaly_score, y_true, paces, epsilon=epsilon, iterations=iterations)

    # Select confident anomalies and normals
    anomaly_tensor, norm_tensor = select_confident(anomaly_score, rate_ano_sel)

    return best_pace, noisy_candi, anomaly_tensor, norm_tensor, nodes_embeddings, anomaly_score


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    # D^-0.5AD^0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def noisy_GAD_train_test_split_v1_40trn_20val_fixTrnAnom(labels, random_seed, true_outliers_num, train_ratio, val_ratio, args):
    """
    From: /home/szhou/baseline_AD_supervised/GCN_semi_AD/pygcn/utils.py

    :param labels: original label, 0 or 1 (outlier)
    :param random_seed:
    :param true_outliers_num: all the outliers with true labels in trn & val

    :return:
    """

    """
    2023.July 2
    The total of outliers num is fixed in trn & val & test, respectively;
    In a graph, if more outliers are labeled, fewer outliers remain in training set;
    The above is reasonable.

    train: valid: test = 40%*N : 20%*N : 40%*N;
        all outliers are split into train: valid: test as 40%:20%:40%;
        but labeled outliers num: 10  (train : valid = 4:2);

    train: 40% normal + 40% * Outlier (only 10*2/3 are labeled outliers, the rest are used as normal)
    valid: 20% normal + 20% * Outlier (only 10*1/3 are labeled outliers, the rest are used as normal)
    test:  40% normal + 40% * Outlier (all nodes are unlabelled)

    trn_val / total_nodes (N) = 60%;
    outlier in trn_val = 10;
    'normal' in trn_val = 60%*N - 10  (must contains contamination: outliers are used as normal);

    Drawback: 
    contamination is not fixed --> contamination will change in trn & val, but not test;
    """
    modified_gnd = cp.deepcopy(labels)
    total_nodes_num = labels.shape[0]
    # collect outliers index & num
    outliers_index = np.where(labels == 1)[0]
    outliers_num = outliers_index.shape[0]
    # collect normal nodes
    normal_index = np.where(labels == 0)[0]
    normal_num = normal_index.shape[0]
    val_ratio_in_Trn_Val = float(val_ratio / (train_ratio + val_ratio))

    # 1. split all normal, respectively in training and test step
    trn_val_norm_indx, test_norm_indx = train_test_split(
        normal_index, test_size=1 - train_ratio - val_ratio, random_state=random_seed + 23)
    trn_norm_indx, val_norm_indx = train_test_split(
        trn_val_norm_indx, test_size=val_ratio_in_Trn_Val, random_state=random_seed + 33)

    # ---------------------------------------- fix labeled outliers ----------------------------------------------------
    # read 40 nodes from files
    fixed_outliers_in_files = np.loadtxt(
        args.root_path + args.dataset + '_fixed_rare_category.csv', delimiter=',', dtype=int)
    fixed_normal_in_files = np.loadtxt(
        args.root_path + args.dataset + '_fixed_normal_nodes.csv', delimiter=',', dtype=int)
    true_labeled_anomalies_num = args.true_outliers_num   # anomalies with true labels
    noisy_labels_num = args.max_labeled_outliers_num - \
        true_labeled_anomalies_num  # (noisy labels)
    # verify true outliers and transfer normal to noisy ones
    fixed_labeled_outliers = fixed_outliers_in_files[0: true_labeled_anomalies_num]
    noisy_labels_indices = fixed_normal_in_files[0: noisy_labels_num]
    # merge true outliers and noisy outliers, all outliers setting (mixed)
    trn_all_labels_in_noisy_setting = np.append(
        fixed_labeled_outliers, noisy_labels_indices)

    # except true outliers the left ones
    exclude_trn_outlier = np.array(
        list(set(outliers_index).difference(set(trn_all_labels_in_noisy_setting))))
    shuffled_left_outliers = shuffle_array(
        exclude_trn_outlier, seed=random_seed + 3)  # shuffle
    # verify test outliers num & index
    test_abnrm_num = int(outliers_num * (1 - train_ratio - val_ratio))
    test_abnrm_indx = shuffled_left_outliers[0:test_abnrm_num]
    # for training outliers as normal
    trn_val_masked_outlier_indx = shuffled_left_outliers[test_abnrm_num:]

    # specify array of training and validation outliers set
    true_trn_labeled_outlier_indx, true_val_labeled_outlier_indx = train_test_split(
        fixed_labeled_outliers, test_size=val_ratio_in_Trn_Val, random_state=random_seed + 33)
    # if there are no nodes with noisy labels
    if noisy_labels_num == 0:
        false_trn_labeled_outlier_indx = np.array([], dtype=np.int64)
        false_val_labeled_outlier_indx = np.array([], dtype=np.int64)
    else:
        false_trn_labeled_outlier_indx, false_val_labeled_outlier_indx = train_test_split(
            noisy_labels_indices, test_size=val_ratio_in_Trn_Val, random_state=random_seed + 33)

    trn_labeled_outlier_indx = np.append(
        true_trn_labeled_outlier_indx, false_trn_labeled_outlier_indx)
    val_labeled_outlier_indx = np.append(
        true_val_labeled_outlier_indx, false_val_labeled_outlier_indx)
    trn_masked_outlier_indx, val_masked_outlier_indx = train_test_split(
        trn_val_masked_outlier_indx, test_size=val_ratio_in_Trn_Val, random_state=random_seed + 13)
    # ---------------------------------------------------------------------------------------------

    # ########################################################
    # 3. change label
    # normal nodes taken as labels (noisy labels), i.e., from 0 to 1.
    for indx_i in noisy_labels_indices:  # because they are mixed
        modified_gnd[indx_i] = 1
    # change label for those unlabeled (or masked) outliers (because they are used as 'normal')
    if noisy_labels_num != 0:
        for indx_k in trn_masked_outlier_indx:
            modified_gnd[indx_k] = 0
        for indx_j in val_masked_outlier_indx:
            modified_gnd[indx_j] = 0

    # 4. merge final data
    # test with outliers and normal nodes
    test_index_final = np.append(test_norm_indx, test_abnrm_indx, axis=0)
    # truely normal + masked outliers
    train_all_normal_set = np.append(
        trn_norm_indx, trn_masked_outlier_indx, axis=0)
    # truely normal + masked outliers
    val_all_normal_set = np.append(
        val_norm_indx, val_masked_outlier_indx, axis=0)
    # contaminated normal + truely outliers
    valid_indx_final = np.append(val_all_normal_set, val_labeled_outlier_indx)
    train_indx_final = np.append(
        train_all_normal_set, trn_labeled_outlier_indx)
    train_indx_final = shuffle_array(train_indx_final, seed=random_seed + 4)
    # ########################################################
    current_outlier_indices = np.where(modified_gnd == 1)[0]
    outliers_as_normal_in_trn_val = list(set(outliers_index).difference(
        set(current_outlier_indices)))  # index of trn and val (outliers-->normal)
    # true_labeled_trn_outliers_list = list(set(current_outlier_indices).intersection(set(trn_labeled_outlier_indx)))
    # true_labeled_val_outliers_list = list(set(current_outlier_indices).intersection(set(val_labeled_outlier_indx)))
    true_labeled_trn_outliers_list = list(true_trn_labeled_outlier_indx)
    true_labeled_val_outliers_list = list(true_val_labeled_outlier_indx)
    # trn_noisy_labels = list(set(normal_index).intersection(set(trn_labeled_outlier_indx)))
    # val_noisy_labels = list(set(normal_index).intersection(set(val_labeled_outlier_indx)))
    trn_noisy_labels = list(false_trn_labeled_outlier_indx)
    val_noisy_labels = list(false_val_labeled_outlier_indx)

    print('_____ some outliers are taken as normal in training & val :',
          len(outliers_as_normal_in_trn_val))
    print('_____ some normal are taken as abnormal in training & val :',
          len(noisy_labels_indices))
    # ########################################################

    # only count the number
    train_data_final_num = train_all_normal_set.shape[0] + \
        trn_labeled_outlier_indx.shape[0]
    valid_data_final_num = valid_indx_final.shape[0]  # only count the number
    print('_____ the total num:', total_nodes_num)
    print('_____ the total labeled num:', outliers_num + normal_num)
    print('_____ the labeled ratio:',
          (outliers_num + normal_num) / total_nodes_num)
    print('_____  train final num, valid final num: ',
          train_data_final_num, valid_data_final_num)
    print('_____  train/valid: ', train_data_final_num /
          (train_data_final_num + valid_data_final_num))
    print('_____  test:', test_index_final.shape[0])
    print('_____  labeled outliers in trn:', trn_labeled_outlier_indx.shape[0])
    print('_____  true_labeled_outliers_list in trn:',
          len(true_labeled_trn_outliers_list))
    print('_____  noisy labels in trn:', len(trn_noisy_labels))
    print('_____  labeled outliers in val:', val_labeled_outlier_indx.shape[0])
    print('_____  true_labeled_outliers_list in val:',
          len(true_labeled_val_outliers_list))
    print('_____  noisy labels in val:',  len(val_noisy_labels))
    if noisy_labels_num != 0:
        print('_____  masked outliers in trn:',
              trn_masked_outlier_indx.shape[0])
        print('_____  masked outliers in val:',
              val_masked_outlier_indx.shape[0])
    print('_____  contamination ratio in train set:',
          (trn_masked_outlier_indx.shape[0] + len(false_trn_labeled_outlier_indx)) / train_data_final_num)
    # print('_____  contamination ratio in train set:', (trn_masked_outlier_indx.shape[0] + false_trn_labeled_outlier_indx)/ train_data_final_num)
    print('_____  labeled outliers in test:', test_abnrm_indx.shape[0])
    print('_____  labeled normal in test:', test_norm_indx.shape[0])

    return train_indx_final, train_all_normal_set, trn_labeled_outlier_indx, valid_indx_final, test_index_final, modified_gnd

def noisy_GAD_train_test_split_v2_TrnVal_fixTrnAnom(labels, random_seed, true_outliers_num, train_ratio, val_ratio, args):
    """
    :param labels: original label, 0 or 1 (outlier)
    :param random_seed:
    :param true_outliers_num: all the outliers with true labels in trn & val

    :return:
    """

    """
    2023.June.27
    -----------------------------------------------------------------------
    The total of outliers num is fixed in trn & test, respectively;
    In a graph, if more outliers are labeled, fewer outliers remain in training set;
    The above is reasonable.

    train : test = 60%*N : 40%*N;
        all outliers are split into train : test as 60%:40%;
        but labeled outliers num: 10;

    train: 60% normal + 60% * Outlier (only 10 are labeled outliers, the rest are used as normal)
    test:  40% normal + 40% * Outlier (all nodes are unlabelled)

    trn / total_nodes (N) = 60%;
    outlier in trn = 10;
    'normal' in trn = 60%*N - 10  (must contains contamination: outliers are used as normal);

    Drawback: 
    contamination is not fixed --> contamination will change in trn, but not test;
    It is reasonable. 
    In a graph, if more labels are used in training while test set is fixed, the contamination level naturally changes.
    -----------------------------------------------------------------------

    There is no validation set (Use all the training data for model training)
    """
    modified_gnd = copy.deepcopy(labels)
    total_nodes_num = labels.shape[0]
    outliers_index = np.where(labels == 1)[0]
    outliers_num = outliers_index.shape[0]
    normal_index = np.where(labels == 0)[0]
    val_ratio_in_Trn_Val = float(val_ratio / (train_ratio + val_ratio))

    # 1. split all outliers & all normal, respectively
    trn_val_norm_indx, test_norm_indx = train_test_split(normal_index, test_size=1 - train_ratio - val_ratio, random_state=random_seed + 23)

    # ---------------------------------------- fix labeled outliers ---------------------------------
    fixed_outliers_in_files = np.loadtxt(args.root_path + args.dataset + '_fixed_rare_category.csv', delimiter=',', dtype=int)
    fixed_normal_in_files = np.loadtxt(args.root_path + args.dataset + '_fixed_normal_nodes.csv', delimiter=',', dtype=int)
    true_labeled_anomalies_num = args.true_outliers_num                         # anomalies with true labels
    noisy_labels_num = args.max_labeled_outliers_num - true_outliers_num  # normal nodes taken as labels (noisy labels)
    fixed_labeled_outliers = fixed_outliers_in_files[0: true_labeled_anomalies_num]  # anomalies with true labels
    noisy_labels_indices = fixed_normal_in_files[0: noisy_labels_num]
    trn_all_labels_in_noisy_setting = np.append(fixed_labeled_outliers, noisy_labels_indices)  # the known labels (in trn & val)
    
    test_abnrm_num = int(outliers_num * (1 - train_ratio - val_ratio))
    exclude_trn_outlier = np.array(list(set(outliers_index).difference(set(trn_all_labels_in_noisy_setting))))
    # shuffle the left outliers 
    shuffled_left_outliers = shuffle_array(exclude_trn_outlier, seed=random_seed + 3)
    test_abnrm_indx = shuffled_left_outliers[0:test_abnrm_num]   # test anomalies num is fixed
    trn_val_masked_outlier_indx = shuffled_left_outliers[test_abnrm_num:]
    # ---------------------------------------------------------------------------------------------

    # ########################################################
    # 3. change label
    # normal nodes taken as labels (noisy labels), i.e., from 0 to 1.
    for indx_i in noisy_labels_indices:
        modified_gnd[indx_i] = 1
    # change label for those unlabeled (or masked) outliers (because they are used as 'normal')
    if noisy_labels_num != 0:
        for indx_k in trn_val_masked_outlier_indx:
            modified_gnd[indx_k] = 0

    # 4. merge final data
    test_index_final = np.append(test_norm_indx, test_abnrm_indx, axis=0)
    train_all_normal_set = np.append(trn_val_norm_indx, trn_val_masked_outlier_indx, axis=0)  # truely normal + masked outliers
    train_data_final_num = train_all_normal_set.shape[0] + trn_all_labels_in_noisy_setting.shape[0]  # only count the number
    
    train_indx_final = np.append(train_all_normal_set, trn_all_labels_in_noisy_setting)
    #train_indx_final = shuffle_array(train_indx_final, seed=random_seed + 4)
    # change shuffle to fix the order of nodes
    # ########################################################
    current_outlier_indices = np.where(modified_gnd == 1)[0]
    outliers_as_normal_in_trn_val = list(set(outliers_index).difference(set(current_outlier_indices)))
    true_labeled_TrnVal_outliers_list = list(set(outliers_index).intersection(set(trn_all_labels_in_noisy_setting)))
    TrnVal_noisy_labels = list(set(normal_index).intersection(set(trn_all_labels_in_noisy_setting)))
    #print('_____ org outliers num:', outliers_num)
    print('_____ some outliers are taken as normal in training & val :', len(outliers_as_normal_in_trn_val))
    print('_____ some normal are taken as abnormal in training & val :', len(noisy_labels_indices))
    # ########################################################
    print('_____ the total num:', total_nodes_num)
    print('_____  train final final num: ', train_data_final_num)
    print('_____  test:', test_index_final.shape[0])
    
    print('_____  labeled outliers in trn:', trn_all_labels_in_noisy_setting.shape[0])
    print('_____  true_labeled_TrnVal_outliers_list in trn:', len(true_labeled_TrnVal_outliers_list))
    print('_____  noisy labels in trn:', len(TrnVal_noisy_labels))
        
    if noisy_labels_num != 0:
        print('_____  masked normal nodes in train:', trn_val_masked_outlier_indx.shape[0])
        print('_____  masked normal nodes in test:', test_abnrm_indx.shape[0])
    
    print('_____  labeled outliers in test:', test_abnrm_indx.shape[0])
    
    return train_indx_final, train_all_normal_set, trn_all_labels_in_noisy_setting, test_index_final, modified_gnd


def shuffle_array(input_arr, seed):
    """
    :param input_arr: [1,2,3,4,5]
    :return: randomly shuffled array, [3,5,1,4,2]
    """
    total_node = input_arr.shape[0]
    randomlist = [i for i in range(total_node)]
    rd.seed(seed)
    rd.shuffle(randomlist)
    ouptut_arr = input_arr[randomlist]
    return ouptut_arr


def normalize(mx):
    """
            Row-normalize sparse matrix
            Code from https://github.com/williamleif/graphsage-simple/
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def optimize_threshold(y_true, scores, mask_id):
    best_threshold = 0
    best_f1 = 0
    for threshold in np.arange(0.0, 1.0, 0.01):  # 尝试从0.0到1.0的所有阈值，步长为0.01
        y_pred = (scores > threshold).astype(int)
        f1 = accuracy_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    labels = (scores > best_threshold).astype(int)
    labels_re = labels.copy()
    set_index = list(set(
            element for sublist in mask_id for element in sublist))
    for i in range(0, len(set_index)):
        if labels_re[set_index[i]] == 0:
            labels_re[set_index[i]] = 1
    return best_threshold, labels_re

#high confident labels 
def select_confident(final_pred_score, rate):
    anomaly_score = final_pred_score.flatten() 
    num_anomaly = int(rate * anomaly_score.shape[0])
                  # 计算前x%的节点数量 可设计为超参数
    topk_scores, anomaly_list = torch.topk(
        anomaly_score, num_anomaly)
    
    lastk_scores, norm_list = torch.topk(
        anomaly_score, num_anomaly, largest=False)
    
    return anomaly_list, norm_list
    
#noisy candidates hyper(num_rate)
def select_noisy_candidates(num_rate, anomaly_score):
    thr_high = np.mean(anomaly_score) + num_rate*np.std(anomaly_score)
    thr_low = np.mean(anomaly_score) - num_rate*np.std(anomaly_score)
    noisy_can = np.where((anomaly_score >= thr_low) & (anomaly_score <= thr_high))[0] 
    return torch.tensor(noisy_can)

def select_candidates(anomaly_score, train_index, num_candidates):
    """
    Selects noisy candidates based on their scores, focusing on those near the mean.
    
    Args:
    - scores (np.array): An array of scores.
    - num_candidates (int): The fixed number of noisy candidates to select.
    
    Returns:
    - np.array: An array of indices for the selected noisy candidates.
    """
    # 计算scores的均值
    mean_score = np.mean(anomaly_score[train_index])
    
    # 计算每个score与均值的绝对差值
    abs_diff_from_mean = np.abs(anomaly_score[train_index] - mean_score)
    
    # 获取绝对差值最小的N个scores的索引
    indices_sorted_by_diff = np.argsort(abs_diff_from_mean)
    selected_indices = indices_sorted_by_diff[:num_candidates]
    
    return torch.tensor(selected_indices)

### new design for candidates ####
def select_nodes_by_pace(scores, pace):
    """根据给定的pace选择分数在中间的节点"""
    mid_point = torch.mean(scores)
    lower_bound = mid_point - pace
    upper_bound = mid_point + pace
    return [i for i, score in enumerate(scores) if lower_bound <= score <= upper_bound]

def calculate_auc(scores, labels, indices):
    """计算给定节点上的AUC"""
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()  # 转换为numpy数组
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy() 
    if len(indices) < 2:  # AUC至少需要两个点计算
        return 0  # 如果不能满足，这个情况肯定不是最佳
    return roc_auc_score([labels[i] for i in indices], [scores[i] for i in indices])

def select_balanced_samples(labels, num_samples_per_class=1):
    labels = labels.detach().cpu().numpy()
    unique_labels = np.unique(labels)
    selected_indices = []

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        selected_indices.extend(np.random.choice(indices, num_samples_per_class, replace=False))

    return selected_indices

def epsilon_greedy(dataset, scores, labels, paces, epsilon=0.02, iterations=100):
    """使用ε-贪心算法寻找最佳pace"""
    best_pace = None
    worst_auc = np.inf  # 我们希望最小化奖励，因此初始化为无穷大
    add_indices = select_balanced_samples(labels, num_samples_per_class=1)
    
    for _ in range(iterations):
        if rd.random() < epsilon:  # 探索
            current_pace = rd.choice(paces)
        else:  # 利用
            current_pace = best_pace if best_pace is not None else rd.choice(paces)
        
        indices = select_nodes_by_pace(scores, current_pace)
        
        if dataset == 'amazon_electronics_computers':
            indices.extend(add_indices)
            current_auc = calculate_auc(scores, labels, indices)
        else: 
            current_auc = calculate_auc(scores, labels, indices)
            
        current_reward = current_auc  # 负AUC作为奖励
        
        # 更新最佳奖励和最佳pace
        if current_reward < worst_auc:
            worst_auc = current_auc
            best_pace = current_pace
    
    torch.cuda.empty_cache()
    return best_pace, worst_auc, indices # 返回最佳pace和对应的AUC

#use tensor to change noisy labels 
def purify_labels(all_label, anomaly_list, norm_list):
    pure_label = all_label.clone()
    anomaly_changes = (pure_label[anomaly_list] != 1).sum().item()  # Count anomaly changes
    norm_changes = (pure_label[norm_list] != 0).sum().item()  # Count norm changes
    
    pure_label[anomaly_list] = 1  # Update anomalies if not already 1
    pure_label[norm_list] = 0  # Update normals if not already 0
    
    return pure_label, anomaly_changes, norm_changes

#the use of cut edges \\\ core nodes records
def get_edge_mask(adj_tensor, noisy_can):
    
    edge_mask = torch.zeros_like(adj_tensor.to_dense())
    # 循环计算太慢，还是应该使用tensor
    edge_mask[noisy_can, :] = 1
    edge_mask[:, noisy_can] = 1
    # edge_mask.fill_diagonal_(0)  # self-loop has no necessaity to be zero
    return edge_mask # a dense matrix

def deviation_loss_torch(y_true, y_pred):
    """
    y_true y_pred are torch tensor

    :return: z-score-based deviation loss
    same as deviation loss in pang's paper
    This code has been checked correctness;
    the result is the same as the numpy version code

    ----------------------------------------
    update on 2021.09.19, to be more robust and sensitive to detect error
    ensure the 2 inputs are the same shape, otherwise the calculated loss will be wrong
    """

    # update on 2021.09.19, to be more robust and sensitive to detect error
    # ----------------------------------------
    # # ensure the 2 inputs are the same shape, otherwise the calculated loss will be wrong
    # print('y_true.detach().cpu().shape:', y_true.detach().cpu().shape)
    # print('y_pred.detach().cpu().shape:', y_pred.detach().cpu().shape)
    assert y_true.detach().cpu().shape == y_pred.detach().cpu().shape
    # ----------------------------------------

    confidence_margin = 5
    device = y_pred.device
    data_num = y_pred.size()[0]

    # size=5000 is the setting of l in algorithm in the paper
    ref = np.random.normal(loc=0., scale=1.0, size=5000)
    ref_torch = torch.from_numpy(ref).to(device)

    dev = (y_pred - torch.mean(ref_torch)) / torch.std(ref_torch)
    inlier_loss = torch.abs(dev)

    zero_array = torch.from_numpy(np.zeros((data_num, 1))).to(device)
    outlier_loss = torch.abs(torch.maximum(
        confidence_margin - dev, zero_array))

    return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)


def train_batch_iter_v2_FullDataTrn(outlier_indices, inlier_indices, batch_size, nb_batch, seed):
    '''
    batchs of samples. This is for csv data.
    Alternates between positive and negative pairs.

    Update on 2021.Nov.25
    The original batch traning will lead to performance having large variance;
    To have stable model, I decide to use all training data (labeled normal and abnormal nodes) for model training.
    '''
    counter = 0
    while counter < nb_batch:
        counter += 1
        rng = np.random.RandomState(seed+counter)
        train_indx_list = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)
        #use this to obtain data randomly
        inlier_indices = shuffle_array(inlier_indices, seed)
        # why *2 then *0.5
        sampled_inliers_num = int(0.5 * batch_size)
        assert sampled_inliers_num <= inlier_indices.shape[0]
        train_indx_list.extend(list(inlier_indices[0:sampled_inliers_num]))

        for i in range(int(batch_size)):
            if (i % 2 == 0):
                sid = rng.choice(n_outliers, 1)
                train_indx_list.extend(list(outlier_indices[sid]))

        yield np.array(train_indx_list)


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def plot_param_test(train_loss, valid_loss, save_name, args):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    # plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r',
                label='Early Stopping Checkpoint')

    val_max_value = max(train_loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, val_max_value)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    if args.test == 'test':
        fig.savefig(args.test_plot + 'loss_plot_' +
                    save_name + '.png', bbox_inches='tight')
    else:
        fig.savefig(args.val_plot + 'loss_plot_' +
                    save_name + '.png', bbox_inches='tight')

def plot_cut_train(train_num, save_name, args):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_num) + 1), train_num, label='Pruned esges')
    # plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = train_num.index(min(train_num)) + 1
    # plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    val_max_value = max(train_num)
    plt.xlabel('epochs')
    plt.ylabel('num')
    plt.ylim(0, val_max_value)  # consistent scale
    plt.xlim(0, len(train_num) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig(args.val_plot + 'edges_plot_' +
                save_name + '.png', bbox_inches='tight')