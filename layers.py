import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from operator import itemgetter
import math
from torch.nn.functional import cosine_similarity
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics import roc_auc_score
import torchmetrics

"""
GCN layers 
"""

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


def find_k_nearest_nodes(feat_emb, node_index, k):
    # compute all distances
    similarities = cosine_similarity(
        feat_emb, feat_emb[node_index].unsqueeze(0))
    distances = 1 - similarities
    # distances = torch.norm(feat_emb - feat_emb[node_index], dim=1)

    # find k node index
    topk_distances, indices = torch.topk(distances, k, largest=False)
    topk_similarities = 1 - topk_distances
    return topk_similarities, indices


def compute_similarity(core_node, node_list, feat_emb):
    # get embedding by index
    core_emb = feat_emb[core_node]
    node_list_emb = feat_emb[node_list]
    # conpute total simi
    simi_total = cosine_similarity(core_emb.unsqueeze(0), node_list_emb)
    simi_mean = simi_total.mean()

    return simi_mean


def RL_Module(anomaly_list, norm_list, batch_nodes, feat_emb, thresholds, threshold_log, args):
    """
    anomaly set 
    normal node set 
    """
    simi_score_list = []
    nearest_nodes_list = []
    stop_flag = True
    k = args.hidden_k
    # 下面是延续caregnn 对前两个epoch不进行rl
    # if len(threshold_log) % len(anomaly_list) != 0 or len(threshold_log) < 2 * len(anomaly_list):
    #     rewards = [0 for i in range(0, len(thresholds))]
    #     new_thresholds = thresholds
    # else:
    rewards_log = []
    # new_thresholds = []
    # compute average neighbor distances for each relation
    for i in range(0, len(anomaly_list)):
        node = anomaly_list[i]
        similarities_near, nearest_nodes = find_k_nearest_nodes(
            feat_emb, node, k+1)
        # to adjust the situation inculding self
        similarities_near = similarities_near[1:]
        nearest_nodes = nearest_nodes[1:]
        # to record the anomaly node neighbors
        nearest_nodes_list.append(nearest_nodes.tolist())
        simi_score_list.append(similarities_near.tolist())
        # this is the standard to be compared (similarity between core and anomaly list)
        std_mean_ab = compute_similarity(node, anomaly_list, feat_emb)
        # compute the simi score of core and normal_list
        std_mean_no = compute_similarity(node, norm_list, feat_emb)
        reward_list = []
        episodes = args.episodes
        step_size = args.step_size
        for eps in range(0, episodes):
            reward = 0
            for j in range(0, len(nearest_nodes)):
                # classify nearest k nodes
                if similarities_near[j] >= thresholds[i]:

                    nei_mean_ab = compute_similarity(
                        nearest_nodes[j], anomaly_list, feat_emb)
                    # update reward
                    if nei_mean_ab >= std_mean_ab:
                        # change reward
                        reward = reward + 1
                    else:
                        reward = reward
                else:  # similarities_near[j] < thresholds[i] j should be normal node
                    nei_mean_no = compute_similarity(
                        nearest_nodes[j], norm_list, feat_emb)
                    if nei_mean_no <= std_mean_no:
                        # change reward
                        reward = reward + 1
                    else:
                        reward = reward

            if reward >= 0.5*k : # 7
                thresholds[i] = thresholds[i] - step_size
            else:
                thresholds[i] = thresholds[i] + step_size

            reward_list.append(reward)
            eps = eps + 1
        rewards_log.append(reward_list)
        # print("update threshold "+str(i))
        # print(reward_list, thresholds[i])

    # print(rewards)
    # new_thresholds = [thresholds[i] - step_size if r >=
    #                   0 else thresholds[i] + step_size for i, r in enumerate(rewards)]
    new_thresholds = thresholds
    # avoid overflow
    new_thresholds = [0.999 if i >= 1.0 else i for i in new_thresholds]
    new_thresholds = [0.001 if i <= 0.0 else i for i in new_thresholds]

    # print(f'rewards: {rewards_log}')
    # print(f'thresholds: {new_thresholds}')

    # TODO: add terminal condition
    return nearest_nodes_list, simi_score_list, rewards_log, new_thresholds, stop_flag


"""
	valuator without rl module layer
	rl targets to learn new anomaly labels
"""
#without rl valuator
class Valuator(nn.Module):
    def __init__(self, embed_dim):
        """
        initialize valuator 
        """
        super(Valuator, self).__init__()

        self.embed_dim = embed_dim

        # label predictor for similarity measure
        self.label_valuator = nn.Sequential(
            nn.Linear(self.embed_dim, 8*self.embed_dim),
            nn.ReLU(True),
            nn.Linear(8*self.embed_dim, 1)
        )

    def forward(self, feat_emb):
        """
        Compute the prediction score for each node.
        """
        pred_score = self.label_valuator(feat_emb)
        return pred_score

class RL_Valu(nn.Module):
    def __init__(self, feature_dim, embed_dim,
                 ano_count, args):
        super(RL_Valu, self).__init__()

        # self.features = feat_emb
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.step_size = args.step_size
        # RL condition flag
        self.RL = True
        # initial filtering thresholds hyperparam
        ini_th = args.ini_th
        self.thresholds = [ini_th for i in range(0, ano_count)]

        # label predictor for similarity measure
        self.label_valuator = nn.Sequential(
            nn.Linear(self.embed_dim, 8*self.embed_dim),
            nn.ReLU(True),
            nn.Linear(8*self.embed_dim, 1)
        )
        # initialize the parameter logs
        self.thresholds_log = [self.thresholds]

    def forward(self, feat_emb, nodes_batch, labels_batch, anomaly_list, norm_list, args, train_flag=True):
        # prepare variables for rl
        pred_score = self.label_valuator(feat_emb)

        # the reinforcement learning module
        if self.RL and train_flag:
            nearest_nodes, simi_score, rewards, thresholds, stop_flag = RL_Module(anomaly_list, norm_list, nodes_batch, feat_emb,
                                                                                  self.thresholds, self.thresholds_log, args)
            self.thresholds = thresholds
            self.RL = stop_flag
            self.thresholds_log.append(self.thresholds)
            mask_index_list = []
            
            for i in range(0, len(self.thresholds)):
                mask_index = []
                nei_index = torch.where(torch.tensor(
                    simi_score[i]) > self.thresholds[i])
                nei_index = nei_index[0].tolist()
                for j in nei_index:
                    mask_index.append(nearest_nodes[i][j])
                mask_index_list.append(mask_index)
        
        return mask_index_list, self.thresholds, pred_score
