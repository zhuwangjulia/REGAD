import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics import roc_auc_score

#basic GNN for GDN 
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


class GDN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GDN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)  # 2*64
        self.gc2 = GraphConvolution(2 * nhid, nhid)  # 64
        self.dropout = dropout
        self.embeddings = None
        self.Outlier_Valuator = nn.Sequential(
            nn.Linear(nhid, 512),  # same as Kaize paper
            nn.ReLU(True),
            nn.Linear(512, 1)
        )

    def forward(self, x, adj):
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout) #, training=self.training
        self.embeddings = self.gc2(x, adj)

        # Outlier_Valuator
        pred_score = self.Outlier_Valuator(self.embeddings)

        return pred_score, self.embeddings

    def get_node_reps(self):
        node_x = self.embeddings
        return node_x