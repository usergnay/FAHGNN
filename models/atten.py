import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ATTEN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,flag = 0):
        """Dense version of GAT."""
        super(ATTEN, self).__init__()
        self.dropout = dropout
        self.flag = flag
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False,flag = self.flag)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # x为2708*64的矩阵
        x = self.out_att(x, adj)
        x = F.elu(x)
        # x为2708*2708的权重矩阵
        return F.log_softmax(x, dim=1)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, flag=0):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.flag = flag

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):   #adj为2708*2708的权重矩阵   初始为一阶邻的有权重  其余权重为0   h为特征矩阵
        Wh = torch.mm(h, self.W) # Wh = 2708*1433  x  1433*64
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(e > 0, e, zero_vec)
        if self.flag == 1:
            att = torch.where(e > 0, e, zero_vec)
            att = F.softmax(att,dim=1)
            att = att.cpu().detach().numpy()
            np.savetxt('att.txt', att, fmt='%f')
            # print(att.shape) #2708*2708
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)   #第一次特征提取到64的时候带着这个 第二次的时候去掉
        # if self.flag == 1:
        #     print(attention)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):  #提取与超边的中心顶点相同类别的特征去更新权重矩阵
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)