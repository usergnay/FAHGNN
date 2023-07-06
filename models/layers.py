import math
import copy
import torch
import time
from torch import nn
from torch.nn.parameter import Parameter
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from utils.layer_utils import sample_ids, sample_ids_v2, cos_dis


class Transform(nn.Module):
    """
    A Vertex Transformation module
    Permutation invariant transformation: (N, k, d) -> (N, k, d)
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        self.convKK = nn.Conv1d(k, k * k, dim_in, groups=k)
        self.activation = nn.Softmax(dim=-1)
        self.dp = nn.Dropout()

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, k, d)
        """
        N, k, _ = region_feats.size()  # (N, k, d)
        conved = self.convKK(region_feats)  # (N, k*k, 1)
        multiplier = conved.view(N, k, k)  # (N, k, k)
        multiplier = self.activation(multiplier)  # softmax along last dimension
        transformed_feats = torch.matmul(multiplier, region_feats)  # (N, k, d)
        return transformed_feats


class VertexConv(nn.Module):
    """
    A Vertex Convolution layer
    Transform (N, k, d) feature to (N, d) feature by transform matrix and 1-D convolution
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        self.trans = Transform(dim_in, k)                   # (N, k, d) -> (N, k, d)
        self.convK1 = nn.Conv1d(k, 1, 1)                    # (N, k, d) -> (N, 1, d)

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, d)
        """
        transformed_feats = self.trans(region_feats)
        pooled_feats = self.convK1(transformed_feats)             # (N, 1, d)
        pooled_feats = pooled_feats.squeeze(1)
        return pooled_feats


class GraphConvolution(nn.Module):
    """
    A GCN layer
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        # dim_in,
        # dim_out,
        # dropout_rate=0.5,
        # activation
        """
        super().__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=kwargs['has_bias'])
        self.dropout = nn.Dropout(p=0.5)
        self.activation = kwargs['activation']

    def _region_aggregate(self, feats, edge_dict):
        N = feats.size()[0]  #2708
        # for i in range(N):
        #     print(edge_dict[i][1])
        #     print(len(edge_dict[i][1]))
        pooled_feats = torch.stack([torch.mean(feats[edge_dict[i][1]], dim=0) for i in range(N)])

        #print(pooled_feats.shape) #2708*256
        return pooled_feats

    def forward(self, ids, feats, edge_dict, G, ite):
        """
        :param ids: compatible with `MultiClusterConvolution`
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats  # (N, d)
        x = self.dropout(self.activation(self.fc(x)))  # (N, d')
        x = self._region_aggregate(x, edge_dict)  # (N, d)
        return x


class EdgeConv(nn.Module):
    """
    A Hyperedge Convolution layer
    Using self-attention to aggregate hyperedges
    """
    def __init__(self, dim_ft, hidden):
        """
        :param dim_ft: feature dimension
        :param hidden: number of hidden layer neurons
        """
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim_ft, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, ft):
        """
        use self attention coefficient to compute weighted average on dim=-2
        :param ft (N, t, d)
        :return: y (N, d)
        """
        scores = []
        n_edges = ft.size(1)
        #print(n_edges)
        for i in range(n_edges):
            scores.append(self.fc(ft[:, i]))
        scores = torch.softmax(torch.stack(scores, 1), 1)
        #print(len(scores))
        #print((scores * ft).sum(1).shape)
        return (scores * ft).sum(1)


class DHGLayer(GraphConvolution):
    """
    A Dynamic Hypergraph Convolution Layer
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ks = kwargs['structured_neighbor'] # number of sampled nodes in graph adjacency
        self.n_cluster = kwargs['n_cluster']              # number of clusters
        self.n_center = kwargs['n_center']                # a node has #n_center adjacent clusters
        self.kn = kwargs['nearest_neighbor']    # number of the 'k' in k-NN
        self.kc = kwargs['cluster_neighbor']    # number of sampled nodes in a adjacent k-means cluster
        self.wu_knn=kwargs['wu_knn']
        self.wu_kmeans=kwargs['wu_kmeans']
        self.wu_struct=kwargs['wu_struct']
        self.alpha=kwargs['alpha']
        self.vc_sn = VertexConv(self.dim_in, self.ks+self.kn)    # structured trans
        self.vc_s = VertexConv(self.dim_in, self.ks)    # structured trans
        self.vc_n = VertexConv(self.dim_in, self.kn)    # nearest trans
        self.vc_c = VertexConv(self.dim_in, self.kc)   # k-means cluster trans
        self.ec = EdgeConv(self.dim_in, hidden=self.dim_in//4)
        self.kmeans = None
        self.structure = None

    def _vertex_conv(self, func, x):
        return func(x)

    def _structure_select(self, ids, feats, edge_dict):
        """
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :param edge_dict: torch.LongTensor
        :return: mapped graph neighbors
        """
        if self.structure is None:
            _N = feats.size(0)
            wait = np.loadtxt('att.txt')
            lst = []
            for i in range(wait.shape[0]):
                d = []
                w = []
                dw = []
                for j in range(wait.shape[1]):
                    if wait[i][j] != 0:
                        d.append(j)
                        w.append(wait[i][j])
                dw.append(d)
                dw.append(w)
                lst.append(dw)
            # idx = torch.LongTensor([sample_ids(edge_dict[i][1], self.ks) for i in range(_N)])    # (_N, ks)
            # idx = torch.LongTensor([sample_ids(lst[i][0], self.ks) for i in range(_N)])    # (_N, ks)
            sampled_ids = []
            two_edge = []
            for i in range(_N):
                one_edge = edge_dict[i][1]
                for e in one_edge:
                    for d in edge_dict[e][1]:
                        if d != i:
                            two_edge.append(d)
                two_edge = one_edge + two_edge
                l = []
                for k in range(len(lst[i][1])):
                    if lst[i][1][k]>self.alpha:
                        l.append(lst[i][0][k])
                re = list(set(two_edge) & set(l))
                # re = list(set(one_edge)& set(re))   #best result
                if len(re) == 0:
                    re = one_edge
                sampled_ids.append(sample_ids(re, self.ks))

            idx = torch.LongTensor(sampled_ids)
            # idx = torch.LongTensor([sample_ids(edge_dict[i][1], self.ks) for i in range(_N)])    # (_N, ks)

            self.structure = idx
        else:
            idx = self.structure

        #print(idx.shape)

        idx = idx[ids]
        #print(idx.shape)
        N = idx.size(0)
        d = feats.size(1)
        #print(d)
        region_feats = feats[idx.view(-1)].view(N, self.ks, d)          # (N, ks, d)
        return region_feats

    def _nearest_select(self, ids, feats):
        """
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :return: mapped nearest neighbors
        """
        dis = cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=1)
        #print(idx)
        idx = idx[ids]
        #print(idx)
        N = len(idx)
        d = feats.size(1)
        # fts_knn = feats[idx]
        # lb = torch.Tensor(lbls).squeeze().long().cuda()
        # mod_knn = ATTEN(nfeat=fts_knn.shape[1],
        #                 nhid=64,
        #                 nclass=n_category,
        #                 dropout=0.6,
        #                 nheads=8,
        #                 alpha=0.2)
        # optimizer = optim.Adam(mod_knn.parameters(),
        #                        lr=0.005,
        #                        weight_decay=5e-4)
        # mod_knn.cuda()
        # fts_knn = fts_knn.cuda()
        #
        # # fts, G, lb = Variable(fts), Variable(G), Variable(lb)
        # # for  1000
        # for i in range(200):  # 768
        #     mod_knn.cuda()
        #     mod_knn.train()
        #     optimizer.zero_grad()
        #     output = mod_knn(fts_knn, H)
        #     output = output.cuda()
        #     lb = lb.cuda()
        #     loss_train = F.nll_loss(output, lb)
        #     loss_train.backward()  # ËðÊ§µüŽú
        #     optimizer.step()
        #     print('Epoch: {:04d}'.format(i + 1),
        #           'loss_train: {:.4f}'.format(loss_train.data.item())
        #           )
        # print(output)
        nearest_feature = feats[idx.view(-1)].view(N, self.kn, d)         # (N, kn, d)
        return nearest_feature

    def _cluster_select(self, ids, feats):
        """
        compute k-means centers and cluster labels of each node
        return top #n_cluster nearest cluster transformed features
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :return: top #n_cluster nearest cluster mapped features
        """
        if self.kmeans is None:
            _N = feats.size(0)
            np_feats = feats.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(np_feats)
            centers = kmeans.cluster_centers_
            #print(self.n_cluster)
            #print(centers.shape)
            dis = euclidean_distances(np_feats, centers)  #2708*400
            _, cluster_center_dict = torch.topk(torch.Tensor(dis), self.n_center, largest=False)
            cluster_center_dict = cluster_center_dict.numpy() #2708*1  zuida wei 399
            point_labels = kmeans.labels_  # equal  cluster_center_dict
            point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(self.n_cluster)]
            idx = torch.LongTensor([[sample_ids_v2(point_in_which_cluster[cluster_center_dict[point][i]], self.kc)   
                        for i in range(self.n_center)] for point in range(_N)])    # (_N, n_center, kc)
            self.kmeans = idx
        else:
            idx = self.kmeans
        
        idx = idx[ids]
        N = idx.size(0)
        d = feats.size(1)
        cluster_feats = feats[idx.view(-1)].view(N, self.n_center, self.kc, d)

        return cluster_feats                    # (N, n_center, kc, d)

    def _edge_conv(self, x):
        return self.ec(x)

    def _fc(self, x):
        return self.activation(self.fc(self.dropout(x)))

    def forward(self, ids, feats, edge_dict, G, ite):
        hyperedges = []
        if ite >= self.wu_kmeans:
            c_feat = self._cluster_select(ids, feats)
            for c_idx in range(c_feat.size(1)):
                xc = self._vertex_conv(self.vc_c, c_feat[:, c_idx, :, :])
                xc  = xc.view(len(ids), 1, feats.size(1))               # (N, 1, d)
                hyperedges.append(xc)
        if ite >= self.wu_knn:
            n_feat = self._nearest_select(ids, feats)
            xn = self._vertex_conv(self.vc_n, n_feat)
            xn  = xn.view(len(ids), 1, feats.size(1))                   # (N, 1, d)
            hyperedges.append(xn)
            #print(xn.shape)
        if ite >= self.wu_struct:
            s_feat = self._structure_select(ids, feats, edge_dict)
            xs = self._vertex_conv(self.vc_s, s_feat)
            xs  = xs.view(len(ids), 1, feats.size(1))                   # (N, 1, d)
            hyperedges.append(xs)
        #print(len(hyperedges))
        x = torch.cat(hyperedges, dim=1)
        #print(x.shape)
        x = self._edge_conv(x)                                          # (N, d)
        x = self._fc(x)                                                 # (N, d')
        return x


class HGNN_conv(nn.Module):
    """
    A HGNN layer
    """
    def __init__(self, **kwargs):
        super(HGNN_conv, self).__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=kwargs['has_bias'])
        self.dropout = nn.Dropout(p=0.5)
        self.activation = kwargs['activation']


    def forward(self, ids, feats, edge_dict, G, ite):
        x = feats
        x = self.activation(self.fc(x))
        x = G.matmul(x)
        x = self.dropout(x)
        return x
