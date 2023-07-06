"""
created by weiyx15 @ 2019.1.4
Cora dataset interface
"""

import random
import numpy as np
from config import get_config
from utils.construct_hypergraph import edge_to_hyperedge
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    """
    Copied from gcn
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def load_citation_data(cfg):
    """
    Copied from gcn
    citeseer/cora/pubmed with gcn split
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(cfg['citation_root'], cfg['activate_dataset'], names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    ##############get cora list
    # for i in range(len(graph)):
    #     sub_list = graph[i]
    #     for e in graph[i]:
    #         edge_one = str(i)+" "+str(e)
    #         with open("cora_edge.txt", "a") as f:
    #             f.writelines(edge_one)
    #             f.write("\n")
    #random realize
    # res = []
    # while len(res) < 54:
    #     res.append(random.randint(0, 1708))
    #     res = set(res)
    #     res = list(res)
    # res.sort()
    # #print(res)
    # with open("{}/ind.cora.allx".format(cfg['citation_root']), 'rb') as d:
    #     allx = pkl.load(d, encoding='latin1')
    # with open("{}/ind.cora.ally".format(cfg['citation_root']), 'rb') as d:
    #     ally = pkl.load(d, encoding='latin1')
    # #print(ally[0])
    # x = sp.vstack((allx[res[0]], allx[res[1]])).tolil()
    # y = np.vstack((ally[res[0]], ally[res[1]]))
    # res.remove(res[0])
    # res.remove(res[0])
    # for i in res:
    #     x = sp.vstack((x, allx[i])).tolil()
    #     y = np.vstack((y, ally[i]))
    ########################
    #print(y.shape)
    #print(x[1])
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(cfg['citation_root'], cfg['activate_dataset']))
    test_idx_range = np.sort(test_idx_reorder)


    if cfg['activate_dataset'] == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    features = features.todense()

    G = nx.from_dict_of_lists(graph)
    edge_list1 = G.adjacency()
    edge_list = []
    for edge_liste in edge_list1:
        edge_list.append(list(edge_liste))   #data source format is error
    for i in range(len(edge_list)):
        edge_list[i][1] = list(edge_list[i][1])
    degree = [0] * len(edge_list)
    if cfg['add_self_loop']:
        for i in range(len(edge_list)):
            edge_list[i].append(i)
            degree[i] = len(edge_list[i][1])
    max_deg = max(degree)
    mean_deg = sum(degree) / len(degree)
    print(f'max degree: {max_deg}, mean degree:{mean_deg}')

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]     # one-hot labels
    n_sample = labels.shape[0]
    n_category = labels.shape[1]
    lbls = np.zeros((n_sample,))
    if cfg['activate_dataset'] == 'citeseer':
        # n_category += 1                                         # one-hot labels all zero: new category
        for i in range(n_sample):
            try:
                lbls[i] = np.where(labels[i]==1)[0]                     # numerical labels
            except ValueError:                              # labels[i] all zeros
                # print(i)
                lbls[i] = 0                        # new category
    else:
        for i in range(n_sample):
            lbls[i] = np.where(labels[i]==1)[0]                     # numerical labels
    # for i in range(n_sample):
    #   lbls[i] = np.where(labels[i]==1)[0]                     # numerical labels
    idx_test = test_idx_range.tolist()  #1708-2708
    idx_train = list(range(len(y)))  #140 270 540
    idx_val = list(range(500, 1500))
    # idx_val = list(range(len(y), len(y) + 500))
    # print(n_category)
    # lbs = lbls.tolist()
    # for i in range(len(lbs)):
    #     lab_one = str(i)+" "+str(int(lbs[i]))
    #     with open("cora_lables.txt", "a") as f:
    #             f.writelines(lab_one)
    #             f.write("\n")
    return features, lbls, idx_train, idx_val, idx_test, n_category, edge_list, edge_list


if __name__ == '__main__':
    cfg = get_config('config/config.yaml')
    load_citation_data(cfg)
