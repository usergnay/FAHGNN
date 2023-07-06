import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import numpy as np
def read_node_label(filename, skip_head=False):
    with open(filename, 'r') as fin:
        X = []
        Y = []
        while 1:
            if skip_head:
                fin.readline()
            l = fin.readline()
            if l == '':
                break
            vec = l.strip().split(' ')
            X.append(vec[0])
            Y.append(vec[1:])
        fin.close()
        return X, Y

def plot_embeddings(embeddings, ):
    X, Y = read_node_label('cora_labels.txt')

    emb_list = []
    for k in range(2708):
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()
X, Y = read_node_label('cora_labels.txt')
# print(X)
# print(Y)