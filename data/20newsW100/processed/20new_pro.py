import numpy as np
from numpy import mat
def load_news_data():
    x = np.load('x.npy')
    x = mat(x)
    y = np.load('y.npy')
    edge = np.load('edge.npy')
    edge = edge.tolist()
    edge_all = []
    edge_one = []
    left = edge[0][0:65451] #len 130902 65450  max16242
    right = edge[1][65451:]
    # for i in range(len(right)):
    #     right[i]-=16242
    for i in range(len(y)): #len(y)
        edge_tem = []
        edge_empty = []
        edge_tem.append(i)
        edge_tem.append(edge_empty)
        edge_tem.append(i)
        edge_all.append(edge_tem)
    # print(edge_all) #[[0, [], 0], [1, [], 1], [2, [], 2]]
    # print(len(edge_all))
    # print(right)
    count=0 # 5000  21050
    for i in range(len(left)):  #len(edge[0])
        edge_all[left[i]][1].append(right[i])
        if left[i]<5000:
            count+=1

    features = x
    lbls = y
    idx_train = list(range(0,500))
    idx_val = list(range(500, 1500))
    idx_test = list(range(1500,2500))
    n_category = 4
    edge_list = edge_all
    return x, lbls, idx_train, idx_val, idx_test, n_category, edge_list, edge_list

load_news_data()

