import numpy as np
from numpy import mat
import numpy
lam = 6000
train = 400
val = 1400
def load_news_data():
    x = np.load('x.npy')
    y = np.load('y.npy')   #y.tolist().count(0): 4605 3519 2656 5462   total:16242
    np.random.seed(1000)
    np.random.shuffle(y)
    x.tolist()
    y.tolist()
    ###########   suffer
    x_suffer = []
    for i in y:
        x_suffer.append(x[i])
    y = np.array(y)
    x = np.array(x_suffer)
    x = mat(x)
    edge = np.load('edge.npy')
    edge = edge.tolist()
    edge_all = []
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
    count = 0
    for i in range(len(left)):  #len(edge[0])
        if right[i]<lam:
            edge_all[left[i]][1].append(right[i])
        else:
            edge_all[left[i]][1].append(left[i])
        # if left[i]<lam:
        #     count+=1   #6000 23934

    features = x[0:lam]
    lbls = y[0:lam]
    idx_train = list(range(0,train))
    idx_val = list(range(train, val))
    idx_test = list(range(lam-1000,lam))
    n_category = 4
    edge_list = edge_all[0:lam]

    # for i in range(len(edge_list)):
    #     sub_list = edge_list[i][1]
    #     # print(sub_list)
    #     for e in sub_list:
    #         edge_one = str(i)+" "+str(e)
    #         # print(edge_one)
    #         with open("newsgroup_edge.txt", "a") as f:
    #             f.writelines(edge_one)
    #             f.write("\n")
    #
    # idx_val = list(range(len(y), len(y) + 500))
    # print(n_category)
    # lbs = lbls.tolist()
    # for i in range(len(lbs)):
    #     lab_one = str(i)+" "+str(int(lbs[i]))
    #     with open("newsgroup_lables.txt", "a") as f:
    #             f.writelines(lab_one)
    #             f.write("\n")
    return features, lbls, idx_train, idx_val, idx_test, n_category, edge_list, edge_list

load_news_data()


