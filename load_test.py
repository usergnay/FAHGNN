# import numpy as np
#
# wait = np.loadtxt('att.txt')
# lst = []
# for i in range(wait.shape[0]):
#     d = []
#     w = []
#     dw = []
#     for j in range(wait.shape[1]):
#         if wait[i][j]!=0:
#             d.append(j)
#             w.append(wait[i][j])
#     dw.append(d)
#     dw.append(w)
#     lst.append(dw)
#
#
# #
# # wait = np.argwhere(wait)  #99596,2
# print(lst[2][0])
import tensorflow
print(tensorflow.__version__)