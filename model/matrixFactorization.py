import numpy as np
from math import pow
from math import sqrt
import matplotlib.pyplot as plt

class Train_data():
    user = []
    iterm = []
    rating = []

u_max = 0
i_max = 0
train_data_list = []
def load_data(path):
    fn = open(path)
    for line in fn:
        line = line.split('::')
        # line = [float(x)for x in line]
        # line = [int(x)for x in line]
        Train_data.user =line[0]
        Train_data.iterm =line[1]
        Train_data.rating = line[2]
        train_data_list.append(Train_data)
        # global u_max,i_max
        # if(u_max<line[0]):                       # 计算最大行数和列数
        #     u_max = line[0]
        # if(i_max<line[1]):
        #     i_max = line[1]
    print("load  is done")

load_data('../data/recsys/train.txt')



# def find_rating(u,i):
#     for j in range(len(Train_data.rating)):
#         if(Train_data.user[j] == u and Train_data.iterm == i):
#             break
#     return Train_data.rating[j]
#
#
# def matrix_factorization(p,q,f,eta,lamuda):
#     step = 0
#     while 1:
#         step += 1
#         rmse = 0
#         e = 0
#         sum = 0
#
#         for u in range(N):
#             print(u)
#             for i in range(M):
#                 rui = 1
#                 deta = rui - np.dot(p[u, :], q[:, i])
#                 e = e + pow(rui - np.dot(p[u, :], q[:, i]), 2)  # e为目标函数
#                 for k in range(f):  # 计算rmse误差
#                     rmse = rmse + deta * deta
#                     e = e + (lamuda) * (pow(p[u][k], 2) + pow(q[k][i], 2))
#                     sum = sum + 1
#
#
#         for u in range(N):
#             for i in range(M):
#                 rui = find_rating(u,i)
#                 print("rui", rui)
#                 deta = rui - np.dot(p[u,:],q[:,i])               # deta为实体rui上的预测误差
#                 print("deta", deta)
#                 for k in range(f):
#                     p[u][k] = p[u][k] + eta * (deta * q[k][i] - lamuda * p[u][k])
#                     q[k][i] = q[k][i] + eta * (deta * p[u][k] - lamuda * q[k][i])0
#
#         rmse = sqrt(rmse/sum)
#         print('迭代次数： ',step,' RMSE：',rmse)


N = u_max
M = i_max
f = 2
eta = 0.001
lamuda = 0.01
p = np.random.rand(N,f)
q = np.random.rand(f,M)

# matrix_factorization(p,q,f,eta,lamuda)