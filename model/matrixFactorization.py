import numpy as np
from math import pow
from math import sqrt
import matplotlib.pyplot as plt
from  model.Mainfuction import trainData


train_data_list = []
test_data_list = []
def load_train_data(path):
    fn = open(path)
    for line in fn:
        line = line.split('::')
        data = trainData(line[0], line[1], line[2])
        train_data_list.append(data)
    return train_data_list

def load_test_data(path):
    fn = open(path)
    for line in fn:
        line = line.split('::')
        data = trainData(line[0], line[1], line[2])
        test_data_list.append(data)
    return test_data_list




def train(round, p, q, train_data_list, test_data_list):

    for i in range(round):
        '''train'''
        for trainData in train_data_list:
            KnownRatingWithBias = trainData.ratio
            PredictionRating = np.matmul(p[int(trainData.user), :], q[int(trainData.item), :])
            err = KnownRatingWithBias - PredictionRating
            tempPu = np.matmul(p[int(trainData.user), :], 1-0.0001)
            tempDeltaPu = np.matmul(p[int(trainData.user), :], err * 0.001)
            np.add(tempPu, tempDeltaPu)


        for testData in test_data_list:


if __name__=="__main__":
    train_data_list = load_train_data('../data/recsys/train.txt')
    test_data_list = load_test_data('../data/recsys/test.txt')
    N = 6000
    M = 4000
    round = 36
    f = 10
    eta = 0.001
    lamuda = 0.01
    p = np.random.rand(N, f)
    q = np.random.rand(M, f)
    train(round, p, q, train_data_list,test_data_list)


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


# matrix_factorization(p,q,f,eta,lamuda)