#!/user/bin/python3
# -*- coding:utf-8 -*-
#@Date      :2018/10/14  15:40
#@Author    :Syler(sylerxin@gmail.com)

import numpy as np
import scipy.optimize as op
from decimal import Decimal

def loadDataSet(filename, strName):
    data = np.loadtxt(filename, delimiter=',', usecols=np.arange(0,4))
    label = np.loadtxt(filename, delimiter=',', usecols=-1, dtype=str)
    for i in range(len(strName)):
        label[np.where(label == strName[i])] = i
    label = np.array(list(map(np.float64, label)))
    return data, label

def sigmoid(data):
    #此处可以尝试实用Decimal进行更加精准的浮点数运算，因为这里的计算有损失值
    return  1.0 / (1.0 + np.exp(-1 * data))

def computeCost(theta, dataSet, label, lam):
    entries = len(label)
    h = sigmoid(dataSet.dot(theta))
    cost = (-1 * np.sum(label * np.log(h) + (1 - label) * np.log(1 - h))) / entries \
           + lam / (2 * entries) * np.sum(theta**2)
    return cost

def computeGrad(theta, dataSet, label, lam):
    entries = len(label)
    h = sigmoid(dataSet.dot(theta))
    grad = np.zeros(np.size(theta))

    grad[0] = dataSet[:,0].dot(h - label) / entries

    grad[1:] = (dataSet[:,1:].T.dot(h - label) / entries \
               + lam / entries * theta[1:]).ravel()
    return grad

def chooseBestValue(init_theta, data, label, lam):
    result = op.minimize(fun=computeCost, x0=init_theta, args=(data, label, lam),
                        jac=computeGrad, method='Newton-CG')
    return result

def mapFeature(X1, X2, X3, X4):
    out = np.ones((len(X1),1))
    for x_1 in range(10):
        for x_2 in range(10):
            for x_3 in range(10):
                for x_4 in range(10):
                    tmp = X1**x_1 * X2**x_2 * X3**x_3 * X4**x_4
                    out = np.column_stack((out, tmp))
    return out

def predict(testData, theta1, theta2, theta3, label):
    # testData = mapFeature(testData[:, 0], testData[:, 1], testData[:, 2], testData[:, 3])

    p1 = sigmoid(np.dot(testData, theta1))#0->1，划分1，2
    p2 = sigmoid(np.dot(testData, theta2))#1->0，划分0，2
    p3 = sigmoid(np.dot(testData, theta3))#2->1，划分1，0

    vote = np.zeros((len(label), 1))
    for i in range(len(label)):
        index = judgeMax(p1[i], p2[i], p3[i])
        if index == 0:
            if returnNum(p1[i]):
                vote[i] = 2
            else:
                vote[i] = 1
        elif index == 1:
            if returnNum(p2[i]):
                vote[i] = 2
            else:
                vote[i] = 0
        else:
            if returnNum(p3[i]):
                vote[i] = 1
            else:
                vote[i] = 0

    error = 0
    for i in range(len(label)):
        if label[i] != vote[i]:
            error += 1
    return 1 - error/len(label)

def judgeMax(i,j,k):
    if i >= j:
        if i >= k:
            max_num = 0
        else:
            max_num = 2
    else:
        if j >= k:
            max_num = 1
        else:
            max_num = 2
    return max_num

def returnNum(i):
    if i >= 0.5:
        return 1
    else:
        return 0

def normData(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

if __name__ == '__main__':
    strName = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    data, label = loadDataSet('iris.data', strName)

    data = normData(data)
    # data = mapFeature(data[:, 0], data[:, 1], data[:, 2], data[:, 3])

    row, col = data.shape
    init_theta = np.zeros((len(strName), col))
    lam = 1

    for i in range(len(strName)):
        tmp = label.copy()
        tmpInd = np.where(tmp == i)
        if i != len(strName) - 1:
            for j in range(len(tmpInd)):
                tmp[tmpInd[j]] = i + 1
        else:
            for j in range(len(tmpInd)):
                tmp[tmpInd[j]] = 0

        locals()['label'+str(i + 1)] = tmp

    result1 = chooseBestValue(init_theta[0], data, label1, lam)#将0转换为1，划分1和2
    result2 = chooseBestValue(init_theta[1], data, label2, lam)#将1转换为0，划分0和2
    result3 = chooseBestValue(init_theta[2], data, label3, lam)#将2转换为1，划分1和0

    data, label = loadDataSet('bezdekIris.data', strName)
    data = normData(data)
    p = predict(data, result1.x, result2.x, result3.x, label)
    print("测试数据集，归类正确率为:{0}".format(p))