#!/user/bin/python3
# -*- coding:utf-8 -*-
#@Date      :2018/10/14  15:40
#@Author    :Syler(sylerxin@gmail.com)

import numpy as np
import scipy.optimize as op
from sklearn.preprocessing import PolynomialFeatures
from decimal import Decimal

def loadDataSet(filename, strName):
    data = np.loadtxt(filename, delimiter=',', usecols=np.arange(0, 4))
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
    cost = -(label.T.dot(np.log(h)) + (1 - label).T.dot(np.log(1 - h))) / entries +\
           np.sum(np.square(theta)) * lam / (2 * entries)
    if np.isnan(cost):
        return np.inf
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
    '''
    在这里，使用默认的梯度下降算法进行优化并不能取得较好的模型，因此我尝试了每一个
    优化算法，发现L-BFGS-B算法达到最优的情况下效果并不是最好、
                TNC不是最优，但是最能拟合数据。两个算法都使得模型的性能提升整整一倍多。
                SLSQP能运行，但是不知道为什么效果很差。
    '''
    result = op.minimize(fun=computeCost, x0=init_theta, args=(data, label, lam),
                        jac=computeGrad, method='TNC')#L-BFGS-B\TNC\SLSQP
    return result

def simpleFe(x, degree):
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(x)

def predict(testData, theta1, theta2, theta3, label, degree):
    testData = simpleFe(testData, degree)

    p1 = sigmoid(np.dot(testData, theta1))#0->1，划分1, 2
    p2 = sigmoid(np.dot(testData, theta2))#1->2，划分2, 0
    p3 = sigmoid(np.dot(testData, theta3))#2->0，划分0, 1

    allP = np.column_stack((p1,p2,p3))
    vote = np.zeros((len(label), 1))

    vote[np.where(allP.argmax(axis=1) == 0)] = 2
    vote[np.where(allP.argmax(axis=1) == 1)] = 0
    vote[np.where(allP.argmax(axis=1) == 2)] = 1

    error = 0.0
    for i in range(len(label)):
        if label[i] != vote[i]:
            error += 1
    return 1 - error/len(label)

def normData(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

if __name__ == '__main__':
    strName = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    fileName = "iris.data"
    data, label = loadDataSet(fileName, strName)

    data = normData(data)
    degree = 6
    data = simpleFe(data, degree)

    row, col = data.shape
    init_theta = np.zeros((len(strName), col))
    lam = 0.01

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
    result2 = chooseBestValue(init_theta[1], data, label2, lam)#将1转换为2，划分0和2
    result3 = chooseBestValue(init_theta[2], data, label3, lam)#将2转换为0，划分1和0

    data, label = loadDataSet(fileName, strName)
    data = normData(data)

    p = predict(data, result1.x, result2.x, result3.x, label, degree)
    print("测试数据集，归类正确率为:{0}".format(p))