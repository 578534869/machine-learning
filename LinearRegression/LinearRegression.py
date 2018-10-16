#!/user/bin/python3
# -*- coding:utf-8 -*-
#@Date      :2018/10/13  21:16
#@Author    :Syler(sylerxin@gmail.com)
import numpy as np
import matplotlib.pyplot as plt
import math

def loadDataSet(filename):
    data = np.loadtxt(filename, skiprows=1, delimiter=";", usecols=np.arange(0,11))
    label = np.loadtxt(filename, skiprows=1, delimiter=";", usecols=-1)
    return data, label

def normalFormal(dataSet):
    mean = np.mean(dataSet, axis=0)#计算每一列的均值
    std = np.std(dataSet, axis=0)#计算标准差
    entries = len(dataSet)
    tmpMean = np.tile(mean, (entries,1))
    data_norm = (dataSet - tmpMean) / std
    return data_norm, mean, std

def computeCost(dataSet, label, theta):
    h = dataSet.dot(theta)
    cost = ((h - label)**2).sum() / (2 * len(label))
    return cost

def gradientDescent(dataSet, label, alpha, theta, num_iters):
    entries = len(label)
    cost_history = []
    for i in range(num_iters):
        h = dataSet.dot(theta)
        tmp = h.T - label
        theta = theta - alpha / entries * dataSet.T.dot(tmp.T)
        tmpCost = computeCost(dataSet, label, theta)
        cost_history.append(tmpCost)
        print("正在第{0}次迭代".format(i))
    new_theta = theta
    return new_theta, np.array(cost_history)

def plotCostChange(num_iters, J_cost):
    num_times = np.arange(1, num_iters+1)
    J_vals = J_cost[:]
    fig, ax = plt.subplots()
    ax.plot(num_times, J_vals)

    ax.set(xlabel='Number of iters', ylabel='Cost',
           title='迭代曲线图')
    ax.grid()

    plt.show()
    return None

def testDataNormal(testData, mean, std):
    resData = (testData - np.tile(mean,1)) / np.tile(std,1)
    return resData

if __name__ == '__main__':
    data, label = loadDataSet('winequality-red.csv')
    data, mean, std = normalFormal(data)

    tmp = np.ones((len(label),1))
    data = np.hstack((tmp,data))

    theta = np.zeros((12,1))
    cost = computeCost(data, label, theta)
    print("未优化前: " + str(cost))
    print("优化中...")

    num_iters = 150
    alpha = 0.03

    theta, J_cost = gradientDescent(data, label, alpha, theta, num_iters)
    plotCostChange(num_iters, J_cost)

    testData = np.array([8,0.5,0,2.5,0.1,15.8,46.46,1,3.3,0.66,10.4])
    normData = testDataNormal(testData, mean, std)
    normData = np.concatenate((np.ones(1), normData), axis=0)
    res = normData.dot(theta)
    print(math.floor(res))