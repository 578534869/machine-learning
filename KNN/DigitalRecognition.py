#!/user/bin/python3
# -*- coding:utf-8 -*-
#@Date      :2018/6/29  19:35
#@Author    :Syler(sylerxin@gmail.com)
import csv
from numpy import *
import operator

# 核心代码
def k_NN(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5

    sortDisn = argsort(distances)

    # print("sortDisn shape: ",sortDisn.shape)
    # print("labels shape:",labels.shape)

    classCount = {}
    for i in range(k):
        # print(sortDisn[i])
        # print(type(sortDisn[i]))

        vote = labels[sortDisn[i]]

        # print("before :",type(vote))
        vote = ''.join(map(str, vote))
        # print("after :", type(vote))

        classCount[vote] = classCount.get(vote, 0) + 1

    sortedD = sorted(classCount.items(), key=operator.itemgetter(1),
                     reverse=True)
    return sortedD[0][0]

#读取Train数据
def loadTrainData():
    filename = 'train.csv'
    with open(filename, 'r') as f_obj:
        f = [x for x in csv.reader(f_obj)]
        f.remove(f[0])
        f = array(f)
        labels = f[:,0]
        datas = f[:,1:]

        # print(shape(labels))

        return normaling(toInt(datas)), toInt(labels)

#读取Test数据
def loadTestData():
    filename = 'test.csv'
    with open(filename, 'r') as f_obj:
        f = [x for x in csv.reader(f_obj)]
        f.remove(f[0])
        f = array(f)

        return normaling(toInt(f))

#归一化数据
def normaling(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    m = dataSet.shape[0]

    denominator = tile(ranges, (m, 1))
    molecular = dataSet - tile(minVals, (m, 1))

    normData = molecular / denominator

    return normData

#字符串数组转换整数
def toInt(array):
    array = mat(array)
    m, n =shape(array)
    newArray = zeros((m, n))
    for i in range(m):
        for j in range(n):
            newArray[i,j] = int(array[i,j])
    return newArray

#保存结果
def saveResult(res):
    with open('res.csv', 'w', newline='') as fw:
        writer = csv.writer(fw)
        writer.writerows(res)

    # 下面这部分代码有问题，需要注意并修改，先提交上面的代码看看是否能得到不错的名次呢?

    '''
    第一次参加 kaggle 1817 66% 没有使用归一化。k值取4
    第二次参加 kaggle 1817 
    '''

    # resu = tile(zeros(shape(res)+1), 2)
    # resu[0][0] = 'ImageId'
    # resu[0][1] = 'Label'
    # resu[1:,1] = res[:,0]
    # for i in range(len(res)):
    #     resu[i+1,0] = i+1
    # with open('resu.csv', 'w', newline='') as fwb:
    #     writer2 = csv.writer(fwb)
    #     writer2.writerows(resu)

if __name__ == '__main__':
    dataSet, labels = loadTrainData()
    testSet = loadTestData()
    row = testSet.shape[0]

    # print("dataSet Shape:",dataSet.shape)
    # print("labels Shape before",shape(labels))
    labels = labels.reshape(labels.shape[1],1)
    # print("labels Shape after reshape ", shape(labels))
    # print("testSet Shape",testSet.shape)

    resList = []
    for i in range(row):
        res = k_NN(testSet[i], dataSet, labels, 4)
        resList.append(res)
        print(i)
    saveResult(resList)