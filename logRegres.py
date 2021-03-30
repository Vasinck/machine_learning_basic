import numpy as np
import random
import matplotlib.pyplot as plt
#MATNAME.transpose()    矩阵转置
#np.mat()   转化为矩阵
#回归系数计算公式：新的回归系数=之前的回归系数 + 设定步长 * 数据矩阵 * 误差向量

def LoadDataSet():
    data_mat = []
    label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        data_mat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        label_mat.append(int(lineArr[2]))
    return data_mat,label_mat

def Sigmoid(inX):
    return 1.0 / (1+np.exp(-inX))

def GradAscent(data_mat_in,class_labels):
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()
    m,n = np.shape(data_matrix)
    ALPHA = 0.001   #目标移动的步长
    MAXCYCLES = 500 #迭代次数
    weights = np.ones((n,1))    #回归系数
    for k in range(MAXCYCLES):
        h = Sigmoid(data_matrix*weights)
        error = label_mat - h
        weights = weights + ALPHA * data_matrix.transpose() * error
    return weights

def stocGradAscent0(dataMatrix,classLabels,numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones((1,n))[0]
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0+j+i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = Sigmoid(np.sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):
    dataMat,lableMat = LoadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(lableMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]     #最佳拟合线
    ax.plot(x,y)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('基于动态alpha的logistic线性回归最佳拟合线')
    plt.rcParams['axes.unicode_minus']=False
    plt.legend(loc=0,fontsize=15,title='红：类别1\n绿：类别2')
    plt.show()

def classifyVector(inX,weights):
    prob = Sigmoid(np.sum(inX*weights))
    if prob > 0.5:
        return 1
    else:
        return 0

def colicTest():
    frTrain = open(r'D:\python源程序\machinelearninginaction-master\Ch05\horseColicTraining.txt')
    frTest = open(r'D:\python源程序\machinelearninginaction-master\Ch05\horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent0(np.array(trainingSet),trainingLabels,numIter=500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print('对这个测试，错误率为：{:.2}'.format(errorRate))
    return errorRate

def multiTest():
    numTests = 3
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('测试了{}份数据，平均错误率为：{:.2}'.format(numTests,errorSum/float(numTests)))

#data_mat,label_mat = LoadDataSet()
#weights = stocGradAscent0(np.array(data_mat),label_mat)
#plotBestFit(weights)
#weights = GradAscent(data_mat,label_mat)
#plotBestFit(weights.getA())
multiTest()