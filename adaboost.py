import numpy as np


def loadSimpData():
    dataMat = np.matrix([
        [168, 1],
        [175, 1],
        [165, 0],
        [155, 0],
        [160, 1],
        [162, 0],
    ])
    classLabels = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(
                    dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=400):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        aggErrors = np.multiply(np.sign(aggClassEst) !=
                                np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum()/m
        if errorRate == 0.0:
            break
    return weakClassArr


def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
    return np.sign(aggClassEst)


dataMat, classLabels = loadSimpData()
D = np.mat(np.ones((6, 1))/6)
weakClassArr = adaBoostTrainDS(dataMat, classLabels)
length = int(input('请输入身高：'))
smoke = int(input('是否抽烟：'))
test = list((length, smoke))
sex = adaClassify(test, weakClassArr)
if sex == 1:
    print('我猜你的性别是男性')
else:
    print('我猜你的性别是女性')
