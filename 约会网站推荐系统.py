from numpy import *
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# 取得训练样本


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 分类器分类


def classifyO(inX, dataSet, labels, k):
    '''
    取得分类的结果;参数说明：

    inX：输入的一维二元素的列表或数组

    dataSet:调用createDataSet所返回的group

    labels:调用createDataSet所返回的labels

    k:选取临近点的个数

    shape:返回矩阵或者列表的维数，其中shape[0]返回的是第一维

    tile:返回一个列表，tile(0,5) ==> [0,0,0,0,0],tile(0,[3,2]) ==> [[0,0],[0,0],[0,0]]

    sum(axis=1)：矩阵行行相加

    X.argsort()：返回一个列表，元素依次为X中所有元素从小到大的索引

    operator.itemgetter(1):指定待排序元素的哪一项进行排序
    '''
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 文件操作 取得训练集


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # 取得文件的行数
    returnMat = zeros((numberOfLines, 3))

    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # 矩阵多维切片
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1

    return returnMat, classLabelVector

# 用散点图显示数据


def showdatas(datingDataMat, datingLabels):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False,
                            sharey=False, figsize=(12, 8))
    numberOfLabels = len(datingLabels)
    LabelsColors = []

    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        elif i == 2:
            LabelsColors.append('orange')
        elif i == 3:
            LabelsColors.append('red')

    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:,
                                                             1], color=LabelsColors, s=15, alpha=.5)
    axs0_title_text = axs[0][0].set_title(
        u'每年步行获得的里程数与玩视屏游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年步行获得的里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(
        u'玩视屏游戏所消耗时间占比', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:,
                                                             2], color=LabelsColors, s=15, alpha=.5)
    axs1_title_text = axs[0][1].set_title(
        u'每年步行获得的里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年步行获得的里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(
        u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:,
                                                             2], color=LabelsColors, s=15, alpha=.5)
    axs2_title_text = axs[1][0].set_title(
        u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(
        u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(
        u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='hate')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='like')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='love')

    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses], loc='best')
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses], loc='best')
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses], loc='best')

    plt.show()

# 归一化


def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 取得矩阵全部行中的的最小行
    maxVals = dataSet.max(0)  # 取得矩阵全部行中的的最大行
    ranges = maxVals - minVals
    #normDataSet = zeros(shape(dataSet))
    datasize = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (datasize, 1))
    normDataSet = normDataSet / tile(ranges, (datasize, 1))

    return normDataSet, ranges, minVals

# 测试函数


def datingClassTest():
    hoRatio = 0.1  # 全部样本集合的10%
    filename = r'D:\python源程序\test.txt'
    datingDataMat, datingLabels = file2matrix(filename)

    normMat, ranges, minVals = autoNorm(datingDataMat)
    normMatSize = normMat.shape[0]
    numTestVeecs = int(normMatSize*hoRatio)
    errorCount = 0.0  # 错误率

    for each in range(numTestVeecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifilerResult = classifyO(normMat[each, :], normMat[numTestVeecs:normMatSize],
                                      datingLabels[numTestVeecs:normMatSize], 4)
        print('分类结果：{}\t真实类别：{}'.
              format(classifilerResult, datingLabels[each]))

        if classifilerResult != datingLabels[each]:
            errorCount += 1.0
    print('错误率为', errorCount/float(numTestVeecs)*100, '%')

# 实例


def classifyPerson():
    resultList = ['讨厌', '有些喜欢', '喜欢']
    filename = r'D:\python源程序\test.txt'

    ffMiles = float(input('每年步行获得的里程数：'))
    precentTats = float(input('玩视频游戏所耗时间百分比：'))
    iceCream = float(input('每周消耗的冰淇淋公升数：'))

    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVales = autoNorm(datingDataMat)
    inArr = array([ffMiles, precentTats, iceCream])
    norminArr = (inArr-minVales) / ranges  # 归一化
    classifierResult = classifyO(norminArr, normMat, datingLabels, 3)

    print('经过系统分析，你可能' + resultList[classifierResult-1] + '这个人。')
    input('请按任意键查看散点图...')
    showdatas(datingDataMat, datingLabels)


if __name__ == '__main__':
    # datingClassTest()
    classifyPerson()
