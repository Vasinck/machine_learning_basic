import operator
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from os import listdir
import imgtoarray


def img2vector(filename):
    f = open(filename)
    returnVector = np.zeros([1, 1024])

    for i in range(32):
        lineStr = f.readline()
        for j in range(32):
            returnVector[0, i*32+j] = int(lineStr[j])

    return returnVector


def handwritingClassTest():
    training_paths = r'D:\python源程序\trainingDigits'
    test_paths = r'D:\python源程序\手写文件测试集' + '\\kNN_number.txt'
    hwLabels = []
    trainingFileList = listdir(training_paths)
    #testFileList = listdir(test_paths)
    #errorCount = 0.0

    filelen = len(trainingFileList)
    training_Mat = np.zeros([filelen, 1024])

    for i in range(filelen):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        training_Mat[i, ] = img2vector(
            '{}\\{}'.format(training_paths, fileNameStr))

    neigh = KNN(3, algorithm='auto')
    neigh.fit(training_Mat, hwLabels)

    # 实例
    #s = input('请输入要测试的手写数字文件：')
    #testname = '{}\\{}'.format(test_paths,s)

    #vectorUnderTest = img2vector(testname)
    vectorUnderTest = img2vector(test_paths)
    classifierResult = neigh.predict(vectorUnderTest)
    print('这个手写数字是：', classifierResult[0])

    # 测试错误率
'''
    filelenTest = len(testFileList)
    for i in range(filelenTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('{}\\{}'.format(test_paths,fileNameStr))

        classifierResult = neigh.predict(vectorUnderTest) 
        print('分类返回的结果是：{}\t真实结果为：{}'.format(classifierResult,classNumber))

        if classifierResult != classNumber:
            errorCount += 1.0
    
    print('在数据量 {} 中，一共错误了 {} 个，错误率为 {:.2f}% 。'.format(filelenTest,int(errorCount),errorCount/filelenTest * 100))
'''
imgtoarray.fileintoarray()
handwritingClassTest()
