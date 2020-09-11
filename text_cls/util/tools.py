import random

"""
01.随机的shuffle
02.按照 trainNum:valNum:testNum 的比例划分数据集
"""
def shuffleAndsplit(data,trainNum,valNum,testNum):
    random.shuffle(data) # shuffle data
    total = trainNum + valNum + testNum
    size = len(data)
    trN = int(trainNum/total *size)
    vaN = int(valNum/total *size)
    teN = int(testNum/total *size)

    trainData = data[:trN]
    valData = data[trN:trN+vaN]
    testData = data[trN+vaN:]
    return trainData,valData,testData


"""
01.均匀的（指定是对应每个特征都均匀）split
这个均匀比较难以实现，暂时不管
"""
def balanceSplit(data,trainNum,valNum,testNum):
    pass


if __name__ == "__main__":
    data = [1,2,3,4,5,6,7,8,9,10]
    a,b,c = shuffleAndsplit(data,2,4,4)
    print(a,b,c)