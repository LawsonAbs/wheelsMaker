'''
Author: LawsonAbs
Date: 2020-10-23 22:00:47
LastEditTime: 2020-10-24 09:54:13
FilePath: /wheels/regression/regression.py
'''
import torch as t
import torch.nn as nn

# 岭回归中的一个参数lambda
lam = -0.3


def getData(dataPath):
    x = [] # 特征值
    y = [] # 房价
    with open(dataPath,'r') as f:
        rows = f.readlines()
        for row in rows:
            row = row.strip("\n")
            row = row.split() # 转成数组的形式
            tempX = row[0:-1]
            tempX = [float(_) for _ in tempX]
           # print(tempX)
            x.append(tempX)

            tempY = row[-1] # 最后一位
            tempY = float(tempY)
            y.append(tempY)
    x = t.tensor(x)
    y = t.tensor(y)
    y.unsqueeze_(1) # 将y变成一个列向量
    return x,y # 返回数据集，tensor

# 计算线性回归
'''
description: 
param {*} x:传入的数据，是个矩阵，tensor类型
return {*}
'''
def linearRegression(x,y):
    # 如果说，直接用了 nn.Linear() 这个操作，就不用手动计算下面的w值，因为它会自动计算出来    
    # linear = nn.Linear(in_features=inFea,out_features=outFea)  # 线性映射操作    
    xT = t.transpose(x,0,1) # 计算得到x的转置

    # 直接计算解析解
    tX =  t.mm(xT, x)  # 矩阵的乘法操作
    tX = t.inverse(tX) # 再算逆
    w = t.mm(tX,xT)
    w = t.mm(w,y)
    
    return w  # 得到的参数w



'''
description: 岭回归
param {*} x
param {*} inFea
param {*} outFea
return {*}
'''
def ridgeRegression(x,y):
    xT = t.transpose(x,0,1) # 计算得到x的转置
    # 直接计算解析解
    tX = t.mm(xT, x)  # 求 (x的转置 * x)
    I = t.eye(tX.size(0)) 
    w = tX + lam * I # 加上单位阵
    w = t.inverse(w) # 算逆
    w = t.mm(w,xT)
    w = t.mm(w,y) 
    return w  # 得到的参数w
    

'''
description: 
计算在训练集上的错误率 
计算在测试集上的错误率
param {*} x： 训练数据
param {*} y： 训练数据
param {*} w： 参数值
return {*} errRate:错误率
return {*}
'''
def calErrorRate(x,y,w):
    w = t.transpose(w,0,1) # 先转置一下
    threshold = 2
    error = 0
    for i in range(len(x)):
        item = x[i]
        item = item.view(item.size(0),1)
        out = t.mm(w,item)
        print(f"模拟out={out.item()}, 实际price={y[i].item()}")
        if (out - y[i]) > threshold:
            error += 1
    errRate = error / len(y)
    return errRate


if __name__ == "__main__":
    dataPath = "/home/lawson/data/program/wheels/housing.txt"
    x,y = getData(dataPath)    
    lW = linearRegression(x,y) # linear w
    rW = ridgeRegression(x,y) # ridge w
    # print(lW)
    linearERate = calErrorRate(x,y,lW)
    ridgeERate = calErrorRate(x,y,rW)
    print(linearERate)
    print(ridgeERate)

