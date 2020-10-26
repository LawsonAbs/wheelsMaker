'''
Author: LawsonAbs
Date: 2020-10-26 08:53:51
LastEditTime: 2020-10-26 09:14:49
FilePath: /wheels/regression/boston2.py
'''
import numpy as np
import copy
from sklearn.datasets import load_boston#导入波士顿房价数据集
 
 
class Regression:
    M_x = []           # 保存训练集中的x
    M_y = []           # 保存训练集中的y
    M_beta = []        # 参数向量
    M_estimate = []    # 预测值
    trained = False    # 用于判断训练是否结束
 
    def __init__(self):
        pass
 
    def dataprocess(self,data,target):
        self.M_x = np.mat(data)
        # 每个向量添加一个分量1，用来对应系数beta0
        fenliang = np.ones((len(data), 1))
        self.M_x = np.hstack((self.M_x, fenliang))
        self.M_y = np.mat(target)

    # 线性回归
    def lineregression(self):     
        M_x_T = self.M_x.T  # 计算X矩阵的转置矩阵
        self.M_beta = (M_x_T * self.M_x).I * M_x_T * self.M_y.T   # 通过最小二乘法计算出参数向量
        print(self.M_beta)
        self.trained = True

    # 岭回归
    def ridgeregression(self, mlambda):
        M_x_T = self.M_x.T  # 计X矩阵的转置矩阵
        tempX = np.identity(14)
        self.M_beta = (mlambda * tempX  + M_x_T * self.M_x ).I* M_x_T * self.M_y.T    # 通过正则化最小二乘法计算出参数向量
        self.trained = True
    # 预测y的值
    def predict(self, vec):
        if not self.trained:
            print("You haven't finished the regression!")
            return
        M_vec = np.mat(vec)
        fenliang = np.ones((len(vec), 1))
        M_vec = np.hstack((M_vec, fenliang))
        self.M_estimate = np.array(np.matmul(M_vec,self.M_beta))    # 根据回归算出来的beta预测y的值

    # 计算均方差
    def meanvalue(self, real):
        M_e = 0.
        for i in range(len(real)):
            M_e += np.square(self.M_estimate[i,0]-real[i])    #计算方差和
        mean = M_e/len(real)    #计算均方差
        return mean
 
if __name__ == '__main__':
    # 从sklearn的数据集中获取相关向量数据集data和房价数据集target
    data, target = load_boston(return_X_y=True)
    print(data.shape)
    # 划分训练集和测试集80%，
    x_train = data[:404]
    y_train = target[:404]
    x_test = data[404:]
    y_test = target[404:]
        
    M_test = np.mat(x_test)  #对测试集中的x进行处理
    real = target  #实际数值real
    
    # 线性回归
    lr = Regression()
    # 数据预处理
    lr.dataprocess(x_train,y_train)
    # 最小二乘法计算参数beta
    lr.lineregression()
    # 使用参数beta对y进行预测
    lr.predict(M_test)
    # 计算均方差
    mean = lr.meanvalue(y_test)
    print("线性回归均方误差:",mean)
   
    # 岭回归
    minmean = 500
    minlambda = 0
    # 使用循环找出最优的lambda
    for i in range (100):
        m_lambda = i  #lambda的取值范围0~100
        lr.ridgeregression(m_lambda)
        lr.predict(M_test)#岭回归预测值
        mean = lr.meanvalue(y_test)
        #print("岭回归均方误差:  ",mean,"lambda:",m_lambda)
        if mean < minmean:
            minmean = mean
            minlambda = m_lambda
    print("岭回归均方误差:  ",minmean,"lambda:",minlambda)
