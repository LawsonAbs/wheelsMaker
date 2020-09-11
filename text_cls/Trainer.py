"""实际训练模型的代码，也就是训练器
"""
import fire
import config as con
from torch.utils.data import DataLoader
from data.CompanyData import CompanyData
from data.ProcessData import ProcessData

# from 包名.模块名 import  类名/方法名
from models.BertCls import BertCls
import torch.nn as nn
import torch as t


class Trainer():
    def __init__(self,
                 model,
                 trainLoader,
                 criterion, # 损失函数
                 optimizer, # 优化器
                 modelPath): # 保存模型的地址
        """初始化一些设置，包括加载dataloader 等等
        """
        self.model = model # 我们就是要用优化这个model
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.criterion = criterion # 指定损失函数
        self.optimizer = optimizer # 指定优化器
        self.modelPath = modelPath # 模型保存到的地址

    def train(self):
        """
        :param trainLoader: 训练数据集
        :return: none
        """
        self.model.train = True
        for epoch in range(con.TRAIN_EPOCH):  # epoch
            print("epoch:",epoch)  # 打印epoch信息
            for i, item in enumerate(self.trainLoader):  # batch
                _da, label = item
                output = self.model(_da)  # 执行结果放到output中
                loss = self.criterion(output,label) # 计算输出和label之间的损失
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # 保存模型

if __name__ == "__main__":
    # step1.定义模型
    # 当程序到这儿的时候，就已经表明所有数据都已经准备好了，可以开始训练了
    model = BertCls()

    # step2.定义数据
    process = ProcessData()
    filePath = "./data/original_dataset/"  # 数据所在的文件目录，是目录，不是地址
    trainSet = CompanyData(filePath,process)

    # 由dataset 得到 dataloader
    trainLoader = DataLoader(trainSet,
                             batch_size=con.BATCH_SIZE,
                             shuffle=False)
    """
    dataloader的值很有意思，长下面这个样子：
    {'input_ids': tensor([[  101,  9160,    110,....],
                          [...]
                          [...]
                          [...]
                          [...]
                          ]
    }  
    """
    for _ in trainLoader: # trainLoader 是
        print(_)

    # step3.后续
    modelPath = "./3339.ckpt"

    # 如果出现：positional argument after keyword argument，则会报红
    trainer = Trainer(model,
                      trainLoader,
                      criterion=nn.BCELoss(),
                      optimizer=t.optim.Adam(model.parameters()),
                      modelPath=modelPath)
    fire.Fire(trainer)