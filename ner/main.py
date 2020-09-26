'''
运行主程序的文件
'''
import torch as t
import torch.nn as nn
from data.ProcessData import getLabels2List
from data.MyData import MyDataSet
from torch.utils.data import DataLoader
from models.PureBert import BaseLine

BATCH_SIZE = 1

# TODO: 后面做一个配置信息
"""
"""
class Config():
    pass

# step1.将数据输入到bert中得到输出向量
def train(filePath):
    # step1.准备数据
    conte, labels = getLabels2List(filePath)
    dataset = MyDataSet(conte,labels)
    train_loader = DataLoader(dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=0)  # 只用主进程加载数据

    # 搞个基准模型跑一跑，对这里的 (768,13) 不是特别熟
    model = BaseLine(768,13)
    criterion = nn.BCELoss()
    optimizer = t.optim.Adam(model.parameters(),lr=1e-4)
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

    # 开始训练
    for epoch in range(20):
        print("=========epoch：", epoch + 1, end = ",")
        # 遍历dataloader 的值
        for i,item in enumerate(train_loader):
            data,label = item
            out = model(data) # 执行model 的 forward() 方法
            outSize = out.size(0)
            out = out.view(outSize)

            out = out.to(device)  # 要将out 放到device  中，否则最后会有一个报错  => 在学院的gpu上，就发现这个才是最重要的！
            loss = criterion(out, label)  # 与分类标签做比较，求出损失

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    filePath = "/home/liushen/brat/data/train"
    train(filePath)