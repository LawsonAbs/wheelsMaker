import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset


'''
继承DataSet类
'''
class MyDataSet(Dataset):
    # 在构造的时候就初始化配置信息
    def __init__(self,data,labels) -> None:
        super().__init__()
        self.data = data
        self.labels = labels

    # 返回的数据就是要送到模型中进行训练的数据
    def __getitem__(self, index: int):
        return self.data[index],self.labels[index]

    def __len__(self):
        return len(self.data)  # 返回总数据集的大小


'''
继承DataLoader
'''
class MyDataLoader(DataLoader):
    def __init__(self):
        super(MyDataLoader).__init__()