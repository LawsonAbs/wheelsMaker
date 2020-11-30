'''
Author: LawsonAbs
Date: 2020-11-29 14:43:00
LastEditTime: 2020-11-29 15:33:40
FilePath: /puns/BaseLine.py
'''

'''
实现的功能:将puns传入到bert中，直接得到一个二分类结果，用于判断是否是puns。用于做一个baseline
'''

import torch as t
from transformers import AutoTokenizer,AutoModel  # 导入transformer的模型
from torch.utils.data import Dataset,DataLoader # 导入数据

# 数据类
class MyDataSet(Dataset):
    def __init__ (self,data):
        super.__init__() 
        self.data = data
    
    # 返回数据集
    def __getitem__(self,index):
        return self.data[index] # 返回数据集
        
        
# 数据加载类
class MyDataLoader(DataLoader):
    pass

