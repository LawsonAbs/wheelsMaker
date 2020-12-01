'''
Author: LawsonAbs
Date: 2020-11-29 22:23:15
LastEditTime: 2020-11-30 22:25:46
FilePath: /wheels/puns_classify/punsMain.py
'''

# xml 文件的path信息
xmlPath = '/home/lawson/program/wheels/puns_classify/data/test/subtask1-homographic-test.xml'

import sys
sys.path.append(r"/home/lawson/program/wheels")
from tqdm import tqdm
import datetime as dt
import torch as t
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import Dataset,DataLoader # 导入数据
from BaseLine import PureBert  # 从BaseLine 中导入所有
import config as con
import tools.util as ut # 因为不在同一文件夹下，所以找不到

# 数据类
class MyDataSet(Dataset):
    def __init__ (self,text,labels):
        super().__init__()
        self.text = text # 这里的text是个 list，每一项是个str
        self.labels = labels # 传入标签
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")        

    # 返回数据集
    def __getitem__(self,index):
        # 根据每条文本，用tokenizer将其转换        
        # self.data  = self.tokenizer(self.text[index], # 处理第index条数据
        #                             padding='max_length', 
        #                             max_length=con.MAX_LENGTH,
        #                             return_tensors='pt'
        #                             )

        return self.text[index], self.labels[index] # 返回训练的数据
    
    def __len__(self):
        return len(self.text) # 原始的数据集大小（即有多少条文本）

# 定义训练的方法
def train(model,tokenizer,trainLoader,devLoader):    
    #step4.执行训练过程    
    bestF1 = 0 # 记录最优的f1值
    for epoch in tqdm(range(1,10)):
        # 遍历dataloader 的值        
        totalLoss = 0 # 计算总的loss 值
        avgLoss = 0 # 计算平均loss 值   # tqdm写在 enumerate 的里面
        for i,item in (enumerate(tqdm(trainLoader))):
            data,label = item[0], item[1]  # 因为返回的值是一个list，取出对应的数据
            # 根据每条文本，用 tokenizer 将其转换        
            data = tokenizer(data,
                            padding='max_length', 
                            max_length=con.MAX_LENGTH,
                            return_tensors='pt'
                            )
            # 将data中的每个value都变成cuda类型
            data = {key:data[key].cuda() for key in data}
            label = label.cuda()
            out = model(data) # 执行model 的 forward() 方法            
            # print(out)
            # 这里的out维度是(BATCH_SIZE,CLS_NUM)
            # label的维度是(BATCH_SIZE)
            
            # 可能不完全按照 BATCH_SIZE 均分，所以这里用 out.size(0)
            firstSize = out.size(0)
            out = out.view(firstSize,con.CLS_NUM) # 转换格式
            loss = criterion(out, label)  # 与分类标签做比较，求出损失
            totalLoss += loss.item()  # loss.item()将loss[是个tensor] 转换成单个数值
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avgLoss = totalLoss / (i+1)        
        
        # 训练解计算acc，计算recall, precision, f1就没啥意义了
        print(f"train_Epoch:{epoch} avgLoss={avgLoss}")

        # 这里加入一个断点保存机制； 和一个验证集
        if epoch % 1 == 0 : # 如果能够整除            
            curF1 = evaluate(model,tokenizer,devLoader)
            if curF1 > bestF1 :
                curTime = dt.datetime.now()
                curTime = curTime.strftime("%Y%m%d_%H%M%S") # 格式化字符串
                curModelPath = f"{con.MODEL_PATH}{curTime}_{epoch}.ckpt"  # 代表第几个epoch 
                t.save(model.state_dict(),curModelPath)
                bestF1 = curF1

    
'''
description: 验证集的使用
param {*} model
param {*} tokenizer
param {*} devLoader
return {*} 
计算预测 puns 的f1值
'''
def evaluate(model,tokenizer,devLoader):    
    correct = 0 # 正确的预测结果    
    totalPre = 0 # 总共预测为 puns 的个数
    totalLabel = 0 # label中总共为 1 的个数
    norm = nn.Softmax(dim=1) # 归一化操作
    for i,item in (enumerate(tqdm(devLoader))):
        data,label = item[0], item[1]  # 因为返回的值是一个list，取出对应的数据
        # 根据每条文本，用 tokenizer 将其转换
        data = tokenizer(data,
                        padding='max_length', 
                        max_length=con.MAX_LENGTH,
                        return_tensors='pt'
                        )
        # 将data中的每个value都变成cuda类型
        data = {key:data[key].cuda() for key in data}
        label = label.cuda()
        
        out = model(data) # 执行model 的 forward() 方法            
        
        # 这里的out维度是(BATCH_SIZE,CLS_NUM)
        firstSize = out.size(0)
        out = out.view(firstSize,con.CLS_NUM) # 转换格式
        res1 = norm(out)
        res2 = t.argmax(res1,dim=1) # 取最大值
        #print(res2) # 得到的应该是个向量
        totalPre += res2.sum() # 预测出为puns的总个数 ，用于算precision
        totalLabel += label.sum() # label中为puns 的个数

        res3 = res2 & label # 计算出总共预测为puns的向量， 用于算recall
        correct += res3.sum()   # 计算出预测正确的数量
    
    totalLabel = totalLabel.item()
    correct = correct.item()
    totalPre = totalPre.item()
    
    precision = correct/totalPre  # 【预测puns，并且正确/预测为puns的】
    recall = correct / totalLabel
    f1 = (2*precision*recall) / (precision+recall)
    print(f"recall:{recall}, precision:{precision}, f1:{f1}")
    return f1 # 返回f1的值，用于挑选最好的模型
    


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("参数不足")
        exit(0)
    elif sys.argv[1] == 'train':
        #step1.得到原始的数据
        text = ut.readXml(con.XML_PATH)
        labels = ut.readLabels(con.XML_LABEL)

        #step2.分割数据，=> train,test,dev等
        tra,dev,test = ut.splitData(text,labels) # 将数据按照 8:1:1 分成三部分

        #step3.加载训练数据
        # 加载train 
        trainData = MyDataSet(tra['x'],tra['y']) 
        trainLoader = DataLoader(trainData,
                                batch_size=con.BATCH_SIZE)
        
        # 加载dev
        devData = MyDataSet(dev['x'],dev['y'])
        devLoader = DataLoader(devData,
                                batch_size=con.BATCH_SIZE)
                
        model = PureBert(768,con.CLS_NUM)
        criterion = nn.CrossEntropyLoss()
        optimizer = t.optim.Adam(model.parameters(),lr=1e-4)
        device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        train(model,tokenizer,trainLoader,devLoader) 
    else: # 执行test操作                
        modelPath = "/home/lawson/program/wheels/puns_classify/ckpt/20201130_214021_1.ckpt"  # 当前这个目录下        

        # 加载已经训练好的模型
        model = PureBert(768,con.CLS_NUM)
        model.load_state_dict(t.load(modelPath))  # 加载模型
        device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

         #step1.得到原始的数据
        text = ut.readXml(con.XML_PATH)
        labels = ut.readLabels(con.XML_LABEL)

        #step2.分割数据，=> train,test,dev等
        tra,dev,test = ut.splitData(text,labels) # 将数据按照 8:1:1 分成三部分

        #step3.加载训练数据
        # 加载train 
        testData = MyDataSet(tra['x'],tra['y']) 
        testLoader = DataLoader(testData,
                                batch_size=con.BATCH_SIZE)
        evaluate(model,tokenizer,testLoader)                                

        