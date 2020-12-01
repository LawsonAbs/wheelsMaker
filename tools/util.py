'''
Author: LawsonAbs
Date: 2020-10-04 14:57:44
LastEditTime: 2020-11-30 22:52:37
FilePath: /wheels/tools/util.py
'''

import  xml.dom.minidom
from tqdm import tqdm

# 预定义一个字符列表
symbol = ['\'',':','\"',',','.','!']


"""
1.读取一个xml文件，然后将得到的值写入到一个txt文件中，结合标签数据，给出歧义词
"""
def readXml(path):
    #打开xml文档
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement #得到文档元素对象
    # print(root.nodeName) #corpus
    
    # 得到所有的text，因为每个text模块下才是word，所以需要先去text，再去word
    texts = root.getElementsByTagName("text")
    puns = [] # 存储双关语的列表
    for text in texts:
        words = text.getElementsByTagName("word") #得到word
        pun = []
        for word in words:
            a = word.firstChild.data        
            pun.append(a)
        puns.append(pun)

    res = []
    maxLen = 0 # 求解句子的最大长度    
    for pun in puns: # 把pun中的每个单词都用空格分开，puns是个列表 [[],[]...]
        str = "" # 形成最后的string
        curPunLen = 0 # 当前句子的长度
        for word in pun:
            if word in symbol and str.endswith(" "): #如果是符号，那么符号前不应该有空格
                str = str.strip() # 去行末空格
            curPunLen += 1
            str = str + word + " "
        maxLen = max(maxLen,curPunLen) # 找出最大长度
        res.append(str)
    
    # 直接将数据放到内存中
    print("\n========数据（puns）的信息=======")
    print("puns的最大长度是：",maxLen)
    print(f"一共有 {len(res)} 句 puns")
    print("下面列举出前10句双关语")    
    for i in range(10): # 只展示 100 条数据
        print(i+1,":",res[i])
    print("================================\n")
    return res #数据格式为：['...','...',...] 


# 读取xml 数据的标签
def readLabels(path):
    labels = []
    with open(path,'r') as f: 
        line = f.readline()
        while line:
            line = line.strip("\n")
            line = line.split()
            labels.append(int(line[1]))
            # print(line)            
            line = f.readline()
    # print(labels[0:10])
    cnt = 0 # 计算多少个双关
    for _ in labels:
        if _ == 1:
            cnt+=1
    print(cnt)
    return labels


'''
description: 
param {*} x: 是个list，数据
param {*} y: 是个list 标签
return {*}
'''
def splitData(x,y):    
    train = {}
    dev = {}
    test = {}

    part1 = int(len(x) * 0.1)  # 强转为int
    part2 = int(len(x)*0.6)   
    x_train = x[:part1]
    y_train = y[:part1]
    x_dev = x[part1:part2]
    y_dev = y[part1:part2]
    x_test = x[part2::] 
    y_test = y[part2::]
    
    # 为防止train中的数据朝正样本倾斜，所以删除其过多的正样本数据
    positiveNum = 0    
    print("x_train中的总样本数：",len(x_train))
    for i in range(len(y_train)):
        if y_train[i] == 1:
            positiveNum += 1 
    print("正样本数：",positiveNum)        
    # 去掉100条正样本
    cnt = 0 # 去掉的样本数
    filter_x_train =[]
    filter_y_train =[]
    for i in range(len(y_train)):
        if y_train[i] == 1 and cnt <100:
            cnt += 1 
        else:
            filter_x_train.append(x_train[i]) # 拿到过滤后的值
            filter_y_train.append(y_train[i]) 

    print("len_filter_x_train:",len(filter_x_train))
    train['x'] = filter_x_train
    train['y'] = filter_y_train

    dev['x'] = x_dev
    dev['y'] = y_dev
    
    test['x'] = x_test
    test['y'] = y_test
    print(len(x))
    print(len(x_train) + len(x_dev) + len(x_test))
    return train,dev,test # 三者都是字典，因为需要带上标签
    

if __name__ =="__main__":
    conte = readXml("/home/lawson/program/wheels/puns_classify/data/test/subtask1-homographic-test.xml")
    labels = readLabels("/home/lawson/program/wheels/puns_classify/data/test/subtask1-homographic-test.gold")
    splitData(conte,labels)