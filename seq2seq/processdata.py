'''
Author: LawsonAbs
Date: 2020-12-06 15:53:06
LastEditTime: 2020-12-06 21:39:37
FilePath: /wheels/seq2seq/processdata.py
处理数据的脚本
01.我修改了en.xml 和 zh.xml 中的若干条数据，因为这些数据导致解析xml时，出现格式错误。

(base) lawson@recall:~/program/wheels/seq2seq/data$ cat en.xml | grep '&'
<seg id="3265"> BEST WESTERN Cleveland Inn & Suites will be on your right. </seg>
<seg id="4539"> I used to haul supplies for J & R and let me tell you, they are very </seg>
<seg id="6106"> Free breakfast with fresh baked waffles & yogurt, high-speed Internet </seg>

'''
import  xml.dom.minidom
import torchtext as tt

def readXml(path):
    #打开xml文档
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement #得到文档元素对象
    # print(root.nodeName) #corpus
    
    # 得到所有的text，因为每个text模块下才是word，所以需要先去text，再去word
    segs = root.getElementsByTagName("seg")
    conts = [] # 存储英语句的列表
    maxLen = 0 # 求解句子的最大长度    
    for text in segs:                
        a = text.firstChild.data        
        a = "<sos>" + a # 加入了起止标识 
        a = a + "<eos>" 
        # print(a)  # 输出某句话
        conts.append(a)
        maxLen = max(maxLen,len(a))

    # 直接将数据放到内存中
    # 下面展示一下数据长什么样子
    print("\n========数据的信息=======")
    print("len的最大长度是：",maxLen)
    print(f"一共有 {len(conts)} 条语料")
    print("下面列举出前10条句子")    
    for i in range(10): # 只展示 100 条数据
        print(i+1,":",conts[i])
    print("================================\n")
    return conts


# 分割数据成train,dev,test 三个部分
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
        
    train['x'] = x_train
    train['y'] = y_train

    dev['x'] = x_dev
    dev['y'] = y_dev
    
    test['x'] = x_test
    test['y'] = y_test
    print(len(x))
    print(len(x_train) + len(x_dev) + len(x_test))
    return train,dev,test # 三者都是字典，因为需要带上标签




if __name__ =="__main__":
    path = "/home/lawson/program/wheels/seq2seq/data/" 
    enConts = readXml(path+"en.xml") # source
    zhConts = readXml(path+"zh.xml") # target
    train,dev,tets = splitData(enConts,zhConts)
    