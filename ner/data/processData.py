"""
处理数据
"""
import os
from tools import utils
import re

def getLabels2List(filePath):
    """
    根据源数据获取文本内容cont 和 生成labels
    :param filePath: '/home/liushen/brat/data/train'
    :return:
    """
    conte = []
    lables = []
    for i in range(1000):  # 文件数在1000以内
        annFileName = filePath + '/' +  str(i) + '.ann'  # 得到标签文件
        txtFileName = filePath + '/' +  str(i) + '.txt'  # 得到标签文件
        with open(txtFileName,encoding='utf8') as f:
            article = f.read()
            # print(cont)  文本内容
            contLen = len(article)  #  得到文本内容的长度，然后生成一个这么大的数组
            # print(contLen)  # 文本长度
            conte.append(article)

        # 生成某个文件的label
        curTab = ['o' for _ in range(contLen)]
        with open(annFileName,encoding='utf8') as f:
            line = f.readline()  # 依次读取每一行
            while line:
                line = line.strip("\n")  # 去换行
                line = re.split('[\t ]', line)

                # 取类别，左标，右标
                cls,left,right = line[1:4]
                left = int(left)
                right = int(right)
                curTab[left] = 'B_'+cls
                for j in range(left+1,right): # 修改这个区间中的标签值
                    curTab[j] = 'I_'+cls
                #print(cls,left,right)
                line = f.readline()
        lables.append(curTab)  # [[],[],...[]]
    return conte,lables  # 返回文本内容list， 标签list


if __name__ == "__main__":
    labels = getLabels2List("/home/liushen/brat/data/train")
    print(labels)
