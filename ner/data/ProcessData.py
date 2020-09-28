"""
处理数据
"""
import re
import config as con

def getLabels2List(filePath,left,right):
    """
    1.功能：根据源数据获取文本内容cont 和 生成labels
    01.这里的labels 是数值化的，也就是通过和id对应过的
    02.如果这里的 cont 长度超过510， 那么就要做相应的分割，分成多个部分
    03.[left,right) 表示一个区间，用于获取文件数据
    :param filePath: '/home/liushen/brat/data/train'
    :return:
    """
    conte = []
    lables2ID = {'o':0}  # 搞成一个字典
    labels = []
    for i in range(left,right):  # 文件数在1000以内
        annFileName = filePath + '/' +  str(i) + '.ann'  # 得到标签文件
        txtFileName = filePath + '/' +  str(i) + '.txt'  # 得到标签文件
        with open(txtFileName,encoding='utf8') as f:
            article = f.read()
            contLen = len(article)  # 得到文本内容的长度，然后生成一个这么大的数组

        # 生成某个文件的label
        curLabel = [0 for _ in range(contLen)]  # o 对应 0
        with open(annFileName,encoding='utf8') as f:
            line = f.readline()  # 依次读取每一行
            while line:
                line = line.strip("\n")  # 去换行
                line = re.split('[\t ]', line)

                # 取类别，左标，右标
                cls,left,right = line[1:4]
                left = int(left)
                right = int(right)
                # leftTab = 'B_' + cls
                # rightTab = 'I_' + cls
                leftTab = rightTab = cls # 全都用一个标签
                if leftTab not in lables2ID.keys():  # 说明不在字典中，就要把键值放进去
                    lables2ID[leftTab] = len(lables2ID)
                    # lables2ID[rightTab] = len(lables2ID)

                curLabel[left] = lables2ID[leftTab]  # 更新当前标签的值
                for j in range(left+1,right):  # 修改这个区间中的标签值
                    curLabel[j] = lables2ID[rightTab]
                line = f.readline()

        if contLen > con.MAX_LENGTH:
            article = splitLine(article)  # 分割文本内容
            conte.extend(article)
            curLabel = splitLine(curLabel)
            labels.extend(curLabel)
        else:
            conte.append(article)
            labels.append(curLabel)  # [[],[],...[]]
    return conte,labels  # 返回文本内容list， 标签list


'''如果文本内容过长，则需要分割成多块
'''
def splitLine(article):
    res = []  # 分割后的list
    contLen = len(article)
    cnt = contLen // con.MAX_LENGTH  # 按照con.MAX_LENGTH整除。cnt：表示需要几个部分才能装下整个数组
    if contLen%con.MAX_LENGTH:
        cnt += 1  # 说明还有余孽，再加一
    left = 0
    for i in range(cnt):
        right = min((i+1) * con.MAX_LENGTH,contLen+1)
        temp = article[left:right]  # 得到内容
        left = right
        res.append(temp)
    return res



def label2IdFromAnn(filePath):
    """
    读取文件夹下的所有.ann 文件，然后形成一个 label -> id 的字典
    :return: 字典 label2Id
    """
    label2Id = {'o': 0}  # 搞成一个字典
    id2Label = {0:'o'} # id -> label
    for i in range(1000):  # 文件数在1000以内
        annFileName = filePath + '/' + str(i) + '.ann'  # 得到标签文件
        with open(annFileName, encoding='utf8') as f:
            line = f.readline()  # 依次读取每一行
            while line:
                line = line.strip("\n")  # 去换行
                line = re.split('[\t ]', line)
                cls = line[1]   # 取类别
                if cls not in label2Id.keys():  # 说明不在字典中，就要把键值放进去
                    label2Id[cls] = len(label2Id)
                    id2Label[len(id2Label)] = cls
                line = f.readline()
    return label2Id,id2Label


if __name__=="__main__":
    conte,id2Label = getLabels2List('/home/liushen/brat/data/train')
    print(conte)
    print(id2Label)


