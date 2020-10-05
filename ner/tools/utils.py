import os
import re  # 正则分割


"""
1.fun:读取单个 .ann 文件，从中获取entity，然后写入到 Dict 中
2.这里以 4.ann 文件作为示例，那么得到的entity 就是[清热]
"""
def getEntity2DictFromFile(entity2Id,filePath):
    with open(filePath, encoding="utf8") as f:
        line = f.readline()  # 依次读取每一行
        while line:
            line = line.strip("\n")  # 去换行
            curKey = re.split('[\t ]',line)[4]  # 取entity
            # 判断字典中是否有过entity
            if curKey not in entity2Id.keys():
                entity2Id[curKey] = len(entity2Id)
            line = f.readline()
    return entity2Id


"""
从某个路径中获取所有的.ann 文件，然后读取其中所有的Entity，返回一个dict
"""
def getEntity2DictFromAnn(filePath):
    entity2Id = {}
    fileNameList = os.listdir(filePath)
    #  注意这里的if 写在了 生成表达式的后面
    fileNameList = [name for name in fileNameList if name.endswith(".ann")]
    # 构造文件名，传入到 上面那个函数中，获取 entity
    for name in fileNameList:
        fileRoute = filePath +'/' +name
        # 读取文件，并生产entity
        getEntity2DictFromFile(entity2Id,fileRoute)

    return entity2Id


def getEntity2TagFromAnn(filePath):
    entity2Tag = {}  # 实体名字到实体类别的映射
    """    
    :return: 
    """
    fileNameList = os.listdir(filePath)
    #  注意这里的if 写在了 生成表达式的后面
    fileNameList = [name for name in fileNameList if name.endswith(".ann")]
    # 构造文件名，传入到 上面那个函数中，获取 entity
    for name in fileNameList:
        fileRoute = filePath +'/' +name
        # 读取文件，并生产entity
        with open(fileRoute,encoding='utf8') as f:  # 打开文件
            line = f.readline()
            while line:
                line = line.strip("\n")
                row = re.split('[\t ]',line)
                tag = row[1]  # entity Tag
                entity = row[4]  # entity Name
                if entity not in entity2Tag.keys(): # 如果不在字典中
                    entity2Tag[entity] = tag
                line = f.readline()
    return entity2Tag

# -----------------rubbish--------------
"""
下面这个代码是根据conte输出，是有问题的，我们应该根据tokens 反向生成text。也就是正在的predict方法
fake predict

"""
def predictFake(predictFilePath, modelPath):
    """
    每次可以预测单个文件，因为多个文件涉及到下标标注问题。这里就没有涉及了
    :param predictFilePath: 待预测文件所在的文件夹路径
    :param modelPath:  训练好的模型
    :return: None
    """
    # step1.准备数据
    conte, labels = getLabels2List(predictFilePath, 115, 116)
    dataset = MyDataSet(conte, labels)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=con.BATCH_SIZE,
                             collate_fn=load_fn,  # 传入一个处理数据的参数
                             shuffle=False,  # 为方便调试，可以设置为False
                             num_workers=0)  # 用几个进程加载数据

    model = BaseLine(768, 14)
    model.load_state_dict(t.load(modelPath))  # 加载模型
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    model.to(device)
    label2Id, id2Label = label2IdFromAnn('/home/liushen/brat/data/train')

    # 开始针对文本数据生成label
    # 这时不需要计算梯度等问题
    with t.no_grad():
        total = 0  # 总的文本下标
        for i, item in (enumerate(tqdm(data_loader))):
            data, label = item['inputs'], item['labels']  # 因为返回的值是一个dict，所以这里取出来
            # 将data中的每个value都变成cuda类型
            data = {key: data[key].cuda() for key in data}
            pred = model(data)  # 执行model 的 forward() 方法
            m = nn.Softmax(-1)  # 操作最后一个维度
            pred = m(pred)
            '''
            for循环打印的结果如下：
            tensor([9.9886e-01, 1.8475e-05, 1.0400e-05, 5.3864e-05, 1.2845e-04, 4.3052e-05,
            8.8519e-06, 2.5577e-05, 1.5128e-05, 1.6781e-05, 6.7736e-04, 9.1378e-05,
            4.6507e-05, 5.3635e-06], device='cuda:0')
            ...
            '''
            res = []  # 表示最后生成的label
            index = pred.argmax(-1)  # 得到下标信息
            batch = i * con.BATCH_SIZE  # 打印下标；用于控制输出当前的测试文本
            for batchNum in index:  # 先遍历batchNum
                cnt = 0  # 重置
                for i in batchNum:
                    val = i.item()  # 得到真正的下标
                    # if-else 防止越界
                    # 这里的 0 代表的就是处理第几个batch的数据
                    if cnt >= len(conte[batch]):
                        word = 'o'
                    else:
                        word = conte[batch][cnt]  # 需要得到字符
                    print(total, word, id2Label[val])
                    cnt += 1
                    total += 1
                batch += 1
                print()

            # 结合输出的概率和 label2Id 这个标签字典，生成最后的结果
            # 取值最大的那个就是得到的类别信息

            # firstSize = pred.size(0)
        print("Prediction Done!")



'''
将字典信息dic写入到文件fileName中
'''
def writeDict2File(dic,fileName):
    with open(fileName,'w') as f:
        dic = str(dic)  # 先转换为string，否则无法写入文本
        f.write(dic)
    


if __name__ == "__main__":
    info = {'name':'lawson','age':24}
    fileName = '/home/liushen/a.txt'
    writeDict2File(info,fileName)
