'''
运行主程序的文件
'''
import torch as t
#t.cuda.set_device(2)
import torch.nn as nn
from data.ProcessData import *
from data.MyData import MyDataSet,MyTestDataSet
from torch.utils.data import DataLoader
from models.PureBert import BaseLine
import config as con
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"  # 三块显卡均可见
from tqdm import tqdm
import datetime as dt
import sys
from transformers import BertTokenizerFast  # 根据tokens反向生成word
from tools.utils import *

label2Id,id2Label = label2IdFromAnn('/home/liushen/brat/data/train')

'''# 通过.ann 文件得到所有的实体名称，实体名称到序号的映射
值形式如下：
{'前列腺增生': 0, '慢性前列腺炎': 1,....}
'''
entity2Id = getEntity2DictFromAnn('/home/liushen/brat/data/train')

'''通过所有 .ann 文件 得到所有的实体名称到类别的映射
值形式如下：
{'前列腺增生': 'DISEASE', '慢性前列腺炎': 'DISEASE',....}
'''
entity2Tag = getEntity2TagFromAnn('/home/liushen/brat/data/train')



"""是对整个batch 的数据进行处理
这个batch 就是一个list
"""
def load_fn(batch):
    batchInputs = [_['inputs'] for _ in batch]  # 拿到这个批次的所有inputs
    batchLabels = [_['label'] for _ in batch]  # 拿到这个批次的所有label

    # 根据 batchInputs
    input_ids = [_['input_ids'] for _ in batchInputs]
    token_type_ids = [_['token_type_ids'] for _ in batchInputs]
    attention_mask = [_['attention_mask'] for _ in batchInputs]
    offset_mapping = [_['offset_mapping'] for _ in batchInputs]

    # 注意这里必须转为tensor，否则得不到正确结果
    input_ids = t.tensor(input_ids,dtype=t.long)
    token_type_ids = t.tensor(token_type_ids, dtype=t.long)
    attention_mask = t.tensor(attention_mask, dtype=t.long)
    offset_mapping = t.tensor(offset_mapping,dtype=t.long)
    label = t.tensor(batchLabels,dtype=t.long)

    res = {
        'inputs':{'input_ids':input_ids,'token_type_ids':token_type_ids,'attention_mask': attention_mask,'offset_mapping':offset_mapping},
        'labels':label
    }

    # 根据生成的inputs_ids 反向查找该句的labels
    return res


"""是对整个batch 的数据进行处理。针对的是predict 的数据
这个batch 就是一个list
"""
def load_fn_predict(batch):
    batchInputs = [_['inputs'] for _ in batch]  # 拿到这个批次的所有inputs

    # 根据 batchInputs
    input_ids = [_['input_ids'] for _ in batchInputs]
    token_type_ids = [_['token_type_ids'] for _ in batchInputs]
    attention_mask = [_['attention_mask'] for _ in batchInputs]
    offset_mapping = [_['offset_mapping'] for _ in batchInputs]

    # 注意这里必须转为tensor，否则得不到正确结果
    input_ids = t.tensor(input_ids,dtype=t.long)
    token_type_ids = t.tensor(token_type_ids, dtype=t.long)
    attention_mask = t.tensor(attention_mask, dtype=t.long)
    offset_mapping = t.tensor(offset_mapping,dtype=t.long)

    res = {
        'inputs':{'input_ids':input_ids,'token_type_ids':token_type_ids,'attention_mask': attention_mask,'offset_mapping':offset_mapping}
    }
    return res


# step1.将数据输入到bert中得到输出向量
# 01. 不能忽略了num 在 tokenizer 中 被合并的过程
def train(filePath,modelPath):
    # step1.准备数据
    # 113.txt 这个文件有多个数据，会修改label
    conte, labels = getLabels2List(filePath, left=0, right=900)
    dataset = MyDataSet(conte,labels)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=con.BATCH_SIZE,
                              collate_fn=load_fn,  # 传入一个处理数据的参数
                              shuffle=False,  # 为方便调试，可以设置为False
                              num_workers=0)  # 用几个进程加载数据

    # 搞个基准模型跑一跑，(768,14) 表示的就是输入，输出维度。
    model = BaseLine(768,14)
    criterion = nn.CrossEntropyLoss() # 采用交叉熵损失函数
    optimizer = t.optim.Adam(model.parameters(),lr=1e-4)
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    model.to(device)

    # 开始训练
    for epoch in tqdm(range(50)):
        # 遍历dataloader 的值
        totalLoss = 0 # 计算总得loss 值
        avgLoss = 0 # 计算平均loss 值   # tqdm写在 enumerate 的里面
        for i,item in (enumerate(tqdm(train_loader))):
            data,label = item['inputs'], item['labels']  # 因为返回的值是一个dict，所以这里取出来
            # 将data中的每个value都变成cuda类型
            data = {key:data[key].cuda() for key in data}
            label = label.cuda()
            if 'offset_mapping' in data.keys():
                del(data['offset_mapping'])  # 删除
            out = model(data) # 执行model 的 forward() 方法
            out.squeeze(0) # 去掉第一维的0
            # 这里的out维度是(BATCH_SIZE,MAX_LENGTH,13)
            # label的维度是(BATCH_SIZE,MAX_LENGTH)
            # 因为对 CrossEntropyLoss() 函数的使用不熟悉，这里改成二维的
            # 可能不完全按照 BATCH_SIZE 均分，所以这里用 out.size(0)
            firstSize = out.size(0)
            #print(firstSize)
            out = out.view(firstSize*con.MAX_LENGTH,14)
            label = label.view( firstSize * con.MAX_LENGTH)
            #print("max(label)=", max(label), "min(label)=", min(label))
            loss = criterion(out, label)  # 与分类标签做比较，求出损失
            totalLoss += loss.item()
            #print(loss.item()) # 转换成单个item值
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avgLoss = totalLoss / (i+1)
        print(f"Epoch:{epoch} end! avgLoss={avgLoss}")
    curTime = dt.datetime.now()
    curTime = curTime.strftime("%Y%m%d_%H%M%S")
    modelPath = modelPath + curTime + ".abc"
    t.save(model.state_dict(),modelPath)


def predict(predictFilePath,modelPath,num):
    """
    每次可以预测单个文件，因为多个文件涉及到下标标注问题。这里就没有涉及了
    :param predictFilePath: 待预测文件所在的文件夹路径
    :param modelPath:  训练好的模型
    :param resPath:  生成结果文件的路径
    :param  num: 第num的文件的序号   
    :return: None
    """
    # step1.准备数据 => 预测单个文件
    conte = getCont2List(predictFilePath,num,num+1)
    fileName = predictFilePath+str(num)+".ann"  # 得到预测结果文件
    print(f"predict {num} ")
    dataset = MyTestDataSet(conte)
    data_loader = DataLoader(dataset=dataset,
                              batch_size=con.BATCH_SIZE,
                              collate_fn=load_fn_predict,  # 传入一个处理数据的参数
                              shuffle=False,  # 为方便调试，可以设置为False
                              num_workers=0)  # 用几个进程加载数据

    model = BaseLine(768,14)
    model.load_state_dict(t.load(modelPath))  # 加载模型
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    model.to(device)
    # 需要反解得到字符，所以用到tokenizer
    # 同时需要用到 return_offset_mapping 这个字段
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    res = {} # 生成最后的结果
    # 开始针对文本数据生成label
    # 这时不需要计算梯度等问题
    with t.no_grad():
        totalOffset = 0 # 总的文本下标
        for i,item in (enumerate(tqdm(data_loader))):  # 这就是拿一个batch的数据
            data = item['inputs']  # 因为返回的值是一个dict，所以这里取出来
            # 将data中的每个value都变成cuda类型
            data = {key:data[key].cuda() for key in data}
            offset_mapping = data['offset_mapping']  # 留下offset_mapping 的值，也是一个多维的数组
            if 'offset_mapping' in data.keys():
                del(data['offset_mapping'])  # 删除
            pred = model(data) # 执行model 的 forward() 方法
            m = nn.Softmax(-1)  # 操作最后一个维度
            pred = m(pred)
            '''
            for循环打印的结果如下：
            tensor([9.9886e-01, 1.8475e-05, 1.0400e-05, 5.3864e-05, 1.2845e-04, 4.3052e-05,
            8.8519e-06, 2.5577e-05, 1.5128e-05, 1.6781e-05, 6.7736e-04, 9.1378e-05,
            4.6507e-05, 5.3635e-06], device='cuda:0')
            ...
            '''
            index = pred.argmax(-1)  # 得到下标信息
            '''
            得到的pred 的维度就是(BATCH_SIZE,MAX_LENGTH,CLS_NUM)，放在这里就是(2,512,13)。
            在经过nn.Softmax()处理之后，得到的维度不变；
            在经过argmax()操作之后，就会变成 (2,512) 的维度。
            '''
            input_ids = data['input_ids'] # 得到这个input_ids，就可以反解拿到汉字

            offset_mapping = offset_mapping.tolist()
            input_ids = input_ids.tolist() # 转为list。这里仍然是多维的

            curBatchNum = min(con.BATCH_SIZE, index.size(0))
            index = index.tolist()  # 转为list

            # 当前有几个batch，就做几次循环
            label = []  # 生成的标签
            outCont = []  # 反解得到的文本内容
            tempMaxOffset = 0 # 用于找出最大的位移值
            for k in range(curBatchNum):  # 这里的con.BATCH_SIZE 值是有问题的，应该是min(con.BATCH_SIZE)
                curInputIds = input_ids[k]
                outCont.extend(tokenizer.convert_ids_to_tokens(curInputIds))  # 得到文本内容
                curIndex = index[k]  # 也是拿到某个batch的label
                curOffsetMapping = offset_mapping[k]  # 当前的offset_mapping
                for j in range(len(curIndex)):
                    label.append(id2Label[curIndex[j]])
                    tempMaxOffset = max(tempMaxOffset, curOffsetMapping[j][1])
                    if curIndex[j]:  # 如果改标签不是0
                        left = curOffsetMapping[j][0]  # 得到第j个标签的左值
                        res[totalOffset+left] = id2Label[curIndex[j]]  # 设置键值
                totalOffset = tempMaxOffset  # 得到最后一个标签的右值

        # 打印文本和标签值
        # for a,b in zip(outCont,label):
        #     print(a,b)
        allConte = "" # 字符串内容
        for i in range(len(conte)):
            allConte += conte[i]
        writePred2Ann(res,allConte,fileName)

    print(f"file {num} is Done!")


'''
res 长如下的样子：
{61: 'PERSON_GROUP', 62: 'PERSON_GROUP',
75: 'PERSON_GROUP', 76: 'PERSON_GROUP', 
82: 'DRUG_EFFICACY', 83: 'DRUG_EFFICACY',84: 'DRUG_EFFICACY', 85: 'DRUG_EFFICACY',
101: 'SYMPTOM', 102: 'SYMPTOM',103: 'SYMPTOM', 104: 'SYMPTOM',
108: 'SYMPTOM', 109: 'SYMPTOM',110: 'SYMPTOM', 111: 'SYMPTOM',
113: 'SYMPTOM', 114: 'SYMPTOM',115: 'SYMPTOM', 116: 'SYMPTOM', 
118: 'SYMPTOM', 119: 'SYMPTOM',
120: 'SYMPTOM', 121: 'SYMPTOM', 122: 'SYMPTOM',123: 'SYMPTOM', 124: 'SYMPTOM', 125: 'SYMPTOM', 
145: 'DRUG_EFFICACY',146: 'DRUG_EFFICACY', 147: 'DRUG_EFFICACY', 148: 'DRUG_EFFICACY'}
'''
def writePred2Ann(res,conte,fileName):
    """
    功能：根据预测结果res 结合关键字，生成预测文件
    :param res: 预测结果
    :param conte:  文本内容
    :param fileName:  文件名
    :return:none
    """
    # previousLeft, previousRight 是区间的左右端点值
    previousLeft = list(res.keys())[0]  # 得到第一个下标
    previousRight = previousLeft
    previousValue = list(res.values())[0] # 得到第一个实体名
    out = [] # 最后的预测输出
    cnt = 1 # 控制输出的实体数
    # 一定要在这里初始化，否则会因为res的长度不够导致进入temp bug
    # 因为entityNum 没有值
    entityNum = "T" + str(cnt)
    for i,item in enumerate(res.items()): # 遍历dict的每一项
        temp = []
        key,value = item # key是下标位置， value 是类别
        if previousRight == key:  # 如果当前的key(也就是index) 与之前的index相等，说明在遍历第一个键值对
            continue  # 继续往下
        # 如果连续，且实体名都相同，则说明在同一组
        elif previousRight + 1 == key and value == previousValue:
            previousRight = key  # 更新区间右端点值
            continue
        else:
            entityNum = "T" + str(cnt)
            temp.extend([entityNum,previousValue,previousLeft,previousRight+1])
            temp.append(conte[previousLeft:previousRight+1])
            previousLeft = key  # 更新区间左端点值
            previousRight = key
            previousValue = value
            cnt += 1
            out.append(temp)
    temp.extend([entityNum, previousValue, previousLeft, previousRight + 1])
    temp.append(conte[previousLeft:previousRight+1])
    out.append(temp)

    # 处理out的值
    out = processPredAnn(out)
    for pred in out:
        print(pred)

    with open(fileName,'w') as f:
        for line in out:
            row = line[0]+'\t'+line[1]+" "+str(line[2])+" "+str(line[3])+"\t"+line[4]+"\n"
            f.write(row)
    return out

'''
处理生成的预测值。处理准则如下：
01.如果是单个值，则直接删除
02.如果句子过长，且可以分为两个，则分成多组，分组过程根据dict来分
'''
def processPredAnn(out):
    res = []
    for pred in out:
        left,right = pred[2:4]
        left = int(left)
        right = int(right)
        if left +1 == right: # 预测中只有一个字，这种要删除
            continue
        else:  # 如果不止一个字，则放到预测结果中
            res.append(pred)
    return res


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("参数不足")
    elif sys.argv[1] == 'test':
        print("in test")
    elif sys.argv[1] == 'predict':
        predictFilePath = "/home/liushen/brat/data/test"
        modelPath = "20200930_091658.abc"  # 当前这个目录下
        for i in range(1364,1365):
            print("in predict")
            predict(predictFilePath, modelPath, i)  # 预测第i个文件的结果
    else:
        print("in train")
        filePath = "/home/liushen/brat/data/train"
        modelPath = "./"
        train(filePath, modelPath)