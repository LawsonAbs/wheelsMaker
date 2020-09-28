'''
运行主程序的文件
'''
import torch as t
#t.cuda.set_device(2)
import torch.nn as nn
from data.ProcessData import getLabels2List,label2IdFromAnn
from data.MyData import MyDataSet
from torch.utils.data import DataLoader
from models.PureBert import BaseLine
import config as con
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"  # 三块显卡均可见
from tqdm import tqdm
import datetime as dt
import sys



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

    input_ids = t.tensor(input_ids,dtype=t.long)
    token_type_ids = t.tensor(token_type_ids, dtype=t.long)
    attention_mask = t.tensor(attention_mask, dtype=t.long)
    label = t.tensor(batchLabels,dtype=t.long)

    res = {
        'inputs':{'input_ids':input_ids,'token_type_ids':token_type_ids,'attention_mask': attention_mask},
        'labels':label
    }
    return res

# step1.将数据输入到bert中得到输出向量
def train(filePath,modelPath):
    # step1.准备数据
    conte, labels = getLabels2List(filePath,left = 0,right = 100)
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
    for epoch in tqdm(range(10)):
        #print("=========epoch：", epoch + 1, end = ",")
        # 遍历dataloader 的值
        totalLoss = 0 # 计算总得loss 值
        avgLoss = 0 # 计算平均loss 值   # tqdm写在 enumerate 的里面
        for i,item in (enumerate(tqdm(train_loader))):
            data,label = item['inputs'], item['labels']  # 因为返回的值是一个dict，所以这里取出来
            # 将data中的每个value都变成cuda类型
            data = {key:data[key].cuda() for key in data}
            label = label.cuda()

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


def predict(predictFilePath,modelPath):
    """
    每次可以预测单个文件，因为多个文件涉及到下标标注问题。这里就没有涉及了
    :param predictFilePath: 待预测文件所在的文件夹路径
    :param modelPath:  训练好的模型
    :return: None
    """
    # step1.准备数据
    conte, labels = getLabels2List(predictFilePath,998,999)
    dataset = MyDataSet(conte,labels)
    data_loader = DataLoader(dataset=dataset,
                              batch_size=con.BATCH_SIZE,
                              collate_fn=load_fn,  # 传入一个处理数据的参数
                              shuffle=False,  # 为方便调试，可以设置为False
                              num_workers=0)  # 用几个进程加载数据

    model = BaseLine(768,14)
    model.load_state_dict(t.load(modelPath))  # 加载模型
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    model.to(device)
    label2Id,id2Label = label2IdFromAnn('/home/liushen/brat/data/train')

    # 开始针对文本数据生成label
    # 这时不需要计算梯度等问题
    with t.no_grad():
        total = 0 # 总的文本下标
        for i,item in (enumerate(tqdm(data_loader))):
            data,label = item['inputs'], item['labels']  # 因为返回的值是一个dict，所以这里取出来
            # 将data中的每个value都变成cuda类型
            data = {key:data[key].cuda() for key in data}
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
            res = []  # 表示最后生成的label
            index = pred.argmax(-1)  # 得到下标信息
            batch = i*con.BATCH_SIZE # 打印下标；用于控制输出当前的测试文本
            for batchNum in index: # 先遍历batchNum
                cnt = 0  # 重置
                for i in batchNum:
                    val = i.item() # 得到真正的下标
                    # if-else 防止越界
                    # 这里的 0 代表的就是处理第几个batch的数据
                    if cnt >= len(conte[batch]):
                        word = 'o'
                    else:
                        word = conte[batch][cnt]  # 需要得到字符
                    print(total,word,id2Label[val])
                    cnt +=1
                    total+=1
                batch+=1
                print()

            # 结合输出的概率和 label2Id 这个标签字典，生成最后的结果
            # 取值最大的那个就是得到的类别信息

            # firstSize = pred.size(0)
        print("Prediction Done!")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("参数不足")
    elif sys.argv[1] == 'test':
        print("in test")
    elif sys.argv[1] == 'predict':
        predictFilePath = "/home/liushen/brat/data/train"
        modelPath = "20200928_115718.abc"  # 当前这个目录下
        predict(predictFilePath,modelPath)
    else:
        print("in train")
        filePath = "/home/liushen/brat/data/train"
        modelPath = "./"
        train(filePath, modelPath)