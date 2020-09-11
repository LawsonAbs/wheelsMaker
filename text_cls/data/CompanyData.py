import torch as t
from torch.utils.data import Dataset
from config import PRETRAINED_MODEL_NAME
import util.csvUtil  as csv  # 导入读取数据的工具包


"""
功能：继承DataSet类这个类就是用于封装数据集
01.可以在DataSet类中对文件中的初始数据进行处理，然后生成想要的类型数据
02.问题就是，因为train, validation ,test都需要数据集，那么怎么解决这个数据集的问题？
是要搞三个实现(Dataset)的类吗？ 
当然不是，这么做的话就显得比较累赘了。
那么正确的做法就是
前提：train+val 和test 不会同时进行，也就是说二者必须用分开的命令执行，这样就可以
"""
class CompanyData(Dataset):
    def __init__(self,filePath,processData):
        """
        :param filePath: 数据文件（包括）的地址
        :param processData:对原始数据的处理，生成可以直接训练的数据
        """
        super(CompanyData, self).__init__()
        """分别保存"标签摘要" ，"标签" 数据。下面看看两个数数据具体的样子
        data = [{'标签摘要': '2019年年报中，格力电器称，根据《暖通空调资讯》发布的数据，....'},{...},...]
        label = [{'标签': '行业龙头'},{'标签': '注册'},{'标签': '资金短缺'}...{}...] 
        """
        self.data,self.labels = csv.readTsv2List(filePath)
        self.process = processData
        self.data = self.process.data2Tokens(self.data,max_length=100)  # 用tokenizer处理数据
        self.labels = self.process.label2vec(self.labels)

    # TODO:暂时不管（针对train/test）返回（不同）数据集的问题
    def __getitem__(self, index):
        # 因为这里返回的并非是tensor，而是原生的字符串
        # 开始对data进行处理，处理成可以正式返回的训练数据
        res = {}
        res["input_ids"] = self.data.get("input_ids")[index]
        res["token_type_ids"] = self.data.get("token_type_ids")[index]
        res["attention_mask"] = self.data.get("attention_mask")[index]
        res["label"] = self.labels[index]  # 标签是0或者1  我认为标签应该是一个one-hot向量
        return res

    def __len__(self):
        # 返回总数据集的长度 => 如果这个长度返回的不正确，那么就会影响你的 __getitem__()的执行
        # 从而导致合并的数据变少
        return len(self.labels)