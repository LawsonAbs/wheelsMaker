"""
01.models表示的是一个可使用的模型集合。
如有bert_cls，也有rnn_cls等等。
02.因为本文件使用方法bert 预测文本分类，故此文件名为 BertCls.py
03.由bert得到的向量大小值是768维，不符合最后的分类值，所以需要使用线性网络进行一个变换，
之后再映射！
"""
import torch as t
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelWithLMHead
from config import PRETRAINED_MODEL_NAME


"""初始化BertCls
01.继承自nn.Module
"""
class BertCls(nn.Module):
    def __init__(self):
        super(BertCls,self).__init__()
        # 定义预训练模型，也就是说这里把预训练的模型作为我们模型的一部分
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        self.bert = AutoModelWithLMHead.from_pretrained(PRETRAINED_MODEL_NAME)
        self.linear = nn.Linear(768,217)  # bert的输出是768维；标签有217种
        self.sig = nn.Sigmoid()  # 对数据进行sigmoid操作

    def forward(self,x):
        """
        :param x: 这里的x是个list，list的大小就是batch_size，每项都是一个标签摘要
            x是输入数据，也就是{inputs_id,token_type_ids,attention_mask} + labels
        :return:x
        """
        # 将数据放入到bert中，但是需要将最后得到的输出调整一下，我们只要拿CLS位的向量就OK了
        # 问题就在于怎么拿CLS位的向量？？
        # 也就是说，这里的x就是bert模型的输入数据，需要知道它长什么样子
        x = self.bert(x)

        x = self.linear(x)  # 进行线性变换
        x = self.sig(x)  # 得到每个的概率
        return x   # 返回的大小是 1*217