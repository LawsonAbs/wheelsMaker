import torch as t
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel  # 导入模型

"""
几乎是只使用纯Bert 做一个baseline
这里就是接了一层 Linear，将bert的输出搞成 13 类
"""
class BaseLine(nn.Module):
    def __init__(self,inFeatures,outFeatures):
        super(BaseLine,self).__init__()
        # 搞到tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        # 对每batch的数据进行tokenizer，然后交由model生成数据
        self.model = AutoModel.from_pretrained('bert-base-chinese')
        self.linear = nn.Linear(in_features=inFeatures,out_features=outFeatures)
        self.sm = nn.Softmax(1)  # 这里的1 不是很理解

    # 传入一个参数——训练数据
    def forward(self,x):
        x = self.tokenizer(x,
                           padding='max_length',
                           return_tensors='pt',
                           max_len=512,
                           truncation=True)  # 根据输入的data，得到bert的输入x
        print(x)
        outputs = self.model(**x)  # 得到输出
        last_hidden_states = outputs[0]  # 得到最后一个隐层的输出  shape=(1,512,768)
        last_hidden_states.squeeze(0) # 压缩第0 维上的1，得到 （512,768） 的tensor
        print(last_hidden_states)
        x = self.linear(last_hidden_states)
        out = self.sm(x)  # 这里处理稍有问题
        return out