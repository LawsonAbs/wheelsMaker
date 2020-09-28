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
        # 对每batch的数据进行tokenizer，然后交由model生成数据
        self.model = AutoModel.from_pretrained('bert-base-chinese')
        self.linear = nn.Linear(in_features=inFeatures,out_features=outFeatures)
        self.sm = nn.Softmax(2)  #

    # 传入一个参数——训练数据
    def forward(self,x):
        outputs = self.model(**x)  # 得到输出  => 这个方法实际会调用什么方法？
        last_hidden_states = outputs[0]  # 得到最后一个隐层的输出  shape=(BATCH_SIZE,MAX_LENGTH,768)
        last_hidden_states.squeeze(0) # 压缩第0 维上的1，得到 （512,768） 的tensor
        #print(last_hidden_states)
        out = self.linear(last_hidden_states)
        #out = self.sm(out)  # 这里处理稍有问题
        return out