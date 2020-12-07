'''
Author: LawsonAbs
Date: 2020-12-06 15:43:34
LastEditTime: 2020-12-07 10:54:37
FilePath: /wheels/seq2seq/decoder.py
'''
import torch.nn as nn
import torch as t
import config as con

# 搞一个两层的LSTM作为 decoder
class Decoder(nn.Module):
    """
    01.output_dim 表示最后LSTM 输出向量的维度，也就是单词表的大小【输入是一个one-hot向量，也是对应单词表的大小】
    02.input_dim 表示的是经过 Encoder 得到的向量的维度
    03.n_layers 表示的是LSTM 该使用几层？    
    04.emb_dim  
    """
    def __init__(self,output_dim,emb_dim,hid_dim,n_layers,dropout):
        super().__init__()
        self.output_dim = output_dim 
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # 因为decoder也需要从上次的输出中获取，所以需要使用 Embedding 
        #TODO 搞清楚这里的Embedding是随机初始化还是找表得到!!!!!!!!!!!!
        self.embedding = nn.Embedding(output_dim,emb_dim)
        self.dropout = nn.Dropout(dropout)        
        self.rnn = nn.LSTM(emb_dim,hid_dim,n_layers,dropout=dropout) # 使用LSTM进行编码
        self.liear = nn.Linear(hid_dim,output_dim) # 线性映射
        
    # 注意每次的输入 hiddn 和 cell 的大小相同。
    # 这个decoder 每次都只能得到一个输出，需要重复的执行这个方法才能得到最后的一个输出
    def forward(self,input,hidden,cell): # 对输入数据进行编码，依次做的处理
        input = input.unsqueeze(0) # 添加新的一维  => input = [1,batch size]
        embedded = self.dropout(self.embedding(input))

        outputs,(hidden,cell) = self.rnn(embedded,(hidden,cell))
        prediction = self.liear(outputs.squeeze(0))
        
        return prediction,hidden,cell

        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]