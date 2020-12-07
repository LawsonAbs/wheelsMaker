'''
Author: LawsonAbs
Date: 2020-12-06 15:43:34
LastEditTime: 2020-12-06 19:18:51
FilePath: /wheels/seq2seq/encoder.py
'''
import torch.nn as nn
import torch as t

# 搞一个两层的LSTM 作为encoder
class Encoder(nn.Module):
    """
    01.hid_dim 表示最后LSTM 输出向量的维度
    02.input_dim 表示的是一个 one-hot 向量的维度
    03.emb_dim 表示的是使用nn.Embedding()得到的向量维度
    04.n_layers 表示的是LSTM 该使用几层？
    05.dropout 表示什么样的概率丢弃
    """
    def __init__(self,input_dim,emb_dim,hid_dim,n_layers,dropout) : # 
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim,emb_dim) # 对输入的英文进行编码，然后再作为lstm的输入
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim,hid_dim,n_layers,dropout=dropout) # 使用LSTM进行编码                
        

    def forward(self,x): # 对输入数据进行编码，依次做的处理
        embedding = self.embedding(x)
        embedding = self.dropout(embedding) 
        outputs,(hidden,cell) = self.rnn(embedding)        
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]        
        #outputs are always from the top hidden layer
        return hidden,cell
        