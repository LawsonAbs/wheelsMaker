'''
Author: LawsonAbs
Date: 2020-12-06 19:13:02
LastEditTime: 2020-12-07 10:47:53
FilePath: /wheels/seq2seq/seq2seq.py
'''
"""
这里综合 Encoder 和 Decoder 给出一个 综合的模型
"""
import torch as t
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
            
    # 解释一下
    # src,trg 表示的都是训练数据，只不过一个是x[src]，一个是y[trg,标签]
    def forward(self, src,trg,teacher_forcing_ratio = 0.5):
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # 为什么叫 batch_size ? => 因为就是实际中的batch_size 
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = t.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        # first input to the decoder is the <sos> tokens
        # tensor中独有的切片操作。 取第0维的所有数，也就是sos的向量表示。
        # 这里得到的结果是所有batch的sos 表示。其值为：tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], device='cuda:0')
        input = trg[0,:]
        
        # 逐个生成得到最后的预测结果
        for i in range(1, trg_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[i] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[i] if teacher_force else top1
        
        return outputs