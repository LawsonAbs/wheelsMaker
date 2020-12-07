'''
Author: LawsonAbs
Date: 2020-12-06 15:18:43
LastEditTime: 2020-12-07 09:48:46
FilePath: /wheels/seq2seq/translate_main.py
01.实现的功能是：将英文翻译成中文
02.这里使用torchtext 将xml文件内容变成可以用torchtext 处理的Dataset 
'''
#from importlib import import_module
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import sys
sys.path.append(r'.')

import processdata as pd
import spacy
import numpy as np
import random
import math
import time
import jieba # 中文分词器
import config as conf
from seq2seq import Seq2Seq
from encoder import Encoder 
from decoder import Decoder 


from torchtext.data import TabularDataset,Field, BucketIterator # 构建数据集
from torchtext.data import Dataset as tDataset # 以防跟torch中的 Dataset 冲突
from torchtext.datasets import Multi30k
#Multi30k.splits()
from torchtext import data

# 我不是很理解这里的SEED 是什么意思
random.seed(conf.SEED)
np.random.seed(conf.SEED)
t.manual_seed(conf.SEED)
t.cuda.manual_seed(conf.SEED)
t.backends.cudnn.deterministic = True

spacy_en = spacy.load("en")

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    01. [::-1]的作用就是 step=-1，start=-1 开始，直到end
    """    
    return [tok.text for tok in spacy_en.tokenizer(text)][::-1]

def tokenize_zh(text):
    return list(jieba.cut(text))

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
    

SRC = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

TRG = Field(tokenize = tokenize_zh, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

# 构建数据集
examples = []
PATH = "/home/lawson/program/wheels/seq2seq/data/" 
src_file = pd.readXml(PATH+"en.xml") # source
trg_file = pd.readXml(PATH+"zh.xml") # target
fields = [("label", SRC), ("text", TRG)]

for src_line, trg_line in zip(src_file, trg_file):
    src_line, trg_line = src_line.strip(), trg_line.strip()
    if src_line != '' and trg_line != '':
        # 需要注意下面这个 fields 的使用
        temp = data.Example.fromlist([src_line, trg_line], fields)
        examples.append(temp)


# 根据得到的example,构建一个Dataset，得到 trainData
allData = tDataset(examples,fields)
trainData,validData,testData = allData.split() #再分割成三份

INPUT_DIM = len(SRC.vocab) 
OUTPUT_DIM = len(TRG.vocab) 
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(encoder,decoder).to(device)
model.apply(init_weights)
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters())     


def train(model, iterator, optimizer, criterion, clip):    
    model.train()    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):        
        src = batch.src
        trg = batch.trg        
        optimizer.zero_grad()        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)        
        loss.backward()        
        t.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()    
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with t.no_grad():    
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0) #turn off teacher forcing
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)            
            epoch_loss += loss.item()        
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):    
    start_time = time.time()    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        t.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


model.load_state_dict(t.load('tut1-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')