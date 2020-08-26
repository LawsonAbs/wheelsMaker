import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context='talk')

"""
1.A standard Encoder-Decoder architecture. Base for this and many 
other models.
上面这句话的意思是：这个类是当前 Attention 也是其他许多模型的基础。也就是encoder-decoder结构的基础
2. 这个类是什么意思？
主要就是一个体现一个架构。
因为我们在调用模型的时候，不可能单个单个的调用，而是想把数据传入一个整体的（融合了各个部分）
模型中，这个类的作用就是如此。 => 即将encoder 和 decoder 融合成一个整的模型
3.在模型中详解每个参数的作用
4.将任务拆分，分解到encoder ,decoder 分别完成
"""
class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_embed,  # 这个是什么意思？ =>同下面的tgt_embed一样，都是个Sequential
                 tgt_embed,
                 generator):  # ？？ =》 见后，就是一个linear transformer + softmax
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):

        """
        1.这个memory 对应的实际含义是什么？，就是输入数据经过Encoder这个模块得到的结果。
        这个 memory 是我自己手动重组过的 2.这里的src 和 tgt的值是相同的
        """
        memory = self.encode(src,src_mask)
        return self.decode(memory,
                           src_mask,
                           tgt,
                           tgt_mask)
    """ 
    1.self.encoder(...) 就会去调用被实例化的Encoder的forward()方法
    2.这里的src_embed()是一个sequential，所以后面会调用sequential的forward()方法"""
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # 调用decoder()
        # 对应架构图中的output Embedding，可以看到这里的 tgt_embed()是一个sequential，集合了Positional Embedding
        return self.decoder(self.tgt_embed(tgt), # 这里的tgt是初始值，就是生成的数字序列
                            memory,
                            src_mask,
                            tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step"
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)  # 这里是取Linear 步骤

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)  # 这里取softmax


"""下面才是主要的实现
"""
def clones(module, N):
    "produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)  # 论文中的N是6层
        self.norm = LayerNorm(layer.size)

    """
    1.这里的mask是全True的tensor; torch.size([30,1,10])
    2.这里的forward()函数把 整个Encoder 中所有的模块都给执行完
    """
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        "这里的-1？？"
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


"""这层主要的功能就是实现每个 encoder layer 之间的连接，包括:LayerNorm功能; 残差网络功能
"""
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        01.这边就能看出来这是一个残差网络的连接
        首先x就是普通 残差网络中的 虚线；
        self.dropout(sublayer(self.norm(x))) 就是经过网络层处理之后的值
        02.这个sublayer 即是传入的参数，但是不理解这里为什么还有一个sublayer传入？ => 因为构成的是残差网络，所以需要这么做
        LayerNorm(x + Sublayer(x))。 最外层的LayerNorm 没有在这里实现
        03.这里sublayer 也可以是一个其它的东西，大多数情况下，它是一个layer，但也有可能是一个 lambda表达式
        """
        return x + self.dropout(sublayer(self.norm(x)))


"""
1.Encoder is made up of self-attn and feed forward.
2.这是单层的Encoder，所以叫EncoderLayer。这是构成整个Encoder 的小零件。N个EncoderLayer就构成了整个Encoder。
上面这个知识很重要
"""
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn  # self-attention 层

        """
        1.这个sublayer 是干什么的？
        => 用于执行LayerNorm 和 Resnet的层，而这两层被放到SublayerConnection中了
        2.因为需要两个SublayerConnection 的实例，所以这里copy了2个
        """
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.feed_forward = feed_forward  # FNN层
        self.size = size

    def forward(self, x, mask):
        """
        :param x:
        :param mask:
        :return:
        1. x = self.sublayer[0](x, lambda x: ...)
        这个步骤调用的过程是什么？  => 这里会把一个<function> 对象作为参数传入到 sublayr[0]中
        2.执行完之后，接着再执行 self.feed_forward，将其作为参数传递给sublayer[1]继续执行
        3.可能会有疑问：既然这里的sublayer[0]操作和sublayer[1]操作是一样的，那么为什么不直接进行一次操作？
        我觉这个原因就是需要每层求导更新时需要，如果使用同一层，则会出乱子。
        4.self.self_attn(...)得到的结果
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)  # 这里的第二个参数self.feed_forward 就是一个sublayer。


"""
1.定义这种Decoder 是干什么的？为什么需要？
这个Decoder 代表的意思就是 整个decoder框架，它是由 N个decoder layer 构成的。
"""
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)  # 传入的是什么layer，就clone什么
        self.norm = LayerNorm(layer.size)  # 这里的layer.size 不是很理解

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:# 这个就会调用DecoderLayer 中的forward()函数
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 为什么这里的数字就是3？
        # 因为 Add&Norm 这种层在decoder layer中有三个
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    # memeory 是干什么的？ =》 是Encoder 块处理后得到的数据
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # 这是self-attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))  # 这是对encoder传入的数据进行attention
        return self.sublayer[2](x, self.feed_forward)

"""Mask out subsequent positions"""
def subsequent_mask(size):
    attn_shape = (1, size, size) # ??
    """
    subsequent_mask 的size 是 (1,9,9)
    [[[0 1 1 1 1 1 1 1 1]
      [0 0 1 1 1 1 1 1 1]
      [0 0 0 1 1 1 1 1 1]
      [0 0 0 0 1 1 1 1 1]
      [0 0 0 0 0 1 1 1 1]
      [0 0 0 0 0 0 1 1 1]
      [0 0 0 0 0 0 0 1 1]
      [0 0 0 0 0 0 0 0 1]
      [0 0 0 0 0 0 0 0 0]]]
    1应该表示被mask了
    """
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') # triu函数为取矩阵上三角形
    print(subsequent_mask)
    return torch.from_numpy(subsequent_mask) == 0


"""
定义至关重要的 attention 函数
01.观察需要哪些参数 mask为全True的tensor，size为(30,1,1,10)
"""
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1) # 难道这里的 transpose(-2,-1)操作就是转置了吗？
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # 计算scores
    if mask is not None:  # 这里是唯一一个在attention中用到mask的地方，这就是进行屏蔽的操作。
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # 针对具体的维度取softmax
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


"""
1.实现multi-head attention。不过这里直接定义了一个类，为什么不是函数呢？
2.
01.继承nn.Module 类
02.
"""
class MultiHeadedAttention(nn.Module):
    """
    :parameter
        h：代表做h次线性投影。我觉得就是multi-head中国head的个数，在本例子中，传入的h为8
        d_model: 通常是512
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        self.d_k = d_model // h  # 因为最后要拼凑起一个维度 为d_model 大小的向量，而每个h(head)均分得到d_k
        self.h = h  # 判断合格之后，开始赋值

        # 这里为什么是4？ => 因为在multi-head Attention 中涉及到4个Linear 操作
        # 三个linear 分别应用在V，K，Q上，最后一个应用在concat的结果上
        # 为什么需要做这个 linear 操作？=> 可以见我个人博客(lawson-t.blog.csdn.net)说明
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # 为什么要做这个unsqueeze()操作？
        nbatches = query.size(0)  # 代表什么意思？

        # 1) Do all the linear projections in batch from d_model => h x d_k
        """ 这里是一个列表生成表达式
        01.功能如下：依次处理（做线性变换，linear projections）query,key,value 这三个tensor
        02.将得到的结果变一下形状，具体是 .view(nbatches,-1,self.h,self.d_k).transpose(1,2)这个操作
        03.这行代码对于python新手不好懂     
        04.下面的temp完全是为了方便看整个过程query 的变化，并在其后注明了每个时刻的维度        
        """
        temp = self.linears[0](query)  # (10,4,512) => (10,4,512)
        temp = temp.view(nbatches,-1,self.h,self.d_k)  # (10,4,8,64)
        temp = temp.transpose(1, 2) # (10,8,4,64)
        # 为什么要transpose?
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value,  # 其实就是一个函数，没有什么特别的地方
                                 mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)  # 最后还有一个linear


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    """
    1.这里的vocab 是什么？
    2.这个类的功能：use learned embeddings to convert the input tokens and output tokens to vectors of dimention d_model
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

"""
1.Here we define a function that takes in hyperparameters and produces a full model.
2.The dimensionality of input and output is d_model = 512, and the inner-layer has dimensionality
d_ff = 2048.
"""
def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)  # 8头的attention
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    # 得到一个实例
    # 注意这里并没有按照架构顺序来写。关键原因在于：这里只是传参，真正的架构实现是在EncoderDecoder()中
    # c(attn)是不是会得到一个全新的引用？
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),c(ff), dropout), N),
        # 为什么这里使用Sequential? => 因为这是叠加了两层，所以需要使用 Sequential
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:  # 这个trg 就是tgt
            self.trg = trg[:, :-1] # 这个相当于直接把数组完整copy过去
            """
            这个没有要任一维的起始元素，如果要的话，则是应该写成 trg[:,0:]，那么其维度即是torch.size([30,9])，
            下面给出一个示例
            tensor([[1, 3, 2],
                    [1, 4, 7]])

            tensor([[3,2],
                    [4,7]])
            但是为什么要这么做呢？
            """
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    """
    1.对下面这个方法不理解？
    2. @staticmethod 这个方法
    """
    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        1.这里的pad是什么意思？ 为什么pad的值为0？ 这里得到的tgt_mask为全 true 的tensor
        2.tgt.size(-1) 是什么意思？  => 取出tensor 最后一维的大小。因为[]中有9个元素，这里就为9
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable( # 这里会执行按位与操作
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    # print(type(data_iter))  # 输出data_iter 的类型  => <class 'generator'>
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    """
    batch 包含的是句子对，在这里就是一个数字对。这一个个数字对就用于训练 Encoder-Decoder 这个架构
    """
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, # size([30,9,512])
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)  # 进入 __call__()函数
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(),
                                    lr=0,
                                    betas=(0.9, 0.98),
                                    eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


"Generate random data for a src-tgt copy task."
def data_gen(V, batch, nbatches):
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 5)))
        data[:, 0] = 1  # 这个操作是将任一维的第1个数据置为1
        src = Variable(data, requires_grad=False) # torch.size([30,5])
        tgt = Variable(data, requires_grad=False)
        """
        这里使用yield 方法，只有在真正需要的时候才会具体返回值
        也就是在run_epoch()函数中的for循环才会真正的得到数据，这里的返回值类型是一个 generator
        """
        yield Batch(src, tgt, 0) # tgt 和 trg 一般应该指的都是target


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    # 熟悉一下 __call__()  方法     # 是在什么时候调用的？  => 计算损失时，就会来调用这个函数
    # x 应该是模型的输出，y是原标签值
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # 这里的loss其实是个Tensor，其size 为torch.Size([1])
        return loss.item() * norm


"""
1.Train the simple copy task.
所以这里的使用的字典大小就 [0—9]总共10个数字
"""
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2) # 注意这里重写了默认参数N的值，改为了2
model_opt = NoamOpt(model.src_embed[0].d_model,
                    1, 400,
                    torch.optim.Adam(model.parameters(),
                                     lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 10, 5), # 10：表示数据的维度；  5表示数据的共有多少个batch
              model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 10, 5),
                    model,
                    SimpleLossCompute(model.generator, criterion, None))) # 可以看到这里的在做测试是，是没有优化器的


# def greedy_decode(model, src, src_mask, max_len, start_symbol):
#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
#     for i in range(max_len - 1):
#         out = model.decode(memory, src_mask,
#                            Variable(ys),
#                            Variable(subsequent_mask(ys.size(1))
#                                     .type_as(src.data)))
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.data[0]
#         ys = torch.cat([ys,
#                         torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
#     return ys
#
#
# model.eval()
# src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
# src_mask = Variable(torch.ones(1, 1, 10))
# print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
