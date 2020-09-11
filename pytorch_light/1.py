"""
lighting 将代码分成三部分：
Research code: 特殊的系统，以及这些系统怎么被训练的。由LightningModule 抽象出来
Engineering code：与训练这个系统相关的代码。
    例如：什么时候停止，分布式训练等。这部分代码由Trainer抽象出来
non-essential code：能够帮助研究者但是其实跟研究的代码关系不大。
    例如：检查梯度；写往tensorboard的 日志
"""
from torch.optim import Adam
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader,random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets,transforms
from pytorch_lightning import Trainer


class LitMNIST(LightningModule):
    def __init__(self):
        super().__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x

    """
    separate the things that write to disk in data-processing 
    from things like transforms which happen in memory
    """
    def prepare_data(self) :
        MNIST(os.getcwd(),train=True,download=True)

    # 实现的这个方法就是pytorch-lighting 的标志
    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)
        return DataLoader(mnist_train, batch_size=64)

    # 同时也可以返回多个优化器
    def configure_optimizers(self):
        return Adam(self.parameters(),lr=1e-3)

    def training_step(self,batch,batch_idx):
        """
        1.这里是没有for循环的，但是其实会在后台用for循环控制(在Trainer中设置)。
        相关的参数有max_epoch,min_epoch等等
        """
        x,y = batch
        logits = self(x)
        loss = F.nll_loss(logits,y)

        # 下面这行代码有问题
        # self.logger.summary.scalar('loss',loss)

        """
        01.为什么要写成这种形式？
        因为第一个要作为loss值返回，后面的都要放到tensorboard中作为一个展示
        02.add logging
        """
        logs = {'train_loss':loss}  # 这是一个字典！！！
        return {'loss':loss,'log':logs}


model = LitMNIST()
trainer = Trainer(gpus=1)
trainer.fit(model)