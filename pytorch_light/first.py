import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer


class LitModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    """
    写了这validation_step ，那么trainer 就会自动调用这个方法
    """
    def validation_step(self, batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        return {'val_loss':F.cross_entropy(y_hat,y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        # TODO: do a real train/val split
        dataset = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=32, num_workers=4)
        return loader

    # you need specially call test
    def test_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        return {'test_loss':F.cross_entropy(y_hat,y)}

    def test_epoch_end(self,outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_logs':avg_loss}
        return {'avg_test_loss':avg_loss,'log':tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
        return loader

    def test_dataloader(self):
        dataset = MNIST(os.getcwd(),train=False,download=True,transform=transforms.ToTensor())
        loader = DataLoader(dataset,batch_size=32,num_workers=4)
        return loader


if __name__ == '__main__':
    model = LitModel()

    # most basic trainer, uses good defaults
    # 我使用gpus=3 就会出现问题，这是为什么？？ => 时不时出现问题
    trainer = Trainer(gpus=3,
                      num_nodes=1,
                      max_epochs=50) # 至少训练 100 epoch

    # 这里的fit*()做的操作就相当于一堆其它代码做的功能
    # 哪里指定epoch？
    trainer.fit(model)
