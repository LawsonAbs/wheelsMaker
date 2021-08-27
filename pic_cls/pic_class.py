'''
将图片分类训练。
使用CNN
使用3层线性网络
'''
# from torch.utils.tensorboard import SummaryWriter   
from visdom import Visdom
from tqdm import tqdm
import torchvision
from torchvision import transforms
import torch as t
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from torchvision.datasets import mnist

transform = transforms.Compose(
    [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
     transforms.Normalize((0.1,), (0.34,)) # 这个方法必须放在ToTensor()之后
     ])


class Model(nn.Module):
    def __init__(self,in_1,out_1,in_2,out_2,in_3,out_3):
        super().__init__() 
        # 定义三个线性层
        self.linear_1 = nn.Linear(in_features= in_1, out_features= out_1)
        self.linear_2 = nn.Linear(in_features= in_2, out_features= out_2)
        self.linear_3 = nn.Linear(in_features= in_3, out_features= out_3)
        
        # 使用激活函数
        # sigmoid 函数可以用多次吗？ => 可以的        
        # self.active = nn.Sigmoid()
        self.active = nn.ReLU()
        
        # 防止过拟合
        self.dropout = nn.Dropout()
    
    # 输入数据是x
    def forward(self,x):
        # 有没有简单的不用使用堆叠的方法？
        tempx = self.linear_1(x)
        tempx = self.active(tempx)
        tempx = self.linear_2(tempx)
        tempx = self.active(tempx)
        tempx = self.linear_3(tempx)
        
        return tempx


if __name__ == "__main__":
    #加载数据
    # 这个raw_mnist_data 是一个 PIL.Image.Image 类型的数据，需要转换成tensor，所以这里用了一个转换操作
    # 得到的数据类型是 torchvision.datasets.MNIST 。其中的数据作为其属性存储
    mnist_data = torchvision.datasets.MNIST('./data/',download=True,train=True,transform=transform)
    train_data = mnist_data.train_data.float()
    #tran = transforms.Normalize((0),(255))
    #train_data = tran(train_data)
    print(train_data[0])
    test_data = mnist_data.test_data.float()
    train_labels = mnist_data.train_labels
    test_labels = mnist_data.test_labels
    
    # 对数据做处理
    batch_size = 16
    # 训练和测试数据
    train_data_loader = DataLoader(train_data,
                            batch_size=batch_size,
                            # shuffle=True
                            )
    train_label_loader =  DataLoader(train_labels,
                            batch_size=batch_size,
                            # shuffle=True
                            )
    test_data_loader = DataLoader(test_data,
                            batch_size=batch_size,
                            # shuffle=True
                            )
    test_label_loader =  DataLoader(test_labels,
                            batch_size=batch_size,
                            # shuffle=True
                            )
    model = Model(784,256,256,128,128,10) #新建一个模型
    train_epoch = 100
    lr = 1e-3
    opti = t.optim.Adam(model.parameters(),lr=lr) # 选择优化器
    criterion = nn.CrossEntropyLoss() # 使用交叉熵作为损失
    
    viz = Visdom()
    win = "train_loss"
    global_step = 0
    model.train()
    for epoch in tqdm(range(train_epoch)):
        for data,gold in tqdm(zip(train_data_loader,train_label_loader)):
            data = data.reshape(batch_size,-1) # 将28*28修改成一维向量
            # print(data) # data.shape (batch_size,28,28)  MNIST 中的数据集格式是28*28
            # print(gold)
            pred = model(data) # 得到返回的标签值
            loss = criterion(pred,gold)
            loss.backward()
            opti.step()
            opti.zero_grad()
            global_step+=1
            viz.line([loss.item()], [global_step],win=win, update="append")
            
            # 打印优化参数的过程 => 验证是否优化了
            for param in model.named_parameters():                
                name,value = param
                if "linear_3.weight" in name:
                    print(value.shape)
                    print(name)
                    print(f"value={value[0,0:10]}")