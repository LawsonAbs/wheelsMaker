from transformers import AutoTokenizer
from torch.utils.data import Dataset
import config as con

'''
DataSet 是一个抽象类，必须由其他类继承，实现其抽象方法
'''
class MyDataSet(Dataset):
    # 在构造的时候就初始化配置信息
    # 数据：这里直接读取到内存中。
    # 标签：这里也是直接读取到内存中。
    # 有的时候如果数据量大，就不能一次性读到内存中
    def __init__(self,data,labels):
        super().__init__()
        self.data = data
        # 搞到tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.labels = labels

        # 处理labels，如果lables不够长，则延伸到512，补0
        for label in labels:
            for i in range(len(label),512):
                label.append(0)

    # 返回的数据就是要送到模型中进行训练的数据
    def __getitem__(self, index: int):
        # 拿到第index 条数据
        text = self.data[index]
        label = self.labels[index]
        # label = t.tensor(label) # 转为tensor

        # 拿到tokenizer 中生成可使用数据，逐条生成【因为这个是在 __getitem__()方法中用到，所以就注定是逐条生成】
        inputs = self.tokenizer(text,
                           padding='max_length',
                           #return_tensors='pt',  # 这里如果使用了pt，则会涉及到数据存放位置的问题
                           max_length=con.MAX_LENGTH, # 这个参数差点儿害了我！是max_length,而不是max_len!!!
                           truncation=True)  # 根据输入的data，得到bert的输入 inputs

        # 注意这里是一个字典，而且inputs 还是一个字典！
        res = {
            'inputs':inputs,
            'label': [0]+label[:-2]+[0]
        }
        return res

    def __len__(self):
        return len(self.data)  # 返回总数据集的大小