from transformers import BertTokenizerFast  # 必须使用这个，才能用 return_offset_mapping 参数
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
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
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

        # 拿到tokenizer 中生成可使用数据，逐条生成【因为这个是在 __getitem__()方法中用到，所以就注定是逐条生成】
        inputs = self.tokenizer(text,
                           padding='max_length',
                           #return_tensors='pt',  # 这里如果使用了pt，则会涉及到数据存放位置的问题
                           max_length=con.MAX_LENGTH, # 这个参数差点儿害了我！是max_length,而不是max_len!!!
                           return_offsets_mapping=True, # 返回下标地址
                           truncation=True)  # 根据输入的data，得到bert的输入 inputs

        '''
        在tokenizer 之后，会导致label进行变换，所以这里需要做一个相应的改变操作        
        '''
        input_ids = inputs["input_ids"]
        offset_mapping = inputs["offset_mapping"]  # 这里的原来的文本到当前的token的下标映射过程，是个list
        tempLabel = [] # 用于放临时被修改的label，因为无法动态的修改label
        for i,item in enumerate(offset_mapping):  # 从1开始遍历list，修改label
            left,right = item  # 拿到每一项
            # 如果对于不存在的id，则需要删除 => 因为动态删除存在问题，所以采取重写的方法
            if (left + 1) < right:  # 说明是多个字母搞一起成为一个token了，这种直接将其label设为0
                tempLabel.append(0)
            elif left+1 == right: # 如果就是单个字符
                tempLabel.extend(label[left:right])  # 插入单个字符
            elif left == right == 0:  # 说明是起始的 CLS向量
                tempLabel.append(0)  # 追加0

            # 下面开始测试这个标注tempLabel的过程对不对。 这个过程和后面的任务是一致的
            # curWord = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            # print(curWord,tempLabel[i])

        for i in range(len(tempLabel),512): # 如果不足512 个label，则需要补齐
            tempLabel.append(0)  # 补0
        # 注意这里是一个字典，而且inputs 还是一个字典！
        res = {
            'inputs': inputs,
            'label': tempLabel
        }

        # 测试 token和label是否对应一致，与上面for的功能相似
        # print(len(input_ids))  # 输出inputs_ids
        ''' 
        out = self.tokenizer.convert_ids_to_tokens(input_ids)
        for a, b in zip(out, tempLabel):
            print(a, b)
        '''
        return res

    def __len__(self):
        return len(self.data)  # 返回总数据集的大小


'''
DataSet 是一个抽象类，必须由其他类继承，实现其抽象方法
01.为实现预测做的一个DataSet
'''
class MyTestDataSet(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = data
        # 搞到tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    def __getitem__(self, index: int):
        text = self.data[index]
        inputs = self.tokenizer(text,
                           padding='max_length',
                           max_length=con.MAX_LENGTH, # 这个参数差点儿害了我！是max_length,而不是max_len!!!
                           return_offsets_mapping=True, # 返回下标地址
                           truncation=True)  # 根据输入的data，得到bert的输入 inputs
        res = {'inputs': inputs}
        return res

    def __len__(self):
        return len(self.data)