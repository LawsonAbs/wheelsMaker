from transformers import AutoTokenizer
import config as con  # 导入配置
from util import csvUtil as csv

LABEL_SET = []  # 所有label的一个集合
LABEL2TAG = {}  # 根据label找到tag，速度更快

"""处理"标签摘要"中的数据 的方法集合
"""
class ProcessData():
    def __init__(self):
        self.initData()  # 初始化数据

    def processData(text):
        """
        1.将*****公司换成公司
        2.去掉一些无用的字段。比如：据报道，日期字段

        :param text: 待处理的文本信息
        :return text: 处理后的文本信息
        """
        # step1.换公司名字
        for name in companyName:
            if name in text and len(name) > 2:
                print(name, '被找到')
                text = text.replace(name, "公司")
                break  # 找到一个后，就直接跳出循环
        return text

    def data2Tokens(self,data,max_length):
        """
        01.功能：将获得的数据data处理成tokens，这里的tokens指的是[{input_id,attention_mask}...]这样的集合
        02.data的样子：
        data = [{'标签摘要': '2019年年报中，格力电器称，根据《暖通空调资讯》发布的数据，....'},{...},...]
        03.tokens 的样子
        tokens =  {'input_ids': tensor([[ 101, 8667,  146,  112,  102],[ 101, 1262, 1330, 5650,  102],[ 101, 1262, 1103, 1304,  102]]),
                    'token_type_ids': tensor([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]),
                    'attention_mask': tensor([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]])
                   }
        """
        summaries = []  # 标签摘要列表
        # 遍历data，拿到每个标签摘要的值
        for item in data:
            summaries.append(item.get("标签摘要"))

        tokenizer = AutoTokenizer.from_pretrained(con.PRETRAINED_MODEL_NAME)
        tokens = tokenizer(summaries,
                  padding='max_length',
                  truncation=True,
                  max_length=max_length,
                  return_tensors='pt')

        return tokens

    def label2vec(self,labels):
        """
        :param labels: 所有标签放在一起的情况
        其值为[{'标签': '行业龙头'}, {'标签': '注册'},{...},....]
        :return: 每个标签对应的one-hot向量
        """
        labelsVec = []
        for label in labels: # 每一项都是一个标签值
            temp = [0 for _ in range(217)]  # 搞一个one-hot 向量，从而
            val = label.get("标签")  # 得到该标签对应的值
            idx = LABEL2TAG.get(val)
            temp[idx] = 1
            labelsVec.append(temp)
        return labelsVec

    def initData(self):
        """
        初始化使用的数据集
        01.读取标签数据集到 LABEL_SET 中
        :return:
        """
        filePath = "/home/liushen/program/wheels/text_cls/data/other/企业新闻舆情正负面标签_关键词参照.xlsx"  # 得到标签文件的地址
        LABEL2TAG = csv.getLabel2Dict(filePath)

    def loadData(filePath, comData):
        """
            参数：
            filepath:数据文件地址
            comData :CompayData的实例
        """
        # TODO： 这里应该写几个标签，然后传入到 readCsv2List 中
        # 将返回的数据集写入到comData中
        data = readCsv2List(filePath)

        # step1.加载bert模型
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

        # 拿到标签摘要的数据
        # step2.根据tokenizer得到相应的输入数据
        labelList = []  # 每条待处理数据的标签
        labelSummary = []  # 将所有的labelSummary 放到一个list中。也就是[[str],[str],...,[str]]这种
        maxLen = 0  # 发现最大的长度是335
        reportCnt = 0  # 标签摘要中出现"据报道"的次数
        for rowDict in data:  # 因为data是一个list，里面装的是dict
            text = rowDict.get('标签摘要')
            label = rowDict.get("标签")
            print(text)
            text = processData(text)  # 数据预处理
            print(text)
            maxLen = max(maxLen, len(text))
            if "据报道" in text:
                reportCnt += 1
            labelSummary.append(text)  # labelSummary对应的就是文本中"标签摘要"一栏
            # print(len(labelSummary))
            labelList.append(label)  # 获取其标签
            # print(labelSummary)
            """
            1.标签内容可能会有重复，因为是同条内容，可能会有多个标签，即如下的情况：
            5月20日晚间，暴风集团发布公告称，收到证监会《调查通知书》，因公司未按期披露定期报告，涉嫌信息披露违法违规，证监会决定对公司进行立案调查
            不良行为              

            5月20日晚间，暴风集团发布公告称，收到证监会《调查通知书》，因公司未按期披露定期报告，涉嫌信息披露违法违规，证监会决定对公司进行立案调查
            专业违规 
            """
            print(reportCnt, "============", label)
        """
        1.input = []  # 待放到bert中处理的数据 
        2.搞清楚这个padding的作用是什么？？？这个见我github    
        """
        input = tokenizer(labelSummary,  # 调用类PreTrainedTokenizerBase  下的__call__方法
                          max_length=MAX_LENGTH,
                          padding=True)
        """temp 是个字典，的内容形式如下：
        {'input_ids': tensor([[  101,  7592,  1010,  2026,  3899,  2003, 10140,   102]]), 
        'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0]]), 
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
        }
        """

        # 往实例中设置值
        comData.data = input.get('input_ids')  # 获取input_ids作为输入，是个list
        comData.label = label