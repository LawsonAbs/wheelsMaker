'''
Author: LawsonAbs
Date: 2020-11-29 22:23:15
LastEditTime: 2020-11-30 08:25:58
FilePath: /wheels/puns_classify/main.py
'''

# xml 文件的path信息
xmlPath = '/home/lawson/program/wheels/puns_classify/data/test/subtask1-homographic-test.xml'

from transformers import AutoTokenizer

def getId():
    str = "\"I ate no soap,\" Tom lied."
    print(len(str)) # 得到的是单个字符的个数
    toknizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    out = toknizer(str)
    print(out)

getId()