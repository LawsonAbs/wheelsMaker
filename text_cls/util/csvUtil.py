import xlrd
import csv
import os
import sys
import re  # 实现对字符串按照多个字符分割

"""读取某个文件夹下的所有tsv文件，生成到一个List中，这个List的每一项都是字典
"""
def readTsv2List(filePath):
    # 判断filePath 是否是一个dir，如果不是dir但是一个.tsv文件
    if (not os.path.isdir(filePath)) and os.path.exists(filePath) and (filePath.endswith(".tsv")):
        # 如果不是dir，那么直接读取
        print(filePath,"is not a dir")
        summary =[]
        label = []
        # 根据每个子文件得到数据
        with open(filePath, encoding="utf8") as fin:
            reader = csv.DictReader(fin, delimiter="\t")
            for row in reader:  # 这里的row是一个list【也很形象，是不是】
                if row['标签公司主体'] in row['标签摘要']:  # 暂定句子中全包含公司主体
                    label.append({"标签": row['标签']})  # str
                    summary.append({"标签摘要": row['标签摘要'].strip("\"[\']\n")})
        return summary, label  # 返回
        sys.exit(0)

    filePathList = os.listdir(filePath)
    summary = []  # 所有的数据
    label = []
    for file in filePathList:
        file = filePath + file
        #根据每个子文件得到数据
        with open(file, encoding="utf8") as fin:
            reader = csv.DictReader(fin, delimiter="\t")
            for row in reader:  # 这里的row是一个list【也很形象，是不是】
                if row['标签公司主体'] in row['标签摘要']:  # 暂定句子中全包含公司主体
                    label.append({"标签": row['标签']})  # str
                    summary.append({"标签摘要": row['标签摘要'].strip("\"[\']\n")})
    return summary,label  # 返回


"""获取公司信息数据
01.参数：文件地址
02.返回值：list
"""
def getCompanyData2List(filePath):
    """get the dict of short-to-full name dict and full-to-short name dict of the companys
    """
    companyName = []  # 公司名
    data = xlrd.open_workbook(filePath)
    table = data.sheets()[0]  # 第1个sheet
    for rown in range(1, table.nrows):  # 将数据放入到其中
        companyName.append(table.cell_value(rown, 1))
        companyName.append(table.cell_value(rown, 0))

    return companyName


"""根据文件得到标签的信息，并将标签和序列拼成一个dict
"""
def getLabel2Dict(filePath):
    label2Tag={}
    s = xlrd.open_workbook(filePath)
    sheet2 = s.sheet_by_index(1)  # 获取第二张表格
    count_rows = sheet2.nrows  # 获取第二张sheet的行数
    for i in range(1, count_rows):  # 遍历每行的内容
        col2 = sheet2.cell(i, 1)  # 获取第2列的内容
        col2 = str(col2)
        col2 = col2.strip("text:'")
        label2Tag[col2] = i-1
    return label2Tag


if __name__ == "__main__":
    filePath = "../data/other/企业新闻舆情正负面标签_关键词参照.xlsx"
    print(getLabel2Dict(filePath))