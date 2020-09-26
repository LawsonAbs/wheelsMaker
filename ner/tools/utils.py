import os
import re  # 正则分割


"""
1.fun:读取单个 .ann 文件，从中获取entity，然后写入到 Dict 中
2.这里以 4.ann 文件作为示例，那么得到的entity 就是[清热]
"""
def getEntity2DictFromFile(entity2Id,filePath):
    with open(filePath, encoding="utf8") as f:
        line = f.readline()  # 依次读取每一行
        while line:
            line = line.strip("\n")  # 去换行
            curKey = re.split('[\t ]',line)[4]  # 取entity
            # 判断字典中是否有过entity
            if curKey not in entity2Id.keys():
                entity2Id[curKey] = len(entity2Id)
            line = f.readline()
    return entity2Id


"""
从某个路径中获取所有的.ann 文件，然后读取其中所有的Entity，返回一个dict
"""
def getEntity2DictFromDir(filePath):
    entity2Id = {}
    fileNameList = os.listdir(filePath)
    #  注意这里的if 写在了 生成表达式的后面
    fileNameList = [name for name in fileNameList if name.endswith(".ann")]
    # 构造文件名，传入到 上面那个函数中，获取 entity
    for name in fileNameList:
        fileRoute = filePath +'/' +name
        # 读取文件，并生产entity
        getEntity2DictFromFile(entity2Id,fileRoute)

    return entity2Id


if __name__ == "__main__":
    entity2Id = getEntity2DictFromDir("/home/liushen/brat/data/train")
    print(entity2Id)