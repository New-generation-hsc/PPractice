import os
import csv
import random


def traverse(root):  # 遍历根目录下所有图片文件
    key_image = []   # 每一个图片文件作为key
    value_type = []  # 图片文件对应的类别作为value
    for root, dirs, files in os.walk(root):
        for file in files:
            image = os.path.join(root, file)  # 图片文件
            key_image.append(image)
            category = root[15:]     # 文件夹是其类别
            value_type.append(category)
    total = key_image + value_type  # total列表先存放所有的文件，后存放对应的类别
    return total


def write():  # 所有结果写入一个csv文件
    with open('res.csv', 'w', newline='') as res:
        with open('res.csv', 'a', newline='') as res:
            num = len(resList)
            key_index = 0      # 图片文件下标
            value_index = (int)(num / 2.0)  # 类别下标

            while key_index < (int)(num / 2.0):
                temp = [resList[key_index], resList[value_index]]
                writer = csv.writer(res)
                writer.writerow(temp)  # 写一行
                key_index += 1        # 图片文件下标增1
                value_index += 1     # 类别下标增1


def count():  # 得到不同类别的图片文件数量
    num = len(resList)
    key_index = 0
    value_index = (int)(num / 2.0)
    cnt = 0
    sum = []   # 新建一个存放 各类别图片文件数量 的列表
    while key_index < (int)(num / 2.0):
        if resList[value_index] == resList[value_index - 1]: # 后一个类别和前一个相等
            cnt += 1
        else:
            sum.append(cnt + 1) # 加入列表
            cnt = 0
        key_index += 1
        value_index += 1
    sum.append(cnt + 1)   # 最后一个类别数量加入列表
    sum[0] -= 1           # 第一个元素为0
    return sum


def index():  # 得到不同类别的下标范围
    cnt = 0
    cal = 0
    res = []
    while cnt < len(sum_type):
        cal = cal + sum_type[cnt]
        res.append(cal)
        cnt += 1
    return res


def get_test_set():  # 得到测试集
    num = len(index_range)  # 测试集类别个数 - 1
    loop = 1
    test = []  # 定义测试集列表
    while loop < num:
        temp = resList[index_range[loop - 1]: index_range[loop]]   # 每一个类别的测试集
        cnt = int(sum_type[loop] * 0.2)  # 每次循环产生随机数的个数
        countlist.append(cnt)
        rdm_res = random.sample(temp, cnt)  # 得到每一个类别的随机文件
        test += rdm_res
        loop += 1
    return test  # 测试集


def get_test():   # 得到测试集并写入test.csv中
    with open('test.csv', 'w', newline='') as test:
        with open('test.csv', 'a', newline='') as test:
            index1 = 0
            index2 = 0
            index3 = 0
            while index2 < len(test_res):
                # 依类别写入测试集，循环 20 次
                while index1 < len(countlist):
                    while index3 < countlist[index1]:
                        print(index1, index2)
                        temp = [test_res[index2], types[index1]]  # 图片文件 类别
                        writer = csv.writer(test)
                        writer.writerow(temp)
                        index2 += 1
                        index3 += 1
                    index3 = 0
                    index1 += 1


def get_train():     # 得到训练集并写入train.csv中
    with open('train.csv', 'w', newline='') as train:
        with open('train.csv', 'a', newline='') as train:
            key_index = 0
            value_index = int(len(resList) / 2.0)
            while key_index < value_index:
                if resList[key_index] not in test_res:
                    temp = [resList[key_index], resList[key_index + value_index]]
                    writer = csv.writer(train)
                    writer.writerow(temp)
                key_index += 1


rootDir = "D:/NEU-dataset"    # 根目录
types = ['bear','bicycle', 'bird', 'car', 'cow', 'elk', 'fox', 'giraffe', 'horse', 'koala', 'lion', 'monkey',
            'plane', 'puppy', 'sheep', 'statue', 'tiger', 'tower', 'train', 'whale', 'zebra']

if __name__ == "__main__":
    resList = traverse(rootDir)
    countlist = []
    write()
    sum_type = count()
    index_range = index()
    test_res = get_test_set()
    get_test()
    get_train()
