
import os   # 加载接口
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class classifier:

    def traverse(self, dir):  # 遍历根目录下所有图片文件
        images = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                image = os.path.join(root, file)  # 图片文件
                print(image)
                images.append(image)
            return images

    def get_matrix(self, image_path):  # 输入图片路径， 输出图片像素矩阵及其类别
        img_matrix = np.array(Image.open(image_path))
        category = image_path.split('\\')
        return (img_matrix, category[1])

root_dir = 'D:/NEU-dataset'  # 测试
str = r"D:/NEU-dataset\statue\75fcd95f-65e6-30be-a05c-c6e1e01fcb19b.jpg"
c = classifier()
(matrix, type) = c.get_matrix(str)
print(matrix)
print(type)


