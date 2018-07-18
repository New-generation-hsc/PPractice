import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#提取图片特征
def getImgV(imgpwd):
    img=Image.open(imgpwd)
    resize_width=64
    resize_height=64
    smaller_image = img.resize((resize_width, resize_height))
    gray = smaller_image.convert("L")
    data=np.array(gray)
    mean=data.mean()
    data=data-mean
    for  i in range(0,64):
        for j in range(0,64):
            if data[i][j]>0:
                data[i][j]=1
            else:
                data[i][j]=0
    v_data=data.sum(1)
    return v_data
#计算两个向量的欧氏距离
#flag 0：相同，1：不同
def check(vec1,vec2):
    dist = np.linalg.norm(vec1 - vec2)
    flag=1
    if dist <45:#相似度阈值初步设置为45
        flag=0
    return flag
#比对pwd1、pwd2两个目录下的图片，并删除pwd1目录下与pwd2中目录相同的图片
def imgCheck(pwd1,pwd2):
    filelist1=os.listdir(pwd1)
    filelist2=os.listdir(pwd2)
    count=0
    for i in range(len(filelist1)):
        newpwd1=os.path.join(pwd1,filelist1[i])
        v1=getImgV(newpwd1)
        for j in range(len(filelist2)):
            newpwd2=os.path.join(pwd2,filelist2[j])
            v2=getImgV(newpwd2)
            if check(v1,v2)==0:
                count+=1
                os.remove(newpwd1)#删除图片
                break
    print(count)

if __name__ == '__main__':
    pwd1='C:/Users/lenovo/Desktop/fox'
    pwd2='C:/Users/lenovo/Desktop/fox2'
    imgCheck(pwd1,pwd2)
    