import numpy as np
import math
img=np.array([[1,2,0],[1,2,0],[1,2,0]])
#img 图片像素矩阵
#灰度共生矩阵相关参数
#d:距离 angle:角度 max_gray:最大灰度阶
def createGlcm(img,d,angle,max_gray):
    w,h=img.shape()
    dx=math.floor(d*math.sin(math.radians(a)))
    dy=math.floor(d*math.cos(math.radians(a)))
    glcm=np.zeros([gray,gray])
    for  i in range(w):
        for j in range(h):
            if (i+dx)<w and (j+dy)<h:
                glcm[img[i,j],img[i+dx][j+dy]]+=1
    print(glcm)
#测试createGlcm函数
if __name__ == '__main__':
    createGlcm(img,1,0,3)


