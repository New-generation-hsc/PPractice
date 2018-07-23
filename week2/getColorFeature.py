from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import math  

#测试图片
path='cover.jpg'
img=io.imread(path)
'''
函数功能：生成图片颜色直方图
img：图片像素矩阵
'''
def getColorHist(img):
	r=img[:,:,0].flatten()
	g=img[:,:,1].flatten()
	b=img[:,:,2].flatten()
	plt.figure(num='colorhist', figsize=(21,8))
	plt.subplot(1,3,1)
	plt.title('red')
	n, bins, patches = plt.hist(r, bins=256, normed=1,edgecolor='None',facecolor='red')
	plt.subplot(132)
	plt.title('green')
	n, bins, patches = plt.hist(g, bins=256, normed=1,edgecolor='None',facecolor='green')
	plt.subplot(133)
	plt.title('blue')
	n, bins, patches = plt.hist(b, bins=256, normed=1,edgecolor='None',facecolor='blue')  
	plt.show()

'''
函数功能：生成颜色自相关相关矩阵，并返回其特征向量
返回值：  图片某一颜色通道自相关直方图特征向量
参数说明：
		  img：图片某一颜色通道
		  d：颜色相关距离
		  max_color：颜色种类个数
'''
def AutoCorrelogram(img,d,max_color):
	w=img.shape[0]
	h=img.shape[1]
	autoCorrelogram=np.zeros([w,h])#颜色自相关矩阵
	temp=np.ones([w,h])
	ext_img=np.zeros([w+2*d,h+2*d])
	ext_img[d:d+w,d:d+h]=img #拓展矩阵
	#获取颜色自相关矩阵
	for i in range(2*d+1):
		if i==0 or i==2*d:
			for j in range(2*d+1):
				autoCorrelogram=autoCorrelogram+temp*(img==ext_img[i:i+w,j:j+h])
		else:
			autoCorrelogram=autoCorrelogram+temp*(img==ext_img[i:i+w,:h])
			autoCorrelogram=autoCorrelogram+temp*(img==ext_img[i:i+w,2*d:2*d+h])
	#获取颜色自相关直方图向量
	autoHist=[]
	lastSum=0
	for i in range(1,max_color+1):
		autoHist.append((autoCorrelogram<i*256/max_color).sum()-lastSum)
		lastSum = (autoCorrelogram<i*256/max_color).sum()
	return autoHist

'''
函数功能：返回图像r,g,b三通道的颜色自相关向量
参数说明：
		img：图片像素矩阵
		  d：颜色相关距离
   max_color:颜色种类个数
返回值 ：图片r,g,b三颜色通道自相关直方图特征向量
'''
def rgbAtuo(img,d,max_color):
	r=img[:,:,0]
	g=img[:,:,1]
	b=img[:,:,2]
	rf=[]
	gf=[]
	bf=[]
	rf=autoCorrelogram(r,d,max_color)
	gf=autoCorrelogram(r,d,max_color)
	bf=autoCorrelogram(r,d,max_color)
	return rf,gf,bf

if __name__ == '__main__':
	rgbAtuo(img,1,64)

