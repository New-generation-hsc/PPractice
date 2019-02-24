import cv2
import numpy as np
from PIL import Image

class Sobel(object):

    def __init__(self):
        klr = [[-1,0,1],[-2,0,2],[-1,0,1]]
        kbt = [[1,2,1],[0,0,0],[-1,-2,-1]]
        ktb = [[-1,-2,-1],[0,0,0],[1,2,1]]
        krl = [[1,0,-1],[2,0,-2],[1,0,-1]]
        kd1 = [[0,1,2],[-1,0,1],[-2,-1,0]]
        kd2 = [[-2,-1,0],[-1,0,1],[0,1,2]]    
        kd3 = [[0,-1,-2],[1,0,-1],[2,1,0]]
        kd4 = [[2,1,0],[1,0,-1],[0,-1,-2]]
        self.sobel = np.asanyarray([klr, kbt, ktb, krl, kd1, kd2, kd3, kd4])

    def get_feature(self, image_path):
        img = np.array(Image.open(image_path).convert('RGB'))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(64,64))
        res =  np.float32([cv2.resize(cv2.filter2D(gray, -1,k),(15,15)) for k in self.sobel])
        feature_vector = res.flatten()
        return feature_vector / feature_vector.sum()


if __name__ == "__main__":

    sobel = Sobel()
    feature = sobel.get_feature("../week2/cover.jpg")
    print(feature.shape)
    print(feature)
    feature = sobel.get_feature("../week2/pic.jpg")
    print(feature.shape)