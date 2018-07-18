import numpy as np
import math
img=np.array([[1,2,0],[1,2,0],[1,2,0]])

def creatGLCM(img,d,a,gray):
    w=3
    h=3
    dx=math.floor(d*math.sin(math.radians(a)))
    dy=math.floor(d*math.cos(math.radians(a)))
    print(dy)
    #dx=0
    #dy=0
    glcm=np.zeros([gray,gray])
    for  i in range(w):
        for j in range(h):
            if (i+dx)<w and (j+dy)<h:
                glcm[img[i,j],img[i+dx][j+dy]]+=1
    print(glcm)

if __name__ == '__main__':
    creatGLCM(img,1,0,3)