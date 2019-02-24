from PIL import Image
import numpy as np
import os


def loaddata(filepath):
    """
    load image data from the speicfic filepath
    `filepath` : the root path of test data include 21 kinds of images
    return : a list and each element is  (images_path,  tag)
    """
    idx = 0
    images = []
    tags = []
    for label in os.listdir(filepath):
        path = os.path.join(filepath, label)
        for img_name in os.listdir(path): # traverse each kind of image
            # img = Image.open(os.path.join(path, img_name))
            idx += 1
            print("loading {}th image feature...".format(idx))
            try:
                fp = open(os.path.join(path, img_name), 'rb')
                pic = Image.open(fp)
                pic_array = np.array(pic)
                fp.close()
                images.append(pic_array)
                tags.append(label)
            except IOError as e:
                pass
    return images, tags