"""
The module define the classifier class
include the following function:
-- load data
-- extract features

"""

import numpy as np
from PIL import Image
import os


class Classifier(object):
    def __init__(self):
        super(Classifier, self).__init__()

    @staticmethod
    def load_img(image_path):
        """
        given an image path, return the image pixel matrix
        and the image label
        """
        img = Image.open(image_path)
        return np.array(img)

    @staticmethod
    def save_feature(feature, feature_path):
        """
        save the feature to binary data such as an image
        feature_path : the path where the feature save
        """
        raise NotImplementedError("Not Implemented Yet")

    @staticmethod
    def extract_feature(img, extract_func):
        """
        the function is responsible for extracting features according to specific function
        """
        raise NotImplementedError("Not Implemented Yet")


if __name__ == "__main__":

    label = get_label_from_path("path/image.jpg")
    print(label)