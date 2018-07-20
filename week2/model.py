"""
The module define the classifier class
include the following function:
-- load data
-- extract features

"""

import numpy as np
from PIL import Image
import re


def get_label_from_path(path):
    """
    the label is the folder name where the image store
    """
    layers = re.split(r'(\\|\\\\|/)', path)
    assert len(layers) >= 3, "Image path must be a valid path include its label"
    return layers[-3]


class Classifier(object):
    def __init__(self):
        super(Classifier, self).__init__()

    @staticmethod
    def load_img(csv_path):
        """
        given an image path, return the image pixel matrix
        and the image label
        """
        raise NotImplementedError("Not Implemented Yet")

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

    label = get_label_from_path("\\path\\new\\image.jpg")
    print(label)