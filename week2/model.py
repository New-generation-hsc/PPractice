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
<<<<<<< HEAD
	the label is the folder name where the image store
	"""
=======
    the label is the folder name where the image store
    """
>>>>>>> 015d3d79c7ba6fd871334f59a845bcb113363acb
    layers = re.split(r'(\\|\\\\|/)', path)
    assert len(layers) >= 3, "Image path must be a valid path include its label"
    return layers[-3]


class Classifier(object):
    def __init__(self):
        super(Classifier, self).__init__()

    @staticmethod
<<<<<<< HEAD
    def load_img(img_path):
        """
		given an image path, return the image pixel matrix
		and the image label
		"""
        label = get_label_from_path(img_path)
        img = Image.open(img_path)
        return np.array(img), label
=======
    def load_img(csv_path):
        """
        given an image path, return the image pixel matrix
        and the image label
        """
        raise NotImplementedError("Not Implemented Yet")
>>>>>>> 015d3d79c7ba6fd871334f59a845bcb113363acb

    @staticmethod
    def save_feature(feature, feature_path):
        """
<<<<<<< HEAD
		save the feature to binary data such as an image
		feature_path : the path where the feature save
		"""
        raise NotImplementedError("Not Implemented Yet")

    def extract_feature(img, extract_func):
        """
		the function is responsible for extracting features according to specific function
		"""
=======
        save the feature to binary data such as an image
        feature_path : the path where the feature save
        """
        raise NotImplementedError("Not Implemented Yet")

    @staticmethod
    def extract_feature(img, extract_func):
        """
        the function is responsible for extracting features according to specific function
        """
>>>>>>> 015d3d79c7ba6fd871334f59a845bcb113363acb
        raise NotImplementedError("Not Implemented Yet")


if __name__ == "__main__":
<<<<<<< HEAD
    label = get_label_from_path("\\path\\new\\image.jpg")
    print(label)
=======

    label = get_label_from_path("\\path\\new\\image.jpg")
    print(label)
>>>>>>> 015d3d79c7ba6fd871334f59a845bcb113363acb
