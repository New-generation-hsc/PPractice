from skimage import feature
import numpy as np
from PIL import Image
from scipy.stats import itemfreq


class LocalBinaryPatterns(object):

    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, "default")
        x = itemfreq(lbp.ravel())
        # Normalize the histogram
        hist = x[:, 1] / sum(x[:, 1])

        # return the histogram of Local Binary Patterns
        return hist

    def get_feature(self, image_path):
        img = Image.open(image_path).convert('L')
        hist = self.describe(img)
        return hist


desc = LocalBinaryPatterns(8, 1)
print(desc.get_feature('../week2/cover.jpg'))
print(desc.get_feature('../week2/pic.jpg').shape)
print(desc.get_feature('../week2/zebra.png').shape)