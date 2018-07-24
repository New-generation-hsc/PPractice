from skimage import feature
import numpy as np
from PIL import Image


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
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # return the histogram of Local Binary Patterns
        return hist

    def get_feature(self, image_path, norm=True, eps=1e-7):
        img = Image.open(image_path).convert('L')
        hist = self.describe(img)
        if norm:
            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
        return hist


desc = LocalBinaryPatterns(24, 8)
print(desc.get_feature('../week2/cover.jpg').shape)
print(desc.get_feature('../week2/pic.jpg').shape)
print(desc.get_feature('../week2/zebra.png').shape)