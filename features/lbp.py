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
                                           self.radius, method="uniform")
        # Normalize the histogram
        # n_bins = 2 ** self.numPoints
        # hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist


    def get_feature(self, image_path):
        img = Image.open(image_path).convert('L')
        hist = self.describe(img)
        return hist


if __name__ == "__main__":
    
    desc = LocalBinaryPatterns(8, 1)
    print(desc.get_feature('../week2/cover.jpg').shape)
    print(desc.get_feature('../week2/pic.jpg').shape)
    print(desc.get_feature('../week2/zebra.png').shape)