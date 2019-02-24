from skimage import feature
from PIL import Image
import sys
import numpy as np


classes_to_index = {
    "bear" : 0,
    "bicycle" : 1,
    "bird" : 2,
    "car" : 3,
    "cow" : 4,
    "elk" : 5,
    "fox" : 6,
    "giraffe" : 7,
    "horse" : 8,
    "koala" : 9,
    "lion" : 10,
    "monkey" : 11,
    "plane" : 12,
    "puppy" : 13,
    "sheep" : 14,
    "statue" : 15,
    "tiger" : 16,
    "tower" : 17,
    "train" : 18,
    "whale" : 19,
    "zebra" : 20
}


class HistgoramOrientedGradient(object):

    def __init__(self, orientation=9):
        super(HistgoramOrientedGradient, self).__init__()
        self.orientation = orientation

    def extract(self, img):
        """
        compute the histograms of Oriented Gradients 
        """
        hog = feature.hog(img, orientations=self.orientation,
                          block_norm='L1',
                          cells_per_block=(2, 2),
                          transform_sqrt=True)
        return hog

    def get_feature(self, img):
        """
        img is the pixel matrix, opened by PIL
        """
        img = Image.fromarray(img)
        img = img.convert('L').resize((64, 64), Image.ANTIALIAS)
        return self.extract(img)


class GreyCoMatrix(object):

    def __init__(self, distance=1, angle=[0, np.pi/2]):
        super(GreyCoMatrix, self).__init__()
        if isinstance(distance, list):
            self.distances = distance
        else:
            self.distances = [distance]
        if isinstance(angle, list):
            self.angles = angle
        else:
            self.angles = [0]

    def extract(self, img):
        matrix = np.array(img) // 4
        matrix = feature.greycomatrix(matrix, self.distances, self.angles, levels=64)
        feature_vector = matrix.flatten()
        feature_vector = feature_vector / feature_vector.sum()
        return feature_vector

    def get_feature(self, img):
        """
        img is the pixel matrix, opened by PIL
        """
        img = Image.fromarray(img)
        img = img.convert('L')
        return self.extract(img)


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
                                           self.radius, "uniform")
        # Normalize the histogram
        n_bins = 2 ** self.numPoints
        hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
        return hist


    def get_feature(self, img):
        """
        img is the pixel matrix, opened by PIL
        """
        img = Image.fromarray(img)
        img = img.convert('L').resize((64, 64), Image.ANTIALIAS)
        hist = self.describe(img)
        return hist


class GreyCoprops(object):

    def __init__(self, distance, angle):
        self.distances = distance
        self.angles = angle
        self.prop = ['contrast', 'dissimilarity', 'homogeneity', 'energy']

    def extract_feature(self, img):
        matrix = np.array(img) // 4
        matrix = feature.greycomatrix(matrix, self.distances, self.angles, levels=64)
        feature_vector = []
        for p in self.prop:
            img_feature = feature.greycoprops(matrix, p).flatten()
            feature_vector.append(img_feature)
        features = np.hstack(feature_vector)
        return features / features.sum()

    def get_feature(self, img):
        img = Image.fromarray(img)
        img = img.convert('L')
        return self.extract_feature(img)


lbp = LocalBinaryPatterns(8, 1)
hog = HistgoramOrientedGradient()
glcm = GreyCoMatrix()
coprop = GreyCoprops([1, 2], [0, np.pi / 2, np.pi, 3 * np.pi / 4])


def extractfeature(images, tags):
    """
    extract the lbp, hog, glcm feature, and concatenate the feature to a long feature
    `img_info` : a list of tuple include img matrix and img tag
    return : a list of tuple include img feature and tag
    """
    idx = 0
    res = [] # each element is a tuple include the image feature and image tag
    for (img, tag) in zip(images, tags):
        idx += 1
        print("extract {}th image feature...".format(idx))
        try:
            #lbp_feature = lbp.get_feature(img) # extract lbp feature
            #hog_feature = hog.get_feature(img) # extract hog feature
            glcm_feature = glcm.get_feature(img) # extract glcm feature
            # # prop_feature = coprop.get_feature(img)
            # feature = np.hstack((lbp_feature, hog_feature, glcm_feature))
            res.append((glcm_feature, classes_to_index[tag]))
        except IOError as e:
            pass
        except OSError as e:
            pass
    return res