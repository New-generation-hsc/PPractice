import numpy as np
from skimage import feature
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import os


class HistgoramOrientedGradient(object):

    def __init__(self, orientation=8):
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
        img = img.convert('L').resize((64, 64))
        return self.extract(img)


class GreyCoMatrix(object):

    def __init__(self, distance=1, angle=0):
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
                                           self.radius, "default")
        # Normalize the histogram
        n_bins = 2 ** self.numPoints
        hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
        return hist


    def get_feature(self, img):
        """
        img is the pixel matrix, opened by PIL
        """
        img = img.convert('L')
        hist = self.describe(img)
        return hist


lbp = LocalBinaryPatterns(8, 1)
hog = HistgoramOrientedGradient()
glcm = GreyCoMatrix()


index_to_classes = ['bear','bicycle', 'bird', 'car', 'cow', 'elk', 'fox', 'giraffe', 'horse', 'koala', 'lion', 'monkey',
            'plane', 'puppy', 'sheep', 'statue', 'tiger', 'tower', 'train', 'whale', 'zebra']
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


def loaddata(filepath):
    """
    load image data from the speicfic filepath
    `filepath` : the root path of test data include 21 kinds of images
    return : a list and each element is  (images_path,  tag)
    """
    res = [] # each element is a tuple include the image matrix and image tag
    idx = 0
    for label in os.listdir(filepath):
        path = os.path.join(filepath, label)
        for img_name in os.listdir(path): # traverse each kind of image
            idx += 1
            print("loading {}th image".format(idx))
            img = Image.open(os.path.join(path, img_name))
            res.append((img, label))
    return res


def extractfeature(img_info):
    """
    extract the lbp, hog, glcm feature, and concatenate the feature to a long feature
    `img_info` : a list of tuple include img matrix and img tag
    return : a list of tuple include img feature and tag
    """
    print("extracting images features...")
    res = [] # each element is a tuple include the image feature and image tag
    idx = 0
    for (img, tag) in img_info:
        idx += 1
        print("extract {}th image feature...".format(idx))
        try:
            lbp_feature = lbp.get_feature(img) # extract lbp feature
            hog_feature = hog.get_feature(img) # extract hog feature
            glcm_feature = glcm.get_feature(img) # extract glcm feature
            feature = np.hstack((lbp_feature, hog_feature, glcm_feature))
            res.append((feature, classes_to_index[tag]))
        except OSError as e:
            print("extract failed:->", img)
    return res


def trainmodel(img_feature_info):
    """
    according to the image feature, train the MLP model
    save the MLP model
    """
    print("train model....")
    x = [info[0] for info in img_feature_info]
    y = [info[1] for info in img_feature_info]
    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(170), random_state=1)
    clf.fit(x, y)
    joblib.dump(clf, 'train_model.m')


def testmodel(img_feature_info):
    """
    according to the image feature, predict the img label
    """
    print("test model...")
    x = [info[0] for info in img_feature_info]
    y = [info[1] for info in img_feature_info]
    clf = joblib.load('train_model.m')
    samples_proba = clf.predict_proba(x) # predict the test images probability
    top5_index = np.argsort(-samples_proba, axis=1)[:, :5].tolist()
    res = []
    for (i, tag) in enumerate(y):
        res.append(tag in top5_index[i])
    return res


if __name__ == "__main__":
    img_info = loaddata('../data/images/train')
    feature_info = extractfeature(img_info)
    trainmodel(feature_info)
    img_info = loaddata('../data/images/test')
    feature_info = extractfeature(img_info)
    result = testmodel(feature_info)
    print("The accuracy is {}.".format(sum(result) / len(result)))