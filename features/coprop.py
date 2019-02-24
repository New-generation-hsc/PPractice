from skimage import feature
import numpy as np
from PIL import Image


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
        img = Image.open(img)
        img = img.convert('L')
        return self.extract_feature(img)


if __name__ == "__main__":

    grey = GreyCoprops([1, 2], [0, np.pi / 2, np.pi, 3 * np.pi / 4])
    img_feature = grey.get_feature("../week2/cover.jpg")
    print(img_feature.shape)
    print(img_feature)

    img_feature = grey.get_feature("../week2/pic.jpg")
    print(img_feature.shape)
    print(img_feature)