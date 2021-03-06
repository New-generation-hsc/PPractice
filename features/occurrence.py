from skimage import feature
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


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
        feature_vector = (feature_vector - np.min(feature_vector))/(np.max(feature_vector)- np.min(feature_vector))

        return feature_vector

    def get_feature(self, image_path):
        img = Image.open(image_path).convert('L')
        return self.extract(img)


if __name__ == "__main__":
    glcm = GreyCoMatrix()
    feature = glcm.get_feature("../week2/cover.jpg")
    print(feature.shape)
    print(feature)
