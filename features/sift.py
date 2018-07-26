
"""
compute the sift feature of every image

Algorithm:
1. extract all image sift feature according to the image category
2. 将每一类图片的SIFT特征聚类为K类，构成该类的visual vocabulary
3. 对于训练数据中每一张图片。统计vocabulary中K个word的词频， 构成相应的直方图
4. 将直方图作为样本向量即可构成SVM的训练数据和测试数据
"""

import cv2
import numpy as np


class SIFT(object):

    def __init__(self, k=200):
        super(SIFT, self).__init__()
        self.k = k
        self.sift = cv2.xfeatures2d.SIFT_create(200)  # the max number of key points

    def calc_sift_feature(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert the image to gray level
        kp, desc = self.sift.detectAndCompute(gray, None)
        return desc

    def build_vocabulary(self, images_path):
        """
        for each category, construct a feature set
        """
        feature_dict = {}
        for img in images_path:
            desc = self.calc_sift_feature(img)
            feature_dict[img] = desc
        # np.save(save_path + "/" + label + ".npy", feature_set)
        return feature_dict

    def calc_centers(self, feature_set):

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(np.array(feature_set), self.k, None, criteria, 20, flags)

        # labels (N * 1), centers (K * 128)
        return centers

    def calc_histogram(self, features, centers):
        """
        compute the number of feature near for center
        """
        hist = np.zeros(self.k)
        for i in range(0, len(features)):
            feature = np.array(features[i])
            diffMat = np.tile(feature, (self.k, 1)) - np.array(centers)
            sqSum = (diffMat**2).sum(axis=1)
            dist = sqSum**0.5
            sortedIndices = dist.argsort()
            idx = sortedIndices[0] # index of the nearest center
            hist[idx] += 1
        return hist



if __name__ == "__main__":

    sift = SIFT()
    desc = sift.calc_sift_feature("../week2/cover.jpg")
    print(np.array(desc).shape)
    desc = sift.calc_sift_feature("../week2/pic.jpg")
    print(np.array(desc).shape)

