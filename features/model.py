"""
This module define the method for store the image feature
"""
from lbp import LocalBinaryPatterns
from hog import HistgoramOrientedGradient
import os
import csv

lbp = LocalBinaryPatterns(8, 1)
hog = HistgoramOrientedGradient()

def load_from_csv(csv_path):
    """
    every row include two column : image_path and image label
    """
    images_path = []
    labels = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            images_path.append(row[0])
            labels.append(row[1])
    return images_path, labels


def store_feature(feature, label):
    """
    store the image in the disk
    """
    with open(label + ".txt", "a+", encoding="utf-8") as file:
        line = ' '.join(map(str, feature.tolist())) + "\n"
        file.write(line)


def extract_feature(images_path, labels, folder):
    """
    extract all images feature and store the images feature in the disk
    """
    index = 0
    for (image, label) in zip(images_path, labels):
        index += 1
        print("extract feature in {}th image...".format(index))
        feature = lbp.get_feature(image)
        print("store feature in {}th image...".format(index))
        store_feature(feature, os.path.join('../data/' + folder, label))


def extract_feature_hog(images_path, labels, folder):
    """
    extract all images feature and store the images feature in the disk
    """
    index = 0
    for (image, label) in zip(images_path, labels):
        index += 1
        print("extract feature in {}th image...".format(index))
        feature = hog.get_feature(image)
        print("store feature in {}th image...".format(index))
        store_feature(feature, os.path.join('../data/' + folder, label))


def extract_feature_lbp_and_hog(images_path, labels, folder):
    """
    extract all images feature and store the images feature in the disk
    """
    index = 0
    for (image, label) in zip(images_path, labels):
        index += 1
        print("extract feature in {}th image...".format(index))
        feature1 = lbp.get_feature(image)
        feature2 = hog.get_feature(image)
        feature = np.hstack((feature1, feature2))
        print("store feature in {}th image...".format(index))
        store_feature(feature, os.path.join('../data/' + folder, label))


if __name__ == "__main__":
    images_path, labels = load_from_csv('../csv/train.csv')
    extract_feature_hog(images_path, labels, 'train_hog')

    images_path, labels = load_from_csv('../csv/test.csv')
    extract_feature_hog(images_path, labels, 'test_hog')
