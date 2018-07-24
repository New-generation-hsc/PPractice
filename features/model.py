"""
This module define the method for store the image feature
"""
from lbp import LocalBinaryPatterns
import os

lbp = LocalBinaryPatterns()

def load_from_csv(csv_path):
    """
    every row include two column : image_path and image label
    """
    images_path = []
    labels = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            images_path.append(row[0])
            labels.append(row[1])
    return images_path, labels


def store_feature(feature, label):
    """
    store the image in the disk
    """
    with open(label + ".txt", "a", encoding="utf-8") as file:
        line = ' '.join(feature) + "\n"
        file.write(line)


def extract_feature(images_path, labels):
    """
    extract all images feature and store the images feature in the disk
    """
    for (image, label) in zip(images_path, labels):
        feature = lbp.get_feature(image)
        store_feature(feature, os.path.join(['data', label]))

