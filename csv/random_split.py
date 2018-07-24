"""
randomly split the whole dataset to train set and test set
"""
import os
import math
import csv


BASE_PATH = "/home/greek/data/ds2018"


def split_dataset():
    train = []
    test = []
    for label in os.listdir(BASE_PATH):
        path = os.path.join(BASE_PATH, label)
        category = []
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            category.append((img_path, label))
        pivot = math.floor(len(category) * 0.8) + 1
        train.extend(category[:pivot])
        test.extend(category[pivot:])
    return train, test


def write_2_csv(train_dataset, test_dataset):

    with open("train.csv", 'w', encoding='utf-8') as train:
        writer = csv.writer(train, delimiter=',')
        for row in train_dataset:
            writer.writerow(row)

    with open("test.csv", 'w', encoding='utf-8') as test:
        writer = csv.writer(test, delimiter=',')
        for row in test_dataset:
            writer.writerow(row)


if __name__ == "__main__":

    train, test = split_dataset()
    write_2_csv(train, test)