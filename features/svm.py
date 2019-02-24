"""
This module define the classifier model
"""

from sklearn import svm
from datetime import datetime
import os
import numpy as np


class Classifier(object):

    def __init__(self, train_x, train_y, test_x, test_y, labels):
        self.train_x = train_x
        self.train_y = self.transform(train_y, labels)
        self.test_x = test_x
        self.test_y = self.transform(test_y, labels)
        
        self.clf = svm.SVC(decision_function_shape='ovo', probability=True)
        self.labels = labels

    @staticmethod
    def transform(data_label, labels):
        return list(map(lambda x : labels.index(x), data_label))

    def train(self):
        print("start training...")
        self.clf.fit(self.train_x, self.train_y)
        print("finish training...") 

    def evaluate(self):
        print("start testing...")
        sample_lables = self.clf.predict_proba(self.test_x)
        print(sample_lables.shape)
        sample_index = np.argsort(-sample_lables, axis=1)[:, :5]
        print(sample_index.shape)
        right_num = 0
        num_n = sample_lables.shape[0]  # the number of samples
        for i in range(num_n):
            right_num += (sample_index[i] == self.test_y[i]).sum()

        return right_num / num_n


def load_feature(path):
    x = []
    y = []
    labels = []
    count = 0
    for file in os.listdir(path):
        label = file.split('.')[0]
        labels.append(label)
        print("loading {} features...".format(label))
        for line in open(os.path.join(path, file), 'r', encoding='utf-8'):
            xx = list(map(float, line.split()))
            x.append(xx)
            y.append(label)

    return x, y, labels
def load_hog_lbp_glcm(hog_path, lbp_path,glcm_path):
    hog_x = []
    lbp_x = []
    glcm_x = []
    y = []
    labels = []
    for file in os.listdir(hog_path):
        label = file.split('.')[0]
        labels.append(label)
        print("loading {} features...".format(label))
        for line in open(os.path.join(hog_path, file), 'r', encoding='utf-8'):
            xx = list(map(float, line.split()))
            hog_x.append(xx)
            y.append(label)
        for line in open(os.path.join(lbp_path, file), 'r', encoding='utf-8'):
            xx = list(map(float, line.split()))
            lbp_x.append(xx)
        for line in open(os.path.join(glcm_path, file), 'r', encoding='utf-8'):
            xx = list(map(float, line.split()))
            glcm_x.append(xx)
    print(np.array(hog_x).shape, np.array(lbp_x).shape, np.array(glcm_x).shape)
    x = np.hstack((hog_x, lbp_x, glcm_x)).tolist()
    return x, y, labels

def train():
    train_x, train_y, labels = load_hog_lbp_glcm('../data/train_hog', '../data/train_lbp', '../data/train_glcm')
    test_x, test_y, labels = load_hog_lbp_glcm('../data/test_hog', '../data/test_lbp', '../data/test_glcm')
    clf = Classifier(train_x, train_y, test_x, test_y, labels)
    print("start training...")
    clf.train()
    print("start evaluate...")
    correction = clf.evaluate()
    print("finish test...")
    print("The classifier correctness is {}.".format(correction))


if __name__ == "__main__":
    start = datetime.now()
    train()
    print("The whole process cost {} time".format(datetime.now() - start))
