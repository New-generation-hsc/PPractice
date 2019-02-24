"""
This module define the classifier model
"""
#parm=(48,13)
from sklearn import svm
from datetime import datetime
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

class Classifier(object):

    def __init__(self, clf,train_x, train_y, test_x, test_y, labels):
        self.train_x = train_x
        self.train_y = self.transform(train_y, labels)
        self.test_x = test_x
        self.test_y = self.transform(test_y, labels)
        self.clf = clf
        #self.clf = svm.SVC(decision_function_shape='ovo', probability=True)
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
        sample_index = np.argsort(-sample_lables, axis=1)[:, :5]
        right_num = 0
        num_n = sample_lables.shape[0]  # the number of samples
        print(sample_index[0])
        print(self.test_y[0])
        for i in range(num_n):
            #if self.test_y[i] in set(sample_index[i]):
                #right_num+=1
            right_num += (sample_index[i] == self.test_y[i]).sum()
        return right_num / num_n


def train(temp_clf):
    train_x, train_y, labels = load_features('../data/train_lbp', '../data/train_new_hog', '../data/train_glcm_4096')
    test_x, test_y, labels = load_features('../data/test_lbp', '../data/test_new_hog', '../data/test_glcm_4096')
    # pca = PCA(n_components=784)
    # train_x = pca.fit_transform(train_x)
    # test_x = pca.fit_transform(test_x)
    assert len(train_x) == len(train_y)
    print("shape:->", np.array(train_x).shape, np.array(test_x).shape)
    clf = Classifier(temp_clf,train_x, train_y, test_x, test_y, labels)
    print("start training...")
    clf.train()
    print("start evaluate...")
    correction = clf.evaluate()
    print("finis test...")
    print("The classifier correctness is {}.".format(correction))
    return correction


def load_features(*pathes):
    """
    load feature from the path, if other path is given, then read other feature from other path
    then hstack the whole feature
    """
    all_x = []
    y = None
    labels = None
    for path in pathes:
        x = []
        y = []
        labels = []
        for file in os.listdir(path):
            label = file.split('.')[0]
            labels.append(label)
            print("loading {} features...".format(label))
            for line in open(os.path.join(path, file), 'r', encoding='utf-8'):
                xx = list(map(float, line.split()))
                x.append(xx)
                y.append(label)
        all_x.append(x)
    return np.hstack(all_x).tolist(), y, labels


if __name__ == "__main__":
    start = datetime.now()
    '''
    maxmax=0
    for i in range(45,55):
        for j in range(10,15):
            parm=(i,j)
            clf=MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=parm, random_state=1)
            temp=train(clf)
            if temp>maxmax:
                maxmax=temp
                best=parm
            print("The whole process cost {} time".format(datetime.now() - start))
    print(maxmax)
    '''
    clf=MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(40), random_state=1)
    temp=train(clf)
    #print(best)