from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import numpy as np


def trainmodel(img_feature_info):
    """
    according to the image feature, train the MLP model
    save the MLP model
    """
    print("train model....")
    x = [info[0] for info in img_feature_info]
    y = [info[1] for info in img_feature_info]
    print("train_x shape:->", np.array(x).shape)
    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(40), random_state=1)
    clf.fit(x, y)
    joblib.dump(clf, 'train_model.m')