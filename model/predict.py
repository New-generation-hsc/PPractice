import numpy as np
from sklearn.externals import joblib
from load import loaddata
from feature import extractfeature


def testmodel(filepath):
    """
    according to the image feature, predict the img label
    """
    print("test model....")
    images, tags = loaddata(filepath)
    feature_info = extractfeature(images, tags)
    
    x = [info[0] for info in feature_info]
    y = [info[1] for info in feature_info]

    clf = joblib.load('train_model.m')
    samples_proba = clf.predict_proba(x) # predict the test images probability
    top5_index = np.argsort(-samples_proba, axis=1)[:, :5].tolist()
    res = []
    for (i, tag) in enumerate(y):
        res.append(tag in top5_index[i])
    return res