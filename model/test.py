from load import loaddata
from feature import extractfeature
from classify import trainmodel
from predict import testmodel
from datetime import datetime


start = datetime.now()
images, tags = loaddata("../data/images/train")
feature_info = extractfeature(images, tags)
trainmodel(feature_info)
result = testmodel("../data/images/test")
print("The accuracy is {}.".format(sum(result) / len(result)))
print("The whole proess cost {}.".format(datetime.now() - start))

