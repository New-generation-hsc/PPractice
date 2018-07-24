"""
This module define the classifier model
"""

from sklearn import svm


class Classifier(object):

	def __init__(self, train_x, train_y, test_x, test_y, labels):
		self.train_x = train_x
		self.train_y = self.transform(train_y, labels)
		self.test_x = test_x
		self.test_y = self.transform(test_y, labels)
		
		self.clf = svm.SVC(decision_function_shape='ovo', probability=True)

	@staticmethod
	def transform(data_label, labels):
		return np.array(list(map(lambda x : labels.index(x), data_label)))

	def train(self):
		self.clf.fit(self.train_x, self.train_y) 

	def evaluate(self):
		sample_lables = self.clf.predict(self.test_x)
		