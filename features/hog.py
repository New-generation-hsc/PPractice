from skimage import feature
import numpy as np
from PIL import Image


class HistgoramOrientedGradient(object):

	def __init__(self, orientation=8):
		super(HistgoramOrientedGradient, self).__init__()
		self.orientation = orientation

	def extract(self, img):
		"""
		compute the histograms of Oriented Gradients 
		"""
		hog = feature.hog(img, orientations=self.orientation,
						  block_norm='L1',
						  transform_sqrt=True,
						  feature_vector=True)
		return hog

	def get_feature(self, image_path):
		img = Image.open(image_path).convert('L')
		return self.extract(img)


if __name__ == "__main__":

	model = HistgoramOrientedGradient()
	print(model.get_feature("../week2/cover.jpg"))