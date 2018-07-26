import numpy as np
import mahotas as mt
from PIL import Image

class Haralick(object):
	"""
	calculate haralick texture feature
	"""
	def __init__(self):
		super(Haralick, self).__init__()

	def extract_features(self, img_path):
		img = np.array(Image.open(img_path).convert('L'))

		# calcuate haralick texture features for 4 types of adjacency
		textures = mt.features.haralick(img).flatten()

		return textures


if __name__ == "__main__":

	model = Haralick()
	print(np.array(model.extract_features('../week2/cover.jpg')).shape)
	print(np.array(model.extract_features('../week2/pic.jpg')).shape)
	print(np.array(model.extract_features('../week2/pic2.jpg')).shape)