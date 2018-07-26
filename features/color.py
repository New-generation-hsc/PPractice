import numpy as np
from PIL import Image
import cv2

class ColorHistogram(object):
	"""
	extract a 3d color histogram
	"""
	def __init__(self):
		super(ColorHistogram, self).__init__()

	@staticmethod
	def image_to_feature_vector(img, size=(64, 64)):
		"""
		resize the image to a fixed size, then flatten the image into
		a list a raw pixel intensities
		"""
		return cv2.resize(img, size).flatten()

	@staticmethod
	def extract_color_histgram(image):
		"""
		extract a 3d color histogram from HSV color space using 
		supplied number of `bins` per channel
		"""
		rgb_img = image.convert('RGB')
		hsv = np.array(rgb_img.convert('HSV'))
		hist_h = cv2.calcHist([hsv], [0], None, [256], [0.0, 255.0])
		hist_h = cv2.normalize(hist_h, hist_h).flatten()
		hist_s = cv2.calcHist([hsv], [1], None, [256], [0.0, 255.0])
		hist_s = cv2.normalize(hist_s, hist_s).flatten()
		hist_v = cv2.calcHist([hsv], [2], None, [256], [0.0, 255.0])
		hist_v = cv2.normalize(hist_v, hist_v).flatten()
		return np.concatenate((hist_h, hist_v, hist_s))

	def get_feature(self, image_path):
		img = Image.open(image_path)
		return self.extract_color_histgram(img)


if __name__ == "__main__":

	color = ColorHistogram()
	print(np.array(color.extract_color_histgram('../week2/cover.jpg')))