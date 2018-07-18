import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def image_to_gray(image_path):
	"""
	given the image path, return a grayscale image
	"""
	return Image.open(image_path).convert('L')


def img_to_arr(image):
	"""
	given the image pixel matrix, return gray array
	"""
	gray_arr = np.zeros(256)
	image_arr = np.array(image).ravel()
	for pixel in image_arr:
		gray_arr[pixel] += 1
	return gray_arr


def plot_hist(arr):
	plt.style.use('seaborn')
	n, bins, patches = plt.hist(arr, bins=256, range=(0, 255))
	plt.show()


def plot_linear(arr):
	plt.style.use('seaborn')
	x = np.arange(256)
	plt.plot(x, arr)
	plt.show()


if __name__ == "__main__":
	img = image_to_gray('cover.jpg')
	arr = img_to_arr(img)
	plot_linear(arr)