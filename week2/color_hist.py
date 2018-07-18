import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def image_to_arr(image_path):
	"""
	given the image path, return image pixel matrix
	"""
	return Image.open(image_path)


def compress(img):
	"""
	compress the image to square image, size 64 * 64
	"""
	return img.resize((64, 64))


def arr_to_image(arr):
	plt.imshow(arr)
	plt.show()


if __name__ == "__main__":
	img = image_to_arr('cover.jpg')
	arr = compress(img)
	arr_to_image(np.array(arr))