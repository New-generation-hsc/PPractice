"""
Pixel Inhomogeneity Factor
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def image_to_gray(image_path):
	"""
	given the image path, return a grayscale image
	"""
	return Image.open(image_path).convert('L')


def calc_numbers(image_matrix, k, sigma):
	shape = image_matrix.shape
	res = np.zeros(image_matrix.shape)
	for i in range(shape[0]):
		for j in range(shape[1]):
			low = 0 if j - k < 0 else j - k
			high = shape[1] if j + k + 1 >= shape[1] else j + k + 1
			left = 0 if i - k < 0 else i - k
			right = shape[0] if i + k + 1 >= shape[0] else i + k + 1
			tmp = np.abs(image_matrix[left:right, low:high] - image_matrix[i, j]) >= sigma
			res[i, j] = tmp.sum()
	return res


def calc_pif(image_matrix, k, sigma):
	return calc_numbers(image_matrix, k, sigma) / ((2 * k + 1) ** 2)


def calc_nif(pif_matrix, k):
	return calc_numbers(pif_matrix, k, 0.5) / ((2 * k + 1) **2 -1)


def calc_average(image_matrix, k):
	shape = image_matrix.shape
	res = np.zeros(image_matrix.shape)
	for i in range(shape[0]):
		for j in range(shape[1]):
			low = 0 if j - k < 0 else j - k
			high = shape[1] if j + k + 1 >= shape[1] else j + k + 1
			left = 0 if i - k < 0 else i - k
			right = shape[0] if i + k + 1 >= shape[0] else i + k + 1
			tmp = np.abs(image_matrix[left:right, low:high] - image_matrix[i, j])
			res[i, j] = tmp.sum() / ((2 * k + 1) ** 2 - 1)
	return res.sum() / (shape[0] * shape[1])


def calc_seeds(pif_matrix, nif_matrix):
	return np.logical_and(pif_matrix >= 0.5, nif_matrix >= 0.5)

def generate(image_path, k):
	img = np.array(Image.open(image_path).convert('L'))
	avg = calc_average(img, k)
	pif = calc_pif(img, k, avg)
	nif = calc_nif(pif, k)
	seeds = calc_seeds(pif, nif)
	plt.imshow(seeds * 255)
	plt.show()

def generate_v2(image_path, k):
	img = np.array(Image.open(image_path).convert('L'))
	avg = calc_average(img, k)
	pif = calc_pif(img, k, 39)
	plt.figure(figsize=(10, 15))
	plt.subplot(121)
	plt.imshow(img, cmap='gray')
	plt.subplot(122)
	plt.imshow(pif, cmap='gray')
	plt.show()

if __name__ == "__main__":
	generate_v2('pic.jpg', 3)
	# img = np.array(Image.open('06c4ab99-9780-3e42-a00d-39696ca39a6e.jpg'))
	# avg = calc_average(img, 3)
	# print("avg:", avg)
	# print(calc_pif(img, 3, avg))