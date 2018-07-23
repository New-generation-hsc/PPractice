"""
Pixel Inhomogeneity Factor
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def calc_pif(image_matrix, k, sigma):
	shape = image_matrix.shape
	ext_image = np.zeros((shape[0] + 2 * k, shape[1] + 2 * k))
	ext_image[k:k+shape[0], k:k+shape[1]] = image_matrix
	res = np.zeros(shape)
	for i in range(k, k + shape[0]):
		for j in range(k, k + shape[1]):
			tmp = np.abs(ext_image[i-k:i+k+1, j-k:j+k+1] - ext_image[i, j]) >= sigma
			res[i - k, j - k] = tmp.sum()
	return res / ((2 * k + 1) ** 2)

def calc_nif(pif_matrix, k):
	shape = pif_matrix.shape
	ext_image = np.zeros((shape[0] + 2 * k, shape[1] + 2 * k))
	ext_image[k:k+shape[0], k:k+shape[1]] = pif_matrix
	res = np.zeros(shape)
	for i in range(k, k + shape[0]):
		for j in range(k, k + shape[1]):
			tmp = ext_image[i-k:i+k+1, j-k:j+k+1] >= 0.5
			res[i - k, j - k] = tmp.sum()
	return res / ((2 * k + 1) ** 2)


def calc_seeds(pif_matrix, nif_matrix):
	return np.logical_and(pif_matrix >= 0.5, nif_matrix >= 0.5)


def calc_average(image_matrix, k):
	shape = image_matrix.shape
	ext_image = np.zeros((shape[0] + 2 * k, shape[1] + 2 * k))
	ext_image[k:k+shape[0], k:k+shape[1]] = image_matrix
	res = np.zeros(shape)
	for i in range(k, k + shape[0]):
		for j in range(k, k + shape[1]):
			tmp = np.abs(ext_image[i-k:i+k+1, j-k:j+k+1] - ext_image[i, j])
			res[i - k, j - k] = tmp.sum() / ((2 * k + 1)**2)
	return res.sum() / (shape[0] * shape[1])


def generate_image(image_path, k):
	img = np.array(Image.open(image_path).convert('L'))
	avg = calc_average(img, k)
	print(avg)
	pif = calc_pif(img, k, avg)
	fig = plt.figure(figsize=(10, 15))
	axes = fig.add_subplot(321)
	axes.imshow(img, cmap='gray')
	axes.set_title("Original Image")
	axes = fig.add_subplot(322)
	axes.imshow(pif, cmap='gray')
	axes.set_title("PIF Image")
	axes = fig.add_subplot(323)
	nif = calc_nif(pif, k)
	axes.imshow(nif, cmap='gray')
	axes.set_title("NIF Image")
	seeds = calc_seeds(pif, nif)
	axes = fig.add_subplot(325)
	axes.imshow(seeds, cmap='gray')
	axes = fig.add_subplot(326)
	combine = seeds * img
	axes.imshow(combine, cmap='gray')
	axes.set_title("Both PIF and NIF")
	plt.show()

if __name__ == '__main__':
	generate_image('fox.jpg', 3)