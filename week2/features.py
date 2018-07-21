import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def split_block(image_path):
	img = np.array(Image.open(image_path))
	w, h, channel = img.shape
	img[w//2,:,:] = 255
	img[:, h//2, :] = 255
	plt.imshow(img)
	plt.show()


if __name__ == "__main__":
	split_block('cover.jpg')