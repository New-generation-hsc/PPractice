import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def show_image(jpg_path, png_path):
	jpg = np.array(Image.open(jpg_path).convert('L'))
	png = np.array(Image.open(png_path).convert('L'))
	plt.imshow(np.abs(jpg - png), cmap='gray')
	plt.show()


if __name__ == "__main__":
	show_image('pic.jpg', 'zebra.png')