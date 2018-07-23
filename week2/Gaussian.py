import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def saliency(img_path):
	img = np.array(Image.open(img_path).convert('lab'))
	(avg_l, avg_a, avg_b) = 