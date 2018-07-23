import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from skimage import color
from scipy.ndimage.filters import gaussian_filter

def saliency(img_path):
    img = np.array(Image.open(img_path))
    img = color.rgb2lab(img)
    print(img)
    mean_matrix = np.zeros(img.shape)
    mean_matrix[0, :, :] = np.mean(img[:,:,0])
    mean_matrix[:, 1, :] = np.mean(img[:,:,1])
    mean_matrix[:, :, 2] = np.mean(img[:,:,2])

    gaussian_matrix = gaussian_filter(img, sigma=(3, 3, 3))
    diff_matrix = np.sqrt(np.sum(np.power(mean_matrix - gaussian_matrix, 2), axis=2))
    plt.imshow(diff_matrix, cmap="gray")
    plt.show()

saliency("pic.jpg")