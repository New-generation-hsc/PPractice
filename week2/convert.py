import matplotlib.pyplot as plt
import numpy as np

SIZE = 256

img = np.zeros([SIZE, SIZE])
img[:, :SIZE // 2] = 255

rand_matrix = np.random.rand(SIZE, SIZE) > 0.97
noise_matrix = ((img > 10) * (-1)  * rand_matrix + (img < 10) * rand_matrix) * 255

combine_img = img + noise_matrix

plt.imshow(combine_img, cmap='gray')
plt.show()