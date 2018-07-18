import matplotlib.pyplot as plt
import numpy as np

SIZE = 256

img = np.zeros([SIZE, SIZE])
img[:, :SIZE // 2] = 255

for i in range(SIZE):
	amplitude = np.random.randint(10)
	direction = np.random.random()
	if direction >= 0.5 and (SIZE - i + amplitude) < SIZE:
		img[i, 255 - i + amplitude] = 255 - img[i, 255 - i + amplitude]
	elif direction < 0.5 and (SIZE - i - amplitude) > 0:
		img[i, 255 - i - amplitude] = 255 - img[i, 255 - i - amplitude]
	else:
		img[i, 255 - i] = 255 - img[i, 255 - i]
plt.imshow(img, cmap='gray')
plt.show()