import matplotlib.pyplot as plt
import random
import numpy as np

plt.style.use('seaborn')

def generate_random_paramater(w_range, b_range):
	w = random.random() * (w_range[1] - w_range[0]) + w_range[0]
	b = random.random() * (b_range[1] - b_range[0]) + b_range[0]
	return w, b


def generate_random_scatter(x_range, w, b, k):
	"""
	the k is the scatter number on each side
	"""
	x_1 = []
	y_1 = []
	x_2 = []
	y_2 = []
	for i in range(k):
		xx = random.random() * (x_range[1] - x_range[0]) + x_range[0]
		x_1.append(xx)
		amplitude = random.randint(4, 15)
		yy = w * xx + b + amplitude
		y_1.append(yy)

		xx = random.random() * (x_range[1] - x_range[0]) + x_range[0]
		x_2.append(xx)
		amplitude = random.randint(4, 15)
		yy = w * xx + b - amplitude
		y_2.append(yy)
	return x_1, y_1, x_2, y_2


def plot_linear(x_range, w, b):
	"""
	plot a linear line in range of x
	"""
	plt.plot(x_range, x_range * w + b)


def plot_scatter(x, y):
	"""
	plot scatter in the figure
	"""
	plt.scatter(x, y)


if __name__ == "__main__":
	w, b = generate_random_paramater([-2, 2], [-4, 5])
	data = generate_random_scatter([-3, 3], w, b, 10)
	plot_linear(np.array([-3, 3]), w, b)
	plot_scatter(data[0], data[1])
	plot_scatter(data[2], data[3])
	plt.show()