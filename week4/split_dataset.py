"""
The train dataset include a train.txt 
each row in the train.txt is image name, image label, tag x1, tag x2, tag y1, tag y2
split the dataset to 
"""
import os
from PIL import Image


SOURCE_PATH = '/home/greek/data/train/train'
SAVE_PATH = '/home/greek/data/storetag'


def split_dataset(train_file_path):
	"""
	`train_file_path`: the train.txt file path
	according to the image tag coordinate, corp the image, and save the image tag to different folder
	"""
	os.chdir(SAVE_PATH)
	idx = 0
	for line in open(train_file_path, 'r', encoding='utf-8'):
		idx += 1
		print("deal with the {}th image...".format(idx))
		row_list = line.split(',')
		path = os.path.join(SAVE_PATH, row_list[1])
		if not os.path.exists(row_list[1]):
			os.mkdir(row_list[1])
		img = Image.open(os.path.join(SOURCE_PATH, row_list[0]))
		region = tuple(map(int, row_list[2:]))
		crop_img = img.crop(region)
		crop_img.save(os.path.join(path, row_list[0]))


if __name__ == "__main__":

	split_dataset('/home/greek/data/train/train.txt')