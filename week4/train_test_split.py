import os
import shutil
import random

ROOT_PATH = "/home/greek/data/storetag"
SAVE_PATH = "/home/greek/data/newstoretag"


train_imgs = {}
test_imgs = {}

for label in os.listdir(ROOT_PATH):
    path = os.path.join(ROOT_PATH, label)
    img_names = os.listdir(path)
    random.shuffle(img_names)
    pivot = len(img_names) * 4 // 5
    train_imgs[label] = img_names[:pivot]
    test_imgs[label] = img_names[pivot:]

idx = 0
os.chdir(SAVE_PATH + "/train")
for label in train_imgs:
    if not os.path.exists(label):
        os.mkdir(label)
    for img_name in train_imgs[label]:
        idx += 1
        print("copying the {}th image...".format(idx))
        shutil.copy(ROOT_PATH + "/" + label + "/" + img_name, SAVE_PATH + "/train/" + label + "/" + img_name)


os.chdir(SAVE_PATH + "/test")
for label in test_imgs:
    if not os.path.exists(label):
        os.mkdir(label)
    for img_name in test_imgs[label]:
        idx += 1
        print("copying the {}th image...".format(idx))
        shutil.copy(ROOT_PATH + "/" + label + "/" + img_name, SAVE_PATH + "/test/" + label + "/" + img_name)