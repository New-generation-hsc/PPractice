from __future__ import print_function
import paddle.v2
import paddle.fluid as fluid
import random
import shutil
import numpy as np
from datetime import datetime
from PIL import Image
import os
import sys


FIXED_IMAGE_SIZE = (32, 32)
params_dirname = "image_classification.inference.model"



def input_program():
    # The image is 32 * 32 * 3 with rgb representation
    data_shape = [3, 32, 32]  # Channel, H, W
    img = fluid.layers.data(name='img', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    return img, label


def simple_conv_net(input):
    def conv_block(ipt, num_filter):
        return fluid.nets.simple_img_conv_pool(
                input=ipt,
                num_filters=num_filter,
                filter_size=3,
                pool_size=2,
                pool_stride=2,
                act='relu'
            )
    conv1 = conv_block(input, 20)
    conv2 = conv_block(conv1, 50)
    fc = fluid.layers.fc(input=conv2, size=100, act='relu')
    predict = fluid.layers.fc(input=fc, size=60, act='softmax')
    return predict


def train_program():
    img, label = input_program()
    predict = simple_conv_net(img)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label, k=5)
    return [avg_cost, accuracy]


def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)


def custom_reader_creator(images_path):
    # return a reader generator
    def reader():
        for label in os.listdir(images_path):
            path = os.path.join(images_path, label)
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                img = load_image(img_path)
                yield img, int(label) - 1
    return reader


def load_image(img_path):
    im = Image.open(img_path)
    im = im.resize(FIXED_IMAGE_SIZE, Image.ANTIALIAS)

    im = np.array(im).astype(np.float32)
    # The storage order of the loaded image is W(width),
    # H(height), C(channel). PaddlePaddle requires
    # the CHW order, so transpose them.
    im = im.transpose((2, 0, 1))  # CHW
    im = im / 255.0
    
    return im


# event handler to track training and testing process
def event_handler(event):
    if isinstance(event, fluid.EndStepEvent):
        if event.step % 100 == 0:
            print("\nTime: [{}] Step {}, Epoch {}, Cost {}, Acc {}".format
                  (datetime.now() - start, event.step, event.epoch, event.metrics[0],
                   event.metrics[1]))
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    if isinstance(event, fluid.EndEpochEvent):
        # Test against with the test dataset to get accuracy.
        avg_cost, accuracy = trainer.test(
            reader=test_reader, feed_order=['img', 'label'])

        print('\nTime:[{}] Test with Epoch {}, Loss {}, Acc {}'.format(datetime.now() - start, event.epoch, avg_cost, accuracy))

        # save parameters
        if params_dirname is not None:
            trainer.save_params(params_dirname)


use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

trainer = fluid.Trainer(train_func=train_program, optimizer_func=optimizer_program, place=place)

# Each batch will yield 128 images
BATCH_SIZE = 128

# Reader for training
train_reader = paddle.batch(
    paddle.reader.shuffle(custom_reader_creator("/home/vibrant/data/newstoretag/train"), buf_size=500),
    batch_size=BATCH_SIZE
)

# Reader for testing
test_reader = paddle.batch(
    custom_reader_creator("/home/vibrant/data/newstoretag/test"),  batch_size=BATCH_SIZE
)

start = datetime.now()
trainer.train(
    reader=train_reader,
    num_epochs=100,
    event_handler=event_handler,
    feed_order=['img', 'label'])
