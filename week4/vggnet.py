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
    # The image is 64 * 64 * 3 with rgb representation
    data_shape = [3, 32, 32]  # Channel, H, W
    img = fluid.layers.data(name='img', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    return img, label


def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    predict = fluid.layers.fc(input=fc2, size=60, act='softmax')
    return predict


def train_program():
    img, label = input_program()
    predict = vgg_bn_drop(img)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    top1_accuracy = fluid.layers.accuracy(input=predict, label=label)
    top5_accuracy = fluid.layers.accuracy(input=predict, label=label, k=5)
    return [avg_cost, top1_accuracy, top5_accuracy]


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
            print("\nTime: [{}] Pass {}, Batch {}, Cost {}, Acc {}".format
                  (datetime.now() - start, event.step, event.epoch, event.metrics[0],
                   event.metrics[1]))
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    if isinstance(event, fluid.EndEpochEvent):
        # Test against with the test dataset to get accuracy.
        avg_cost, top1_accuracy, top5_accuracy = trainer.test(
            reader=test_reader, feed_order=['img', 'label'])

        print('\nTime:[{}] Test with Pass {}, Loss {}, Acc {} Top5 Acc: {}'.format(datetime.now() - start, event.epoch, avg_cost, top1_accuracy, top5_accuracy))

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
