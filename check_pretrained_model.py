'''将训练好的三层layer重新存储成6层完整的layer'''
import os
import time
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import numpy as np

from network import SSDA, MultiSSDA8
from config import *

# Network
input1 = tf.placeholder(dtype=tf.float32, shape=[None, 289], name='input1')
net = MultiSSDA8(inputs=input1)
# optimizer
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate = lr,
                                           global_step = global_step,
                                           decay_steps = 2000,
                                           decay_rate = 0.2,
                                           staircase = True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./pretrained_model/')
steps = ckpt.model_checkpoint_path[len('./pretrained_model/pretrained-model-'):]
saver.restore(sess, ckpt.model_checkpoint_path)

# Obtain the first 3 layers parameters
w1s = sess.run([x.w1 for x in [net.da1,net.da2,net.da3, net.da4]])
b1s = sess.run([x.b1 for x in [net.da1,net.da2,net.da3, net.da4]])
b1_s = sess.run([x.b1_ for x in [net.da1,net.da2,net.da3, net.da4]])

w1s = [np.transpose(x) for x in w1s]
net.initial_decoder(w1s[::-1], b1s=b1_s[::-1], b1_s=b1s[::-1])

# 初始化还未初始化的变量
def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

initialize_uninitialized(sess)

# important! you must be recreate a saver object
saver = tf.train.Saver()
saver.save(sess, './pretrained_model/whole-pretrained-model-{}'.format(steps))

########################## 测试推理 ##########################

import cv2

img_path = '/home/wuyanxue/Data/StandardTestImages/dataset/origin/lena.pgm'
im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
im = im / 255.

def add_gaussian_noise(im_gray):
    height, width = im_gray.shape[:2]
    mean = 0
    var = 0.1
    sigma = (np.random.uniform() * var**2)**0.5
    gauss = np.random.normal(mean, sigma, (height, width))
    _im_gray_noise = im_gray + gauss
    _im_gray_noise = np.clip(_im_gray_noise, 0, 1)
    return _im_gray_noise

im_noise = add_gaussian_noise(im)
height, width = im_noise.shape[:2]

im_reconstruct = np.zeros_like(im_noise)
im1 = np.zeros_like(im_noise)
for x in range(0, width - 17, 3):
    for y in range(0, height - 17, 3):
        c_patch = im_noise[y:y+17, x:x+17].reshape(-1)
        c_patch = c_patch[None, :]
        rec = sess.run(net.das[-1].y1, feed_dict={input1: c_patch})
        im_reconstruct[y:y + 17, x:x + 17] += rec[0].reshape(17, 17)
        im1[y:y + 17, x:x + 17] += 1

im1[im1 == 0] = 1
im_reconstruct /= im1
im_reconstruct = np.dstack([im_reconstruct, im_reconstruct, im_reconstruct])
im = np.dstack([im, im, im])
im_noise = np.dstack([im_noise, im_noise, im_noise])

im_reconstruct = np.clip(im_reconstruct * 255, 0, 255).astype(np.uint8)
im = np.clip(im * 255, 0, 255).astype(np.uint8)
im_noise = np.clip(im_noise * 255, 0, 255).astype(np.uint8)

if not os.path.exists('./test_result/test_pretrain'):
    os.mkdir('./test_result/test_pretrain')

# 效果一般
cv2.imwrite('./test_result/test_pretrain/lena_rec.jpg', im_reconstruct)
cv2.imwrite('./test_result/test_pretrain/lena_noise.jpg', im_noise)
cv2.imwrite('./test_result/test_pretrain/lena_origin.jpg', im)



