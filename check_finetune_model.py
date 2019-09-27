import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import numpy as np

from network import MultiSSDA8

import tensorflow as tf
input1 = tf.placeholder(dtype=tf.float32, shape=[None, 289], name='input1')
net = MultiSSDA8(inputs=input1)
net.initial_decoder()

print(net)

# session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./finetuned_model/')
ckpt_version = ckpt.model_checkpoint_path[len('./finetuned_model/finetuned-model.ckpt-'):]
saver.restore(sess, ckpt.model_checkpoint_path)
img_path = '/home/wuyanxue/Data/StandardTestImages/dataset/origin/lena.pgm'
im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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


def darkened(im_gray):
    gamma = np.random.uniform(2, 5)
    return im_gray ** gamma


def add_gaussian_noise_darkened(im_gray):
    _im_gray_darkened = darkened(im_gray)
    return add_gaussian_noise(_im_gray_darkened)


# im_noise = add_gaussian_noise_darkened(im)
height, width = im.shape[:2]

im_reconstruct = np.zeros_like(im)
im1 = np.zeros_like(im)
im_noise = np.zeros_like(im)
for x in range(0, width - 17, 3):
    for y in range(0, height - 17, 3):
        # c_patch = im_noise[y:y+17, x:x+17].reshape(-1)
        c_patch = im[y:y + 17, x:x + 17]
        c_patch = add_gaussian_noise_darkened(c_patch)
        im_noise[y:y+17, x:x+17] = c_patch
        c_patch = c_patch.reshape(-1)
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

cv2.imwrite('./test_result/lena_rec-{}.jpg'.format(ckpt_version), im_reconstruct)
cv2.imwrite('./test_result/lena_noise-patch.jpg', im_noise)
cv2.imwrite('./test_result/lena_origin.jpg', im)