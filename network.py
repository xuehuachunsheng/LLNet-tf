
import tensorflow as tf
import numpy as np

from config import rho, beta, lambda_reg, layer_sizes

########################## Construct the SSDA #########################
class SSDA(object):
    def __init__(self, inputs, f_unit=289, h_unit=867, activation=tf.sigmoid, w1=None, b1=None, b1_=None):
        self.inputs = inputs
        corrupted_input1 = self.get_corrupted_input(self.inputs)

        # 使用truncated normal是不合适的
        if w1 is None:
            self.w1 = tf.Variable(initial_value=tf.random_uniform(shape=[f_unit, h_unit],
                                                              minval=-4 * np.sqrt(6. / (f_unit + h_unit)),
                                                              maxval=4 * np.sqrt(6. / (f_unit + h_unit))),
                                    name='w1')
        else:
            self.w1 = w1
        if b1 is None:
            self.b1 = tf.Variable(initial_value=tf.zeros(shape=[h_unit]), name='b1')
        else:
            self.b1 = b1
        self.y1 = activation(tf.matmul(corrupted_input1, self.w1) + self.b1)
        if b1_ is None:
            self.b1_ = tf.Variable(initial_value=tf.zeros(shape=[f_unit]), name='b1_')
        else:
            self.b1_ = b1_
        self.w1_ = tf.transpose(tf.identity(self.w1))
        self.z1 = activation(tf.matmul(self.y1, self.w1_) + self.b1_)

    def get_corrupted_input(self, x, corrupted_level=0.1):
        added = tf.random.normal(shape=tf.shape(x), mean=0, stddev=corrupted_level) + x
        return tf.clip_by_value(added, 0, 1)

    def loss(self):
        l2norm = tf.reduce_sum((self.inputs - self.z1) ** 2, axis=1)
        self.l2norm = tf.reduce_mean(l2norm)
        self.rho_j = tf.reduce_mean(self.y1, axis=0) # 一个batch的均值
        kl = rho * tf.log(rho / self.rho_j) + (1 - rho) * tf.log((1 - rho) / (1 - self.rho_j))
        self.kl = tf.reduce_sum(kl)
        self.weight_decay = tf.reduce_sum(tf.pow(self.w1, 2))
        self.loss = self.l2norm + beta * self.kl + lambda_reg * self.weight_decay
        return self.loss

    def get_hidden_value(self):
        return self.y1


# 多层堆叠
class MultiSSDA(object):
    def __init__(self, inputs, layers=3):
        self.inputs = inputs
        self.layers = layers
        # encoder
        self.da1 = SSDA(inputs, f_unit=289, h_unit=layer_sizes[0])
        self.da2 = SSDA(self.da1.y1, f_unit=layer_sizes[0], h_unit=layer_sizes[1])
        self.da3 = SSDA(self.da2.y1, f_unit=layer_sizes[1], h_unit=layer_sizes[2])
        self.y_true = tf.placeholder(shape=[None, 289], dtype=tf.float32)

    def initial_decoder(self, w1s=None, b1s=None, b1_s=None):
        '''input is the ndarray'''
        if w1s is not None:
            w1 = tf.Variable(tf.convert_to_tensor(w1s[0]), dtype=tf.float32, name='w1')
            b1 = tf.Variable(tf.convert_to_tensor(b1s[0]), dtype=tf.float32, name='b1')
            b1_ = tf.Variable(tf.convert_to_tensor(b1_s[0]), dtype=tf.float32, name='b1_')
            self.da4 = SSDA(self.da3.y1, f_unit=layer_sizes[2], h_unit=layer_sizes[1], w1=w1, b1=b1, b1_=b1_)

            w1 = tf.Variable(tf.convert_to_tensor(w1s[1]), dtype=tf.float32, name='w1')
            b1 = tf.Variable(tf.convert_to_tensor(b1s[1]), dtype=tf.float32, name='b1')
            b1_ = tf.Variable(tf.convert_to_tensor(b1_s[1]), dtype=tf.float32, name='b1_')
            self.da5 = SSDA(self.da4.y1, f_unit=layer_sizes[1], h_unit=layer_sizes[0], w1=w1, b1=b1, b1_=b1_)

            w1 = tf.Variable(tf.convert_to_tensor(w1s[2]), dtype=tf.float32, name='w1')
            b1 = tf.Variable(tf.convert_to_tensor(b1s[2]), dtype=tf.float32, name='b1')
            b1_ = tf.Variable(tf.convert_to_tensor(b1_s[2]), dtype=tf.float32, name='b1_')
            self.da6 = SSDA(self.da5.y1, f_unit=layer_sizes[0], h_unit=289, w1=w1, b1=b1, b1_=b1_)
        else:
            self.da4 = SSDA(self.da3.y1, f_unit=layer_sizes[2], h_unit=layer_sizes[1])
            self.da5 = SSDA(self.da4.y1, f_unit=layer_sizes[1], h_unit=layer_sizes[0])
            self.da6 = SSDA(self.da5.y1, f_unit=layer_sizes[0], h_unit=289)

        self.das = [self.da1, self.da2, self.da3, self.da4, self.da5, self.da6]

    def loss(self):
        y_pred = self.da6.y1
        y_true = self.y_true
        self.l2norm = tf.reduce_mean(tf.reduce_sum((y_pred - y_true)**2, axis=-1))
        self.weight_decay = tf.reduce_sum(self.da1.w1**2) + \
                  tf.reduce_sum(self.da2.w1**2) + \
                  tf.reduce_sum(self.da3.w1**2) + \
                  tf.reduce_sum(self.da4.w1**2) + \
                  tf.reduce_sum(self.da5.w1**2) + \
                  tf.reduce_sum(self.da6.w1**2)
        self.loss = self.l2norm + lambda_reg / 2 * self.weight_decay
        return self.loss

    # metric
    def ssim(self, size=(17, 17)):
        im1 = tf.reshape(self.da6.y1, (-1, size[0], size[1]))[..., None]
        im2 = tf.reshape(self.y_true, (-1, size[0], size[1]))[..., None]
        batch_ssim = tf.image.ssim(im1, im2, max_val=1.)
        return batch_ssim

    def psnr(self, size=(17, 17)):
        im1 = tf.reshape(self.da6.y1, (-1, size[0], size[1]))[..., None]
        im2 = tf.reshape(self.y_true, (-1, size[0], size[1]))[..., None]
        batch_psnr = tf.image.psnr(im1, im2, max_val=1.)
        return batch_psnr


# 多层堆叠
class MultiSSDA8(object):
    def __init__(self, inputs):
        self.inputs = inputs
        # encoder
        self.da1 = SSDA(inputs, f_unit=289, h_unit=layer_sizes[0])
        self.da2 = SSDA(self.da1.y1, f_unit=layer_sizes[0], h_unit=layer_sizes[1])
        self.da3 = SSDA(self.da2.y1, f_unit=layer_sizes[1], h_unit=layer_sizes[2])
        self.da4 = SSDA(self.da3.y1, f_unit=layer_sizes[2], h_unit=layer_sizes[3])
        self.y_true = tf.placeholder(shape=[None, 289], dtype=tf.float32)

    def initial_decoder(self, w1s=None, b1s=None, b1_s=None):
        '''input is the ndarray'''
        if w1s is not None:
            w1 = tf.Variable(tf.convert_to_tensor(w1s[0]), dtype=tf.float32, name='w1')
            b1 = tf.Variable(tf.convert_to_tensor(b1s[0]), dtype=tf.float32, name='b1')
            b1_ = tf.Variable(tf.convert_to_tensor(b1_s[0]), dtype=tf.float32, name='b1_')
            self.da5 = SSDA(self.da4.y1, f_unit=layer_sizes[3], h_unit=layer_sizes[2], w1=w1, b1=b1, b1_=b1_)

            w1 = tf.Variable(tf.convert_to_tensor(w1s[1]), dtype=tf.float32, name='w1')
            b1 = tf.Variable(tf.convert_to_tensor(b1s[1]), dtype=tf.float32, name='b1')
            b1_ = tf.Variable(tf.convert_to_tensor(b1_s[1]), dtype=tf.float32, name='b1_')
            self.da6 = SSDA(self.da5.y1, f_unit=layer_sizes[2], h_unit=layer_sizes[1], w1=w1, b1=b1, b1_=b1_)

            w1 = tf.Variable(tf.convert_to_tensor(w1s[2]), dtype=tf.float32, name='w1')
            b1 = tf.Variable(tf.convert_to_tensor(b1s[2]), dtype=tf.float32, name='b1')
            b1_ = tf.Variable(tf.convert_to_tensor(b1_s[2]), dtype=tf.float32, name='b1_')
            self.da7 = SSDA(self.da6.y1, f_unit=layer_sizes[1], h_unit=layer_sizes[0], w1=w1, b1=b1, b1_=b1_)

            w1 = tf.Variable(tf.convert_to_tensor(w1s[3]), dtype=tf.float32, name='w1')
            b1 = tf.Variable(tf.convert_to_tensor(b1s[3]), dtype=tf.float32, name='b1')
            b1_ = tf.Variable(tf.convert_to_tensor(b1_s[3]), dtype=tf.float32, name='b1_')
            self.da8 = SSDA(self.da7.y1, f_unit=layer_sizes[0], h_unit=289, w1=w1, b1=b1, b1_=b1_)

        else:
            self.da5 = SSDA(self.da4.y1, f_unit=layer_sizes[3], h_unit=layer_sizes[2])
            self.da6 = SSDA(self.da5.y1, f_unit=layer_sizes[2], h_unit=layer_sizes[1])
            self.da7 = SSDA(self.da6.y1, f_unit=layer_sizes[1], h_unit=layer_sizes[0])
            self.da8 = SSDA(self.da7.y1, f_unit=layer_sizes[0], h_unit=289)

        self.das = [self.da1, self.da2, self.da3, self.da4, self.da5, self.da6, self.da7, self.da8]

    def loss(self):
        y_pred = self.das[-1].y1
        y_true = self.y_true
        self.l2norm = tf.reduce_mean(tf.reduce_sum((y_pred - y_true)**2, axis=-1))
        self.weight_decay = tf.reduce_sum(self.da1.w1**2) + \
                  tf.reduce_sum(self.da2.w1**2) + \
                  tf.reduce_sum(self.da3.w1**2) + \
                  tf.reduce_sum(self.da4.w1**2) + \
                  tf.reduce_sum(self.da5.w1**2) + \
                  tf.reduce_sum(self.da6.w1**2) + \
                  tf.reduce_sum(self.da7.w1**2) + \
                  tf.reduce_sum(self.da8.w1**2)
        self.loss = self.l2norm + lambda_reg / 2 * self.weight_decay
        return self.loss

    # metric
    def ssim(self, size=(17, 17)):
        im1 = tf.reshape(self.das[-1].y1, (-1, size[0], size[1]))[..., None]
        im2 = tf.reshape(self.y_true, (-1, size[0], size[1]))[..., None]
        batch_ssim = tf.image.ssim(im1, im2, max_val=1.)
        return batch_ssim

    def psnr(self, size=(17, 17)):
        im1 = tf.reshape(self.das[-1].y1, (-1, size[0], size[1]))[..., None]
        im2 = tf.reshape(self.y_true, (-1, size[0], size[1]))[..., None]
        batch_psnr = tf.image.psnr(im1, im2, max_val=1.)
        return batch_psnr