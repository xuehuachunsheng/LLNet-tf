
import os
import time
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import tensorflow as tf

from config import *
from network import SSDA
vis = True

########################## Pretrained hyper-parameters #########################
# According to the official implementations
# https://github.com/kglore/llnet_color/blob/master/library.py
########################## End of pretrained_model hyper-parameters #########################

########################## Load data #########################

root_path = '/home/wuyanxue/Data/StandardTestImages/dataset/'
train_path = os.path.join(root_path, 'train_2500patch_per_image.tfrecords')

def parse_func(example_proto):
    feature_desc = {
        'origin': tf.io.FixedLenFeature([289], tf.float32),
        'noise': tf.io.FixedLenFeature([289], tf.float32),
        'darked': tf.io.FixedLenFeature([289], tf.float32),
        'noise_darked': tf.io.FixedLenFeature([289], tf.float32),
    }
    features = tf.io.parse_single_example(example_proto, feature_desc)
    return features['origin']


train_ds = tf.data.TFRecordDataset(train_path)
train_ds = train_ds.map(parse_func).shuffle(25000).repeat().batch(batch_size)
train_ds = train_ds.prefetch(buffer_size=batch_size)

########################## End of load data #########################

########################## Training #########################
# 这个结构是逐层训练的
# Layer 1
global_steps = []
lrs = []
optimizers = []

for i in range(len(layer_sizes)):
    global_steps.append(tf.Variable(0, trainable=False))
    lrs.append(tf.train.exponential_decay(learning_rate=lr[i],
                                           global_step=global_steps[i],
                                           decay_steps=20000,
                                           decay_rate=1,
                                           staircase=True))
    optimizers.append(tf.train.AdamOptimizer(learning_rate=lrs[i]))

input1 = tf.placeholder(dtype=tf.float32, shape=[None, 289], name='input1')
da1 = SSDA(input1, f_unit=289, h_unit=layer_sizes[0])
loss1 = da1.loss()
train_op1 = optimizers[0].minimize(loss1, var_list=[da1.w1, da1.b1, da1.b1_], global_step=global_steps[0])

# Layer 2
da2 = SSDA(da1.y1, f_unit=layer_sizes[0], h_unit=layer_sizes[1])
loss2 = da2.loss()
train_op2 = optimizers[1].minimize(loss2, var_list=[da2.w1, da2.b1, da2.b1_], global_step=global_steps[1])

# Layer 3
da3 = SSDA(da2.y1, f_unit=layer_sizes[1], h_unit=layer_sizes[2])
loss3 = da3.loss()
train_op3 = optimizers[2].minimize(loss3, var_list=[da3.w1, da3.b1, da3.b1_], global_step=global_steps[2])

# Layer 4
da4 = SSDA(da3.y1, f_unit=layer_sizes[2], h_unit=layer_sizes[3])
loss4 = da4.loss()
train_op4 = optimizers[3].minimize(loss4, var_list=[da4.w1, da4.b1, da4.b1_], global_step=global_steps[3])

losses = [loss1, loss2, loss3, loss4]
train_ops = [train_op1, train_op2, train_op3, train_op4]

iterator = train_ds.make_one_shot_iterator()
batch_data_tf = iterator.get_next()

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

if vis:
    # layer 1
    weight1_sum = tf.summary.histogram('weight1', da1.w1)
    loss1_sum = tf.summary.scalar('loss1', da1.loss)
    l2_norm1_sum = tf.summary.scalar('l2_norm1', da1.l2norm)
    kl1_sum = tf.summary.scalar('kl1', da1.kl)
    weight_decay1_sum = tf.summary.scalar('weight_decay1', da1.weight_decay)

    # layer 2
    weight2_sum = tf.summary.histogram('weight2', da2.w1)
    loss2_sum = tf.summary.scalar('loss2', da2.loss)
    l2_norm2_sum = tf.summary.scalar('l2_norm2', da2.l2norm)
    kl2_sum = tf.summary.scalar('kl2', da2.kl)
    weight_decay2_sum = tf.summary.scalar('weight_decay2', da2.weight_decay)

    # layer 3
    weight3_sum = tf.summary.histogram('weight3', da3.w1)
    loss3_sum = tf.summary.scalar('loss3', da3.loss)
    l2_norm3_sum = tf.summary.scalar('l2_norm3', da3.l2norm)
    kl3_sum = tf.summary.scalar('kl3', da3.kl)
    weight_decay3_sum = tf.summary.scalar('weight_decay3', da3.weight_decay)

    weight4_sum = tf.summary.histogram('weight4', da4.w1)
    loss4_sum = tf.summary.scalar('loss4', da4.loss)
    l2_norm4_sum = tf.summary.scalar('l2_norm4', da4.l2norm)
    kl4_sum = tf.summary.scalar('kl4', da4.kl)
    weight_decay4_sum = tf.summary.scalar('weight_decay4', da4.weight_decay)

    merge1 = tf.summary.merge([weight1_sum, loss1_sum, l2_norm1_sum, kl1_sum, weight_decay1_sum])
    merge2 = tf.summary.merge([weight2_sum, loss2_sum, l2_norm2_sum, kl2_sum, weight_decay2_sum])
    merge3 = tf.summary.merge([weight3_sum, loss3_sum, l2_norm3_sum, kl3_sum, weight_decay3_sum])
    merge4 = tf.summary.merge([weight4_sum, loss4_sum, l2_norm4_sum, kl4_sum, weight_decay4_sum])
    merges = [merge1, merge2, merge3, merge4]

    c_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    train_writer = tf.summary.FileWriter('./logs/pretrain/{}'.format(c_time), sess.graph)


# 逐层训练
total_train_steps = 0
for c_layer in range(len(layer_sizes)):
    c_num_train_steps = 0
    for epoch in range(pretrain_epochs):
        for i in range(int(num_train_samples / batch_size)):
            sess.run(global_steps[c_layer])
            batch_data = sess.run(batch_data_tf)
            if vis:
                c_merge, _, c_loss = sess.run([merges[c_layer], train_ops[c_layer], losses[c_layer]], feed_dict={input1: batch_data})
                train_writer.add_summary(c_merge, c_num_train_steps)
            else:
                _, c_loss = sess.run([train_ops[c_layer], losses[c_layer]], feed_dict={input1: batch_data})
            print('Current layer: {}, epoch: {}, batch_id: {}/{}, loss: {}'.format(c_layer, epoch, i, int(num_train_samples / batch_size), c_loss))
            c_num_train_steps += 1
            total_train_steps += 1
            # if total_train_steps % stored_ckpt_batches == 0:
            #     saver = tf.train.Saver()
            #     saver.save(sess, './pretrained_model/pretrained-model', global_step=int(total_train_steps*batch_size))

saver = tf.train.Saver()
saver.save(sess, './pretrained_model/pretrained-model', global_step=int(total_train_steps*batch_size))

print('End of pretraining...')
