
import os
import time
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf

from network import SSDA, MultiSSDA
vis = True

continue_train = True

epochs = 30
lr = 0.0001 # when the learning rate is 0.1, the result may be nan, 1e-5
num_train_samples = 27485
num_test_samples = 13765
batch_size = 128
stored_ckpt_batches = 1000
eval_frequency_epochs = 3
log_dir = './logs/finetune'

########################## Pretrained hyper-parameters #########################
# According to the official implementations
# https://github.com/kglore/llnet_color/blob/master/library.py
########################## End of pretrained_model hyper-parameters #########################

########################## Load data #########################

root_path = '/home/wuyanxue/Data/StandardTestImages/dataset/'
train_path = os.path.join(root_path, 'train.txt')
test_path = os.path.join(root_path, 'test.txt')


def parse_func(a_line):
    a_line = tf.strings.strip(a_line) # Remove the '\n' char
    split_ = tf.string_split([a_line], delimiter=' ')
    split_ = split_.values
    split_ = tf.map_fn(lambda x: tf.string_to_number(x), split_, dtype=tf.float32)
    # noise_darkened_patch, origin_patch
    # x, y
    return split_[-289:], split_[:289]


train_ds = tf.data.TextLineDataset(train_path)
train_ds = train_ds.map(parse_func).shuffle(25000).repeat().batch(batch_size)
train_ds = train_ds.prefetch(buffer_size=batch_size)


test_ds = tf.data.TextLineDataset(test_path)
test_ds = test_ds.map(parse_func).repeat().batch(batch_size)
test_ds = test_ds.prefetch(buffer_size=batch_size)

########################## End of load data #########################

########################## Training #########################
# 这个结构是逐层训练的
# Layer 1

iterator = train_ds.make_one_shot_iterator()
batch_train_data_tf = iterator.get_next()
iterator = test_ds.make_one_shot_iterator()
batch_test_data_tf = iterator.get_next()

# Network
input1 = tf.placeholder(dtype=tf.float32, shape=[None, 289], name='input1')
net = MultiSSDA(inputs=input1)
net.initial_decoder()

# session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
if not continue_train:
    saver.restore(sess, './pretrained_model/whole-pretrained-model-246400')
    global_step = tf.Variable(0, trainable=False)
    num_global_step = 0
    start_epochs = 0
    global_var_step = 0
else:
    ckpt = tf.train.get_checkpoint_state('./finetuned_model/')
    saver.restore(sess, ckpt.model_checkpoint_path)
    str_global_step = ckpt.model_checkpoint_path[len('./finetuned_model/finetuned-model.ckpt-'):]
    num_global_step = int(int(str_global_step) / batch_size)
    global_step = tf.Variable(num_global_step, trainable=False)
    start_epochs = int(np.round(int(str_global_step) / num_train_samples))
    global_var_step = int(num_test_samples/num_train_samples * num_global_step)

# optimizer

learning_rate = tf.train.exponential_decay(learning_rate=lr,
                                           global_step=global_step,
                                           decay_steps=2000,
                                           decay_rate=0.96,
                                           staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
net.loss()
loss = net.l2norm

train_op = optimizer.minimize(loss)

batch_ssim_tf = net.ssim()
mean_ssim_tf = tf.reduce_mean(batch_ssim_tf)

batch_psnr_tf = net.psnr()
mean_psnr_tf = tf.reduce_mean(batch_psnr_tf)


# 初始化剩余变量
def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


initialize_uninitialized(sess)

if vis:
    weights_sum = []
    for i, da in enumerate(net.das):
        # layer 1
        weighti_sum = tf.summary.histogram('weight{}'.format(i), da.w1)
        weights_sum.append(weighti_sum)
    loss_sum = tf.summary.scalar('loss', net.loss)
    l2_norm_sum = tf.summary.scalar('l2_norm', net.l2norm)
    weight_decay_sum = tf.summary.scalar('weight_decay', net.weight_decay)

    ssim_sum = tf.summary.scalar('mean_ssim', mean_ssim_tf)
    psnr_sum = tf.summary.scalar('mean_psnr', mean_psnr_tf)

    merge = tf.summary.merge_all()
    merge_test = tf.summary.merge([loss_sum, ssim_sum, psnr_sum])

    if continue_train:
        times = os.listdir(log_dir)
        c_time = sorted(times)[-1]
    else:
        c_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    train_dir = os.path.join(log_dir, '{}/train'.format(c_time))
    test_dir = os.path.join(log_dir, '{}/test'.format(c_time))
    train_writer = tf.summary.FileWriter(train_dir, sess.graph)
    test_writer = tf.summary.FileWriter(test_dir, sess.graph)


# 逐层训练
c_num_train_steps = num_global_step
c_num_val_steps = global_var_step
for epoch in range(start_epochs, start_epochs+epochs):
    # Train
    for i in range(int(num_train_samples / batch_size)):
        sess.run(global_step)
        batch_x, batch_y = sess.run(batch_train_data_tf)
        if vis:
            c_merge, _, c_loss = sess.run([merge, train_op, loss], feed_dict={input1: batch_x, net.y_true: batch_y})
            train_writer.add_summary(c_merge, c_num_train_steps)
        else:
            _, c_loss = sess.run([train_op, loss], feed_dict={input1: batch_x, net.y_true: batch_y})

        print('Training... Epoch: {}, Batch: {}/{}, loss: {}'.format(epoch, i, int(num_train_samples / batch_size), c_loss))
        c_num_train_steps += 1
        if c_num_train_steps % stored_ckpt_batches == 0:
            saver = tf.train.Saver()
            saver.save(sess, './finetuned_model/finetuned-model.ckpt', global_step=int(c_num_train_steps*batch_size))

    # Evaluate
    if (epochs + 1) % eval_frequency_epochs == 0:
        for i in range(int(num_test_samples / batch_size)):
            batch_x, batch_y = sess.run(batch_test_data_tf)
            if vis:
                c_merge, c_loss, batch_ssim, batch_psnr = sess.run([merge_test, loss, batch_ssim_tf, batch_psnr_tf], feed_dict={input1: batch_x, net.y_true: batch_y})
                test_writer.add_summary(c_merge, c_num_val_steps)
            else:
                c_loss, batch_ssim, batch_psnr = sess.run([loss, batch_ssim_tf, batch_psnr_tf], feed_dict={input1: batch_x, net.y_true: batch_y})

            print('Evaluating... Batch ID: {}, Validation loss: {}, Mean ssim: {}, Mean psnr: {}'.format(i, c_loss, np.mean(batch_ssim), np.mean(batch_psnr)))
            c_num_val_steps += 1


print('End of finetune mix model')
