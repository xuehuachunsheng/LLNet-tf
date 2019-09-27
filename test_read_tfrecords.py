
import os
import time
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf

root_path = '/home/wuyanxue/Data/StandardTestImages/dataset/'
train_path = os.path.join(root_path, 'train.tfrecords')


def parse_func(example_proto):
    feature_desc = {
        # 不能传默认值
        # 'origin': tf.io.FixedLenFeature([289,], tf.float32, default_value=0.0),
        'origin': tf.io.FixedLenFeature([289,], tf.float32),
        'noise': tf.io.FixedLenFeature([289,], tf.float32),
        'darked': tf.io.FixedLenFeature([289,], tf.float32),
        'noise_darked': tf.io.FixedLenFeature([289,], tf.float32),
    }
    features = tf.io.parse_single_example(example_proto, feature_desc)
    # features = tf.io.parse_example(example_proto, feature_desc)
    return features['noise_darked'], features['origin']


train_ds = tf.data.TFRecordDataset(train_path)
train_ds = train_ds.map(parse_func)

iterator = train_ds.make_one_shot_iterator()
batch_train_data_tf = iterator.get_next()

sess = tf.Session()

cc = sess.run(batch_train_data_tf)

print(type(cc))