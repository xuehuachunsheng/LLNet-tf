
import os
import time
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf

root_path = '/home/wuyanxue/Data/StandardTestImages/dataset/'
train_path = os.path.join(root_path, 'train.txt')
test_path = os.path.join(root_path, 'test.txt')


out_train_tfrecords = os.path.join(root_path, 'train.tfrecords')
out_test_tfrecords = os.path.join(root_path, 'test.tfrecords')

train_writer = tf.data.experimental.TFRecordWriter(out_train_tfrecords)
test_writer = tf.data.experimental.TFRecordWriter(out_test_tfrecords)

def parse_func(a_line):
    a_line = tf.strings.strip(a_line) # Remove the '\n' char
    split_ = tf.string_split([a_line], delimiter=' ')
    split_ = split_.values
    split_ = tf.map_fn(lambda x: tf.string_to_number(x), split_, dtype=tf.float32)
    # noise_darkened_patch, origin_patch
    # x, y
    # return split_[-289:], split_[:289]
    return split_[:289], split_[289:289*2], split_[289*2:289*3], split_[289*3:]


train_ds = tf.data.TextLineDataset(train_path)
train_ds = train_ds.map(parse_func)
test_ds = tf.data.TextLineDataset(test_path)
test_ds = test_ds.map(parse_func)

def serialize_example(origin, noise, darked, noise_darked):
    feature = {
        'origin': tf.train.Feature(float_list=tf.train.FloatList(value=origin)),
        'noise': tf.train.Feature(float_list=tf.train.FloatList(value=noise)),
        'darked': tf.train.Feature(float_list=tf.train.FloatList(value=darked)),
        'noise_darked': tf.train.Feature(float_list=tf.train.FloatList(value=noise_darked)),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def generator():
    for features in train_ds:
        yield serialize_example(*features)

serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())

train_writer.write(serialized_features_dataset)

def generator():
    for features in test_ds:
        yield serialize_example(*features)

serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())

test_writer.write(serialized_features_dataset)
