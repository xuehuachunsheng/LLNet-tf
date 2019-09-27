
import os
import tensorflow as tf

root_path = '/home/wuyanxue/Data/StandardTestImages/dataset/'
train_path = os.path.join(root_path, 'train_2500patch_per_image.txt')
test_path = os.path.join(root_path, 'test_2500patch_per_image.txt')

out_train_tfrecords = os.path.join(root_path, 'train_2500patch_per_image.tfrecords')
out_test_tfrecords = os.path.join(root_path, 'test_2500patch_per_image.tfrecords')

train_writer = tf.io.TFRecordWriter(out_train_tfrecords)
test_writer = tf.io.TFRecordWriter(out_test_tfrecords)

def serialize_example(origin, noise, darked, noise_darked):
    feature = {
        'origin': tf.train.Feature(float_list=tf.train.FloatList(value=origin)),
        'noise': tf.train.Feature(float_list=tf.train.FloatList(value=noise)),
        'darked': tf.train.Feature(float_list=tf.train.FloatList(value=darked)),
        'noise_darked': tf.train.Feature(float_list=tf.train.FloatList(value=noise_darked)),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


with open(train_path, 'r') as f:
    s = f.readlines()
    for i, x in enumerate(s):
        if i % 1000 == 0:
            print('{}-th done...'.format(i))
        xx = x.strip().split(' ')
        xx = [float(c) for c in xx]
        origin = xx[:289]
        noise = xx[289:289*2]
        darked = xx[289*2:289*3]
        noise_darked = xx[289*3:289*4]
        example = serialize_example(origin, noise, darked, noise_darked)
        train_writer.write(example)

train_writer.close()

with open(test_path, 'r') as f:
    s = f.readlines()
    for i, x in enumerate(s):
        if i % 1000 == 0:
            print('{}-th done...'.format(i))
        xx = x.strip().split(' ')
        xx = [float(c) for c in xx]
        origin = xx[:289]
        noise = xx[289:289*2]
        darked = xx[289*2:289*3]
        noise_darked = xx[289*3:289*4]
        example = serialize_example(origin, noise, darked, noise_darked)
        test_writer.write(example)

test_writer.close()

