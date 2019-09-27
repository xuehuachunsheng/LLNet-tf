

import tensorflow as tf


ckpt = tf.train.get_checkpoint_state('./finetuned_model/')

print(ckpt)

print(ckpt.model_checkpoint_path)