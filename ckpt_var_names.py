
from tensorflow.python import pywrap_tensorflow
checkpoint_path = './pretrained_model/pretrained-model-246400'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    print(key, end='\t')
