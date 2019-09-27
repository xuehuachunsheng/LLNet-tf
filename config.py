
layer_sizes = [1000, 1000, 1000, 289]
lr = [0.0001, 0.0001, 0.0001, 0.0001] # when the learning rate is 0.1, the result may be nan.
# lr = [0.1, 0.1, 0.1, 0.01] # when the learning rate is 0.1, the result may be nan.
# num_train_samples = 27485
num_train_samples = 275287
# num_test_samples = 13765
num_test_samples = 137213

batch_size = 10
rho = 0.2
lambda_reg = 0.00001
beta = 0.01
stored_ckpt_batches = 3000
pretrain_epochs = 3
