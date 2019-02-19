import numpy as np
import solver
import softmax as soft
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
N, D = x_train.shape

validation_per = 0.9
train_len_tr = int(validation_per * N)

train_data = x_train
train_tar = y_train
test_data = x_test
test_tar = y_test

validation_data = train_data[train_len_tr:]
validation_tar = train_tar[train_len_tr:]

data = {
    'X_train': train_data,  # training data
    'y_train': train_tar,  # training labels
    'X_val': validation_data,  # validation data
    'y_val': validation_tar  # validation labels
}
model = soft.SoftmaxClassifier(hidden_dim=500, reg=0.1)
solver = solver.Solver(model, data,
                       update_rule='adam',
                       optim_config={
                           'learning_rate': 1e-4,
                       },
                       lr_decay=0.95,
                       num_epochs=20, batch_size=128,
                       print_every=100)
solver.train()

acc = solver.check_accuracy(test_data, test_tar)

print('Test Accuracy: %f' % acc)
