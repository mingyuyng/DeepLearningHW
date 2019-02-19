import numpy as np
import pickle
import gzip
import solver
import cnn
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#mndata = MNIST('samples')
#images, labels = mndata.load_training()
#images = np.asarray(images)
#labels = np.asarray(labels)

N, H, W = x_train.shape
x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)

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
model = cnn.ConvNet(reg=0.000, hidden_dim=300)
solver = solver.Solver(model, data,
                       update_rule='adam',
                       optim_config={
                           'learning_rate': 5e-4,
                       },
                       lr_decay=0.95,
                       num_epochs=10, batch_size=128,
                       print_every=1)
solver.train()

acc = solver.check_accuracy(test_data, test_tar)

print('Test Accuracy: %f' % acc)
