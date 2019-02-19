import pickle
import logistic as log
import solver


with open('data.pkl', 'rb') as f:
  u = pickle._Unpickler(f)
  u.encoding = 'latin1'
  p = u.load()

data = p[0]
target = p[1]

N, D = data.shape
train_len = 500
valid_len = 250
test_len = 250

train_data = data[:train_len]
train_tar = target[:train_len]
valid_data = data[train_len:train_len + valid_len]
valid_tar = target[train_len:train_len + valid_len]
test_data = data[train_len + valid_len:]
test_tar = target[train_len + valid_len:]

data = {
    'X_train': train_data,  # training data
    'y_train': train_tar,  # training labels
    'X_val': valid_data,  # validation data
    'y_val': valid_tar  # validation labels
}

model = log.LogisticClassifier(input_dim=D, hidden_dim=200, reg=0.001)

solver = solver.Solver(model, data,
                       update_rule='adam',
                       optim_config={
                           'learning_rate': 1e-3,
                       },
                       lr_decay=1,
                       num_epochs=3000, batch_size=500,
                       print_every=1)
solver.train()

acc = solver.check_accuracy(test_data, test_tar)

print('Test Accuracy: %f' % acc)
