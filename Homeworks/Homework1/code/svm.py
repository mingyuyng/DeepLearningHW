import numpy as np
import pickle
import gzip
from layers import *


class SVM(object):
    """
    A binary SVM classifier with optional hidden layers.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the model. Weights            #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases (if any) using the keys 'W2' and 'b2'.                        #
        ############################################################################
        if hidden_dim is not None:
            self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
            self.params["b1"] = np.zeros(hidden_dim)
            self.params["W2"] = weight_scale * np.random.randn(hidden_dim)
            self.params["b2"] = 0
        else:
            self.params["W1"] = weight_scale * np.random.randn(input_dim)
            self.params["b1"] = 0
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, D)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N,) where scores[i] represents the probability
        that X[i] belongs to the positive class.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the model, computing the            #
        # scores for X and storing them in the scores variable.                    #
        ############################################################################
        if "W2" in self.params:
            W1 = self.params["W1"]
            b1 = self.params["b1"]
            W2 = self.params["W2"]
            b2 = self.params["b2"]
            x1 = X.dot(W1) + b1
            a1 = np.maximum(x1, 0)
            scores = a1.dot(W2) + b2
        else:
            W1 = self.params["W1"]
            b1 = self.params["b1"]
            scores = X.dot(W1) + b1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the model. Store the loss          #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss and make sure that grads[k] holds the gradients for self.params[k]. #
        # Don't forget to add L2 regularization.                                   #
        #                                                                          #
        ############################################################################
        if 0 in y:
            y = 2 * y - 1
        loss, dLds = svm_loss(scores, y)
        if "W2" in self.params:
            loss = loss + 0.5 * self.reg * (np.sum(W1**2) + np.sum(b1**2) + np.sum(W2**2) + np.sum(b2**2))
            dLdx2 = dLds
            grads["W2"] = a1.T.dot(dLdx2) + self.reg * W2
            grads["b2"] = np.sum(dLdx2) + self.reg * b2
            dLda1 = np.outer(dLdx2, W2)
            dLdx1 = dLda1
            dLdx1[x1 <= 0] = 0
            grads["W1"] = X.T.dot(dLdx1) + self.reg * W1
            grads["b1"] = np.sum(dLdx1, axis=0) + self.reg * b1
        else:
            loss = loss + 0.5 * self.reg * (np.sum(W1**2) + np.sum(b1**2))
            dLdx1 = dLds
            grads["W1"] = X.T.dot(dLdx1) + self.reg * W1
            grads["b1"] = np.sum(dLdx1) + self.reg * b1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


if __name__ == '__main__':

        # Load data.pkl
    with open('data.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    data = p[0]
    target = p[1]
    target[target == 0] = -1

    N, D = data.shape
    training_per = 0.7
    training_len = int(training_per * N)

    train_data = data[:training_len]
    train_tar = target[:training_len]
    test_data = data[training_len:]
    test_tar = target[training_len:]

    one_layer_net = SVM(input_dim=D, hidden_dim=4 * D, weight_scale=1e-3, reg=0.05)

    steps = 5000
    lr = 5e-5
    loss_set = []
    for i in range(steps):
        loss, grads = one_layer_net.loss(train_data, train_tar)

        one_layer_net.params["W1"] -= lr * grads["W1"]
        one_layer_net.params["b1"] -= lr * grads["b1"]
        one_layer_net.params["W2"] -= lr * grads["W2"]
        one_layer_net.params["b2"] -= lr * grads["b2"]
        loss_set.append(loss)
        print("Iteration " + str(i) + " Loss: " + str(loss))

    score_test = one_layer_net.loss(test_data)
    score_test[score_test >= 0] = 1
    score_test[score_test < 0] = -1
    test_err = np.sum(abs(score_test - test_tar)) / (N - training_len) / 2
    print(test_err)
