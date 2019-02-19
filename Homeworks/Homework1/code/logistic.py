import numpy as np
from layers import *


class LogisticClassifier(object):
    """
    A logistic regression model with optional hidden layers.

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
            y = y * 2 - 1
        loss, dLds = logistic_loss(scores, y)
        if "W2" in self.params:
            loss = loss + 0.5 * self.reg * (np.sum(W1**2) + np.sum(b1**2) + np.sum(W2**2) + np.sum(b2**2))
            dLdx2 = np.copy(dLds)
            grads["W2"] = a1.T.dot(dLdx2) + self.reg * W2
            grads["b2"] = np.sum(dLdx2) + self.reg * b2
            dLda1 = np.outer(dLdx2, W2)
            dLdx1 = np.copy(dLda1)
            dLdx1[x1 <= 0] = 0
            grads["W1"] = X.T.dot(dLdx1) + self.reg * W1
            grads["b1"] = np.sum(dLdx1, axis=0) + self.reg * b1
        else:
            loss = loss + 0.5 * self.reg * (np.sum(W1**2) + np.sum(b1**2))
            dLdx1 = np.copy(dLds)
            grads["W1"] = X.T.dot(dLdx1) + self.reg * W1
            grads["b1"] = np.sum(dLdx1) + self.reg * b1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
