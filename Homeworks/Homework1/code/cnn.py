import numpy as np

from layers import *


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - fc - softmax

    You may also consider adding dropout layer or batch normalization layer.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.params["W1"] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
        self.params["W2"] = weight_scale * np.random.randn(int(num_filters * (input_dim[1] - filter_size + 1) * (input_dim[2] - filter_size + 1) / 4), hidden_dim)
        self.params["b2"] = np.zeros(hidden_dim, dtype=np.float32)
        self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b3"] = np.zeros(num_classes, dtype=np.float32)
        self.params["gamma"] = weight_scale * np.random.randn(int(num_filters * (input_dim[1] - filter_size + 1) * (input_dim[2] - filter_size + 1)))
        self.params["beta"] = weight_scale * np.random.randn(int(num_filters * (input_dim[1] - filter_size + 1) * (input_dim[2] - filter_size + 1)))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Test version 1:
        X -> conv -> x1 -> relu -> x3 -> pool -> x5 ->
        hidden1 -> x6 -> relu -> x7 -> hidden2 -> scores -> softmax -> loss

        Full architecture
        X -> conv -> x1 -> batchnorm -> x2 -> relu -> x3 -> drop out -> x4 -> pool -> x5 ->
        hidden1 -> x6 -> relu -> x7 -> hidden2 -> scores -> softmax -> loss

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1 = self.params['W1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        gamma, beta = self.params['gamma'], self.params['beta']

        # pass conv_param to the forward pass for the convolutional layer
        # filter_size = W1.shape[2]
        N, C, H, W = X.shape
        # conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        if y is None:
            bn_param = {'mode': 'test'}
            dropout_param = {'p': 0.9, 'mode': 'test'}
        else:
            bn_param = {'mode': 'train'}
            dropout_param = {'p': 0.9, 'mode': 'train'}

        # pass pool_param to the forward pass for the max-pooling layer
        #
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        x, cache1 = conv_forward(X, W1)
        x, cache2 = batchnorm_forward(x, gamma, beta, bn_param)
        x, cache3 = relu_forward(x)
        x, cache4 = dropout_forward(x, dropout_param)
        x, cache5 = max_pool_forward(x, pool_param)
        x, cache6 = fc_forward(x, W2, b2)
        x, cache7 = relu_forward(x)
        scores, cache8 = fc_forward(x, W3, b3)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dLds = softmax_loss(scores, y)
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss = loss + 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(b2**2) + np.sum(W3**2) + np.sum(b3**2) + np.sum(gamma**2) + np.sum(beta ** 2))
        dLdx, dLdw3, dLdb3 = fc_backward(dLds, cache8)
        dLdx = relu_backward(dLdx, cache7)
        dLdx, dLdw2, dLdb2 = fc_backward(dLdx, cache6)
        dLdx = max_pool_backward(dLdx, cache5)
        dLdx = dropout_backward(dLdx, cache4)
        dLdx = relu_backward(dLdx, cache3)
        dLdx, dgamma, dbeta = batchnorm_backward(dLdx, cache2)
        dx, dLdw1 = conv_backward(dLdx, cache1)

        grads["W3"] = dLdw3 + self.reg * W3
        grads["b3"] = dLdb3 + self.reg * b3
        grads["W2"] = dLdw2 + self.reg * W2
        grads["b2"] = dLdb2 + self.reg * b2
        grads["W1"] = dLdw1 + self.reg * W1
        grads["gamma"] = dgamma + self.reg * gamma
        grads["beta"] = dbeta + self.reg * beta
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


pass
