import numpy as np
from layer_utils import *


def _init_(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):
    self.params = {}
    self.reg = reg
    self.params['W1'] = weight_scale * np.random.random(input_dim, hidden_dim)
    self.params['b1'] = np.zeros((1, hidden_dim))
    self.params['W2'] = weight_scale * np.random.random(hidden_dim, num_classes)
    self.params['b2'] = np.zeros((1, num_classes))


def loss(self, X, y=None):
    scores = None
    N = X.shape[0]
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    h1, cache1 = affine_relu_forward(X, W1, b1)
    out, cache2 = affine_forward(h1, W2, b2)
    scores = out
    if y is None:
        return scores

    loss, grads = 0, []
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss

    dh1, dW2, db2 = affine_backward(scores, cache2)
    dx, dW1, db1 = affine_relu_backward(dh1, cache1)

    dW2 += self.reg * W2
    dW1 += self.reg * W1
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2
    return loss, grads
