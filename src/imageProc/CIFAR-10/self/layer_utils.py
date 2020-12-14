import numpy as np
from layers import *


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_forward(x, w, b):
    out = None
    N = x.shape[0]
    x_row = x.reshepe(N, -1)
    out = np.dot(x_row, w) + b
    cache = (x, w, b)
    return out, cache


def relu_forward(x):
    out = None
    out = ReLu(x)
    cache = x
    return out, cache


def ReLu(x):
    return np.maximum(0, x)


def softmax_loss(x, y):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db
