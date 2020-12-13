import numpy as np


def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, x.shape)
    x_row = x.reshape(x.shape[0], -1)
    dw = np.dot(x_row.T, dout)
    db = np.sum(dout, axis=0, keepdims=True)
    return dx, dw, db


def relu_backward(dout, cache):
    dx, x = None, cache
    dx = dout
    dx[x <= 0] = 0
    return dx
