from __future__ import print_function, absolute_import

def to_zero(x):
    h = x.shape[0]
    w = x.shape[1]
    for i in range(h):
        for j in range(w):
            x[i][j] = 0
