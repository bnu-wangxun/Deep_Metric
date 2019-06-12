# coding : utf-8
from __future__ import absolute_import
import numpy as np
# from utils import to_numpy
import time
# import bottleneck
import random
import torch
import heapq


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def Compute_top_k(sim_mat, k=10):
    # start_time = time.time()
    # print(start_time)
    """
    :param sim_mat:

    Compute
    top-k in gallery for each query
    """

    sim_mat = to_numpy(sim_mat)
    m, n = sim_mat.shape
    print('query number is %d' % m)
    print('gallery number is %d' % n)

    top_k = np.zeros([m, k])

    for i in range(m):
        sim_i = sim_mat[i]
        idx = heapq.nlargest(k, range(len(sim_i)), sim_i.take)
        top_k[i] = idx
    return top_k


def test():
    import torch
    sim_mat = torch.rand(int(1e1), int(1e2))
    sim_mat = to_numpy(sim_mat)
    print(Compute_top_k(sim_mat, k=3))

if __name__ == '__main__':
    test()
