# coding : utf-8
from __future__ import absolute_import
import numpy as np
import torch
from utils import to_numpy
import time
import random


def Recall_at_ks(sim_mat, data='cub', query_ids=None, gallery_ids=None):
    # start_time = time.time()
    # print(start_time)
    """
    :param sim_mat:
    :param query_ids
    :param gallery_ids
    :param data

    Compute  [R@1, R@2, R@4, R@8]
    """

    ks_dict = dict()
    ks_dict['cub'] = [1, 2, 4, 8, 16, 32]
    ks_dict['car'] = [1, 2, 4, 8, 16, 32]
    ks_dict['jd'] = [1, 2, 4, 8]
    ks_dict['product'] = [1, 10, 100, 1000]
    ks_dict['shop'] = [1, 10, 20, 30, 40, 50]

    if data is None:
        data = 'cub'
    k_s = ks_dict[data]

    sim_mat = to_numpy(sim_mat)
    m, n = sim_mat.shape
    gallery_ids = np.asarray(gallery_ids)
    if query_ids is None:
        query_ids = gallery_ids
    else:
        query_ids = np.asarray(query_ids)

    num_max = int(1e6)

    if m > num_max:
        samples = list(range(m))
        random.shuffle(samples)
        samples = samples[:num_max]
        sim_mat = sim_mat[samples, :]
        query_ids = [query_ids[k] for k in samples]
        m = num_max

    # Hope to be much faster  yes!!
    num_valid = np.zeros(len(k_s))
    neg_nums = np.zeros(m)
    for i in range(m):
        x = sim_mat[i]

        pos_max = np.max(x[gallery_ids == query_ids[i]])
        neg_num = np.sum(x > pos_max)
        neg_nums[i] = neg_num

    for i, k in enumerate(k_s):
        if i == 0:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp
        else:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp - num_valid[i-1]
    # t = time.time() - start_time
    # print(t)
    return num_valid / float(m)


def test():
    sim_mat = torch.rand(int(7e2), int(14e2))
    sim_mat = to_numpy(sim_mat)
    query_ids = int(1e2)*list(range(7))
    gallery_ids = int(2e2)*list(range(7))
    gallery_ids = np.asarray(gallery_ids)
    query_ids = np.asarray(query_ids)
    print(Recall_at_ks(sim_mat,  query_ids=query_ids, gallery_ids=gallery_ids, data='shop'))

if __name__ == '__main__':
    test()
