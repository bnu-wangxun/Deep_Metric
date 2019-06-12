from __future__ import absolute_import
from .meters import *
from .sampler import RandomIdentitySampler, FastRandomIdentitySampler
import torch
from .osutils import mkdir_if_missing
from .orthogonal_regularizaton import orth_reg
from .str2nums import chars2nums
from .HyperparamterDisplay import display
from .Batch_generator import BatchGenerator
from .cluster import cluster_
from .numpy_tozero import to_zero

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

