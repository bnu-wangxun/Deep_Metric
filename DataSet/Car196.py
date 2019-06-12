from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
from PIL import Image

import os
from torchvision import transforms
from collections import defaultdict

from DataSet.CUB200 import MyData, default_loader, Generate_transform_Dict


class Cars196:
    def __init__(self, root=None, origin_width=256, width=227, ratio=0.16, transform=None):
        if transform is None:
            transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)
        if root is None:
            root = 'data/Cars196/'

        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')
        self.train = MyData(root, label_txt=train_txt, transform=transform_Dict['rand-crop'])
        self.gallery = MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop'])


def testCar196():
    data = Cars196()
    print(len(data.gallery))
    print(len(data.train))
    print(data.train[1])


if __name__ == "__main__":
    testCar196()


