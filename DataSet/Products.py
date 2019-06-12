from __future__ import absolute_import

import os
import os.path as osp
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from DataSet.CUB200 import default_loader, Generate_transform_Dict, MyData


class Products:
    def __init__(self, width=224, origin_width=256, ratio=0.16, root=None, transform=None):
        transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)
        if root is None:
            root = '../data/Products'
        
        train_txt = osp.join(root, 'train.txt')
        test_txt = osp.join(root, 'test.txt')

        self.train = MyData(root, label_txt=train_txt, transform=transform_Dict['rand-crop'])
        self.gallery = MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop'])
    
def test():
    data = Products()
    print(data.train[1][0][0][0])
    print(len(data.gallery), len(data.train))



if __name__=='__main__':
    test()




