from __future__ import absolute_import, print_function
"""
In-shop-clothes data-set for Pytorch
"""
import torch
import torch.utils.data as data
from PIL import Image

import os
from torchvision import transforms
from collections import defaultdict

from DataSet.CUB200 import default_loader, Generate_transform_Dict


class MyData(data.Dataset):
    def __init__(self, root=None, label_txt=None,
                 transform=None, loader=default_loader):

        # Initialization data path and train(gallery or query) txt path

        if root is None:
            root = "/home/xunwang"
            label_txt = os.path.join(root, 'train.txt')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if transform is None:
            transform = transforms.Compose([
                # transforms.CovertBGR(),
                transforms.Resize(256),
                transforms.RandomResizedCrop(scale=(0.16, 1), size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        # read txt get image path and labels
        file = open(label_txt)
        images_anon = file.readlines()

        images = []
        labels = []

        for img_anon in images_anon:
            img_anon = img_anon.replace(' ', '\t')

            [img, label] = (img_anon.split('\t'))[:2]
            images.append(img)
            labels.append(int(label))

        classes = list(set(labels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels):
            Index[label].append(i)

        # Initialization Done
        self.root = root
        self.images = images
        self.labels = labels
        self.classes = classes
        self.transform = transform
        self.Index = Index
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.images[index], self.labels[index]
        # print(os.path.join(self.root, fn))
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)


class InShopClothes:
    def __init__(self, root=None, crop=False, origin_width=256, width=224, ratio=0.16):
        # Data loading code
        transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)

        if root is None:
            root = 'data/In_shop_clothes'

        train_txt = os.path.join(root, 'train.txt')
        gallery_txt = os.path.join(root, 'gallery.txt')
        query_txt = os.path.join(root, 'query.txt')

        self.train = MyData(root, label_txt=train_txt, transform=transform_Dict['rand-crop'])
        self.gallery = MyData(root, label_txt=gallery_txt, transform=transform_Dict['center-crop'])
        self.query = MyData(root, label_txt=query_txt, transform=transform_Dict['center-crop'])



def testIn_Shop_Clothes():
    data = InShopClothes()
    print(len(data.gallery), len(data.train))
    print(len(data.query))
    print(data.train[1][0][0][0][1])


if __name__ == "__main__":
    testIn_Shop_Clothes()