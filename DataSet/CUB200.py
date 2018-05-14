from __future__ import absolute_import

import os
# import torchvision.transforms as transforms
import torchvision.datasets as datasets
from DataSet import transforms


class CUB200:
    def __init__(self, root, train=True, test=True, transform=None):
        # Data loading code

        std_values = [0.229, 0.224, 0.225]
        mean_values = [104 / 255.0, 117 / 255.0, 128 / 255.0]

        if transform is None:
            transform = [transforms.Compose([
                transforms.CovertBGR(),
                transforms.Resize(256),
                transforms.RandomResizedCrop(scale=(0.16, 1), size=227),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_values,
                                     std=std_values),
            ]),
                transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.Resize(256),
                    transforms.CenterCrop(227),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean_values,
                                         std=std_values),
                ])]

        if root is None:
            root = '/opt/intern/users/xunwang/DataSet/CUB_200_2011'

        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'test')
        if train:
            self.train = datasets.ImageFolder(traindir, transform[0])
        if test:
            self.test = datasets.ImageFolder(testdir, transform[1])




