from __future__ import absolute_import, print_function
import os
from DataSet import transforms
import torchvision.datasets as datasets


class Car196:
    def __init__(self, root, train=True, test=True, transform=None):
        # Data loading code

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if transform is None:
            transform = [transforms.Compose([
                transforms.CovertBGR(),
                transforms.Resize(256),
                transforms.RandomResizedCrop(scale=(0.16, 1), size=227),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
                transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.Resize(256),
                    transforms.CenterCrop(227),
                    transforms.ToTensor(),
                    normalize,
                ])]

        if root is None:
            root = '/opt/intern/users/xunwang/DataSet/Car196'
        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'test')

        if train:
            self.train = datasets.ImageFolder(traindir, transform[0])
        if test:
            self.test = datasets.ImageFolder(testdir, transform[1])

