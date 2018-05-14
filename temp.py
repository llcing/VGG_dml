# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import models

model = models.create(args.net, pretrained=True)

model.features = torch.nn.Sequential(
    model.features,
    torch.nn.MaxPool2d(7),
    torch.nn.Dropout(p=0.01)
)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(512, 512)
)
print(model)
