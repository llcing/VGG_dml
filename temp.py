# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import models
net = 'vgg'
model = models.create(net, pretrained=True)

model.features = torch.nn.Sequential(
    model.features,
    torch.nn.MaxPool2d(7),
    torch.nn.Dropout(p=0.01)
)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(512, 512)
)
print(model)
model_dict = model.state_dict()
print('initialize the FC layer orthogonally')
w = model_dict['classifier.0.weight']
print(w[0][0])
model_dict['classifier.0.weight'] = torch.nn.init.orthogonal_(w)

model_dict = model.state_dict()
w = model_dict['classifier.0.weight']
print(w[0][0])

