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

R@1     R@10    R@20    R@30    R@40    R@50
87.3    96.7    97.9    98.2    98.5    98.7(%)

0.8621  0.9594  0.9723  0.9777  0.9814  0.9846

0.8656  0.9618  0.9733  0.9786  0.9819  0.9838
