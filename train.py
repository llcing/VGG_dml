# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import models

vgg16 = models.create('vgg16_bn', pretrained=True)
print(vgg16)