# coding=utf-8
from __future__ import absolute_import, print_function
import time
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
# import models
import losses
from utils import RandomIdentitySampler, FastRandomIdentitySampler, mkdir_if_missing, logging, display
import DataSet
import numpy as np
cudnn.benchmark = True


def save_model(model, filename):
    state = model.state_dict()
    for key in state:
        state[key] = state[key].clone().cpu()
    torch.save(state, filename)

for i in range(135, 200, 5):
    r = '/opt/intern/users/xunwang/checkpoints/bin/jd/512-BN-alpha40/%d_model.pkl' % i
    t = '/opt/intern/users/xunwang/checkpoints/bin/jd/512-BN-alpha40/%d_model.pth' % i
    model = torch.load(r)
    save_model(model, t)

