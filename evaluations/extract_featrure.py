from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
from utils import to_numpy
import numpy as np

# from .evaluation_metrics import cmc, mean_ap
from utils.meters import AverageMeter
from .cnn import extract_cnn_feature


def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x


def extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # features = OrderedDict()
    # labels = OrderedDict()
    features = list()
    labels = list()
    end = time.time()
    for i, (imgs, pids) in enumerate(data_loader):
        data_time.update(time.time() - end)
        # print(imgs.size())
        outputs = extract_cnn_feature(model, imgs)
        # print(outputs.size())
        for output, pid in zip(outputs, pids):
            features.append(output)
            labels.append(pid)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))
    return features, labels


def pairwise_distance(features, metric=None):
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    # normalize feature before test
    x = normalize(x)
    # print(4*'\n', x.size())
    if metric is not None:
        x = metric.transform(x)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True)
    # print(dist.size())
    dist = dist.expand(n, n)
    dist = dist + dist.t()
    dist = dist - 2 * torch.mm(x, x.t()) + 1e5 * torch.eye(n)
    dist = torch.sqrt(dist)
    return dist


def pairwise_similarity(x, y=None):

    if y is None:
        
        n = len(x)
        x = torch.cat(x)
        x = x.view(n, -1)
        x = normalize(x)
        # print(4*'\n', x.size())
        similarity = torch.mm(x, x.t()) - 1e5 * torch.eye(n)
        return similarity
        
    else:

        m = len(y)
        y = torch.cat(y)
        y = y.view(m, -1)
        y = normalize(y)

        n = len(x)
        x = torch.cat(x)
        x = x.view(n, -1)
        x = normalize(x)

        similarity = torch.mm(x, y.t())
        return similarity
        
    
    
