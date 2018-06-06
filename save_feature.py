from __future__ import absolute_import, print_function

import torch
from torch.backends import cudnn
from evaluations import extract_features
import DataSet
import numpy as np
from utils import RandomIdentitySampler, mkdir_if_missing, logging, display

# torch.cuda.set_device(7)

cudnn.benchmark = True
r = '/opt/intern/users/xunwang/checkpoints/bin/cub/512/800_model.pkl'

data = 'cub'
dim = 512
model = torch.load(r)
model = model.cuda()

data = DataSet.create(data, train=True)

data_loader = torch.utils.data.DataLoader(
    data.test, batch_size=64, shuffle=False,
    sampler=RandomIdentitySampler(data.train, num_instances=32), drop_last=False)

features, labels = extract_features(model, data_loader, print_freq=4, metric=None)

features = [feature.resize_(1, dim) for feature in features]
features = torch.cat(features)

U, S, V = torch.svd(features)

print(S)





# np.save('represent.npy', features.numpy())






