from __future__ import absolute_import, print_function

import torch
from torch.backends import cudnn
from evaluations import extract_features
import DataSet
import numpy as np
torch.cuda.set_device(1)

cudnn.benchmark = True
r = '/opt/intern/users/xunwang/checkpoints/neighbour/car/2-0.05/500_model.pkl'

data = 'car'
dim = 64

# model = inception_v3(dropout=0.5)
model = torch.load(r)
model = model.cuda()

data = DataSet.create(data, train=True)
data_loader = torch.utils.data.DataLoader(
    data.test, batch_size=8, shuffle=False, drop_last=False)

features, labels = extract_features(model, data_loader, print_freq=32, metric=None)
features = [feature.resize_(1, dim) for feature in features]
features = torch.cat(features)

np.save('represent.npy', features.numpy())






