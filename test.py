# coding=utf-8
from __future__ import absolute_import, print_function
import argparse

import torch
from torch.backends import cudnn
from evaluations import extract_features, pairwise_similarity
from evaluations import Recall_at_ks, Recall_at_ks_products, Recall_at_ks_shop
import models
import DataSet

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('-data', type=str, default='cub')
parser.add_argument('-r', type=str, default='model.pkl', metavar='PATH')

parser.add_argument('-test', type=int, default=1,
                    help='evaluation on test set or train set')
parser.add_argument('-dim', type=int, default=512,
                    help='Dimension of Embedding Feather')
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('--nThreads', '-j', default=16, type=int, metavar='N',
                    help='number of data loading threads (default: 2)')

args = parser.parse_args()

PATH = args.r
model = models.create('vgg', dim=args.dim, pretrained=False)
model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load(PATH))

model = model.cuda()
# print(model)
temp = args.r.split('/')
name = temp[-1][:-10]


if args.data == 'shop':
    data = DataSet.create(args.data)
    gallery_loader = torch.utils.data.DataLoader(
        data.gallery, batch_size=args.batch_size, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=args.nThreads)
    query_loader = torch.utils.data.DataLoader(
        data.query, batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        pin_memory=True, num_workers=args.nThreads)
    
    gallery_feature, gallery_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None)
    query_feature, query_labels = extract_features(model, query_loader, print_freq=1e5, metric=None)

    sim_mat = pairwise_similarity(x=query_feature, y=gallery_feature)
    result = Recall_at_ks_shop(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels)

elif args.data == 'jd':
    if args.test == 1:
        data = DataSet.create(args.data)
        data_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=args.batch_size, 
            shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.nThreads)
    else:
        data = DataSet.create(args.data)
        data_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=args.batch_size, shuffle=False, drop_last=False,
            pin_memory=True, num_workers=args.nThreads)
    features, labels = extract_features(model, data_loader, print_freq=5, metric=None)

    sim_mat = pairwise_similarity(features)
    result = Recall_at_ks(sim_mat, query_ids=labels, gallery_ids=labels)

else:
    if args.test == 1:
        data = DataSet.create(args.data, train=False)
        data_loader = torch.utils.data.DataLoader(
            data.test, batch_size=args.batch_size, shuffle=False, drop_last=False,
            pin_memory=True, num_workers=args.nThreads)
    else:
        data = DataSet.create(args.data, test=False)
        data_loader = torch.utils.data.DataLoader(
            data.train, batch_size=args.batch_size, shuffle=False, drop_last=False,
            pin_memory=True, num_workers=args.nThreads)
    features, labels = extract_features(model, data_loader, print_freq=1e5, metric=None)

    num_class = len(set(labels))
    
    sim_mat = pairwise_similarity(features)
    if args.data == 'product':
        result = Recall_at_ks_products(sim_mat, query_ids=labels, gallery_ids=labels)
    else:
        result = Recall_at_ks(sim_mat, query_ids=labels, gallery_ids=labels)

result = ['%.4f' % r for r in result]
temp = '  '
result = temp.join(result)
print('Epoch-%s' % name, result)

