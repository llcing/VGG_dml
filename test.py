# coding=utf-8
from __future__ import absolute_import, print_function
import argparse

import torch
from torch.backends import cudnn
from evaluations import extract_features, pairwise_distance, pairwise_similarity
from evaluations import Recall_at_ks, Recall_at_ks_products, Recall_at_ks_shop
import DataSet

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('-data', type=str, default='cub')
parser.add_argument('-r', type=str, default='model.pkl', metavar='PATH')

parser.add_argument('-test', type=int, default=1,
                    help='evaluation on test set or train set')

args = parser.parse_args()

# model = inception_v3(dropout=0.5)
model = torch.load(args.r)
model = model.cuda()
# print(model)
temp = args.r.split('/')
name = temp[-1][:-10]


if args.data == 'shop':
    data = DataSet.create(args.data)
    gallery_loader = torch.utils.data.DataLoader(
        data.gallery, batch_size=48, shuffle=False, drop_last=False)
    query_loader = torch.utils.data.DataLoader(
        data.query, batch_size=48, shuffle=False, drop_last=False)
    
    gallery_feature, gallery_labels = extract_features(model, gallery_loader, print_freq=10, metric=None)
    query_feature, query_labels = extract_features(model, query_loader, print_freq=10, metric=None)

    sim_mat = pairwise_similarity(x=query_feature, y=gallery_feature)
    result = Recall_at_ks_shop(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels)

else:
    if args.test == 1:
        data = DataSet.create(args.data, train=False)
        data_loader = torch.utils.data.DataLoader(
            data.test, batch_size=48, shuffle=False, drop_last=False)
    else:
        data = DataSet.create(args.data, test=False)
        data_loader = torch.utils.data.DataLoader(
            data.train, batch_size=48, shuffle=False, drop_last=False)
    features, labels = extract_features(model, data_loader, print_freq=1000, metric=None)
    
    num_class = len(set(labels))
    
    sim_mat = -pairwise_distance(features)
    if args.data == 'product':
        result = Recall_at_ks_products(sim_mat, query_ids=labels, gallery_ids=labels)
    else:
        result = Recall_at_ks(sim_mat, query_ids=labels, gallery_ids=labels)

result = ['%.4f' % r for r in result]
temp = '  '
result = temp.join(result)
print('Epoch-%s' % name, result)

#
#
# # coding=utf-8
#
# # from __future__ import absolute_import, print_function
# import argparse
#
# import torch
# from torch.backends import cudnn
# from evaluations import extract_features, pairwise_distance, pairwise_similarity
# from evaluations import Recall_at_ks_shop, Recall_at_ks_products
# import DataSet
# import  os
#
# print(torch.__version__)
#
# r = '/opt/intern/users/xunwang/checkpoints/bin/shop/512/100_model.pkl'
# model = torch.load(r)
# model = model.cuda()
#
# data = DataSet.create('product')
# gallery_loader = torch.utils.data.DataLoader(
#     data.test, batch_size=32, shuffle=False, drop_last=False)
#
#
# gallery_feature, gallery_labels = extract_features(model, gallery_loader, print_freq=10, metric=None)
# # query_feature, query_labels = extract_features(model, query_loader, print_freq=10, metric=None)
#
# sim_mat = pairwise_similarity(x=gallery_feature, y=gallery_feature)
# result = Recall_at_ks_products(sim_mat, query_ids=gallery_labels, gallery_ids=gallery_labels)
#
