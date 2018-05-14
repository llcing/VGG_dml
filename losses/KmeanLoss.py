from __future__ import print_function, absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from utils import BatchGenerator
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


def pair_euclidean_dist(inputs_x, inputs_y):
    n = inputs_x.size(0)
    m = inputs_y.size(0)
    xx = torch.pow(inputs_x, 2).sum(dim=1, keepdim=True).expand(
        n, m)
    yy = torch.pow(inputs_y, 2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy
    dist.addmm_(1, -2, inputs_x, inputs_y.t())
    # dist = dist.clamp(min=1e-12).sqrt()
    return dist


class KmeanLoss(nn.Module):
    def __init__(self, alpha=16, n_cluster=2, beta=0.5):
        super(KmeanLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_cluster

    def cluster(self, inputs, targets):
        X = inputs.data.cpu().numpy()
        y = targets.data.cpu().numpy()

        result = dict()
        for y_ in set(y):
            # print(y_)
            idx_ = np.where(y == y_)
            # print(idx_[0])
            X_i = X[idx_[0]]
            # print(X_i)
            kmeans = KMeans(n_clusters=3, random_state=1).fit(X_i)
            pred_cluster = kmeans.labels_
            print(pred_cluster)

        for i in range(len(y)):

            k = str(y[i]) + ' ' + str(pred_cluster[i])
            if k in result:
                result[k].append(i)
            else:
                result[k] = [i]
        split_ = result.values()
        return split_

    def forward(self, inputs, targets):
        split_ = self.cluster(inputs, targets)

        if not np.random.randint(64) == 1:
            print('split batch into %d clusters' % len(split_))
            for t in split_:
                print(t)

        num_dim = inputs.size(1)
        n = inputs.size(0)
        centers = []
        inputs_list = []
        targets_ = []

        cluster_mat = np.ones([n, len(split_)])
        for i, split_i in enumerate(split_):
            # print(split_i)
            size_ = len(split_i)
            if size_ > 1:
                for k in split_i:
                    cluster_mat[k][i] = float(size_*size_)/((size_-1)*(size_-1))
            targets_.append(targets[split_i[0]])
            input_ = torch.cat([inputs[i].resize(1, num_dim) for i in split_i], 0)
            centers.append(torch.mean(input_, 0))
            inputs_list.append(input_)

        cluster_mat = Variable(torch.FloatTensor(cluster_mat)).cuda().detach()

        # print(cluster_mat.requires_grad)

        targets_ = torch.cat(targets_)

        centers = [center.resize(1, num_dim) for center in centers]
        centers = torch.cat(centers, 0)
        # norm = centers.norm(dim=1, p=2, keepdim=True)
        # print(norm)
        # centers = centers.div(norm.expand_as(centers))
        # norm = centers.norm(dim=1, p=2, keepdim=True)
        # print(norm)

        # no gradient on centers
        # centers = centers.detach()

        centers_dist = pair_euclidean_dist(inputs, centers)*cluster_mat

        loss = []
        dist_ap = []
        dist_an = []
        num_match = 0
        for i, target in enumerate(targets):
            # for computation stability
            dist = centers_dist[i]
            pos_pair_mask = (targets_ == target)
            pos_pair = torch.masked_select(dist, pos_pair_mask)

            dist = torch.masked_select(dist, dist > 1e-3)
            pos_pair = torch.masked_select(pos_pair, pos_pair > 1e-3)
            dist_ap.append(torch.mean(pos_pair))
            dist_an.append(torch.mean(dist))

            base = (torch.max(dist) + torch.min(dist)).data[0]/2
            pos_exp = torch.sum(torch.exp(-self.alpha*(pos_pair - base)))
            a_exp = torch.sum(torch.exp(-self.alpha*(dist - base)))
            loss_ = - torch.log(pos_exp/a_exp)
            loss.append(loss_)
            if loss_.data[0] < 0.3:
                num_match += 1
        loss = torch.mean(torch.cat(loss))
        # print(dist_an, dist_ap)
        dist_an = torch.mean(torch.cat(dist_an)).data[0]
        dist_ap = torch.mean(torch.cat(dist_ap)).data[0]

        accuracy = float(num_match)/len(targets)
        return loss, accuracy, dist_ap, dist_an


def main():
    features = np.load('0_feat.npy')
    labels = np.load('0_label.npy')

    num_instances = 32
    batch_size = 128
    Batch = BatchGenerator(labels, num_instances=num_instances, batch_size=batch_size)
    batch = Batch.batch()

    inputs = Variable(torch.FloatTensor(features[batch, :])).cuda()
    targets = Variable(torch.LongTensor(labels[batch])).cuda()
    print(KmeanLoss(n_cluster=32)(inputs, targets))

if __name__ == '__main__':
    main()
    print('Congratulations to you!')
