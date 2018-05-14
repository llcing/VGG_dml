from __future__ import print_function, absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from utils import BatchGenerator, cluster_


def pair_euclidean_dist(inputs_x, inputs_y):
    n = inputs_x.size(0)
    m = inputs_y.size(0)
    xx = torch.pow(inputs_x, 2).sum(dim=1, keepdim=True).expand(n, m)
    yy = torch.pow(inputs_y, 2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy
    dist.addmm_(1, -2, inputs_x, inputs_y.t())
    # dist = dist.clamp(min=1e-12).sqrt()
    return dist


class MCALoss(nn.Module):
    def __init__(self, alpha=16, centers=None, cluster_counter=None, center_labels=None):
        super(MCALoss, self).__init__()
        self.alpha = alpha
        self.centers = centers
        self.center_labels = center_labels
        self.cluster_counter = cluster_counter

    def forward(self, inputs, targets, _mask):
        centers_dist = pair_euclidean_dist(inputs, self.centers)
        loss = []
        dist_ap = []
        # dist_an = []
        num_match = 0
        for i, target in enumerate(targets):
            # for computation stability
            dist = centers_dist[i]

            pos_pair_mask = (self.center_labels == target)
            # print(pos_pair_mask[:7])
            neg_pair_mask = (self.center_labels != target)

            pos_pair = torch.masked_select(dist, pos_pair_mask)

            # count the closest cluster
            pos_idx = torch.sort(pos_pair)[1][0].data[0]
            self.cluster_counter[target.data[0]][pos_idx] += 1

            # delete the dead cluster
            pos_pair = torch.masked_select(pos_pair, _mask[target.data[0]])
            neg_pair = torch.sort(torch.masked_select(dist, neg_pair_mask))

            # only consider neighbor negative clusters
            neg_pair = neg_pair[0][:32]

            # if i == 1:
            #     print(neg_pair)

            dist_ap.extend(pos_pair)

            base = (torch.max(neg_pair) + torch.min(dist)).data[0]/2
            pos_exp = torch.sum(torch.exp(-self.alpha*(pos_pair - base)))
            neg_exp = torch.sum(torch.exp(-self.alpha*(neg_pair - base)))
            loss_ = - torch.log(pos_exp/(pos_exp + neg_exp))
            loss.append(loss_)
            if loss_.data[0] < 0.32:
                num_match += 1
        loss = torch.mean(torch.cat(loss))
        # print(dist_an, dist_ap)
        dist_an = torch.mean(centers_dist).data[0]
        dist_ap = torch.mean(torch.cat(dist_ap)).data[0]

        accuracy = float(num_match)/len(targets)
        return loss, accuracy, dist_ap, dist_an


def main():
    features = np.load('0_feat.npy')
    labels = np.load('0_label.npy')

    centers, center_labels = cluster_(features, labels, n_clusters=3)
    centers = Variable(torch.FloatTensor(centers).cuda(),  requires_grad=True)
    center_labels = Variable(torch.LongTensor(center_labels)).cuda()
    cluster_counter = np.zeros([100, 3])
    num_instances = 3
    batch_size = 120
    Batch = BatchGenerator(labels, num_instances=num_instances, batch_size=batch_size)
    batch = Batch.batch()

    # _mask = Variable(torch.ByteTensor(np.ones([num_class_dict[args.data], args.n_cluster])).cuda())
    _mask = Variable(torch.ByteTensor(np.ones([100, 3])).cuda())
    inputs = Variable(torch.FloatTensor(features[batch, :])).cuda()
    targets = Variable(torch.LongTensor(labels[batch])).cuda()
    # print(torch.mean(inputs))
    mca = MCALoss(alpha=16, centers=centers,
                  center_labels=center_labels, cluster_counter=cluster_counter)
    for i in range(2):
        # loss, accuracy, dist_ap, dist_an =
            # MCALoss(alpha=16, centers=centers, center_labels=center_labels)(inputs, targets)
        loss, accuracy, dist_ap, dist_an = \
            mca(inputs, targets, _mask)
        # print(loss.data[0])
        loss.backward()
        # print(centers.grad.data)
        centers.data -= centers.grad.data
        centers.grad.data.zero_()
        # print(centers.grad)

    print(cluster_counter)


if __name__ == '__main__':
    main()
    print('Congratulations to you!')

