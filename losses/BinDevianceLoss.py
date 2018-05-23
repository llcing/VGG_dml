from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

"""
Baseline loss function in BIER

Deep Metric Learning with BIER: Boosting Independent Embeddings Robustly
"""


def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


class BinDevianceLoss(nn.Module):
    def __init__(self, alpha=20, margin=0.5):
        super(BinDevianceLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, inputs, targets):
        inputs = normalize(inputs)
        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = similarity(inputs)
        # print(sim_mat)
        targets = targets.cuda()
        # split the positive and negative pairs
        eyes_ = Variable(torch.eye(n, n)).cuda()
        # eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        num_instances = len(pos_sim)//n + 1
        num_neg_instances = n - num_instances

        pos_sim = pos_sim.resize(len(pos_sim)//(num_instances-1), num_instances-1)
        neg_sim = neg_sim.resize(
            len(neg_sim) // num_neg_instances, num_neg_instances)

        #  clear way to compute the loss first
        loss = 0
        c = 0

        base = np.max([torch.max(sim_mat) - 0.1, self.margin + 0.2])

        # print(40*'#')
        for i, pos_pair in enumerate(pos_sim):
            # print(i)
            pos_pair_ = torch.sort(pos_pair)[0]
            neg_pair_ = torch.sort(neg_sim[i])[0]
            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > pos_pair_[0] - 0.05)
            pos_pair = torch.masked_select(pos_pair_, pos_pair_ < base)

            # for train stability
            if len(neg_pair) < 1:
                c += 1
                neg_pair = neg_pair_[-1]

            if len(pos_pair) < 1:
                pos_pair = pos_pair_[0]

            pos_loss = torch.mean(torch.log(1 + torch.exp(-2*(pos_pair - self.margin))))
            neg_loss = (float(2)/self.alpha) * torch.mean(torch.log(1 + torch.exp(self.alpha*(neg_pair - self.margin))))
            loss_ = pos_loss + neg_loss
            loss = loss + loss_

        loss = loss/n
        prec = float(c)/n
        neg_d = torch.mean(neg_sim).item()
        pos_d = torch.mean(pos_sim).item()

        return loss, prec, pos_d, neg_d


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w).cuda()
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_)).cuda()

    print(BinDevianceLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


