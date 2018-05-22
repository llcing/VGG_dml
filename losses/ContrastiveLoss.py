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


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        # self.alpha = alpha

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

        for i, pos_pair in enumerate(pos_sim):

            # pos_pair = torch.sort(pos_pair)[0]
            neg_pair = torch.sort(neg_sim[i])[0]

            pos_pair = torch.masked_select(neg_pair, neg_pair < self.margin + 0.05)
            neg_pair = torch.masked_select(neg_pair, neg_pair > self.margin - 0.05)

            # pos_pair = pos_pair[1:]
            if len(neg_pair) < 2:
                c += 1
                continue

            # neg_pair = torch.sort(neg_pair)[0]

            # if i == 1 and np.random.randint(99) == 1:
            #     print('neg_pair is ---------', neg_pair)
            #     print('pos_pair is ---------', pos_pair)
            if len(pos_pair) == 0:
                pos_loss = 0
            else:
                pos_loss = torch.sum(self.margin + 0.05 - pos_pair)

            if len(neg_pair) == 0:
                neg_loss = 0
            else:
                neg_loss = torch.sum(neg_pair - self.margin + 0.05)
            # print(pos_loss)
            # print(neg_loss)

            loss = loss + (pos_loss + neg_loss)

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

    print(ContrastiveLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


