from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

"""

Decorrelation Loss 

To decorrelate the relation between vector
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


class DecorLoss(nn.Module):
    def __init__(self, margin=0.6):
        super(DecorLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs):
        # normalize to input
        inputs = normalize(inputs)
        # normalize for features
        inputs = normalize(inputs.t())

        sim_mat = torch.abs(similarity(inputs))
        if np.random.randint(4) == 1:
            sim_big = torch.masked_select(sim_mat, sim_mat > self.margin + 0.05)
            sim_big = torch.masked_select(sim_big, sim_big < 0.99)
            print(40*'#', sim_big)
            print(40*'#', len(sim_big))
        loss = torch.mean(torch.pow(torch.clamp(sim_mat - self.margin, min=0), 2))

        return loss


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

    print(DecorLoss()(inputs))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


