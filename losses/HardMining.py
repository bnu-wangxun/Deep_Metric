from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


class HardMiningLoss(nn.Module):
    def __init__(self, beta=None, margin=0, **kwargs):
        super(HardMiningLoss, self).__init__()
        self.beta = beta
        self.margin = 0.1

    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets

        base = 0.5
        loss = list()
        c = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets==targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets!=targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > pos_pair_[0] - self.margin)
            pos_pair = torch.masked_select(pos_pair_, pos_pair_ < neg_pair_[-1] + self.margin)
            # pos_pair = pos_pair[1:]
            if len(neg_pair) < 1:
                c += 1
                continue

            pos_loss = torch.mean(1 - pos_pair)
            neg_loss = torch.mean(neg_pair)
            # pos_loss = torch.mean(torch.log(1 + torch.exp(-2*(pos_pair - self.margin))))
            # neg_loss = 0.04*torch.mean(torch.log(1 + torch.exp(50*(neg_pair - self.margin))))
            loss.append(pos_loss + neg_loss)

        loss = sum(loss)/n
        prec = float(c)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return loss, prec, mean_pos_sim, mean_neg_sim


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(HardMiningLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


