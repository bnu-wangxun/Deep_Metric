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


class SemiHardLoss(nn.Module):
    def __init__(self, alpha=0, beta=None, margin=0, **kwargs):
        super(SemiHardLoss, self).__init__()
        self.beta = beta
        self.margin = margin
        self.alpha = alpha

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute similarity matrixr®
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
        loss = list()
        c = 0
        base = 0.5
        for i, pos_pair_ in enumerate(pos_sim):
            # print(i)
            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_sim[i])[0]
        
            neg_pair = torch.masked_select(neg_pair_, neg_pair_ < pos_pair_[-1])
            pos_pair = torch.masked_select(pos_pair_, pos_pair_ > neg_pair_[0])


                pos_loss =   2.0/self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base))))      
            else:
                pos_loss = 0*torch.mean(1 - pos_pair_)
            
            if len(neg_pair)>0:
                # neg_loss = torch.mean(neg_pair)
                neg_loss = 2.0/self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base))))
            else:
                neg_loss = 0*torch.mean(neg_pair_)
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

    print(SemiHardLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


