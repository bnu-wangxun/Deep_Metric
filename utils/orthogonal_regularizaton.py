from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.autograd import Variable


def orth_reg(net, loss, cof=1):
    orth_loss = 0

    if cof == 0:
        return orth_loss

    for m in net.modules():
        if isinstance(m, nn.Linear):
            w = m.weight
            mat_ = torch.matmul(w, w.t())
            diff = mat_ - torch.diag(torch.diag(mat_))
            orth_loss = torch.mean(torch.pow(diff, 2))
            loss = loss + cof*orth_loss
    return loss
