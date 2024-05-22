# Author: Jie Chang <changjie@megvii.com>
# pylint: disable=not-callable, import-error,no-name-in-module
"""Functions for Pytorch"""

import torch
def Uncertainty_Regression_Loss(mu, logvar, labels, mean):
    means = mean[labels]
    sigama_sq = torch.exp(logvar)
    #print(mu)
    #print(means)
    #loss_v1 = (((mu - means) * (mu - means)) / (1e-10 + sigama_sq)) + sigama_sq
    loss_v1 = 0.5 * (((mu - means) * (mu - means)) / (1e-10 + sigama_sq)) + 0.5 * torch.log(sigama_sq)
    #dis = torch.sum(((mu - means) * (mu - means)),dim=1)
    #print(sigama_sq.mean().item())
    #print(dis.mean().item(),"++++++++++++++++++++++")
    loss_v2 = torch.mean(loss_v1, 1)
    loss = torch.mean(loss_v2)
    return loss
