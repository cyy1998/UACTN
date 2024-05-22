import torch
from torch.nn.functional import cross_entropy


class CenterLoss:

    def __init__(self,num_classes,feat_dim) -> None:
        self.num_classes = num_classes
        self.feat_dim = feat_dim


    def __call__(self, features: torch.Tensor, weights: torch.LongTensor,targets: torch.LongTensor) -> torch.Tensor:
        weights=weights.t()
        batch_size = features.size(0)
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(weights, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(features, weights.t(),beta=1,alpha=-2)

        classes = torch.arange(self.num_classes).long().cuda()
        targetss = targets.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = targetss.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss