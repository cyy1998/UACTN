import torch
from torch.nn.functional import cross_entropy
import torch.nn.functional as F


class FocalAMSoftMaxLoss:
    r"""Additive Margin Softmax loss as described in the paper `"Additive Margin Softmax for Face Verification"
    <https://arxiv.org/abs/1801.05599>`_. AM Softmax Loss is identical to `CosFace <https://arxiv.org/abs/1801.09414>`_.

    .. math::

        \frac{1}{N}\sum_i-log(\frac{e^{s\cdot (\cos\theta_{y_i,i}-m)}}{e^{s\cdot (\cos\theta_{y_i,i}-m)}+
        \sum_{k\neq y_i}e^{s\cdot \cos\theta_{k,i}}})

    where :math:`s` is the scale factor, :math:`m` denotes margin and :math:`i` indexes the :math:`i`-th sample.

    .. note::
        We do not do embedding normalization, weight normalization or linear transform in this function. This function is
        a post-process. Please use :meth:`~megrec.torch_model.modules.TransformEmbeddingToLogit` function first to convert
        embeddings into logits.

    Args:
        scale (float): Specifies the scale factor. Default: 64.0
        margin (float): Margin value. Default: 0.35
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    """

    def __init__(self, scale: float = 15.0, margin: float = 0.35, reduction: str = "mean",gamma: float=2.0) -> None:
        self.scale = scale
        self.margin = margin
        self.gamma=gamma
        self.reduction = reduction

    def __call__(self, logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        """call function as forward

        Args:
            logits (torch.Tensor): The predicted logits before softmax with shape of :math:`(N, C)`
            targets (torch.LongTensor): The ground-truth label long vector with shape of :math:`(N,)`

        Returns:
            torch.Tensor: loss
                the computed loss
        """

        one_hot_labels = torch.zeros_like(logits).scatter_(dim=1, index=targets.unsqueeze(1), value=1.0)
        am_logits = self.scale * (logits - one_hot_labels * self.margin)
        preds_softmax = F.softmax(am_logits, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, targets.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, targets.view(-1, 1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        if self.reduction=='mean':
            loss=loss.mean()
        else:
            loss=loss.sum()

        return loss