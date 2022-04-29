from abc import abstractmethod

import torch
from torch import nn


class _MeanMetric(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.classes = classes
        self._corrects = torch.tensor([0 for _ in range(self.classes)])
        self._totals = torch.tensor([0 for _ in range(self.classes)])

    def forward(self, epoch: bool = False, *args, **kwargs):
        return self.get_epoch_metric() if epoch else self.get_batch_metric(*args, **kwargs)

    def get_epoch_metric(self):
        mean = self._corrects / self._totals
        self._corrects *= 0
        self._totals *= 0
        return mean

    @abstractmethod
    def get_batch_metric(self, predictions, targets):
        raise NotImplementedError()


class MeanPixelAccuracy(_MeanMetric):

    def get_batch_metric(self, predictions, targets):
        correct = torch.sum(predictions * targets, dim=(0, 2, 3))
        total = targets.sum(dim=(0, 2, 3))
        self._corrects += correct.to(self._corrects.dtype)
        self._totals += total
        return correct / total


class MeanIoU(_MeanMetric):

    def get_batch_metric(self, predictions, targets):
        intersect = torch.mul(predictions, targets).sum(dim=(0, 2, 3))
        union = torch.maximum(predictions, targets).sum(dim=(0, 2, 3))

        self._corrects += intersect.to(self._corrects.dtype)
        self._totals += union.to(self._totals.dtype)

        return intersect / union
