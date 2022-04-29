import torch
import cv2
from torch import nn, Tensor
from torch.nn import functional as F


class SegmentationCrossEntropy(nn.CrossEntropyLoss):

    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        multiple_class_idx = ~target.reshape(*target.shape[:2], -1).all(-1).any(1)

        return F.cross_entropy(input_[multiple_class_idx], target[multiple_class_idx],
                               weight=self.weight, ignore_index=self.ignore_index,
                               reduction=self.reduction, label_smoothing=self.label_smoothing)


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        # ToDo
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)

        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1. - dsc
