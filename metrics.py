import numpy as np
import torch
from torch.nn import BCELoss


class IoU:
    def __init__(self, thresh=0.5):
        self.threshold = thresh

    def __call__(self, outputs, targets):
        outputs = outputs.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
        outputs[outputs >= self.threshold] = 1
        outputs[outputs <= self.threshold] = 0

        intersection = (targets * outputs).sum()
        union = targets.sum() + outputs.sum()
        return (2 * intersection + 1) / (union + 1)


class DICELoss:
    eps = 1e-7

    def __init__(self, size_average):
        self.size_average = size_average

    def __call__(self, outputs, targets):
        batch_size = outputs.size(0)
        outputs = outputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        nominator = (outputs * targets).sum(1)
        denominator = outputs.sum(1) + targets.sum(1)

        if self.size_average:
            return torch.clamp(2 * ((nominator + DICELoss.eps) / (denominator + DICELoss.eps)).sum() / batch_size, 0, 1)
        return 2 * (nominator + DICELoss.eps) / (denominator + DICELoss.eps)


class BCEDICELoss:
    def __init__(self, loss_weights, size_average=True):
        self.bce_loss = BCELoss(reduction='elementwise_mean')
        self.dice_loss = DICELoss(size_average=size_average)
        self.loss_weights = loss_weights

    def __call__(self, outputs, targets):
        return self.loss_weights['bce'] * self.bce_loss(outputs, targets) \
               + self.loss_weights['dice'] * (1 - self.dice_loss(outputs, targets))
