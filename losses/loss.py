import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchBasedClassificationLoss(nn.Module):
    def __init__(self):
        super(BatchBasedClassificationLoss, self).__init__()

    def forward(self, predicted_features, tar_features):
        prediction = 100 * predicted_features @ tar_features.T
        labels = torch.arange(0, predicted_features.size(0)).long().to(predicted_features.device)

        return F.cross_entropy(prediction, labels)
