import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class BatchBasedClassificationLoss(nn.Module):
    def __init__(self):
        super(BatchBasedClassificationLoss, self).__init__()

    def forward(self, predicted_features, tar_features):
        prediction = 100 * predicted_features @ tar_features.T
        # prediction = self.cos_sim(predicted_features, tar_features)
        labels = torch.arange(0, predicted_features.size(0)).long().to(predicted_features.device)

        return F.cross_entropy(prediction, labels)


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, a, b):
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        return cos(a, b)
