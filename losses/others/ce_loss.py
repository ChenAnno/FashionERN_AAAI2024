import torch
import torch.nn as nn


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        total_loss = self.loss(input, target)
        return total_loss


def accuracy(output, target, topk=(1,)):
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
