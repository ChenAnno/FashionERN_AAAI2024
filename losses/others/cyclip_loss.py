import torch
import torch.nn as nn
import numpy as np

class CyCLIPLoss(nn.Module):

    def __init__(self, batch_size):
        super(CyCLIPLoss, self).__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.batch_size = batch_size
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        self.ground_truth = torch.arange(batch_size).cuda()

    def forward(self, image_emb, text_emb):
        logit_scale = self.logit_scale
        logits_image_per_text = logit_scale * image_emb @ text_emb.t()
        logits_text_per_image = logit_scale * text_emb @ image_emb.t()
        logits_image_per_image = logit_scale * image_emb @ image_emb.t()
        logits_text_per_text = logit_scale * text_emb @ text_emb.t()

        # contrastive_loss
        loss_img = self.loss_img(logits_image_per_text, self.ground_truth)
        loss_txt = self.loss_txt(logits_text_per_image, self.ground_truth)
        contrastive_loss = (loss_img + loss_txt) / 2.0
        # crossmodal_cyclic_loss
        crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / (
                    logit_scale * logit_scale) * self.batch_size
        # inmodal_cyclic_loss
        inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (
                    logit_scale * logit_scale) * self.batch_size
        # total_loss
        cylambda1 = cylambda2 = 0.25
        cyclic_loss = cylambda1 * inmodal_cyclic_loss + cylambda2 * crossmodal_cyclic_loss
        # print("contrastive_loss: {}, cyclic_loss: {}".format(contrastive_loss, cyclic_loss))
        total_loss = contrastive_loss + cyclic_loss

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
