import torch
import torch.nn as nn
import numpy as np


class TextRelevanceLoss(nn.Module):
    def __init__(self, batch_size, emb_dim=512):
        super(TextRelevanceLoss, self).__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        self.loss_1 = nn.CrossEntropyLoss()
        self.loss_2 = nn.CrossEntropyLoss()
        self.ground_truth = torch.arange(batch_size).cuda()

        self.K = int(batch_size * 10)
        self.query_feats_bank = torch.zeros(self.K, emb_dim).cuda()
        self.doc_feats_bank = torch.zeros(self.K, emb_dim).cuda()
        self.ptr = 0
        self.is_full = False

    def enqueue_dequeue(self, query_feat, doc_feat):
        q_size = query_feat.shape[0]
        if self.ptr + q_size > self.K:
            self.ptr = q_size
            self.is_full = True

        tmp_query = self.query_feats_bank[0: q_size]
        tmp_doc = self.doc_feats_bank[0: q_size]
        self.query_feats_bank[self.ptr: self.ptr + q_size] = tmp_query
        self.doc_feats_bank[self.ptr: self.ptr + q_size] = tmp_doc
        self.query_feats_bank[0: q_size] = query_feat
        self.doc_feats_bank[0: q_size] = doc_feat
        self.ptr += q_size

    def get(self):
        if self.is_full:
            return self.query_feats_bank, self.doc_feats_bank
        else:
            return self.query_feats_bank[:self.ptr], self.doc_feats_bank[:self.ptr]

    def forward(self, image_emb, text_emb):
        logit_scale = self.logit_scale
        logits_per_image = logit_scale * image_emb @ text_emb.t()
        logits_per_text = logit_scale * text_emb @ image_emb.t()
        loss_img = self.loss_img(logits_per_image, self.ground_truth)
        loss_txt = self.loss_txt(logits_per_text, self.ground_truth)

        self.enqueue_dequeue(image_emb.detach(), text_emb.detach())
        query_bank, doc_bank = self.get()

        logits_1 = logit_scale * image_emb @ doc_bank.t()  # N * K
        loss_1 = self.loss_1(logits_1, self.ground_truth)
        logits_2 = logit_scale * text_emb @ query_bank.t()
        loss_2 = self.loss_2(logits_2, self.ground_truth)
        total_loss = (loss_img + loss_txt + loss_1 + loss_2) / 4

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
