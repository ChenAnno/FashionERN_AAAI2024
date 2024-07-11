import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionPooling(nn.Module):
    def __init__(self, emb_dim, emb_num):
        super(AttentionPooling, self).__init__()
        self.emb_dim = emb_dim
        self.emb_num = emb_num
        self.projection = nn.Linear(emb_dim * emb_num, emb_num)

    def forward(self, inputs):
        # (B, T, H) -> (B, T)
        energy = self.projection(inputs.view(inputs.shape[0], -1))
        weights = F.softmax(energy, dim=1)
        # weights[:, 0] = 0.55
        # weights[:, 1] = 0.45
        # print(weights)
        # (B, T, H) * (B, T, 1) -> (B, H)
        outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1)

        return outputs
