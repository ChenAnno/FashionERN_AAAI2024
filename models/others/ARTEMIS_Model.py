import torch
from torch import nn
import torch.nn.functional as F
from utils import l2norm

class Artemis(nn.Module):
    # ICLR 2023
    def __init__(self, clip_feature_dim) -> None:
        super().__init__()
        self.embed_dim = clip_feature_dim
        self.Transform_m = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), L2Module())
        self.Attention_EM = AttentionMechanism(self.embed_dim)
        self.Attention_IS = AttentionMechanism(self.embed_dim)
        self.temperature = nn.Parameter(torch.FloatTensor((2.65926,)))

    def apply_attention(self, a, x):
        return l2norm(a * x)
	
    def compute_score_artemis(self, r, m, t, store_intermediary=False):
        EM = self.compute_score_EM(r, m, t, store_intermediary)
        IS = self.compute_score_IS(r, m, t, store_intermediary)
        if store_intermediary:
            self.hold_results["EM"] = EM
            self.hold_results["IS"] = IS
        return EM + IS

    def compute_score_broadcast_artemis(self, r, m, t):
        return self.compute_score_broadcast_EM(r, m, t) + self.compute_score_broadcast_IS(r, m, t)

    def compute_score_EM(self, r, m, t, store_intermediary=False):
        Tr_m = self.Transform_m(m)
        A_EM_t = self.apply_attention(self.Attention_EM(m), t)
        if store_intermediary:
            self.hold_results["Tr_m"] = Tr_m
            self.hold_results["A_EM_t"] = A_EM_t
        return (Tr_m * A_EM_t).sum(-1)

    def compute_score_broadcast_EM(self, r, m, t):
        batch_size = r.size(0)
        A_EM = self.Attention_EM(m) # shape (Bq, d)
        Tr_m = self.Transform_m(m) # shape (Bq, d)
		# apply each query attention mechanism to all targets
        A_EM_all_t = self.apply_attention(A_EM.view(batch_size, 1, self.embed_dim), t.view(1, batch_size, self.embed_dim)) # shape (Bq, Bt, d)
        EM_score = (Tr_m.view(batch_size, 1, self.embed_dim) * A_EM_all_t).sum(-1) # shape (Bq, Bt) ; coefficient (i,j) is the IS score between query i and target j
        return EM_score

    def compute_score_IS(self, r, m, t, store_intermediary=False):
        A_IS_r = self.apply_attention(self.Attention_IS(m), r)
        A_IS_t = self.apply_attention(self.Attention_IS(m), t)
        if store_intermediary:
            self.hold_results["A_IS_r"] = A_IS_r
            self.hold_results["A_IS_t"] = A_IS_t
        return (A_IS_r * A_IS_t).sum(-1)

    def compute_score_broadcast_IS(self, r, m, t):
        batch_size = r.size(0)
        A_IS = self.Attention_IS(m) # shape (Bq, d)
        A_IS_r = self.apply_attention(A_IS, r) # shape (Bq, d)
		# apply each query attention mechanism to all targets
        A_IS_all_t = self.apply_attention(A_IS.view(batch_size, 1, self.embed_dim), t.view(1, batch_size, self.embed_dim)) # shape (Bq, Bt, d)
        IS_score = (A_IS_r.view(batch_size, 1, self.embed_dim) * A_IS_all_t).sum(-1) # shape (Bq, Bt) ; coefficient (i,j) is the IS score between query i and target j
        return IS_score


class L2Module(nn.Module):

	def __init__(self):
		super(L2Module, self).__init__()

	def forward(self, x):
		x = l2norm(x)
		return x

class AttentionMechanism(nn.Module):
	"""
	Module defining the architecture of the attention mechanisms in ARTEMIS.
	"""

	def __init__(self, embed_dim):
		super(AttentionMechanism, self).__init__()

		self.embed_dim = embed_dim
		input_dim = self.embed_dim

		self.attention = nn.Sequential(
			nn.Linear(input_dim, self.embed_dim),
			nn.ReLU(),
			nn.Linear(self.embed_dim, self.embed_dim),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		return self.attention(x)