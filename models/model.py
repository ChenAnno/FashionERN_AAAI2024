from torch import nn
import torch.nn.functional as F

from models.fusion_model import CombinerSimple, VisualSR, DVR_module
from models.clip_model import ImageCLIP, TextCLIP


class ERN(nn.Module):
    def __init__(self, clip_model, feature_dim, device):
        super(ERN, self).__init__()
        self.image_clip = ImageCLIP(clip_model)
        self.text_clip = TextCLIP(clip_model)

        self.DVR = DVR_module(feature_dim=feature_dim, device=device)
        self.SR_module = VisualSR(embed_dim=feature_dim)
        self.Combiner_module = CombinerSimple(feature_dim, feature_dim * 4, feature_dim * 8)

    def forward(
        self,
        image=None,
        text=None,
        ref_feats=None,
        ref_local_feats=None,
        text_feats=None,
        text_seq_feats=None,
        tar_feats=None,
        tar_local_feats=None,
        mode="train",
    ):
        """
        For query side = TME (CLIP enhanced) + DVR
        For target side = SR + Combiner
        :param image: visual embeddings
        :param text: text embeddings
        :param ref_feats: [b, dim]
        :param ref_local_feats: [b, patch_num, dim]
        :param text_feats: [b, dim]
        :param text_seq_feats: [b ,seq_num, dim]
        :param tar_feats
        :param tar_local_feats
        :param mode:
        :return: judged by the mode
        """
        if mode == "image":
            return self.image_clip(image)
        elif mode == "text_global":
            return self.text_clip(text, mode="global", visual_emb=ref_local_feats)[0]  # [b, dim]
        elif mode == "text_seq":
            return self.text_clip(text, mode="seq", visual_emb=ref_local_feats)  # [b, seq_num, dim]
        elif mode == "local":
            return self.SR_module(ref_local_feats)  # [b, dim]
        elif mode == "index":
            return F.normalize(self.Combiner_module(tar_feats, tar_local_feats), dim=-1)
        elif mode == "test":
            fusion_feat = self.DVR(ref_local_feats, text_seq_feats, ref_feats, text_feats)
            return fusion_feat
        else:  # train
            fusion_feat = self.DVR(ref_local_feats, text_seq_feats, ref_feats, text_feats)
            tar_local_attn = self.SR_module(tar_local_feats)
            tar_local_attn = self.Combiner_module(tar_feats, tar_local_attn)
            tar_local_attn = F.normalize(tar_local_attn, dim=-1)
            return fusion_feat, tar_local_attn


if __name__ == '__main__':
    pass
