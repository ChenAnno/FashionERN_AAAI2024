from torch import nn

from models.fusion_model import CombinerSimple, VisualSR, DVR_module
from models.clip_model import ImageCLIP, TextCLIP


class ERN(nn.Module):
    def __init__(self, clip_model, feature_dim, device):
        super(ERN, self).__init__()

        # TME (CLIP-based) model
        self.image_clip = ImageCLIP(clip_model)
        self.text_clip = TextCLIP(clip_model)

        # DVR Model
        self.DVR = DVR_module(feature_dim=feature_dim, device=device)

        # Target-side Model
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
        For query side = TME (CLIP enhanced) + DVR (MR + SR)
        For target side = SR + Combiner
        :param image: images from dataloader
        :param text: text after tokenizer
        :param ref_feats: [b, dim]
        :param ref_local_feats: [b, patch_num, dim]
        :param text_feats: [b, dim]
        :param text_seq_feats: [b ,seq_num, dim]
        :param tar_feats: [b, dim]
        :param tar_local_feats: [b ,seq_num, dim]
        :param mode:
            1. "image": extract visual global embedding
            2. "text_global": extract text global embedding
            3. "text_seq": extract text sequence embedding
            4. "index": extract gallary-side embedding
            5. "test": extract query-side embedding in the test phase
            6. "train" or else: extract query-side embedding during training

        :return: judged by the mode
        """
        if mode == "image":
            return self.image_clip(image)  # [b, dim]
        
        elif mode == "text_global":
            return self.text_clip(text, mode="global", visual_emb=ref_local_feats)[0]  # [b, dim]
        
        elif mode == "text_seq":
            return self.text_clip(text, mode="seq", visual_emb=ref_local_feats)  # [b, seq_num, dim]
        
        elif mode == "index":
            tar_local_center_feats = self.SR_module(tar_local_feats)  # [b, dim]
            return self.Combiner_module(tar_feats, tar_local_center_feats)  # [b, dim]
        
        elif mode == "test":
            return self.DVR(ref_local_feats, text_seq_feats, ref_feats, text_feats)  # [b, dim]
        
        else:  # train
            fusion_feat = self.DVR(ref_local_feats, text_seq_feats, ref_feats, text_feats)  # [b, dim]
            tar_feat_attn = self.SR_module(tar_local_feats)
            tar_feat_attn = self.Combiner_module(tar_feats, tar_feat_attn)  # [b, dim]
            return fusion_feat, tar_feat_attn
