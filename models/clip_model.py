import torch
from torch import nn


class ImageCLIP(nn.Module):
    def __init__(self, clip_model):
        super(ImageCLIP, self).__init__()
        self.clip_model = clip_model

    def forward(self, images, mode="eval"):
        self.clip_model.eval()
        if mode == "train":
            return self.clip_model.encode_image(images)
        with torch.no_grad():
            return self.clip_model.encode_image(images)


class TextCLIP(nn.Module):
    def __init__(self, clip_model):
        super(TextCLIP, self).__init__()
        self.clip_model = clip_model

    def forward(self, text, mode="global", visual_emb=None):
        self.clip_model.eval()
        with torch.no_grad():
            if mode == "seq":
                return self.clip_model.encode_text(text, mode="seq", visual_emb=visual_emb)
            elif mode == "global":
                return self.clip_model.encode_text(text, visual_emb=visual_emb)
            else:
                return self.clip_model.encode_text(text, visual_emb=visual_emb)  # ([batch, 512], [batch, 77, 512])

