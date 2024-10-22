import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from transformers import BertConfig, BertModel


class DVR_module(nn.Module):
    def __init__(self, feature_dim=640, device=None):
        super(DVR_module, self).__init__()
        self.device = device

        # Initialize transformer models
        self.transformer_layer = PlusModel(feature_dim=feature_dim, device=device, layers=2).to(device)

        # Initialize sub-modules
        self.SR_module = VisualSR(embed_dim=feature_dim)
        self.MR_component = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, dropout=0.1, batch_first=True
        )
         # Initialize Combine Modules
        self.combiner_global = CombinerSimple(feature_dim, feature_dim * 4, feature_dim * 8)
        self.combiner_local = CombinerSimple(feature_dim, feature_dim * 4, feature_dim * 8)
        self.combiner = CombinerSimple(feature_dim, feature_dim * 4, feature_dim * 8)

    def forward(self, ref_patch_features, text_seq_features, ref_global_feats, text_global_feats):
        """
        Fuse multiple features
        :param text_global_feats: [b, dim]
        :param ref_global_feats: [b, dim]
        :param ref_patch_features: [b, 13, dim]
        :param text_seq_features: [b, 77, dim]
        :return: fusion feats: [b, dim]
        """
        _, last_hidden_state, _ = self.transformer_layer(ref_patch_features, text_seq_features)
        patch_num = ref_patch_features.shape[1]

        image_feats = last_hidden_state[:, 1 : patch_num + 1, :]
        text_feats = last_hidden_state[:, patch_num + 1 :, :]
        image_feats_norm = F.normalize(image_feats, dim=2)
        text_feats_norm = F.normalize(text_feats, dim=2)

        # Cross attention between normalized text and image features
        cross_vision_feats, _ = self.MR_component(
            query=text_feats_norm, key=image_feats_norm, value=image_feats_norm
        )
        cross_vision_feats = cross_vision_feats[:, :patch_num, :]
        patch_vision_mean = self.SR_module(cross_vision_feats)
        seq_text_mean = torch.mean(text_feats_norm, dim=1)  # [b, feature_dim]

        # Combine global and local features
        global_feats = self.combiner_global(ref_global_feats, text_global_feats)
        local_feats = self.combiner_local(patch_vision_mean, seq_text_mean)
        fusion_feature = self.combiner(global_feats, local_feats)
        return fusion_feature


class CombinerSimple(nn.Module):
    """
    Combiner module that fuses textual and visual information.
    """

    def __init__(
        self, clip_feature_dim=512, projection_dim=512 * 4, hidden_dim=512 * 8
    ):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: Projection dimension
        :param hidden_dim: Hidden dimension
        """
        super(CombinerSimple, self).__init__()

        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.text_projection_layer = self._create_projection_layer(clip_feature_dim, projection_dim)
        self.image_projection_layer = self._create_projection_layer(clip_feature_dim, projection_dim)

    def _create_projection_layer(self, input_dim, output_dim):
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):
        text_projected_features = self.text_projection_layer(text_features)
        image_projected_features = self.image_projection_layer(image_features)

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), dim=-1)
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)

        output = dynamic_scalar * text_features + (1 - dynamic_scalar) * image_features
        return F.normalize(output, dim=-1)


class VisualSR(nn.Module):
    """
    Build global representations within patch embeds.
    Args:
        - local: patch embeddings, shape: (batch_size, patch_num, 640)
    Returns:
        - new_global: final represention, shape: (batch_size, 640).
    """

    def __init__(self, embed_dim=512, dropout_rate=0.5, num_region=13):
        super(VisualSR, self).__init__()
        self.embedding_local = self._create_embedding_layer(
            embed_dim, num_region, dropout_rate
        )
        self.embedding_global = self._create_embedding_layer(
            embed_dim, embed_dim, dropout_rate
        )
        self.embedding_common = nn.Linear(embed_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def _create_embedding_layer(self, embed_dim, num_features, dropout_rate):
        return nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(num_features),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(module.in_features + module.out_features)
                nn.init.uniform_(module.weight, -r, r)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def l2norm(self, X, dim=-1, eps=1e-8):
        """L2-normalize columns of X"""
        norm = torch.sqrt(torch.sum(X**2, dim=dim, keepdim=True)) + eps
        return X / norm

    def forward(self, local_feature):
        raw_global = torch.mean(local_feature, dim=1)

        # Compute embeddings of local regions and raw global image
        l_emb = self.embedding_local(local_feature)
        g_emb = (self.embedding_global(raw_global).unsqueeze(1).repeat(1, l_emb.size(1), 1))
        
        # Compute attention weights
        common = l_emb * g_emb
        weights = self.softmax(self.embedding_common(common).squeeze(2))

        # Compute final image representation
        new_global = torch.sum(weights.unsqueeze(2) * local_feature, dim=1)
        return self.l2norm(new_global)


class EncoderModel(nn.Module):
    def __init__(self, fp16=False, feature_dim=640, layers=3):
        super(EncoderModel, self).__init__()
        self.fp16 = fp16

        self.config = BertConfig(
            hidden_size=feature_dim,
            num_attention_heads=8,
            max_position_embeddings=512,
            num_hidden_layers=layers,
            pad_token_id=None,
            vocab_size=0,
            type_vocab_size=2,
        )
        self.bert_model = BertModel(self.config)
        del self.bert_model.embeddings.word_embeddings

    def forward(self, bert_input):
        with torch.cuda.amp.autocast(self.fp16):
            bert_output = self.bert_model(**bert_input)
        return bert_output


class PlusModel(nn.Module):
    def __init__(self, fp16=False, feature_dim=640, device=None, layers=3):
        super(PlusModel, self).__init__()
        self.bert_encoder = EncoderModel(fp16, feature_dim, layers)
        self.device = device
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim)).to(device)

    def forward(self, reference_features, text_features):
        """
        fuse ref_features and text_features
        :param reference_features: [b, 13, dim]
        :param text_features: [b, 77, dim]
        :return: fusion features
        """

        # [CLS]
        batch_size, patch_num, dim = reference_features.shape
        _, seq_num, _ = text_features.shape

        inputs_embeds = torch.cat((reference_features, text_features), dim=1).to(self.device)  # notice the dimension
        cls_token = self.cls_token.expand(batch_size, -1, -1).to(self.device)
        inputs_embeds = torch.cat((cls_token, inputs_embeds), dim=1).to(self.device)
        token_ref, token_text = torch.zeros([batch_size, patch_num + 1]), torch.ones([batch_size, seq_num])
        token_type_ids = (torch.cat((token_ref, token_text), dim=-1).long().to(self.device))
        attention_mask = (torch.ones([batch_size, patch_num + seq_num + 1]).long().to(self.device))
        # adopt bert-like
        bert_input = {
            "inputs_embeds": inputs_embeds,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "output_attentions": True,
        }
        output = self.bert_encoder(bert_input)
        # pooler_output is [CLS] after the full connection layer and activation function
        last_hidden_state, pooler_output = output[:2]
        fusion_feature = F.normalize(pooler_output, dim=-1)
        return fusion_feature, last_hidden_state, pooler_output
