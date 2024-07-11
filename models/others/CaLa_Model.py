import torch
from torch import nn


class TwinAttentionCompositorBLIP2(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()
        # self.fusion = nn.Linear(512,256)
        # self.relu1 = nn.ReLU(inplace=True)
        self.reference_as_query_attention = torch.nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=1, dropout=0.0, batch_first=True
        )
        self.target_as_query_attention = torch.nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=1, dropout=0.0, batch_first=True
        )

    def forward(
        self, reference_embeddings: torch.tensor, target_embeddings: torch.tensor
    ):
        # embeddings to tokens  bs x length x hidden    bs x 32 x 256

        # 4 layers of attention
        output1, _ = self.reference_as_query_attention(
            query=reference_embeddings, key=target_embeddings, value=target_embeddings
        )
        output1, _ = self.reference_as_query_attention(
            query=reference_embeddings, key=output1, value=output1
        )
        output1, _ = self.reference_as_query_attention(
            query=reference_embeddings, key=output1, value=output1
        )
        output1, _ = self.reference_as_query_attention(
            query=reference_embeddings, key=output1, value=output1
        )

        # 4 layers of attention
        output2, _ = self.target_as_query_attention(
            query=target_embeddings,
            key=reference_embeddings,
            value=reference_embeddings,
        )
        output2, _ = self.target_as_query_attention(
            query=target_embeddings, key=output2, value=output2
        )
        output2, _ = self.target_as_query_attention(
            query=target_embeddings, key=output2, value=output2
        )
        output2, _ = self.target_as_query_attention(
            query=target_embeddings, key=output2, value=output2
        )

        # share weight
        # output2, _ = self.reference_as_query_attention(query=target_embeddings, key=reference_embeddings, value=reference_embeddings)
        # output2, _ = self.reference_as_query_attention(query=target_embeddings, key=output2, value=output2)
        # output2, _ = self.reference_as_query_attention(query=target_embeddings, key=output2, value=output2)
        # output2, _ = self.reference_as_query_attention(query=target_embeddings, key=output2, value=output2)

        # use 0 token 作为 features bs x 256  两个features平均
        output1_features = output1[:, 0, :]
        output2_features = output2[:, 0, :]
        output_features = (output1_features + output2_features) / 2
        return output_features
