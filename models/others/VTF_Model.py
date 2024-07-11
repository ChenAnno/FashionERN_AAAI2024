from torch import nn
import torch
import torch.nn.functional as F


class VTFModule(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(VTFModule, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim * 2, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim * 2, projection_dim)

        self.image_mlp = nn.Sequential(
            nn.Linear(clip_feature_dim * 2, clip_feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(clip_feature_dim * 2, 1),
            nn.Softmax(dim=-1),
        )
        self.text_mlp = nn.Sequential(
            nn.Linear(clip_feature_dim * 2, clip_feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(clip_feature_dim * 2, 1),
            nn.Softmax(dim=-1),
        )

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(
            hidden_dim, clip_feature_dim * 2
        )  # [batch_size, 1280]
        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.dropout11 = nn.Dropout(0.5)
        self.dropout22 = nn.Dropout(0.5)
        self.combiner_layer2 = nn.Linear(projection_dim, hidden_dim)
        self.output_layer2 = nn.Linear(
            hidden_dim, clip_feature_dim * 2
        )  # [batch_size, 1280]
        self.dropout33 = nn.Dropout(0.5)
        self.dynamic_scalar2 = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, image_features: torch.tensor, text_features: torch.tensor
    ) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: scaled logits
        """

        text_projected_features = self.dropout1(
            F.relu(self.text_projection_layer(text_features))
        )
        image_projected_features = self.dropout2(
            F.relu(self.image_projection_layer(image_features))
        )  # [b, 640*4]
        raw_combined_features = torch.cat(
            (text_projected_features, image_projected_features), -1
        )  # [b, 640*8]
        combined_features = self.dropout3(
            F.relu(self.combiner_layer(raw_combined_features))
        )  # [b, 640*8]
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)

        text_sa = self.dropout11(
            F.relu(text_features * self.text_mlp(text_features))
        )  # [b, 640*2]
        image_sa = self.dropout11(
            F.relu(image_features * self.text_mlp(image_features))
        )
        raw_combined_features2 = torch.cat((text_sa, image_sa), -1)  # [b, 640*4]
        combined_feature2 = self.dropout33(
            F.relu(self.combiner_layer2(raw_combined_features2))
        )  # [b, 640*8]
        dynamic_scalar2 = self.dynamic_scalar2(raw_combined_features2)
        output = (
            self.output_layer(combined_features)
            + dynamic_scalar * text_features
            + (1 - dynamic_scalar) * image_features
            + self.output_layer2(combined_feature2)
            + dynamic_scalar2 * text_sa
            + (1 - dynamic_scalar2) * image_sa
        )
        predicted_features = F.normalize(output, dim=-1)
        return predicted_features
