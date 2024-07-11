from operator import itemgetter
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import open_clip

from utils.utils import collate_fn


def compute_cirr_val_metrics(relative_val_dataset,
                             clip_model,
                             index_features,
                             index_local_features,
                             index_names,
                             model,
                             device,
                             feature_dim
                             ):
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = generate_cirr_val_predictions(clip_model,
                                                                                                     relative_val_dataset,
                                                                                                     model, index_names,
                                                                                                     index_features,
                                                                                                     device,
                                                                                                     feature_dim)
    index_features = F.normalize(index_features, dim=-1).float()
    tar_local_feats = model(ref_local_feats=index_local_features, mode="local").float()
    index_features = model(tar_feats=index_features, tar_local_feats=tar_local_feats, mode="index").float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(clip_model,
                                  relative_val_dataset,
                                  model,
                                  index_names,
                                  index_features,
                                  device,
                                  feature_dim,
                                  ):
    tokenizer = open_clip.get_tokenizer('RN50x4')
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=4, pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)
    name_to_feat = dict(zip(index_names, index_features))
    predicted_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    all_reference_names = []

    for ref_names, batch_target_names, captions, ref_patch_feats, batch_group_members in relative_val_loader:
        text_inputs = tokenizer(captions, context_length=77).to(device)
        ref_patch_feats = ref_patch_feats.to(device)
        batch_group_members = np.array(batch_group_members).T.tolist()

        """Compute the predicted features"""
        with torch.no_grad():
            ref_patch_feats_trans = ref_patch_feats.transpose(0, 1)
            text_features, _ = clip_model.encode_text(text_inputs, visual_emb=ref_patch_feats_trans)
            text_seq_feats = clip_model.encode_text(text_inputs, mode="seq", visual_emb=ref_patch_feats_trans)
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*ref_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*ref_names)(name_to_feat))
            text_features, text_seq_feats = text_features.to(device), text_seq_feats.to(device)
            batch_predicted_features = model(
                ref_feats=reference_image_features,
                ref_local_feats=ref_patch_feats,
                text_feats=text_features,
                text_seq_feats=text_seq_feats,
                mode="test"
            )
        predicted_features = torch.vstack((predicted_features, batch_predicted_features))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        all_reference_names.extend(ref_names)

    return predicted_features, all_reference_names, target_names, group_members

