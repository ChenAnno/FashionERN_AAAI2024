from argparse import ArgumentParser
from operator import itemgetter
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import open_clip

from utils.utils import collate_fn, extract_index_features, setup_seed
from dataloader.dataset import targetpad_transform
from dataloader.cirr import CIRRDataset
from models.model import ERN
# from X_clip.factory import create_model_and_transforms

setup_seed(42)


def compute_cirr_val_metrics(
    relative_val_dataset,
    clip_model,
    index_features,
    index_local_features,
    index_names,
    model,
    device,
    feature_dim,
    batch_size,
    num_workers,
    clip_model_name
):
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = (
        generate_cirr_val_predictions(
            clip_model,
            relative_val_dataset,
            model,
            index_names,
            index_features,
            device,
            feature_dim,
            batch_size,
            num_workers,
            clip_model_name,
        )
    )
    index_features = F.normalize(index_features, dim=-1).float()
    index_features = model(tar_feats=index_features, tar_local_feats=index_local_features, mode="index").float()

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


def generate_cirr_val_predictions(
    clip_model,
    relative_val_dataset,
    model,
    index_names,
    index_features,
    device,
    feature_dim,
    batch_size,
    num_workers,
    clip_model_name,
):
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='CIRR', type=str, help="'CIRR' or 'fashionIQ' or 'fashion200k" or 'shoes')
    parser.add_argument("--input-dim", default=288, type=int)
    parser.add_argument("--feature-dim", default=640, type=int)
    parser.add_argument("--patch-num", default=13, type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="'ViT-B-16', 'RN50x4'")
    parser.add_argument("--clip-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--fusion-model-path", type=str, help="Path to the fine-tuned fusion model")
    args = parser.parse_args()

    device = torch.device("cuda")

    """Define the model and the train dataset"""
    clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_model_name, device=device)
    saved_state_dict = torch.load(args.clip_path, map_location=device)
    clip_model.load_state_dict(saved_state_dict["CLIP"])
    clip_model.eval()
    clip_model = clip_model.float()
    tokenizer = open_clip.get_tokenizer(args.clip_model_name)

    model = ERN(clip_model, args.feature_dim, device).to(device, non_blocking=True)
    model.load_state_dict(torch.load(args.fusion_model_path))

    """Define the test dataset"""
    preprocess = targetpad_transform(args.target_ratio, args.input_dim)
    classic_val_dataset = CIRRDataset("val", "classic", preprocess)
    relative_val_dataset = CIRRDataset("val", "relative", preprocess)
    val_index_features, val_index_names, val_index_patches = extract_index_features(
        classic_val_dataset, clip_model, args.patch_num, device, args.feature_dim
    )
    model.eval()
    model = model.float()
    results = compute_cirr_val_metrics(
        relative_val_dataset,
        clip_model,
        val_index_features,
        val_index_patches,
        val_index_names,
        model,
        device,
        args.feature_dim,
        args.batch_size,
        args.num_workers,
        args.clip_model_name,
    )
    group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

    print("R@1:", recall_at1)
    print("R@5:", recall_at5)
    print("R@10: ", recall_at10)
    print("R@50: ", recall_at50)
    print("G@1:", group_recall_at1)
    print("G@2:", group_recall_at2)
    print("G@3:", group_recall_at3)
    print("Average: ", (recall_at5 + group_recall_at1) / 2)
