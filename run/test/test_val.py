from argparse import ArgumentParser
from operator import itemgetter
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import open_clip
from statistics import mean

from dataloader.dataset import targetpad_transform
from dataloader.fashioniq import FashionIQDatasetVAL
from utils.utils import collate_fn, extract_index_features, setup_seed
from models.model import ERN

setup_seed(42)


def compute_fiq_val_metrics(
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
    clip_model_name,
):
    predicted_features, target_names = generate_fiq_val_predictions(
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

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()
    tar_local_feats = model(ref_local_feats=index_local_features, mode="local").float()
    index_features = model(tar_feats=index_features, tar_local_feats=tar_local_feats, mode="index").float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at15 = (torch.sum(labels[:, :15]) / len(labels)).item() * 100
    recall_at20 = (torch.sum(labels[:, :20]) / len(labels)).item() * 100
    recall_at30 = (torch.sum(labels[:, :30]) / len(labels)).item() * 100
    recall_at40 = (torch.sum(labels[:, :40]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at1, recall_at5, recall_at10, recall_at15, recall_at20, recall_at30, recall_at40, recall_at50


def generate_fiq_val_predictions(
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
    for ref_names, batch_target_names, captions, ref_patch_feats in relative_val_loader:
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}"
            for i in range(0, len(flattened_captions), 2)
        ]
        text_inputs = tokenizer(input_captions, context_length=77).cpu()
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
            ref_patch_feats, reference_image_features = ref_patch_feats.to(device), reference_image_features.to(device)
            batch_predicted_features = model(
                ref_feats=reference_image_features,
                ref_local_feats=ref_patch_feats,
                text_feats=text_features,
                text_seq_feats=text_seq_feats,
                mode="test"
            )
        predicted_features = torch.vstack((predicted_features, batch_predicted_features))
        target_names.extend(batch_target_names)
    return predicted_features, target_names


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='fashionIQ', type=str, help="'CIRR' or 'fashionIQ' or 'fashion200k" or 'shoes')
    parser.add_argument("--input-dim", default=224, type=int, help="224 for ViT, 288 for RN50x4")
    parser.add_argument("--feature-dim", default=512, type=int, help="512 for ViT, 640 for RN50x4")
    parser.add_argument("--patch-num", default=13, type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--clip-model-name", default="ViT-B-16", type=str, help="'ViT-B-16', 'RN50x4'")
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
    idx_to_dress_mapping = {}
    relative_val_datasets = []
    index_whole_features_list, index_names_list, index_local_list = [], [], []
    for _, dress_type in enumerate(['dress', 'toptee', 'shirt']):
        idx_to_dress_mapping[_] = dress_type
        relative_val_dataset = FashionIQDatasetVAL('val', [dress_type], 'relative', preprocess)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDatasetVAL('val', [dress_type], 'classic', preprocess)
        index_list = extract_index_features(
            classic_val_dataset, clip_model, args.patch_num, device, args.feature_dim
        )
        index_whole_features_list.append(index_list[0])
        index_names_list.append(index_list[1])
        index_local_list.append(index_list[2])
    model.eval()
    model = model.float()

    recalls_at1 = []
    recalls_at5 = []
    recalls_at10 = []
    recalls_at15 = []
    recalls_at20 = []
    recalls_at30 = []
    recalls_at40 = []
    recalls_at50 = []

    for relative_val_dataset, index_features, index_names, index_local_feats, cloth_type in zip(
            relative_val_datasets,
            index_whole_features_list,
            index_names_list,
            index_local_list,
            idx_to_dress_mapping,
    ):
        recall_at1, recall_at5, recall_at10, recall_at15, recall_at20, recall_at30, recall_at40, recall_at50 = compute_fiq_val_metrics(
            relative_val_dataset,
            clip_model,
            index_features,
            index_local_feats,
            index_names,
            model,
            device,
            args.feature_dim,
            args.batch_size,
            args.num_workers,
            args.clip_model_name
        )
        recalls_at1.append(recall_at1)
        recalls_at5.append(recall_at5)
        recalls_at10.append(recall_at10)
        recalls_at15.append(recall_at15)
        recalls_at20.append(recall_at20)
        recalls_at50.append(recall_at50)

    r_1, r_5, r_15, r_20 = mean(recalls_at1), mean(recalls_at5), mean(recalls_at15),mean(recalls_at20)
    r_10, r_50 = mean(recalls_at10), mean(recalls_at50)
    r_average = (r_10 + r_50) / 2

    print("R@1: ", r_1)
    print("R@5: ", r_5)
    print("R@10: ", r_10)
    print("R@15: ", r_15)
    print("R@20: ", r_20)
    print("R@50: ", r_50)
    print("Average: ", r_average)
