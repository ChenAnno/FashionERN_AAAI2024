from typing import List
import numpy as np
import torch
import os
from argparse import ArgumentParser
from torch import optim
from torch.utils.data import DataLoader
import open_clip

from dataloader.dataset import targetpad_transform
from dataloader.fashioniq import FashionIQDataset
from utils.utils import (
    extract_index_features,
    generate_randomized_fiq_caption,
    AverageMeter,
    setup_seed,
    extract_text_features,
    extract_image_features
)
from run.valid.validate_fiq import compute_fiq_val_metrics
from models.model import ERN
from losses.loss import BatchBasedClassificationLoss
from run.train.base_trainer import BaseTrainer

# from X_clip.factory import create_model_and_transforms

setup_seed(42)


class FashionIQTrainer(BaseTrainer):
    def __init__(
        self,
        train_dress_types: List[str],
        val_dress_types: List[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_dress_types = train_dress_types
        self.val_dress_types = val_dress_types
        # Initialize specific datasets, samplers, and other components here
        (
            self.relative_val_datasets,
            self.index_whole_features_list,
            self.index_names_list,
            self.index_local_list,
        ) = self.define_val_datasets()

    def define_model(self):
        clip_model, _, _ = open_clip.create_model_and_transforms(self.clip_model_name, device=self.device)
        clip_model.eval()
        clip_model = clip_model.float()
        tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        model = ERN(clip_model, self.feature_dim, self.device).to(self.device, non_blocking=True)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )
        return model, tokenizer

    def define_train_loader(self):
        preprocess = targetpad_transform(self.target_ratio, self.input_dim)
        train_dataset = FashionIQDataset("train", self.train_dress_types, "relative", preprocess)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )
        return train_loader

    def define_val_datasets(self, target_ratio, input_dim):
        preprocess = targetpad_transform(target_ratio, input_dim)
        relative_val_datasets = []
        index_whole_features_list = []
        index_names_list = []
        index_local_list = []
        for dress_type in self.val_dress_types:
            relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
            relative_val_datasets.append(relative_val_dataset)
            classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
            index_whole_name_local = extract_index_features(classic_val_dataset, self.model, self.patch_num, self.device, self.feature_dim)
            index_whole_features_list.append(index_whole_name_local[0])
            index_names_list.append(index_whole_name_local[1])
            index_local_list.append(index_whole_name_local[2])
        return relative_val_datasets, index_whole_features_list, index_names_list, index_local_list

    def define_optimizer_and_loss(self):
        fusion_params = []
        other_params = []
        for name, param in self.model.module.named_parameters():
            if all(keyword not in name for keyword in ["new_text_projection", "image_clip", "text_clip"]):
                fusion_params.append(param)
            else:
                other_params.append(param)
        optimizer = optim.Adam(fusion_params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100 * len(self.train_loader))
        BBC_criterion = BatchBasedClassificationLoss()
        scaler = torch.cuda.amp.GradScaler()
        return optimizer, scheduler, BBC_criterion, scaler

    def train_one_epoch(self):
        self.model.train()
        losses = AverageMeter()
        for idx, (ref_images, tar_images, captions, ref_patch_feats, tar_patch_feats) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            ref_images, tar_images, ref_patch_feats, tar_patch_feats = map(
                lambda x: x.to(self.device, non_blocking=True),
                [ref_images, tar_images, ref_patch_feats, tar_patch_feats],
            )
            flattened_captions = np.array(captions).T.flatten().tolist()
            input_captions = generate_randomized_fiq_caption(flattened_captions)
            text_inputs = self.tokenizer(input_captions, context_length=77).to(self.device, non_blocking=True)

            with torch.no_grad():
                ref_feats = extract_image_features(self.model, ref_images, self.clip_bs)
                tar_feats = extract_image_features(self.model, tar_images, self.clip_bs)
                text_feats, text_seq_feats = extract_text_features(self.model, text_inputs, ref_patch_feats)

            with torch.cuda.amp.autocast():
                fusion_feat, target_feat = self.model(
                    ref_feats=ref_feats,
                    ref_local_feats=ref_patch_feats,
                    text_feats=text_feats,
                    text_seq_feats=text_seq_feats,
                    tar_feats=tar_feats,
                    tar_local_feats=tar_patch_feats,
                    mode="train",
                )
                loss = self.criterion(fusion_feat, target_feat)
            losses.update(loss.detach().cpu().item())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            if (idx == len(self.train_loader) - 1 or idx % self.kwargs["print_frequency"] == 0):
                if self.local_rank == 0:
                    print(
                        f"Train Epoch: [{self.epoch}][{idx}/{len(self.train_loader)}]\t Loss {losses.val:.4f} ({losses.avg:.4f})"
                    )

    def validate(self):
        self.model.eval()
        recalls_at10, recalls_at50 = [], []
        for relative_val_dataset, index_features, index_names, index_local_feats in zip(
            self.relative_val_datasets,
            self.index_whole_features_list,
            self.index_names_list,
            self.index_local_list,
        ):
            recall_at10, recall_at50 = compute_fiq_val_metrics(
                relative_val_dataset,
                self.model,
                index_features,
                index_local_feats,
                index_names,
                self.device,
                self.feature_dim,
            )
            recalls_at10.append(recall_at10)
            recalls_at50.append(recall_at50)
        r_10, r_50 = np.mean(recalls_at10), np.mean(recalls_at50)
        r_average = (r_10 + r_50) / 2
        if r_average > best_r_average:
            best_r_average = round(r_average, 5)
            if self.local_rank == 0:
                print("Best Mean Now: ", best_r_average, "*" * 30)
            checkpoint_name = "ckpt/fashioniq-best" + ".pth"
            torch.save(self.model.module.state_dict(), checkpoint_name)
        else:
            if self.local_rank == 0:
                print("Mean Now: {:.5f} Best Before: {:.5f} {}".format(r_average, best_r_average, "-" * 20))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--dataset", default='fashionIQ', type=str, help="'CIRR' or 'fashionIQ' or 'fashion200k" or 'shoes')
    parser.add_argument("--input-dim", default=288, type=int)
    parser.add_argument("--feature-dim", default=640, type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--num-epochs", default=300, type=int)
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="'ViT-B-16', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--lr", default=4e-5, type=float, help="Combiner learning rate")
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--patch-num", default=13, type=int)
    parser.add_argument("--clip-bs", default=4, type=int, help="Batch size during CLIP feature extraction")
    parser.add_argument("--validation-bs", default=1024, type=int, help="Batch size during validation")
    parser.add_argument("--validation-frequency", default=3, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--print-frequency", default=100, type=int)
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true', help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true', help="Save only the best model during training")
    return parser.parse_args()


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    dist_url = "tcp://{}:{}".format(
        os.environ.get("MASTER_ADDR", "127.0.0.1"),
        os.environ.get("MASTER_PORT", "12584"),
    )
    args.world_size = world_size
    args.rank = rank
    args.dist_url = dist_url
    print("=> world size:", world_size)
    print("=> rank:", rank)
    print("=> dist_url:", dist_url)
    print("=> local_rank:", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method=dist_url, rank=rank, world_size=world_size
    )

    training_hyper_params = {
        "input_dim": args.input_dim,
        "feature_dim": args.feature_dim,
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "clip_model_name": args.clip_model_name,
        "clip_model_path": args.clip_model_path,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "patch_num": args.patch_num,
        "clip_bs": args.clip_bs,
        "valid_bs": args.validation_bs,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "print_frequency": args.print_frequency,
        "device": device,
        "local_rank": args.local_rank,
    }
    training_hyper_params.update(
        {
            "train_dress_types": ["dress", "toptee", "shirt"],
            "val_dress_types": ["dress", "toptee", "shirt"],
        }
    )

    trainer = FashionIQTrainer(**training_hyper_params)
    trainer.train()


if __name__ == "__main__":
    main()
