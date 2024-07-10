import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import open_clip
import torch.distributed as dist
from argparse import ArgumentParser

from dataloader.dataset import ShoesDataset, targetpad_transform
from utils.utils import extract_index_features, AverageMeter, setup_seed, extract_image_features, extract_text_features
from run.valid.validate_shoes import compute_shoes_val_metrics
from models.model import ERN
from loss.loss import BatchBasedClassificationLoss
from run.train.base_trainer import BaseTrainer

# from X_clip.factory import create_model_and_transforms

setup_seed(42)


class ShoesTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        relative_train_dataset = ShoesDataset('train', 'relative', preprocess)
        train_sampler = torch.utils.data.distributed.DistributedSampler(relative_train_dataset)
        train_loader = DataLoader(
            dataset=relative_train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )
        return train_loader

    def define_val_datasets(self):
        preprocess = targetpad_transform(self.target_ratio, self.input_dim)
        classic_val_dataset = ShoesDataset('val', 'classic', preprocess)
        relative_val_dataset = ShoesDataset('val', 'relative', preprocess)
        val_index_features, val_index_names, val_index_patches = extract_index_features(
            classic_val_dataset, self.model, self.patch_num, self.device, self.feature_dim
        )
        return relative_val_dataset, val_index_features, val_index_names, val_index_patches

    def define_optimizer_and_loss(self):
        fusion_params = []
        other_params = []
        for name, param in self.model.module.named_parameters():
            if all(keyword not in name for keyword in ['new_text_projection', 'image_clip', 'text_clip']):
                fusion_params.append(param)
            else:
                other_params.append(param)
        optimizer = optim.Adam(fusion_params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100 * len(self.train_loader))
        criterion = BatchBasedClassificationLoss()
        scaler = torch.cuda.amp.GradScaler()
        return optimizer, scheduler, criterion, scaler

    def train_one_epoch(self):
        self.model.train()
        losses = AverageMeter()
        for idx, (ref_images, tar_images, captions, ref_patch_feats, tar_patch_feats) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            ref_images, tar_images, ref_patch_feats, tar_patch_feats = map(
                lambda x: x.to(self.device, non_blocking=True),
                [ref_images, tar_images, ref_patch_feats, tar_patch_feats],
            )
            text_inputs = self.tokenizer(captions, context_length=77).to(self.device, non_blocking=True)

            with torch.no_grad():
                ref_img_feats = extract_image_features(self.model, ref_images, self.clip_bs)
                tar_img_feats = extract_image_features(self.model, tar_images, self.clip_bs)
                text_feats, text_seq_feats = extract_text_features(self.model, text_inputs, ref_patch_feats)

            with torch.cuda.amp.autocast():
                fusion_feat, target_feat = self.model(
                    ref_feats=ref_img_feats,
                    ref_local_feats=ref_patch_feats,
                    text_feats=text_feats,
                    text_seq_feats=text_seq_feats,
                    tar_feats=tar_img_feats,
                    tar_local_feats=tar_patch_feats,
                    mode="train"
                )
                loss = self.criterion(fusion_feat, target_feat)
            losses.update(loss.detach().cpu().item())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            if idx == len(self.train_loader) - 1 or idx % self.print_frequency == 0:
                if self.local_rank == 0:
                    print(
                        "Train Epoch: [{0}][{1}/{2}]\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                            self.epoch,
                            idx,
                            len(self.train_loader),
                            loss=losses,
                        )
                    )

    def validate(self):
        self.model.eval()
        relative_val_dataset, val_index_features, val_index_names, val_index_patches = self.define_val_datasets()
        r_10, r_50 = compute_shoes_val_metrics(
            relative_val_dataset,
            self.model,
            val_index_features,
            val_index_patches,
            val_index_names,
            self.device,
            self.feature_dim,
        )
        r_average = (r_10 + r_50) / 2

        if self.local_rank == 0:
            print("R@10: {:.5f}    R@50: {:.5f}".format(r_10, r_50))
        if r_average > self.best_r_average:
            self.best_r_average = round(r_average, 5)
            if self.local_rank == 0:
                print("Best Mean Now: {:.5f} {}".format(self.best_r_average, "*" * 30))
            checkpoint_name = "ckpt/shoes-best" + ".pth"
            torch.save(self.model.module.state_dict(), checkpoint_name)
        else:
            if self.local_rank == 0:
                print("Mean Now: {:.5f} Best Mean Before: {:.5f} {}".format(r_average, self.best_r_average, "-" * 20))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--dataset", default='shoes', type=str, help="'CIRR' or 'fashionIQ' or 'fashion200k' or 'shoes'")
    parser.add_argument("--input-dim", default=288, type=int)
    parser.add_argument("--feature-dim", default=640, type=int)
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--num-epochs", default=300, type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--lr", default=4e-5, type=float, help="Combiner learning rate")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--patch-num", default=13, type=int)
    parser.add_argument("--clip-bs", default=4, type=int, help="Batch size during CLIP feature extraction")
    parser.add_argument("--validation-bs", default=1024, type=int, help="Batch size during validation")
    parser.add_argument("--validation-frequency", default=3, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--print-frequency", default=100, type=int)
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str, help="Preprocess pipeline")
    parser.add_argument("--save-training", dest="save_training", action='store_true', help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true', help="Save only the best model during training")

    args = parser.parse_args()

    try:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        dist_url = "tcp://{}:{}".format(
            os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]
        )
    except KeyError:
        world_size = 1
        rank = 0
        dist_url = "tcp://127.0.0.1:12584"
    args.world_size = world_size
    args.rank = rank
    args.dist_url = dist_url
    print("=> world size:", world_size)
    print("=> rank:", rank)
    print("=> dist_url:", dist_url)
    print("=> local_rank:", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dist.init_process_group(
        backend="nccl", init_method=args.dist_url, rank=args.rank, world_size=args.world_size
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

    trainer = ShoesTrainer(**training_hyper_params)
    trainer.train()
