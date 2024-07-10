import os
import torch
import torch.distributed as dist
from argparse import ArgumentParser
from torch import optim
from torch.utils.data import DataLoader
import open_clip

from dataloader.dataset import targetpad_transform
from dataloader.fashion200k_patch import Fashion200kDataset
from utils.utils import AverageMeter, setup_seed, extract_image_features, extract_text_features
from models.model import ERN
from run.train.base_trainer import BaseTrainer
from loss.loss import BatchBasedClassificationLoss
# from X_clip.factory import create_model_and_transforms

setup_seed(42)


class Fashion200kTrainer(BaseTrainer):
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
        relative_train_dataset = Fashion200kDataset(split='train', img_transform=preprocess)
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
        for idx, (ref_images, tar_images, captions, _, _, _, ref_patch_feats, tar_patch_feats) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            ref_images = ref_images.to(self.device, non_blocking=True)
            tar_images = tar_images.to(self.device, non_blocking=True)
            ref_patch_feats = ref_patch_feats.to(self.device, non_blocking=True)
            tar_patch_feats = tar_patch_feats.to(self.device, non_blocking=True)
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

if __name__ == "__main__":
    setup_seed(42)

    parser = ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--dataset", default='fashion200k', type=str, help="'CIRR' or 'fashionIQ' or 'fashion200k' or 'shoes'")
    parser.add_argument("--input-dim", default=288, type=int)
    parser.add_argument("--feature-dim", default=640, type=int)
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--num-epochs", default=300, type=int)
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--lr", default=3e-5, type=float, help="Combiner learning rate")
    parser.add_argument("--batch-size", default=1024, type=int)
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
        "local_rank": args.local_rank
    }

    trainer = Fashion200kTrainer(**training_hyper_params)
    trainer.train()
