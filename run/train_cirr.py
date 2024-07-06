import torch
import os
import torch.distributed as dist
from argparse import ArgumentParser
from torch import optim
from torch.utils.data import DataLoader
import open_clip

from dataloader.dataset import CIRRDataset, targetpad_transform
from utils.utils import extract_index_features, AverageMeter, setup_seed
from run.validate_cirr import compute_cirr_val_metrics
from models.model import ERN
from loss.loss import BatchBasedClassificationLoss
# from X_clip.factory import create_model_and_transforms

setup_seed(42)

def train_cirr(num_epochs, lr, batch_size, patch_num, clip_bs, clip_model_name, **kwargs):
    global best_r_average

    device = kwargs['device']
    local_rank = kwargs['local_rank']

    """Define the model"""
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, device=device)
    clip_model.eval()
    clip_model = clip_model.float()
    tokenizer = open_clip.get_tokenizer(clip_model_name)

    input_dim = kwargs["input_dim"]
    feature_dim = kwargs["feature_dim"]

    model = ERN(clip_model, feature_dim, device).to(device, non_blocking=True)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True,
    )

    """Define the training datasets"""
    preprocess = targetpad_transform(kwargs['target_ratio'], input_dim)
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    train_sampler = torch.utils.data.distributed.DistributedSampler(relative_train_dataset)
    train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=kwargs["num_workers"],
        pin_memory=True,
        sampler=train_sampler,
    )

    """Define the validation dataset and extract the validation index features for each dress_type"""
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    val_index_features, val_index_names, val_index_patches = extract_index_features(
        classic_val_dataset, clip_model, patch_num, device, feature_dim
    )

    """Define the optimizer, the loss and the grad scaler"""
    fusion_params = []
    other_params = []
    for name, param in model.module.named_parameters():
        if all(keyword not in name for keyword in ['new_text_projection', 'image_clip', 'text_clip']):
            fusion_params.append(param)
        else:
            other_params.append(param)
    optimizer = optim.Adam(fusion_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100 * len(train_loader))
    BBC_criterion = BatchBasedClassificationLoss()
    scaler = torch.cuda.amp.GradScaler()

    """Train the model, also including validation to witness the best results"""
    best_r_average = 0
    if local_rank == 0:
        print("Begin to train")
    for epoch in range(num_epochs):
        losses = AverageMeter()
        model.train()
        for idx, (ref_images, tar_images, captions, ref_patch_feats, tar_patch_feats) in enumerate(train_loader):
            optimizer.zero_grad()
            ref_images = ref_images.to(device, non_blocking=True)
            tar_images = tar_images.to(device, non_blocking=True)
            ref_patch_feats = ref_patch_feats.to(device, non_blocking=True)
            tar_patch_feats = tar_patch_feats.to(device, non_blocking=True)
            text_inputs = tokenizer(captions, context_length=77).to(device, non_blocking=True)

            """Extract the features"""
            with torch.no_grad():
                ref_imgs_list = torch.split(ref_images, clip_bs)
                ref_img_feats = torch.vstack([model(image=mini_batch, mode="image").float() for mini_batch in ref_imgs_list])
                ref_patch_feats_trans = ref_patch_feats.transpose(0, 1)
                tar_img_list = torch.split(tar_images, clip_bs)
                tar_img_feats = torch.vstack([model(image=mini_batch, mode="image").float() for mini_batch in tar_img_list])

                text_feats = model(text=text_inputs, mode="text_global", ref_local_feats=ref_patch_feats_trans).float()
                text_seq_feats = model(text=text_inputs, mode="text_seq", ref_local_feats=ref_patch_feats_trans).float()

            """Compute the logits and the loss"""
            with torch.cuda.amp.autocast():
                fusion_feat, target_feat = model(
                    ref_feats=ref_img_feats,
                    ref_local_feats=ref_patch_feats,
                    text_feats=text_feats,
                    text_seq_feats=text_seq_feats,
                    tar_feats=tar_img_feats,
                    tar_local_feats=tar_patch_feats,
                    mode="train"
                )
                loss = BBC_criterion(fusion_feat, target_feat)
            losses.update(loss.detach().cpu().item())

            """Backpropagation and update the weights"""
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if idx == len(train_loader) - 1 or idx % kwargs["print_frequency"] == 0:
                if local_rank == 0:
                    print(
                        'Train Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        .format(
                            epoch, idx, len(train_loader),
                            loss=losses,
                        )
                    )
                if idx == len(train_loader) - 1:
                    model = model.float()
                    model.eval()
                    results = compute_cirr_val_metrics(
                        relative_val_dataset,
                        clip_model,
                        val_index_features,
                        val_index_patches,
                        val_index_names,
                        model,
                        device,
                        feature_dim,
                    )
                    group_recall_at1, _, _, _, recall_at5, _, _ = results
                    r_average = (recall_at5 + group_recall_at1) / 2
                    if local_rank == 0:
                        print("R@5: {:.5f}    GR@1: {:.5f}".format(recall_at5, group_recall_at1))
                    if r_average > best_r_average:
                        best_r_average = round(r_average, 5)
                        if local_rank == 0:
                            print("Best Mean Now: {:.5f} {}".format(best_r_average, "*" * 30))
                        # checkpoint name
                        checkpoint_name = "ckpt/cirr-best" + ".pth"
                        torch.save(model.module.state_dict(), checkpoint_name)
                    else:
                        if local_rank == 0:
                            print("Mean Now: {:.5f} Best Before: {:.5f} {}".format(r_average, best_r_average, "-" * 20))


if __name__ == "__main__":
    setup_seed(42)

    parser = ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--dataset", default='shoes', type=str,
                        help="'CIRR' or 'fashionIQ' or 'fashion200k" or 'shoes')
    parser.add_argument("--input-dim", default=288, type=int)
    parser.add_argument("--feature-dim", default=640, type=int)
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--num-epochs", default=300, type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="'RN50', 'RN50x4'")
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
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")

    args = parser.parse_args()

    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
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
    dist.init_process_group(backend="nccl", init_method=dist_url, rank=rank, world_size=world_size)

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

    train_cirr(**training_hyper_params)
