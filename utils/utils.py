import torch
from PIL import ImageDraw
import numpy as np
import random
import math
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy
from typing import Tuple, List
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn


# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def concat_global_local_feats(global_feats, local_feats):
    """
    concat
    :param global_feats: [batch_size, feature_dim]
    :param local_feats: [batch_size, patch_num, feature_dim]
    :return: [batch_size, patch_num+1, feature_dim]
    """
    global_feats = global_feats.unsqueeze(dim=1)
    concat_feats = torch.cat((global_feats, local_feats), dim=1)
    return concat_feats


def extract_index_features(dataset, clip_model, patch_num, device, feature_dim):
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param clip_model: CLIP model
    :param patch_num: number of split patches
    :param device: device set
    :param feature_dim: 512 for ViT, 640 for ResNetx50
    :return: index_whole_features, index_names, index_local_features
    """

    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=4,
                                    pin_memory=True, collate_fn=collate_fn)
    index_whole_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_names = []
    index_local_features = torch.empty((0, patch_num, feature_dim)).to(device, non_blocking=True)
    for names, images, local_feats in classic_val_loader:
        images = images.to(device, non_blocking=True)
        local_feats = local_feats.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_whole_features = torch.vstack((index_whole_features, batch_features))
            index_names.extend(names)
            index_local_features = torch.vstack((index_local_features, local_feats))

    return index_whole_features, index_names, index_local_features


def extract_index_features_vit(dataset, clip_model, patch_num, device, feature_dim):
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param clip_model: CLIP model
    :param patch_num: number of split patches
    :param device: device set
    :param feature_dim: 512 for ViT, 640 for ResNetx50
    :return: index_whole_features, index_names, index_local_features
    """

    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=4,
                                    pin_memory=True, collate_fn=collate_fn)
    index_whole_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_names = []
    index_local_features = torch.empty((0, patch_num, feature_dim)).to(device, non_blocking=True)
    index_q = torch.empty((0, 197, feature_dim))
    for names, images, local_feats in classic_val_loader:
        images = images.to(device, non_blocking=True)
        local_feats = local_feats.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features, batch_q = clip_model.encode_image(images)
            index_whole_features = torch.vstack((index_whole_features, batch_features))
            index_names.extend(names)
            index_local_features = torch.vstack((index_local_features, local_feats))
            index_q = torch.vstack((index_q, batch_q.cpu()))

    return index_whole_features, index_names, index_local_features, index_q


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


def generate_shoes_caption(flattened_captions: List[str]):
    captions = []
    for i in range(0, len(flattened_captions)):
        captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
    return captions


def element_wise_sum(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def update_train_running_results(train_running_results,
                                 loss,
                                 loss_ag,
                                 loss_prompt,
                                 images_in_batch):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param loss_ag
    :param loss_prompt
    :param images_in_batch: num images in the batch
    """
    train_running_results['accumulated_train_loss'] += loss.to('cpu',
                                                               non_blocking=True).detach().item() * images_in_batch
    train_running_results['ag_loss'] += loss_ag.to('cpu',
                                                   non_blocking=True).detach().item() * images_in_batch
    train_running_results['prompt_loss'] += loss_prompt.to('cpu',
                                                           non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        # self.module = deepcopy(model)
        self.module = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)
            self.module.text_clip.to(device)
            self.module.combiner.to(device)
            self.module.visual_attn.to(device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
