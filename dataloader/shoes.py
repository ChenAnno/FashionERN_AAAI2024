import PIL
import PIL.Image
import json as jsonmod
import os
import torch
from torch.utils.data import Dataset

from dataloader.dataset import targetpad_transform


class ShoesDataset(Dataset):
    def __init__(self, split, mode="relative", preprocess=targetpad_transform(target_ratio=1.25, dim=640)):
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.shoes_path = "/mnt/vision_retrieval/chenyanzhe/shoes_dataset/"
        self.local_feature_path = '/mnt/vision_retrieval/chenyanzhe/shoes_dataset/shoes_local_feature_13/'
        self.image_id2name = self.load_file(os.path.join(self.shoes_path, f'split.{split}.json'))
        if self.mode == "relative":
            self.annotations = self.load_file(os.path.join(self.shoes_path, f'triplet.{split}.json'))

    def __getitem__(self, index):
        if self.mode == "relative":  # 返回三元组形式
            ann = self.annotations[index]
            caption = ann['RelativeCaption']
            reference_path = self.shoes_path + ann['ReferenceImageName']
            target_path = self.shoes_path + ann['ImageName']
            reference_name = reference_path.split('/')[-1].split(".jpg")[0]
            target_name = target_path.split('/')[-1].split(".jpg")[0]

            ref_local_path = self.local_feature_path + f"{reference_name}.pth"
            ref_patch_feat = torch.load(ref_local_path)
            tar_local_path = self.local_feature_path + f"{target_name}.pth"
            tar_patch_feat = torch.load(tar_local_path)

            if self.split == "train":
                reference_image = self.preprocess(PIL.Image.open(reference_path))
                target_image = self.preprocess(PIL.Image.open(target_path))
                return reference_image, target_image, caption, ref_patch_feat, tar_patch_feat
            else:  # val
                return reference_name, target_name, caption, ref_patch_feat, tar_patch_feat
        else:
            image_path = self.shoes_path + self.image_id2name[index]
            image_name = image_path.split('/')[-1].split(".jpg")[0]
            image = self.preprocess(PIL.Image.open(image_path))

            local_patch_path = self.local_feature_path + image_name.split(".jpg")[0] + ".pth"
            local_feature = torch.load(local_patch_path)

            return image_name, image, local_feature

    def __len__(self):
        if self.mode == "relative":
            return len(self.annotations)
        else:
            return len(self.image_id2name)

    def load_file(self, f):
        with open(f, "r") as jsonfile:
            ann = jsonmod.loads(jsonfile.read())
        return ann
