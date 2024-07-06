import json
import os
from typing import List
import PIL
import PIL.Image
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import json as jsonmod


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim=288):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio=1.25, dim=288):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class FashionIQDatasetViT(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split, dress_types, mode="relative", preprocess=targetpad_transform()):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.base_path = "./"

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(self.base_path + 'fashion-iq/captions/' + f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(self.base_path + 'fashion-iq/image_splits/' + f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        # print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']
                reference_feature_path = self.base_path + 'fashion-iq/fashioniq_13_vit_2b/' + f"{reference_name}.pth"
                ref_patch_feature = torch.load(reference_feature_path)

                if self.split == 'train':
                    reference_image_path = self.base_path + 'fashion-iq/images/' + f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    target_image_path = self.base_path + 'fashion-iq/images/' + f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    target_feature_path = self.base_path + 'fashion-iq/fashioniq_13_vit_2b/' + f"{target_name}.pth"
                    tar_patch_feature = torch.load(target_feature_path)
                    return reference_image, target_image, image_captions, ref_patch_feature, tar_patch_feature

                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    target_feature_path = self.base_path + 'fashion-iq/fashioniq_13_vit_2b/' + f"{target_name}.pth"
                    tar_patch_feature = torch.load(target_feature_path)
                    return reference_name, target_name, image_captions, ref_patch_feature

                elif self.split == 'test':
                    reference_image_path = self.base_path + 'fashion-iq/images/' + f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = self.base_path + 'fashion-iq/images/' + f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))

                local_feature_path = self.base_path + 'fashion-iq/fashioniq_13_vit_2b/' + f"{image_name}.pth"
                local_feature = torch.load(local_feature_path)

                return image_name, image, local_feature

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split, dress_types, mode="relative", preprocess=targetpad_transform()):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.base_path = "./"

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(self.base_path + 'fashion-iq/captions/' + f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(self.base_path + 'fashion-iq/image_splits/' + f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        # print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']
                reference_feature_path = self.base_path + 'fashion-iq/fashion_local13/' + f"{reference_name}.pth"
                ref_patch_feature = torch.load(reference_feature_path)

                if self.split == 'train':
                    reference_image_path = self.base_path + 'fashion-iq/images/' + f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    target_image_path = self.base_path + 'fashion-iq/images/' + f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    target_feature_path = self.base_path + 'fashion-iq/fashion_local13/' + f"{target_name}.pth"
                    tar_patch_feature = torch.load(target_feature_path)
                    return reference_image, target_image, image_captions, ref_patch_feature, tar_patch_feature

                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    target_feature_path = self.base_path + 'fashion-iq/fashion_local13/' + f"{target_name}.pth"
                    tar_patch_feature = torch.load(target_feature_path)
                    return reference_name, target_name, image_captions, ref_patch_feature

                elif self.split == 'test':
                    reference_image_path = self.base_path + 'fashion-iq/images/' + f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = self.base_path + 'fashion-iq/images/' + f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                local_feature_path = self.base_path + 'fashion-iq/fashion_local13/' + f"{image_name}.pth"
                local_feature = torch.load(local_feature_path)

                return image_name, image, local_feature

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


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


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.base_path = './'

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(self.base_path + 'cirr_dataset/cirr/captions/' + f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(self.base_path + 'cirr_dataset/cirr/image_splits/' + f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']
                ref_patch_path = self.base_path + 'cirr_dataset/cirr_local_13/' + f"{reference_name}.pth"
                ref_patch_feature = torch.load(ref_patch_path)

                if self.split == 'train':
                    reference_image_path = self.base_path + 'cirr_dataset/' + self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = self.base_path + 'cirr_dataset/' + self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))

                    target_patch_path = self.base_path + 'cirr_dataset/cirr_local_13/' + f"{target_hard_name}.pth"
                    target_patch_feature = torch.load(target_patch_path)

                    return reference_image, target_image, rel_caption, ref_patch_feature, target_patch_feature

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']

                    return reference_name, target_hard_name, rel_caption, ref_patch_feature, group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = self.base_path + 'cirr_dataset/' + self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)

                local_feature_path = self.base_path + 'cirr_dataset/cirr_local_13/' + f"{image_name}.pth"
                local_feature = torch.load(local_feature_path)

                return image_name, image, local_feature

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class FashionIQDatasetVAL(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split, dress_types, mode="relative", preprocess=targetpad_transform(1.25, 288)):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.base_path = "./"

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(self.base_path + 'fashion-iq/captions/' + f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for i in self.triplets:
            if i['target'] not in self.image_names:
                self.image_names.append(i['target'])
            if i['candidate'] not in self.image_names:
                self.image_names.append(i['candidate'])
        print(len(self.image_names))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']
                reference_feature_path = self.base_path + 'fashion-iq/fashion_local13/' + f"{reference_name}.pth"
                ref_patch_feature = torch.load(reference_feature_path)

                if self.split == 'train':
                    reference_image_path = self.base_path + 'fashion-iq/images/' + f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    target_image_path = self.base_path + 'fashion-iq/images/' + f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    target_feature_path = self.base_path + 'fashion-iq/fashion_local13/' + f"{target_name}.pth"
                    tar_patch_feature = torch.load(target_feature_path)
                    return reference_image, target_image, image_captions, ref_patch_feature, tar_patch_feature

                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_captions, ref_patch_feature

                elif self.split == 'test':
                    reference_image_path = self.base_path + 'fashion-iq/images/' + f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = self.base_path + 'fashion-iq/images/' + f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))

                local_feature_path = self.base_path + 'fashion-iq/fashion_local13/' + f"{image_name}.pth"
                local_feature = torch.load(local_feature_path)

                return image_name, image, local_feature

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
