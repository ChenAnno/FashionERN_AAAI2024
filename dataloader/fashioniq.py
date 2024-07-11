import json
from typing import List
import PIL
import PIL.Image
import torch
from torch.utils.data import Dataset

from dataloader.dataset import targetpad_transform


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
