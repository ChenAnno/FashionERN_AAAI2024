import torch
from PIL import Image
import clip
import os
import glob
from tqdm import tqdm
import time
import sys
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode


def _convert_image_to_rgb(image):
    return image.convert("RGB")


# 第二步：将图片切割成九宫格
def cut_image_4(image):
    width, height = image.size
    # 一行放3张图
    item_width = int(width / 2)
    item_height = int(height / 2)
    box_list = []
    for i in range(0, 2):
        for j in range(0, 2):
            box = (j * item_width, i * item_height, (j + 1) * item_width, (i + 1) * item_height)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list


# 第二步：将图片切割成九宫格
def cut_image_9(image):
    width, height = image.size
    # 一行放3张图
    item_width = int(width / 3)
    item_height = int(height / 3)
    box_list = []
    for i in range(0, 3):
        for j in range(0, 3):
            box = (j * item_width, i * item_height, (j + 1) * item_width, (i + 1) * item_height)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list


def cut_image_16(image):
    width, height = image.size
    # 一行放4张图
    item_width = int(width / 4)
    item_height = int(height / 4)
    box_list = []
    for i in range(0, 4):
        for j in range(0, 4):
            box = (j * item_width, i * item_height, (j + 1) * item_width, (i + 1) * item_height)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list


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


def targetpad_transform(target_ratio=1.25, dim=224):
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


if __name__ == '__main__':
    device = "cuda"
    model, _ = clip.load("RN50", device=device)
    model = model.eval()
    model_path = "/mnt/vision_retrieval/chenyanzhe/Prompt-my/ckpt/fashion200k_tuned_6182.pt"
    saved_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(saved_state_dict["CLIP"])
    print('CLIP model loaded successfully')

    preprocess = targetpad_transform()

    target_path = "./fashion200k_13_patch"
    image_paths = []
    for f in tqdm(glob.glob("./labels/*.txt")):
        print(f)
        with open(f) as fr:
            lines = fr.readlines()
            for line in lines:
                path = line.strip().split("\t")[0]
                image_paths.append(path)

    print("all image paths: {}".format(len(image_paths)))

    has_processed = set()
    with open("./dir.txt") as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            has_processed.add(line.strip())

    with torch.no_grad():
        for i, image_path in tqdm(enumerate(image_paths)):
            save_path = image_path.replace("women", "fashion200k_13_patch") + ".pth"
            if save_path in has_processed:
                continue

            if not (i % 3 == 0):
                continue

            image = Image.open(image_path)
            image = image.resize((360, 360), Image.ANTIALIAS)

            start = time.time()
            image_list_4 = cut_image_4(image)
            image_list_9 = cut_image_9(image)
            image_list = image_list_4 + image_list_9
            end = time.time()
            # print("crop time: ", end - start)

            start = time.time()
            feature_list = []
            for image in image_list:
                image = preprocess(image).unsqueeze(0).to(device)
                feature = model.encode_image(image)
                feature_list.append(feature)
            feature_all = torch.cat(feature_list, dim=0)
            feature_all = feature_all.detach().float().cpu()
            end = time.time()
            # print("inference time: ", end - start)

            try:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
            except Exception as e:
                print(e)
            torch.save(feature_all, save_path)
            # print(gpuid, i, len(image_paths))
