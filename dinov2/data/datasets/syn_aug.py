import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import csv
from tqdm import tqdm
import copy 
import logging
import torch
from torchvision import transforms

logger = logging.getLogger("dinov2")

# from .transforms import (
#     GaussianBlur,
#     make_normalize_transform,
# )


logger = logging.getLogger("dinov2")

from PIL import ImageFilter
import random

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max),
            )
        )


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean, std
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(
                   mean=(0.48145466, 0.4578275, 0.40821073),
                   std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
    

import os
import json
class SyntAugDataset(ImageFolder):
    def __init__(self, root, transforms=None, num_views=0,):
        super().__init__(
            root,
            transform=None)
        
        num_classes = len(self.classes)
        
        self.num_views = num_views
        self.transforms = transforms
       
        ## 4 views for SyntAugment
        self.views = []
        for img, label in self.imgs:
            view1 = (img,label)
            view2 = (view1[0].replace('view1','view2'), label)
            view3 = (view1[0].replace('view1','view3'), label)
            view4 = (view1[0].replace('view1','view4'), label)
            all_views = [view1, view2,view3,view4]
            self.views.append(all_views)
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        
        target = self.targets[idx]
        views = self.views[idx]
        random.shuffle(views)
        
        output = {}
        
        # global crop 1:
        image = Image.open(views[0][0]).convert('RGB')
        global_crop = self.transforms.geometric_augmentation_global(image)
        global_crop = self.transforms.global_transfo1(global_crop)
        output["global_crops"] = [global_crop]
        output["local_crops"] = []
        
        # global crop 2:
        image = Image.open(views[1][0]).convert('RGB')
        global_crop = self.transforms.geometric_augmentation_global(image)
        global_crop = self.transforms.global_transfo2(global_crop)
        output["global_crops"].append(global_crop)
            
        # local crops:
        for _ in range(self.num_views - 2):
            image = Image.open(self.views[0][0]).convert('RGB')
            local_crop = self.transforms.geometric_augmentation_local(image)
            local_crop = self.transforms.local_transfo(local_crop)
            output["local_crops"].append(local_crop)
        
        # output["local_crops"] = []
        output["offsets"] = ()
        output["global_crops_teacher"] = output["global_crops"]
        
        return output, torch.tensor([target] * self.num_views)
