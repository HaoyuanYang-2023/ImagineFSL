import os
import random
from PIL import Image
from torchvision import transforms

from collections import defaultdict

import torchvision
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pathlib
from typing import Any, Callable, Optional, Tuple



import torch
from typing import Dict, List, Optional, Tuple
from torchvision.transforms.autoaugment import _apply_op
from torchvision.transforms import functional as F, InterpolationMode
from torch import Tensor
class RandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.1, num_bins), True), # modified from 0.3
            "ShearY": (torch.linspace(0.0, 0.1, num_bins), True), # modified from 0.3
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 5.0, num_bins), True), # modified from 30.0
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

from PIL import ImageFilter, ImageOps
class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __repr__(self):
        return "{}(p={}, radius_min={}, radius_max={})".format(
            self.__class__.__name__, self.p, self.radius_min, self.radius_max
        )

    def __call__(self, img):
        if random.random() <= self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __repr__(self):
        return "{}(p={})".format(self.__class__.__name__, self.p)

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class FewShotDataset():

    def __init__(self, root, num_shots, preproces=None, args=None):

        self.image_dir = root

        aux_transform = transforms.Compose([
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(0.2),
            # Solarization(0.2),
        ])

        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            RandAugment(),
            aux_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        if preproces is None:
            test_preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            test_preprocess = preproces

        self.train = torchvision.datasets.ImageFolder(root=os.path.join(self.image_dir, 'train'),
                                                      transform=train_preprocess)
        self.num_classes = len(self.train.classes)

        self.val = torchvision.datasets.ImageFolder(root=os.path.join(self.image_dir, 'val'),
                                                        transform=test_preprocess)
        if os.path.exists(os.path.join(self.image_dir, 'test')):
            self.test = torchvision.datasets.ImageFolder(root=os.path.join(self.image_dir, 'test'),
                                                         transform=test_preprocess)
        else:
            self.test = self.val
        
        self.class_name = self.train.classes

        if num_shots:
            split_by_label_dict = defaultdict(list)
            for i in range(len(self.train.imgs)):
                # split_by_label_dict is a dict, key is label, value is a list of (image path, label)
                split_by_label_dict[self.train.targets[i]].append(self.train.imgs[i])
            imgs = []
            targets = []
            for label, items in split_by_label_dict.items():
                imgs = imgs + random.sample(items, num_shots)
                targets = targets + [label for i in range(num_shots)]
            self.train.imgs = imgs
            self.train.targets = targets
            self.train.samples = imgs


class SynDataset():
    """Syn datasets
        ├── root/
        │   ├── class_0/
        │   │   ├── 00000001.jpg
        │   │   └── ...
        │   ├── class_1/
        │   │   ├── 00000001.jpg
        │   │   └── ...
        │   └── ...

    """

    def __init__(self, root, preproces=None, args=None):

        self.image_dir = root

        print("==>  Using aux_transforms")
        aux_transform = transforms.Compose([
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(0.2),
            # Solarization(0.2),
        ])

        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            RandAugment(),
            aux_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])


        if preproces is None:
            test_preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            test_preprocess = preproces

        self.train = torchvision.datasets.ImageFolder(root=self.image_dir, transform=train_preprocess)
        self.num_classes = len(self.train.classes)

        self.class_name = self.train.classes


