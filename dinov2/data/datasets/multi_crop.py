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


import os
import json
class MultiCropDataset(ImageFolder):
    def __init__(self, root, transforms=None, num_views=0,):
        super().__init__(
            root,
            transform=None)
        
        num_classes = len(self.classes)
        cls_idx = [ [] for _ in range(num_classes)]
        for idx, (_, label) in enumerate(self.samples):
            cls_idx[label].append(idx)
        self.cls_idx = cls_idx
        self.num_views = num_views
        self.transforms = transforms
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        
        target = self.targets[idx]
        target_class_index =  copy.deepcopy(self.cls_idx[target])
        target_class_index.remove(idx)
        img_idxs = random.sample(target_class_index, self.num_views - 1)
        
        output = {}
        
        # global crop 1:
        image = Image.open(self.imgs[idx][0]).convert('RGB')
        global_crop_1 = self.transforms.geometric_augmentation_global(image)
        global_crop_1 = self.transforms.global_transfo1(global_crop_1)
        output["global_crops"] = [global_crop_1]
        output["local_crops"] = []
        
        # global crop 2:
        image = Image.open(self.imgs[img_idxs[0]][0]).convert('RGB')
        global_crop = self.transforms.geometric_augmentation_global(image)
        global_crop = self.transforms.global_transfo1(global_crop)
        output["global_crops"].append(global_crop)
            
        # local crops:
        for _ in range(self.num_views - 2):
            image = Image.open(self.imgs[idx][0]).convert('RGB')
            local_crop = self.transforms.geometric_augmentation_local(image)
            local_crop = self.transforms.local_transfo(local_crop)
            output["local_crops"].append(local_crop)
        
        # output["local_crops"] = []
        output["offsets"] = ()
        output["global_crops_teacher"] = output["global_crops"]
        
        return output, torch.tensor([target] * self.num_views)
    
