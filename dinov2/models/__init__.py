# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from .adapter import AdapterDr
from dinov2.layers.embed import HoMPool
logger = logging.getLogger("dinov2")

import torch.nn as nn
def convert_weights(model):
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
    model.apply(_convert_weights_to_fp16)
    


from clip.model import VisionTransformer
def build_clip_model(cfg, only_teacher=False):

    if cfg.arch == "ViT-B/16":
        teacher = VisionTransformer(
            input_resolution=224,
            patch_size=16,
            width=768, 
            layers=12,
            heads=12, output_dim=512
        )
        
        if only_teacher:
            return teacher, teacher.output_dim
        
        student = VisionTransformer(
            input_resolution=224,
            patch_size=16,
            width=768, 
            layers=12,
            heads=12, output_dim=512
        )
        embed_dim = student.output_dim
        
    if cfg.arch == "ViT-L/14":
        
        teacher = VisionTransformer(
            input_resolution=224,
            patch_size=14,
            width=1024, 
            layers=24,
            heads=16, output_dim=768
        )
        
        if only_teacher:
            return teacher, teacher.output_dim
        
        student = VisionTransformer(
            input_resolution=224,
            patch_size=14,
            width=1024, 
            layers=24,
            heads=16, output_dim=768
        )
        embed_dim = 768
        
    return student, teacher, embed_dim

def build_clip_model_from_cfg(cfg, only_teacher=False):
    return build_clip_model(cfg.student, only_teacher=only_teacher)


def build_hom_pool3(args, only_teacher=False):

    args = args.student
    teacher = HoMPool()
    if only_teacher:
            return teacher
    student = HoMPool()
    return student, teacher

def bulid_adapter_dr(args, only_teacher=False):
    teacher = AdapterDr(embed_dim=args.embed_dim,
                        qkv_dim=args.qkv_dim,
                        depth=args.depth,
                        num_heads=args.num_heads,
                        mlp_ratio=args.mlp_ratio
                        )
    if only_teacher:
            return teacher
    student = AdapterDr(
            embed_dim=args.embed_dim,
            qkv_dim=args.qkv_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )


    return student, teacher