# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from .adapter import AdapterDr
from dinov2.layers.gauss_embed import HoMPool
logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim

import torch.nn as nn
def convert_weights(model):
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
    model.apply(_convert_weights_to_fp16)
    

def bulid_adapter(args,only_teacher=False,):
    teacher = Adapter()
    if only_teacher:
            return teacher, teacher.embed_dim
    student = Adapter(
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
    embed_dim = student.embed_dim
    return student, teacher, embed_dim

def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)

from clip.model import VisionTransformer
def build_clip_model(cfg, only_teacher=False):
    # print(cfg.arch)
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
        
    # if cfg.arch == "ViT-L/14":
        
    # teacher = VisionTransformer(
    #     input_resolution=224,
    #     patch_size=14,
    #     width=1024, 
    #     layers=24,
    #     heads=16, output_dim=768
    # )
    
    # if only_teacher:
    #     return teacher, teacher.output_dim
    
    # student = VisionTransformer(
    #     input_resolution=224,
    #     patch_size=14,
    #     width=1024, 
    #     layers=24,
    #     heads=16, output_dim=768
    # )
    # embed_dim = 768
        
    return student, teacher, embed_dim

def build_clip_model_from_cfg(cfg, only_teacher=False):
    return build_clip_model(cfg.student, only_teacher=only_teacher)

def build_clip_text_model(clip_model):
    return TextEncoder(clip_model)
        

def bulid_text_adapter(attn_mask,args=None, only_teacher=False,):
    # import pdb;pdb.set_trace()
    teacher = TextAdapter(attn_mask=attn_mask,
            drop_path_rate=0.0,
            drop_path_uniform=True,
    )
    # import pdb;pdb.set_trace()
    if only_teacher:
            return teacher, teacher.embed_dim
    student = TextAdapter(
        attn_mask=attn_mask,
        drop_path_rate=args.drop_path_rate,
        drop_path_uniform=args.drop_path_uniform,
        )
    embed_dim = student.embed_dim
    
    return student, teacher, embed_dim


def build_gauss_pool(args, only_teacher=False):
    embed_dim=512
    args = args.student
    teacher = GaussPool(
        embed_dim,
        args.gauss_embed_dim
    )
    if only_teacher:
            return teacher
    student = GaussPool(
        embed_dim,
        args.gauss_embed_dim
    )
    return student, teacher

# def build_gauss_pool2(args, only_teacher=False):
#     embed_dim=512
#     args = args.student
#     teacher = GaussPool2(
#         embed_dim,
#         args.gauss_embed_dim
#     )
#     if only_teacher:
#             return teacher
#     student = GaussPool2(
#         embed_dim,
#         args.gauss_embed_dim
#     )
#     return student, teacher,

def build_gauss_pool3(args, only_teacher=False):
    embed_dim=512
    args = args.student
    teacher = HoMPool()
    if only_teacher:
            return teacher
    student = HoMPool()
    return student, teacher,

def bulid_adapter_dr(args,only_teacher=False,):
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
    # embed_dim = student.embed_dim
    return student, teacher