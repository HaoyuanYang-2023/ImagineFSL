# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
from copy import deepcopy
import torch
import json
from torch import nn
import torch.distributed as dist
from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import *
from dinov2.layers import DINOHead,GaussHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.utils.template import TEMPLATES
from dinov2.utils.classnames import CLASSNAMES
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from dinov2.models.vision_transformer import BlockChunk

from dinov2.layers.embed import HoMPool

try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")

from clip import clip


def reload_model(model, ckpt):
    model.load_state_dict(ckpt, strict=False)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None
        
        clip_arch = cfg.student.arch
        
        clip_model,_ = clip.load(clip_arch, download_root='./clip', device='cpu')
        ckpt = clip_model.visual.state_dict()
        
        student_model_dict = dict()
        teacher_model_dict = dict()
        
        student_adapter, teacher_adapter = bulid_adapter_dr(cfg.student.adapter)
        
        student_backbone, teacher_backbone, embed_dim = build_clip_model_from_cfg(cfg)
        
        adapter_params = count_parameters(student_adapter)
        
        self.text_encoder, self.text_adapter = None, None
        self.text_adapter = None

        student_backbone = reload_model(student_backbone, ckpt)
        teacher_backbone = reload_model(teacher_backbone, ckpt)
                
        
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        
        student_model_dict["adapter"] = student_adapter
        teacher_model_dict["adapter"] = teacher_adapter
        
        gauss_pool = HoMPool()
        
        gauss_params = count_parameters(gauss_pool)
        
        student_model_dict["gauss_pool"] = deepcopy(gauss_pool)
        teacher_model_dict["gauss_pool"] = deepcopy(gauss_pool)
        
        torch.cuda.empty_cache()
        
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")
        logger.info(f"OPTIONS -- architecture : adapter_params: {adapter_params}")
        logger.info(f"OPTIONS -- architecture : gauss_params: {gauss_params}")
        
        # text_adapter_params = count_parameters(self.text_adapter)
        # logger.info(f"OPTIONS -- architecture : text_adapter_params: {text_adapter_params}")
        
        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                GaussHead,
                in_dim=embed_dim*4,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            dino_head_params = count_parameters(dino_head())
            logger.info(f"OPTIONS -- DINO -- head_params: {dino_head_params}")
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                logger.info("OPTIONS -- IBOT -- separate head")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")
    
  
    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()
        

    def forward_backward(self, images, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True) # [BS x n_global(2), 3, 224, 224]
        
        if n_local_crops > 0:
            local_crops = images["collated_local_crops"].cuda(non_blocking=True) # [BS x n_local, 3, 96, 96]
        masks = images["collated_masks"].cuda(non_blocking=True) # [BS x n_global,  196]
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)  # [BS x n_global x mask_patches]
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)
        
        # local crops loss terms, locals loss computed with all global crops
        # we get n_local_crops_loss_terms local loss pairs
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        # global crops loss terms, globals loss computed with different views
        # we get n_global_crops_loss_terms global loss pairs
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops 

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops
        # teacher output
        @torch.no_grad()
        def get_teacher_output(global_crops):
            x, n_global_crops_teacher = global_crops, n_global_crops
            with torch.no_grad():
                x = self.teacher.backbone(x, is_training=True)
            teacher_backbone_output_dict = self.teacher.adapter(x)
            
            
            # TODO: Gauss Pooling for teacher
            # get all tokens
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"] # [2BS, 512] 
            teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"] # [2BS x 196 x 512]
            teacher_tokens = torch.cat((teacher_cls_tokens.unsqueeze(1), teacher_patch_tokens), dim=1) # [2BS x 197 x 512]

            teacher_gauss = self.teacher.gauss_pool(teacher_tokens) # [2BS x D_gauss]

            # chunk 2 views
            teacher_gauss = teacher_gauss.chunk(n_global_crops_teacher) 
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_gauss = torch.cat((teacher_gauss[1], teacher_gauss[0]))

            # get patch tokens
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = teacher_cls_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]
            if do_ibot and not self.ibot_separate_head:
                # Do DINO and IBOT together with the same head
                # Set up the buffer tensor for the teacher. max number of patches is upperbound
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                # Copy the cls tokens
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                # Copy the masked patch tokens
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                # Do DINO and IBOT with separate heads
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_gauss)

                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
                

            else:
                # Do DINO only
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                # centering cls tokens
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    # centering patch tokens
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            elif self.cfg.train.centering == "none":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_patch_tokens_after_head[:, :n_masked_patches]
            
            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered
       
        # 4: get teacher output
        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output(
            global_crops, 
        )
        reshard_fsdp_model(self.teacher)
        loss_dict = {}
        loss_accumulator = 0  # for backprop
        if do_ibot:
            masks_t = masks
        else:
            masks_t = None
        if n_local_crops == 0:
            x = self.student.backbone(global_crops, masks=masks_t, is_training=True)
            student_global_backbone_output_dict = self.student.adapter(x)
        else:
            x_global, x_local = self.student.backbone(
                [global_crops, local_crops], masks=[masks_t, None], is_training=True, patch_drop=self.patch_drop
            )
            student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.adapter(
                [x_global, x_local]
            )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        if n_local_crops > 0:
            student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
            inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        student_global_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
        student_global_tokens = torch.cat((student_global_cls_tokens.unsqueeze(1), student_global_patch_tokens), dim=1)
        # student_global_tokens = self.student.dimension_reduction(student_global_tokens)
        student_global_gauss = self.student.gauss_pool(student_global_tokens)
        inputs_for_student_head_list.append(student_global_gauss.unsqueeze(0))
        # inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            # Set up the buffer tensor for the student. max number of patches is upperbound
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            # Copy the masked patch tokens
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                # Do DINO and IBOT together with the same head
                # append patch tokens to the list
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                # DO DINO and IBOT with separate heads
                # DO iBOT with separate head
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]
        
        # 2: run; split the inputs from BlockDiagonalMask,DO DINO
        # TODO: do guass pooling for student dino loss
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        if n_local_crops > 0:
            student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
        
        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            # Do DINO and IBOT together with the same head
            # Get the masked patch tokens for iBOT
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 0:
            # compute L_DINO for local crops
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=[teacher_dino_softmaxed_centered_list[1]], # note the reverse
            ) / (n_local_crops_loss_terms) # mean over pairs

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process 2 global crops togethe
        
        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms) # mean over pairs
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                # KOLEO Regularization
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_global_gauss.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually
                
           
        
        if do_ibot:
            # compute iBOT loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales # 2 global views
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2 # mean over global views

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss
        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            for attr in {"_unshard_stream", "_post_backward_stream", "_pre_unshard_stream", "_all_reduce_stream", "_default_stream"}:
                stream = getattr(self.teacher.backbone, attr)
                setattr(self.student.dino_head, attr, stream)
                setattr(self.teacher.dino_head, attr, stream)
                setattr(self.student.adapter, attr, stream)
                setattr(self.teacher.adapter, attr, stream)
                setattr(self.student.backbone, attr, stream)
                
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    # Only update the backbone mask token for backpropagation
                    if hasattr(ms, "mask_token"):
                        student_param_list.append(ms.mask_token)
                        teacher_param_list.append(mt.mask_token)
                    else:
                        student_param_list += ms.params
                        teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.student.backbone.eval()
        self.teacher.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for mm in self.student.values():      
            all_params_groups += self.get_maybe_fused_params_for_submodel(mm)
        if self.text_adapter is not None:
            all_params_groups += self.get_maybe_fused_params_for_submodel(self.text_adapter)
            
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
            
        
        
                                            