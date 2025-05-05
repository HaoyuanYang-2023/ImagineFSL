# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from dinov2.models import build_model_from_cfg, build_clip_model_from_cfg, bulid_adapter_dr, build_hom_pool3

from dinov2.utils.config import setup
import dinov2.utils.utils as dinov2_utils
from dinov2.logging import MetricLogger

import logging
from typing import Dict, Optional
from torchmetrics import MetricCollection

logger = logging.getLogger("dinov2")

def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model


def setup_and_build_model(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args)
    model = build_model_for_eval(config, args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype


def build_model_clip_for_eval(config, pretrained_weights, only_backbone):
    model, _ = build_clip_model_from_cfg(config, only_teacher=True)

    adapter = bulid_adapter_dr(config.student.adapter, only_teacher=True)
    gauss_pool = build_hom_pool3(config, only_teacher=True)
    
   
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    if not only_backbone:
        dinov2_utils.load_pretrained_weights(adapter, pretrained_weights, "teacher")
        dinov2_utils.load_pretrained_weights(gauss_pool, pretrained_weights, "teacher")

        
    model.eval()
    model.cuda()
    adapter.eval()
    adapter.cuda()

    gauss_pool.eval()
    gauss_pool.cuda()
    
    
    return model, adapter, gauss_pool


def setup_and_build_model_clip(args, only_backbone) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args)
    # return config
    model, adapter, gauss_pool = build_model_clip_for_eval(config, args.pretrained_weights, only_backbone)
    autocast_dtype = get_autocast_dtype(config)
    return model, adapter, gauss_pool, autocast_dtype


@torch.inference_mode()
def evaluate_with_clip_v4_logits(
    visual_logits,
    visual_text_logits,
    targets,
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    fused_weight = 1.0
):
    assert criterion is None

    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    fused = fused_weight * visual_logits + visual_text_logits
    # fused = fused_weight * (visual_logits / visual_logits.norm(dim=-1, keepdim=True) )+ (1-fused_weight) * (visual_text_logits / visual_text_logits.norm(dim=-1, keepdim=True))
    # fused = fused_weight * F.softmax(visual_logits) + F.softmax(visual_text_logits)


    outputs = {
        "visual": visual_logits,
        "text": visual_text_logits,
        "fused": fused
    }

    # if len(metrics.keys()) > 1:
    for k, metric in metrics.items():
        
        metric_inputs = {
                "preds": outputs[k],
                "target": targets,
            }
        metric.update(**metric_inputs)


    metric_logger.synchronize_between_processes()
    # logger.info(f"Alpha: {alpha}")
    logger.info(f"Weight: {fused_weight}, Averaged stats: {metric_logger}")
    # import pdb; pdb.set_trace()

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


