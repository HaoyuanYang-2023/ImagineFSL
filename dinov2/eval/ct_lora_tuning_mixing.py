# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

from dinov2.data import SamplerType, make_data_loader
import dinov2.distributed as distributed
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model_clip
from dinov2.eval.utils import ModelWithIntermediateLayers, evaluate_with_clip, evaluate_with_clip_v4_logits
from dinov2.logging import MetricLogger
from dinov2.data.datasets import ImageNetFS, FewShotDataset, SynDataset
import copy
from dinov2.models.lora_v2 import lora_replace_attention_layers_vis
from dinov2.eval.ct.finetune_with_text_utils import ImageClassifierWithCLIP

logger = logging.getLogger("dinov2")
real_dataset_root = {
    'imagenet': '/home/Dataset/ImageNet',
    'food101': '/home/Dataset/CLIP_Dataset/Food101',
    'flowers': '/home/Dataset/CLIP_Dataset/Flowers',
    'cars': '/home/Dataset/CLIP_Dataset/Cars',
    'dtd': '/home/Dataset/CLIP_Dataset/DTD',
    'pets': '/home/Dataset/CLIP_Dataset/Pets',
    'aircraft': '/home/Dataset/CLIP_Dataset/Aircraft',
    'caltech101': '/home/Dataset/CLIP_Dataset/Caltech101',
    'eurosat': '/home/Dataset/CLIP_Dataset/EuroSAT',
    'ucf101': '/home/Dataset/CLIP_Dataset/UCF101',
    'sun397': '/home/Dataset/CLIP_Dataset/SUN397',
}

synth_dataset_root = {
    'imagenet': '/SSD2T/synthetic0302/imagenet/view1',
    'food101': '/SSD2T/synthetic1005/food101',
    'flowers': '/SSD2T/synthetic1005/flowers',
    'cars': '/SSD2T/synthetic1005/cars_new',
    'dtd': '/SSD2T/synthetic1005/dtd',
    'pets': '/SSD2T/synthetic1005/pets',
    'aircraft': '/SSD2T/synthetic1005/aircraft',
    'caltech101': '/SSD2T/synthetic1005/caltech101',
    'eurosat': '/SSD2T/synthetic1005/eurosat',
    'ucf101': '/SSD2T/synthetic1005/ucf101_frames/clean',
    'sun397': '/SSD2T/synthetic1005/sun397/clean',
}



def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Path to the root directory of the dataset",
        default="/media/yhy/YHYSSD1T/Dataset"
    )
    
    parser.add_argument(
        "--category",
        type=str,
        default="ImageNet"
    )
    
    parser.add_argument(
        "--num_shots",
        type=int,
        help="Number of shots",
        default=1,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-iterations",
        type=int,
        help="Number of iterations between two evaluations.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-4,
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--val-metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--test-metric-types",
        type=MetricType,
        choices=list(MetricType),
        nargs="+",
        help="Evaluation metric",
    )
    parser.add_argument(
        "--clip-path",
        dest="clip_path",
        type=str,
        help="Clip model (Vit-B/16) path",
    )
    parser.add_argument(
        "--is_synth",
        action="store_true"
    )
    parser.add_argument(
        "--visual_classifier_resume",
        type=str,
        default=''
    )
    parser.add_argument(
        "--text_classifier_resume",
        type=str,
        default=''
    )
    parser.add_argument(
        "--visual_adapter_resume",
        type=str,
        default=''
    )
    parser.add_argument(
        "--visual_model_resume",
        type=str,
        default=''
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16
    )
    parser.add_argument(
        "--lora_scale",
        type=int,
        default=32
    )
    parser.add_argument(
        "--real_loss_weight",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--synth_dataset_root",
        type=str,
        default=None
    )

    parser.add_argument(
        "--lora_start_block",
        type=int,
        default=0
    )
    parser.add_argument(
        "--visual_classifier_dropout",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--is_test",
        type=str,
        default="False"
    )
    
    parser.add_argument(
        "--template_type",
        type=str,
        default="cafo+simple"
    )

    parser.add_argument(
        "--fuse_weight",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--eval_only",
        type=bool,
        default=False
    )
    
    parser.set_defaults(
        epochs = 10,
        batch_size = 128,
        num_shots = 1,
        num_workers= 12,
        save_checkpoint_frequency=1e8,
        eval_period_iterations=1e8,
        learning_rates=[1e-4, 1e-1, 1e-1, 1e-1],
        val_metric_type=MetricType.MEAN_ACCURACY,
        test_metric_types=None,
    )
    return parser


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


def _pad_and_collate(batch):
    maxlen = max(len(targets) for image, targets in batch)
    padded_batch = [
        (image, np.pad(targets, (0, maxlen - len(targets)), constant_values=-1)) for image, targets in batch
    ]
    return torch.utils.data.default_collate(padded_batch)


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes, bias=False)

    def forward(self, x_tokens_list):
        output = x_tokens_list
        return self.linear(output)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": preds,
            "target": targets,
        }


def setup_classifiers(lr, out_dim, num_classes=1000, optim_param_groups=None):
    """
    Setup linear classifiers for evaluation
    """
    
    if optim_param_groups is None:
        optim_param_groups = []
    
    out_dim = out_dim
    linear_classifier = LinearClassifier(
        out_dim, num_classes=num_classes
    )

    # linear_classifier.linear.requires_grad_=False
    linear_classifier = linear_classifier.cuda()
    linear_classifier.eval()

    if lr <= 0:
        linear_classifier.eval()
        linear_classifier.linear.requires_grad_=False
    else:
        linear_classifier.train()
        optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    return linear_classifier, optim_param_groups


# 合并两个dataset
# 将数据量较小的dataset重复多次，使得两个dataset的数据量相等
class ConcatRepeatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.length = max(self.lengths)
        self.num_datasets = len(datasets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return [d[idx % self.lengths[i]] for i, d in enumerate(self.datasets)]


def make_dataset(args, category, num_shots, is_synth=False, is_mixing=False):

    root = real_dataset_root[category]
    dataset = FewShotDataset(root, num_shots=num_shots, args=args)
    training_num_classes = dataset.num_classes
    real_train_dataset = dataset.train
    val_dataset = dataset.val
    test_dataset = dataset.test
    class_name = dataset.class_name

    if is_synth:
        root = synth_dataset_root[category]
        syn_dataset = SynDataset(root, args=args)
        syn_training_num_classes = syn_dataset.num_classes
        syn_train_dataset = syn_dataset.train
        syn_class_name = syn_dataset.class_name

        assert syn_training_num_classes == training_num_classes
        assert syn_class_name == class_name

    if is_mixing and is_synth:
        train_dataset = ConcatRepeatDataset([real_train_dataset, syn_train_dataset]) # Concatenate the two datasets
    elif not is_mixing and is_synth:
        train_dataset = syn_train_dataset
    else:
        train_dataset = real_train_dataset

    return train_dataset, val_dataset, test_dataset, training_num_classes, class_name

@torch.no_grad()
def evaluate_classifiers(
    visual_model,
    visual_adapter,
    visual_classifier,
    text_classifier,
    data_loader,
    test_data_loader,
    metric_type,
    training_num_classes,
    prefixstring="",
    class_mapping=None,
    output_dir='.',
    category='',
):
    logger.info("running validation !")

    num_classes = len(class_mapping) if class_mapping is not None else training_num_classes
    
    visual_model.eval()
    visual_classifier.eval()
    text_classifier.eval()
    visual_adapter.eval()
    
    if not args.eval_only:
      
        visual_logits_list = []
        visual_text_logits_list = []
        targets_list = []
        
        for samples, targets in data_loader:
        
            samples = samples.to(torch.cuda.current_device())
            targets = targets.to(torch.cuda.current_device())

            visual_feat = visual_model(samples)
            adapter_feat = visual_adapter(visual_feat)
            visual_feat = visual_feat[:, 0] / visual_feat[:, 0].norm(dim=1, keepdim=True)

            visual_text_logits = text_classifier(visual_feat) / 0.01
            visual_logits = visual_classifier(adapter_feat)

            visual_logits_list.append(visual_logits)
            visual_text_logits_list.append(visual_text_logits)
            targets_list.append(targets)

        visual_logits = torch.cat(visual_logits_list, dim=0)
        visual_text_logits = torch.cat(visual_text_logits_list, dim=0)
        targets = torch.cat(targets_list, dim=0)

        fused_weight_range = torch.arange(0.0, 1.01, 0.01)
        
        best_acc_fuse = 0.0
        best_acc_visual = 0.0
        best_acc_text = 0.0
        
        best_fuse_weight = 0.0
        
        for fused_weight in fused_weight_range:

            metric = build_metric(metric_type, num_classes=num_classes)
            fused_metric = {'visual': metric.clone(), 'text': metric.clone(), 'fused': metric.clone()}
        
            _, results_dict_temp = evaluate_with_clip_v4_logits(
                visual_logits,
                visual_text_logits,
                targets,
                metrics=fused_metric,
                device=torch.cuda.current_device(),
                fused_weight=fused_weight
            )

            fuse_acc = results_dict_temp['fused']["top-1"].item()
            
            if fuse_acc > best_acc_fuse:
                best_acc_fuse = fuse_acc
                best_acc_visual = results_dict_temp['visual']["top-1"].item()
                best_acc_text = results_dict_temp['text']["top-1"].item()
                best_fuse_weight = fused_weight

        print(f"Val Acc: {best_acc_fuse}, {best_acc_text}, {best_acc_visual}")
        print(f"weight: {best_fuse_weight}")
    
    else:
        best_fuse_weight = args.fuse_weight
        
    if args.is_test == "True":
        
        visual_logits_list_test = []
        visual_text_logits_list_test = []
        targets_list_test = []
        
        metric = build_metric(metric_type, num_classes=num_classes)
        fused_metric = {'visual': metric.clone(), 'text': metric.clone(), 'fused': metric.clone()}

        for samples, targets in test_data_loader:
    
            samples = samples.to(torch.cuda.current_device())
            targets = targets.to(torch.cuda.current_device())

            visual_feat = visual_model(samples)
            adapter_feat = visual_adapter(visual_feat)
            visual_feat = visual_feat[:, 0] / visual_feat[:, 0].norm(dim=1, keepdim=True)

            visual_text_logits = text_classifier(visual_feat) / 0.01
            visual_logits = visual_classifier(adapter_feat)

            visual_logits_list_test.append(visual_logits)
            visual_text_logits_list_test.append(visual_text_logits)
            targets_list_test.append(targets)

        visual_logits_test = torch.cat(visual_logits_list_test, dim=0)
        visual_text_logits_test = torch.cat(visual_text_logits_list_test, dim=0)
        targets_test = torch.cat(targets_list_test, dim=0)
    
        _, results_dict_temp = evaluate_with_clip_v4_logits(
            visual_logits_test,
            visual_text_logits_test,
            targets_test,
            metrics=fused_metric,
            device=torch.cuda.current_device(),
            fused_weight=best_fuse_weight,
        )
        
        best_acc_fuse = results_dict_temp['fused']["top-1"].item()
        best_acc_visual = results_dict_temp['visual']["top-1"].item()
        best_acc_text = results_dict_temp['text']["top-1"].item()
    
    print(f"FINAL_ACCURACY: {best_acc_fuse}, {best_acc_text}, {best_acc_visual}")
    print(f"BEST FUSE WEIGHT: {best_fuse_weight}")
    if not args.eval_only:
        torch.save(remove_ddp_wrapper(visual_model).state_dict(), os.path.join(output_dir, f"{category}_{args.num_shots}shot_visual_model.pth"))
        torch.save(remove_ddp_wrapper(visual_adapter).state_dict(), os.path.join(output_dir, f"{category}_{args.num_shots}shot_visual_adapter.pth"))
        torch.save(remove_ddp_wrapper(visual_classifier).state_dict(), os.path.join(output_dir, f"{category}_{args.num_shots}shot_visual_classifier.pth"))
        torch.save(remove_ddp_wrapper(text_classifier).state_dict(),
                os.path.join(output_dir, f"{category}_{args.num_shots}shot_text_classifier.pth"))
    
    return best_acc_fuse

def fine_tune(
    *,
    visual_model,
    visual_adapter,
    visual_classifier,
    text_classifier,
    train_data_loader,
    val_data_loader,
    test_data_loader,
    optimizer,
    scheduler,
    output_dir,
    max_iter,
    checkpoint_period,  # In number of iter, creates a new file every period
    metric_type,
    training_num_classes,
    val_class_mapping=None,
    real_loss_weight=0.8,
    visual_classifier_dropout=0.0
):
    start_iter = 0
    iteration = start_iter
    if not args.eval_only:
        visual_adapter.train()

        visual_model.train()
        visual_classifier.train()
        text_classifier.train()

        if visual_classifier_dropout > 0:
            visual_dropout = nn.Dropout(visual_classifier_dropout)
        else:
            visual_dropout = None
        
        

        checkpointer_adapter = Checkpointer(visual_adapter, output_dir, optimizer=optimizer, scheduler=scheduler)
        periodic_checkpointer_adapter = PeriodicCheckpointer(checkpointer_adapter, checkpoint_period, max_iter=max_iter)


        checkpointer_visual_classifier = Checkpointer(visual_classifier, output_dir, optimizer=optimizer, scheduler=scheduler)
        periodic_checkpointer_visual_classifier = PeriodicCheckpointer(checkpointer_visual_classifier, checkpoint_period, max_iter=max_iter)
        
        checkpointer_text_classifier = Checkpointer(text_classifier, output_dir, optimizer=optimizer, scheduler=scheduler)
        periodic_checkpointer_text_classifier = PeriodicCheckpointer(checkpointer_text_classifier, checkpoint_period, max_iter=max_iter)

        
        logger.info("Starting training from iteration {}".format(start_iter))
        metric_logger = MetricLogger(delimiter="  ")
        header = "Training"
        scaler = torch.cuda.amp.GradScaler()
        for real_data, syn_data in metric_logger.log_every(
            train_data_loader,
            100,
            header,
            max_iter,
            start_iter,
        ):

            real_x, real_y = real_data
            syn_x, syn_y = syn_data
            real_x = real_x.cuda(non_blocking=True)
            real_y = real_y.cuda(non_blocking=True)
            syn_x = syn_x.cuda(non_blocking=True)
            syn_y = syn_y.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                image_feats = visual_model(real_x)
                    
                adapter_feats = visual_adapter(image_feats)

                image_feats = image_feats[:, 0] / image_feats[:, 0].norm(dim=1, keepdim=True)

                visual_text_logits = text_classifier(image_feats) / 0.01

                if visual_dropout:
                    adapter_feats = visual_dropout(adapter_feats)

                visual_logits = visual_classifier(adapter_feats)
                real_loss = {"real_vt_loss": nn.CrossEntropyLoss()(visual_text_logits, real_y)*real_loss_weight, "real_v_loss": nn.CrossEntropyLoss()(visual_logits, real_y)*real_loss_weight}


                image_feats = visual_model(syn_x)
                    
                adapter_feats = visual_adapter(image_feats)

                image_feats = image_feats[:, 0] / image_feats[:, 0].norm(dim=1, keepdim=True)

                visual_text_logits = text_classifier(image_feats) / 0.01

                if visual_dropout:
                    adapter_feats = visual_dropout(adapter_feats)

                visual_logits = visual_classifier(adapter_feats)
                syn_loss = {"syn_vt_loss": nn.CrossEntropyLoss()(visual_text_logits, syn_y)*(1-real_loss_weight), "syn_v_loss": nn.CrossEntropyLoss()(visual_logits, syn_y)*(1-real_loss_weight)}

            loss = sum(real_loss.values()) + sum(syn_loss.values())

            # compute the gradients
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # step
            scheduler.step()

            # log
            if iteration % 1 == 0:
                metric_logger.update(real_v_loss=real_loss['real_v_loss'].item())
                metric_logger.update(real_vt_loss=real_loss['real_vt_loss'].item())
                metric_logger.update(syn_v_loss=syn_loss['syn_v_loss'].item())
                metric_logger.update(syn_vt_loss=syn_loss['syn_vt_loss'].item())
                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # if visual_adapter:
            periodic_checkpointer_adapter.step(iteration)
            periodic_checkpointer_visual_classifier.step(iteration)
            periodic_checkpointer_text_classifier.step(iteration)


            iteration = iteration + 1


    visual_model.eval()
    acc  = evaluate_classifiers(
        visual_model=visual_model,
        visual_adapter=remove_ddp_wrapper(visual_adapter) if visual_adapter else None,
        visual_classifier=remove_ddp_wrapper(visual_classifier),
        text_classifier=remove_ddp_wrapper(text_classifier),
        data_loader=val_data_loader,
        test_data_loader=test_data_loader,
        prefixstring=f"ITER: {iteration}",
        metric_type=metric_type,
        training_num_classes=training_num_classes,
        class_mapping=val_class_mapping,
        output_dir=output_dir,
        category=args.category
    )

    


def make_eval_data_loader(val_dataset, batch_size, num_workers, metric_type):
    test_dataset = val_dataset
    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=_pad_and_collate if metric_type == MetricType.IMAGENET_REAL_ACCURACY else None,
    )
    return test_data_loader


def run_eval_linear(
    args,
    visual_model,
    visual_adapter,
    output_dir,
    category,
    num_shots,
    batch_size,
    epochs,
    num_workers,
    save_checkpoint_frequency,
    learning_rates,
    val_metric_type=MetricType.MEAN_ACCURACY,
    is_synth=False,
    weight_decay=1e-3,
    real_loss_weight=0.8,
    visual_classifier_dropout=0.0,
):
    seed = 0

    train_dataset, val_dataset, test_dataset, training_num_classes, class_name = make_dataset(args, category, num_shots, is_synth, is_mixing=True)
    
    sampler_type = SamplerType.INFINITE
    
    visual_model.cuda()
    visual_model.eval()
    
    clip_text_model = ImageClassifierWithCLIP(args, class_name, device='cuda')
    zeroshot_weight = clip_text_model.zeroshot_weights
    
    if visual_adapter:
        visual_adapter.eval()
        visual_adapter.cuda()

    if len(learning_rates) == 4:
        visual_lr, adapter_lr, linear_lr, text_lr = learning_rates
    else:
        visual_lr, linear_lr = learning_rates[0], learning_rates[1]
    

    optim_param_groups = []

    visual_classifier, optim_param_groups = setup_classifiers(
        linear_lr,
        out_dim=2048,
        num_classes=training_num_classes,
        optim_param_groups=optim_param_groups
    )

    text_classifier, optim_param_groups = setup_classifiers(
        text_lr,
        out_dim=512,
        num_classes=training_num_classes,
        optim_param_groups=optim_param_groups
    )
    
    assert text_classifier.linear.weight.data.shape == zeroshot_weight.T.shape
    text_classifier.linear.weight.data = copy.deepcopy(zeroshot_weight.T)
    
    if args.visual_classifier_resume:
        state_dict = torch.load(args.visual_classifier_resume)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        visual_classifier.load_state_dict(state_dict)
    
    if args.text_classifier_resume:
        state_dict = torch.load(args.text_classifier_resume)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        text_classifier.load_state_dict(state_dict)
    
    if args.visual_adapter_resume:
        state_dict = torch.load(args.visual_adapter_resume)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        visual_adapter.load_state_dict(state_dict)
    
    if args.visual_model_resume:
        state_dict = torch.load(args.visual_model_resume)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        visual_model.load_state_dict(state_dict)
        
    
    optim_param_groups.append({"params": visual_adapter.parameters(), "lr": adapter_lr})
    

    for name, p in visual_model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    optim_param_groups.append({"params": [p for p in visual_model.parameters() if p.requires_grad], "lr": visual_lr})

    optimizer = torch.optim.AdamW(optim_param_groups, weight_decay=weight_decay)
    
    models = {
        "visual_model": visual_model, 
        "visual_adapter": visual_adapter,  
        "visual_classifier": visual_classifier, 
        "text_classifier": text_classifier
    }

    total_params_in_millions = 0
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_in_millions += total_params / 1e6
        print(name, total_params/1e6)
    print(f"total has {total_params_in_millions:.2f}M trainable parameters")

    # ---- Note that epoch length is manually computed here ----
    epoch_length = len(train_dataset) // batch_size
    max_iter = epochs * epoch_length
    # ----------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=1e-7)
    start_iter = 0
    
    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        sampler_type=sampler_type,
        sampler_advance=start_iter,
        drop_last=False,
        persistent_workers=True,
    )
    
    val_data_loader = make_eval_data_loader(val_dataset, 128, num_workers, val_metric_type)

    test_data_loader = make_eval_data_loader(test_dataset, 128, num_workers, val_metric_type)
        

    checkpoint_period = save_checkpoint_frequency * epoch_length

    
    fine_tune(
        visual_model=visual_model,
        visual_adapter=visual_adapter,
        visual_classifier=visual_classifier,
        text_classifier=text_classifier,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        test_data_loader=test_data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        max_iter=max_iter,
        metric_type=val_metric_type,
        training_num_classes=training_num_classes,
        checkpoint_period=checkpoint_period,
        real_loss_weight=real_loss_weight,
        visual_classifier_dropout=visual_classifier_dropout
    )


class Adapter(nn.Module):
    def __init__(self, adapter, hom_pool, gauss_head=None):
        super().__init__()
        self.adapter = adapter
        self.hom_pool = hom_pool
        self.gauss_head = gauss_head
    def forward(self, x):
        x = self.adapter(x, is_training=True)
        cls_tokens = x["x_norm_clstoken"] # [2BS, 512] 
        patch_tokens = x["x_norm_patchtokens"] # [2BS x 196 x 512]
        x = torch.cat((cls_tokens.unsqueeze(1), patch_tokens), dim=1) # [2BS x 197 x 512]
        x = self.hom_pool(x)
        return x

def main(args):
    
    model, adapter, hom_pool, autocast_dtype= setup_and_build_model_clip(args, only_backbone=False)

    adapter = Adapter(adapter, hom_pool)

    model.transformer = lora_replace_attention_layers_vis(
            model.transformer,
            lora_r=args.lora_r,
            lora_alpha=args.lora_scale,
            lora_dropout=0.1,
            start_block=args.lora_start_block,
            q_lora=True,
            k_lora=True,
            v_lora=True,
            o_lora=True,
        )

    print(synth_dataset_root[args.category])
    run_eval_linear(
        args=args,
        visual_model=model,
        visual_adapter=adapter,
        output_dir=args.output_dir,
        num_shots = args.num_shots,
        category=args.category,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        learning_rates=args.learning_rates,
        is_synth=args.is_synth,
        weight_decay=args.weight_decay,
        real_loss_weight=args.real_loss_weight,
        visual_classifier_dropout=args.visual_classifier_dropout
    )
    return 0


if __name__ == "__main__":
    description = "DINOv2+CLIP Finetune evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    if args.synth_dataset_root:
        synth_dataset_root[args.category] = args.synth_dataset_root

    logger.info(args)
    sys.exit(main(args))