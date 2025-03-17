import sys
import time
import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional
from tensorboardX import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

from dinov2.data import SamplerType, make_data_loader
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model_clip
from dinov2.logging import MetricLogger

from dinov2.data.datasets import *
from finetune_with_text_utils import ImageClassifierWithCLIP, ClassificationHead
logger = logging.getLogger("imgainefsl")

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
    'ucf101_midframes': '/home/Dataset/CLIP_Dataset/UCF101',
    'sun397': '/home/Dataset/CLIP_Dataset/SUN397',
}

syn_dataset_root = {
    'imagenet': '/SSD2T/synthetic0302/imagenet/view1',
    'food101': '/SSD2T/synthetic1005/food101',
    'flowers': '/SSD2T/synthetic1005/flowers',
    'cars':  '/SSD2T/synthetic1005/cars',
    'dtd': '/SSD2T/synthetic1005/dtd',
    'pets': '/SSD2T/synthetic1005/pets',
    'aircraft': '/SSD2T/synthetic1005/aircraft',
    'caltech101': '/SSD2T/synthetic1005/caltech101',
    'eurosat': '/SSD2T/synthetic1005/eurosat',
    'ucf101_midframes': '/SSD2T/synthetic1005/ucf101_frames',
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
        default=""
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
    )
    parser.add_argument(
        "--text_path",
        type=str,
    )

    parser.add_argument(
        "--normalize",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--num_shots",
        help="Number of shots",
        default=1,
    )
    parser.add_argument(
        '--weight_v',
        type=float,
        default=1.0,
        help="Parameter for visual loss weight, visual-text weight would be (1 - weight_v)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--ra",
        type=bool,
        default=False,
        help="Use Rand Augmentation",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--clip-path",
        dest="clip_path",
        type=str,
        help="Clip model (Vit-B/16) path",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="imagenet"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="syn"
    )
    parser.add_argument(
        "--template-type",
        dest="template_type",
        default="simple+cafo",
        type=str,
    )
    parser.add_argument(
        '--range-params',
        nargs="+",
        default=[0.0, 1.02, 0.02],
        type=float,
        help="Parameters for weight range, should be three floats: start, end, and step.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
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
        default=[1e-4, 1e-4, 1e-4, 1e-4],
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--is_test",
        type=str,
        help="Whether to test",
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
        "--num_workers",
        type=int,
        help="Number of workers for data loading",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for clip logits",
    )
    parser.add_argument(
        "--fuse_weight",
        type=float,
        help="Weight for fusion",
    )
    parser.add_argument(
        "--eval_only",
        type=bool,
        default=False,
        help="Whether to only evaluate",
    )

    parser.set_defaults(
        batch_size=256,
        num_workers=16,
        epoch_length=1 * 1000 / 128,
        save_checkpoint_frequency=12500,
        eval_period_iterations=12500,
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
        
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = x_tokens_list
        return self.linear(output)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }

def setup_classifiers(out_dim, lr, num_classes=1000):
    """
    Setup linear classifiers for evaluation
    """
    optim_param_groups = []
    
    linear_classifier = LinearClassifier(
        out_dim, num_classes=num_classes
    )
    
    linear_classifier = linear_classifier.cuda()
    
    optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    return linear_classifier, optim_param_groups


@torch.no_grad()
def evaluate_classifiers(
    clip_model,
    feature_model,
    test_loader,
    classifier_head,
    temperature,
    linear_classifiers,
    data_loader,
    metric_type,
    training_num_classes,
    class_mapping=None,
):
    logger.info("running validation !")

    num_classes = len(class_mapping) if class_mapping is not None else training_num_classes
    metric = build_metric(metric_type, num_classes=num_classes)
    postprocessors = LinearPostprocessor(linear_classifiers, class_mapping)
    metrics = {linear_classifiers: metric.clone() }

    start, end, step = args.range_params
    test_fuse_weight_range = torch.arange(start, end, step)

    test_acc, clip_acc, visual_acc = evaluate_with_text(
        clip_model,
        feature_model,
        test_loader,
        test_fuse_weight_range,
        classifier_head,
        temperature,
        data_loader,
        postprocessors,
        metrics,
        torch.cuda.current_device()
    )

    return test_acc, clip_acc, visual_acc


def evaluate_with_text(
    clip_model,
    feature_model: nn.Module,
    test_loader,
    test_fuse_weight,
    classifier_head,
    temperature,
    data_loader,
    postprocessors,
    metrics,
    device: torch.device
):
    clip_model.eval()
    feature_model.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    best_acc = 0.0
    best_weight = 0.0

    for weight in test_fuse_weight:
        weight = weight.item() if hasattr(weight, 'item') else weight
        clip_num = 0
        dino_num = 0

        for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
            features = clip_model(samples.to(device))
            outputs = feature_model(features)
            targets = targets.to(device)

            clip_logit = classifier_head(features[:, 0, :]) * torch.exp(temperature)

            metric_inputs = postprocessors(outputs, targets)

            dino_num += (torch.argmax(metric_inputs['preds'], dim=-1) == targets).sum()

            fused_outputs = (weight * metric_inputs['preds'] / metric_inputs['preds'].norm(dim=1, keepdim=True) * 100
                             + (1 - weight) * clip_logit / clip_logit.norm(dim=1, keepdim=True) * 100)


            metric_inputs['preds'] = fused_outputs

            metric.update(**metric_inputs)
            clip_num += (torch.argmax(clip_logit, dim=-1) == targets).sum()

        clip_acc = clip_num / len(data_loader.dataset)
        dino_acc = dino_num / len(data_loader.dataset)
        acc = [metric.compute() for k, metric in metrics.items()][0]['top-1']
        print(f"clip_acc: {clip_acc}, visual_acc: {dino_acc}, weight: {weight}, fuse acc: {acc}")
        
        metric.reset()
        
        if acc >= best_acc:
            best_acc = acc
            best_weight = weight
        else:
            break
        test_acc = best_acc
        
    if args.is_test == "True":
        clip_num = 0
        dino_num = 0
        for samples, targets, *_ in metric_logger.log_every(test_loader, 10, header):
            features = clip_model(samples.to(device))
            outputs = feature_model(features)
            targets = targets.to(device)
            clip_logit = classifier_head(features[:, 0, :]) * torch.exp(temperature)

            metric_inputs = postprocessors(outputs, targets)
            dino_num += (torch.argmax(metric_inputs['preds'], dim=-1) == targets).sum()
            fused_outputs = (best_weight * metric_inputs['preds'] / metric_inputs['preds'].norm(dim=1, keepdim=True)
                                + (1 - best_weight) * clip_logit / clip_logit.norm(dim=1, keepdim=True))


            metric_inputs['preds'] = fused_outputs

            metric.update(**metric_inputs)
            clip_num += (torch.argmax(clip_logit, dim=-1) == targets).sum()

        test_acc = [metric.compute() for k, metric in metrics.items()][0]['top-1']
        test_acc = test_acc.item() * 100
        clip_acc = clip_num / len(test_loader.dataset) * 100
        dino_acc = dino_num / len(test_loader.dataset) * 100

    print(f"BEST FUSE WEIGHT: {best_weight}")
    return test_acc, clip_acc, dino_acc


def fine_tune(
    *,
    clip_model,
    classifier_head,
    temperature,
    feature_model,
    linear_classifiers,
    train_data_loader,
    val_data_loader,
    test_loader,
    optimizer,
    scheduler,
    output_dir,
    max_iter,
    checkpoint_period,  
    metric_type,
    training_num_classes,
    category,
    val_class_mapping=None,
):
    iteration = 0
    if not args.eval_only:
        writer = SummaryWriter(f'{args.output_dir}/runs/{category}')

        feature_model.train()
        linear_classifiers.train()
        classifier_head.train()
        
        best_acc = 0.0
        checkpointer_adapter = Checkpointer(feature_model, output_dir, optimizer=optimizer, scheduler=scheduler)
        checkpointer_linear = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
        start_iter = 0
        periodic_checkpointer_adapter = PeriodicCheckpointer(checkpointer_adapter, checkpoint_period, max_iter=max_iter)
        periodic_checkpointer_linear = PeriodicCheckpointer(checkpointer_linear, checkpoint_period, max_iter=max_iter)
        iteration = start_iter
        logger.info("Starting training from iteration {}".format(start_iter))
        metric_logger = MetricLogger(delimiter="  ")
        header = "Training"
        scaler = torch.cuda.amp.GradScaler()

        pre_data_loader_time = time.time()
        data_loader_time = 0
        train_time = 0
        
        for real_data, syn_data in metric_logger.log_every(
            train_data_loader,
            100,
            header,
            max_iter,
            start_iter,
        ):

            post_data_loader_time = time.time()
            data_loader_time += post_data_loader_time - pre_data_loader_time
            print(
                f"Time to load data: {(post_data_loader_time - pre_data_loader_time) / 60} min, total load time: {data_loader_time / 60} min")
            

            real_x, real_y = real_data
            syn_x, syn_y = syn_data
            real_x = real_x.cuda(non_blocking=True)
            real_y = real_y.cuda(non_blocking=True)
            syn_x = syn_x.cuda(non_blocking=True)
            syn_y = syn_y.cuda(non_blocking=True)

            print(real_x.shape, real_y.shape, syn_x.shape, syn_y.shape)


            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    real_features = clip_model(real_x)
                real_clip_logit = classifier_head(real_features) * torch.exp(temperature)
                real_features = feature_model(real_features)
                real_outputs = linear_classifiers(real_features)

                with torch.no_grad():
                    syn_features = clip_model(syn_x)
                syn_clip_logit = classifier_head(syn_features) * torch.exp(temperature)
                syn_features = feature_model(syn_features)
                syn_outputs = linear_classifiers(syn_features)


            real_visual_loss = nn.CrossEntropyLoss()(real_outputs, real_y)
            real_clip_loss = nn.CrossEntropyLoss()(real_clip_logit[:, 0, :], real_y)
            real_loss = real_visual_loss + args.weight_v * real_clip_loss

            syn_visual_loss = nn.CrossEntropyLoss()(syn_outputs, syn_y)
            syn_clip_loss = nn.CrossEntropyLoss()(syn_clip_logit[:, 0, :], syn_y)
            syn_loss = syn_visual_loss + args.weight_v * syn_clip_loss

            loss = 0.8 * real_loss + 0.2 * syn_loss

            post_train_time = time.time()
            train_time += post_train_time - post_data_loader_time
            print(
                f"Time to train an epoch: {(post_train_time - post_data_loader_time) / 60} min, total train time: {train_time / 60} min")

            writer.add_scalars('Loss',
                            {
                                'Loss/visual': real_visual_loss,
                                'Loss/fuse': real_loss,
                                'Loss/text': real_clip_loss,
                                'Loss/syn_visual': syn_visual_loss,
                                'Loss/syn_fuse': syn_loss,
                                'Loss/syn_text': syn_clip_loss,
                                'Loss/real_syn_text': loss,
                                }, iteration)

            # compute the gradients
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # step
            scheduler.step()

            # log
            if iteration % 10 == 0:
                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
            periodic_checkpointer_adapter.step(iteration)
            periodic_checkpointer_linear.step(iteration)
            iteration = iteration + 1

            pre_data_loader_time = time.time()


        print(f"Train time: {train_time / 60} min; load data: {data_loader_time / 60} min")
    
    if args.eval_only:
        temperature = torch.tensor(args.temperature, device='cuda')
        fuse_weight = torch.tensor(args.fuse_weight, device='cuda')
        
    feature_model.eval()
    linear_classifiers.eval()
    test_acc, clip_acc, visual_acc = evaluate_classifiers(
                clip_model=clip_model,
                feature_model=remove_ddp_wrapper(feature_model),
                classifier_head=classifier_head,
                test_loader=test_loader,
                temperature=temperature,
                linear_classifiers=remove_ddp_wrapper(linear_classifiers),
                data_loader=val_data_loader,
                metrics_file_path=metrics_file_path,
                prefixstring=f"ITER: {iteration}",
                metric_type=metric_type,
                training_num_classes=training_num_classes,
                iteration=iteration,
                class_mapping=val_class_mapping,
                fuse_weight=fuse_weight,
            )
    print(f"FINAL_ACCURACY:{test_acc},{clip_acc},{visual_acc},{temperature.item()}")
    print(f"Temperature: {temperature}")

    if not args.eval_only:
        torch.save(remove_ddp_wrapper(feature_model).state_dict(), os.path.join(output_dir, f"{category}_{args.num_shots}shot_adapter.pth"))
        torch.save(remove_ddp_wrapper(linear_classifiers).state_dict(), os.path.join(output_dir, f"{category}_{args.num_shots}shot_linear.pth"))
        torch.save(remove_ddp_wrapper(classifier_head).state_dict(),
                os.path.join(output_dir, f"{category}_{args.num_shots}shot_text.pth"))



def make_eval_data_loader(val_dataset, batch_size, num_workers, metric_type):
    test_dataset = val_dataset
    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # sampler_type=SamplerType.DISTRIBUTED,
        sampler_type=None,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=_pad_and_collate if metric_type == MetricType.IMAGENET_REAL_ACCURACY else None,
    )
    return test_data_loader


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


def make_dataset(category, num_shots):
    
    root = real_dataset_root[category]

    dataset = FewShotDataset(root, num_shots=num_shots)
    training_num_classes = dataset.num_classes
    real_train_dataset = dataset.train
    val_dataset = dataset.val
    test_dataset = dataset.test
    real_classnames = dataset.class_name
    
    if args.phase == "syn":
        root = syn_dataset_root[category]
        dataset = SynDataset(root)
        syn_train_dataset = dataset.train
        syn_classnames = dataset.class_name
        assert training_num_classes == dataset.num_classes, "Real and synthetic dataset should have the same number of classes"
        assert real_train_dataset.classes == syn_train_dataset.classes, "Real and synthetic dataset should have the same classes"

        train_dataset = ConcatRepeatDataset([real_train_dataset, syn_train_dataset])
    
    return train_dataset, val_dataset, test_dataset, training_num_classes, real_classnames


def run_eval_linear(
    clip_model, adapter,
    output_dir,
    num_shots,
    batch_size,
    epochs,
    epoch_length,
    num_workers,
    save_checkpoint_frequency,
    eval_period_iterations,
    learning_rates,
    val_metric_type=MetricType.MEAN_ACCURACY,
):
    seed = 0

    train_dataset, val_dataset, test_dataset, training_num_classes, classnames = make_dataset(category=args.category,
                                                                                  num_shots=int(num_shots))
  
    clip_text_model = ImageClassifierWithCLIP(args, classnames, device='cuda')
    args.epoch_length = len(train_dataset) / args.batch_size
    
    # sampler_type = SamplerType.DISTRIBUTED
    sampler_type = SamplerType.INFINITE
    
    feature_model = adapter
    feature_model.eval()
    clip_model.eval()
    
    
    adapter_lr, linear_lr, temp_lr, text_lr = learning_rates
    
    linear_classifiers, optim_param_groups = setup_classifiers(
        2048,
        linear_lr,
        training_num_classes,
    )
    
    if args.classifier_path:
        state_dict = torch.load(args.classifier_path)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        linear_classifiers.load_state_dict(state_dict)
    
    if args.adapter_path:
        state_dict = torch.load(args.adapter_path)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        feature_model.load_state_dict(state_dict)
        
    optim_param_groups.append({"params": feature_model.parameters(), "lr": adapter_lr})

    temperature = nn.Parameter(torch.tensor(4.6052, device='cuda'), requires_grad=True)
    optim_param_groups.append({"params": temperature, "lr": temp_lr})

    if args.text_path:
        state_dict = torch.load(args.text_path)
    else:
        state_dict = None
    zeroshot_weight = clip_text_model.zeroshot_weights #* torch.exp(temperature)

    classifier_head = ClassificationHead(state_dict=state_dict, weights=zeroshot_weight)
    classifier_head.to(torch.cuda.current_device())


    optim_param_groups.append({"params": classifier_head.parameters(), "lr": text_lr})

    ### TODO: 5e-4?? 1e-4??
    optimizer = torch.optim.AdamW(optim_param_groups, weight_decay=0.0001)
    max_iter = epochs * args.epoch_length
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    start_iter = 0
    
    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        sampler_type=sampler_type,
        sampler_advance=start_iter,
        drop_last=True,
        persistent_workers=True,
    )
    
    val_data_loader = make_eval_data_loader(val_dataset, 128, num_workers, val_metric_type)
    test_loader = make_eval_data_loader(test_dataset, 128, num_workers, val_metric_type)
    
    checkpoint_period = save_checkpoint_frequency * epoch_length
    
    fine_tune(
        clip_model=clip_model,
        clip_text_model=clip_text_model,
        classifier_head=classifier_head,
        temperature=temperature,
        feature_model=feature_model,
        linear_classifiers=linear_classifiers,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        max_iter=max_iter,
        checkpoint_period=checkpoint_period,
        metric_type=val_metric_type,
        training_num_classes=training_num_classes,
        category=args.category
    )




class Adapter(nn.Module):
    def __init__(self, adapter, gauss_pool):
        super().__init__()
        self.adapter = adapter
        self.gauss_pool = gauss_pool

    def forward(self, x):
        x = self.adapter(x, is_training=True)
        cls_tokens = x["x_norm_clstoken"] # [2BS, 512] 
        patch_tokens = x["x_norm_patchtokens"] # [2BS x 196 x 512]
        x = torch.cat((cls_tokens.unsqueeze(1), patch_tokens), dim=1) # [2BS x 197 x 512]
        x = self.gauss_pool(x)
        return x

    
def main(args):
    model, adapter, gauss_pool, autocast_dtype= setup_and_build_model_clip(args, only_backbone=False)
    
    adapter = Adapter(adapter, gauss_pool)
    
    run_eval_linear(
        clip_model=model,
        adapter=adapter,
        output_dir=args.output_dir,
        num_shots=args.num_shots,
        root=args.root,
        category=args.category,
        batch_size=args.batch_size,
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        num_workers=args.num_workers,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        eval_period_iterations=args.eval_period_iterations,
        learning_rates=args.learning_rates,
    )
    return 0


if __name__ == "__main__":
    description = "DINOv2+CLIP Finetune evaluation"
    args_parser = get_args_parser(description=description)
    start_time = time.time()

    args = args_parser.parse_args()

    sys.exit(main(args))
