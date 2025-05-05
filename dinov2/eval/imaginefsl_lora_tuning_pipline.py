import os
import subprocess
import numpy as np
import json
import time
import sys
import argparse


def get_args_parser(
    description=None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        add_help=add_help,
    )
    parser.add_argument("--pretrain-weights", dest="pretrain_weights", type=str, default="HoM_4views.pth")
    parser.add_argument('--exp_name', type=str, default='exp', help='exp_name')

    
    
    return parser


def run_phase_tuning(category, lora_lr_list, adapter_lr_list, vis_ratio, text_ratio,
                     lora_r_list, lora_scale_list, rlw_list, drp_list, epochs, weight_decay,
                     code_file_path, pretrain_weights, num_shots=None,batch_size=64):

    start_time = time.time()
    data = []
    
    ckpt_path = "ckpts"
    max_patient_counter = 2

    if not num_shots:
        num_shots = "0"

    json_file_path = f"./dinov2/eval/mix_train_lora/{args.exp_name}/{category}/{category}_{num_shots}_shot.json"
    os.makedirs(f"./dinov2/eval/mix_train_lora/{args.exp_name}/{category}", exist_ok=True)
    output_dir = f"./dinov2/eval/mix_train_lora/{ckpt_path}/{args.exp_name}"

    if not os.path.exists(json_file_path):
        with open(json_file_path, 'w') as f:
            json.dump(data, f)

    print(f"Training with {num_shots} shots")
    best_acc = 0
    for epoch in epochs:
        for lr1 in lora_lr_list:
            for lr2 in adapter_lr_list:
                for lora_r in lora_r_list:
                    for lora_scale in lora_scale_list:
                        for rlw in rlw_list:
                            for drp in drp_list:
                                for wd in weight_decay:
                                    
                                    lr3 = lr2 * vis_ratio
                                    lr4 = lr2 * text_ratio

                                    result = subprocess.run([
                                        "python",
                                        f'dinov2/eval/{code_file_path}',
                                        "--learning-rates",
                                        str(lr1), str(lr2), str(lr3), str(lr4),
                                        "--epochs",
                                        str(epoch),
                                        "--num_shots",
                                        str(num_shots),
                                        "--batch-size", str(batch_size),
                                        "--config-file",
                                        "dinov2/configs/eval/clip_b16.yaml",
                                        "--pretrained-weights", pretrain_weights,
                                        "--clip-path",
                                        "clip/ViT-B-16.pt",
                                        "--output-dir", output_dir,  
                                        "--category", str(category),
                                        "--is_test", "False",
                                        "--lora_r", str(lora_r),
                                        "--lora_scale", str(lora_scale),
                                        "--lora_start_block", str(0),
                                        "--real_loss_weight", str(rlw),
                                        "--visual_classifier_dropout", str(drp),
                                        "--weight-decay", str(wd),
                                        "--is_synth",
                                        ],
                                        capture_output=True,
                                        text=True
                                    )
                                    
                                    if result.returncode != 0:
                                        print(category)
                                        print(result.stderr)
                                        accuracy = 'error'

                                    accuracy, record = analyze_and_record(result, json_file_path, num_shots, lrs = [lr1, lr2, lr3, lr4], epoch=epoch, lora_r=lora_r, lora_scale=lora_scale, rlw=rlw,drp=drp, wd=wd, batch_size = batch_size)

                                    if accuracy > best_acc:
                                        best_acc = accuracy
                                        best_param = record
                                    
                                    curr_time = time.time() - start_time
                                    curr_time = curr_time / 3600
                                    print(f"Time: {curr_time:.2f} hours")
                                    print(f"lr1: {lr1}, lr2: {lr2}, lr3: {lr3}, lr4: {lr4}, lora_r: {lora_r}, lora_scale: {lora_scale}, acc: {accuracy}")  
                                

    test_result = subprocess.run([
        "python",
        f'dinov2/eval/{code_file_path}',
        "--learning-rates",
        best_param["lora_lr"], best_param["adapter_lr"], best_param["visual_lr"], best_param["text_lr"],
        "--epochs",
        best_param['epoch'],
        "--num_shots",
        str(num_shots),
        "--batch-size", str(batch_size),
        "--config-file",
        "dinov2/configs/eval/clip_b16.yaml",
        "--pretrained-weights", pretrain_weights,
        "--clip-path",
        "clip/ViT-B-16.pt",
        "--output-dir", output_dir,  
        "--category", str(category),
        "--is_test", "True",
        "--lora_r", best_param["lora_r"],
        "--lora_scale", best_param["lora_scale"],
        "--lora_start_block", str(0),
        "--real_loss_weight", best_param["rlw"],
        "--visual_classifier_dropout", str(drp),
        "--weight-decay", best_param["wd"],
        "--is_synth",
        ],
        capture_output=True,
        text=True
    )

    if test_result.returncode != 0:
        print(category)
        print(test_result.stderr)
        test_accuracy = 'error'
        return
    
    test_file_path = f"./dinov2/eval/mix_train_lora/{args.exp_name}/{category}/test_results.json"
    os.makedirs(f"./dinov2/eval/mix_train_lora/{args.exp_name}/{category}", exist_ok=True)

    test_accuracy, record = analyze_and_record(test_result, test_file_path, num_shots, best_param=best_param)

    end_time = time.time()
    interval = (end_time - start_time) / 3600
    print(f"{category} Final Acc: {test_accuracy}")
    print(f"{category} training time: {interval:.2f} hours")
    return interval


def analyze_and_record(result, json_file_path, shot, lrs=None, epoch=None, lora_r=None, lora_scale=None, rlw=None, best_param=None, drp=None, wd=None, batch_size=None):
    if lrs:
        lr1, lr2, lr3, lr4 = lrs
    if not os.path.exists(json_file_path):
        with open(json_file_path, 'w') as f:
            json.dump([], f)

    if result.returncode != 0:
        print(category)
        print(result.stderr)
        accuracy = 'error'
        return None, None

    output_lines = result.stdout.splitlines()
    for line in output_lines:
        if line.startswith("FINAL_ACCURACY:"):
            accuracy = float(line.split("FINAL_ACCURACY:")[1].split(",")[0].strip())
            clip_acc = float(line.split("FINAL_ACCURACY:")[1].split(",")[1].strip())
            visual_acc = float(line.split("FINAL_ACCURACY:")[1].split(",")[2].strip())
            accuracy = max(accuracy, clip_acc, visual_acc)
        if line.startswith("BEST FUSE WEIGHT:"):
            best_fuse_weight = float(line.split("BEST FUSE WEIGHT:")[1].strip())
        
    if best_param:
        lr1 = best_param["lora_lr"]
        lr2 = best_param["adapter_lr"]
        lr3 = best_param["visual_lr"]
        lr4 = best_param["text_lr"]
        lora_r = best_param["lora_r"]
        lora_scale = best_param["lora_scale"]
        rlw = best_param["rlw"]
        epoch = best_param["epoch"]
        drp = best_param["drp"]
        wd = best_param["wd"]
        batch_size = best_param["batch_size"]
        
    record = {
        "lora_lr": str(lr1),
        "adapter_lr": str(lr2),
        "visual_lr": str(lr3),
        "text_lr": str(lr4),
        "epoch": str(epoch),
        "accuracy": str(accuracy),
        "clip acc": str(clip_acc),
        "visual acc": str(visual_acc),
        "lora_scale": str(lora_scale),
        "lora_r": str(lora_r),
        "shot": str(shot),
        "rlw": str(rlw),
        "wd": str(wd),
        "drp": str(drp),
        "batch_size": str(batch_size),
        "fuse_weight": str(best_fuse_weight),
    }

    has_data = False
    if os.path.exists(json_file_path):
        has_data = True
    with open(json_file_path, 'r+') as f:
        if has_data == True:
            data = json.load(f)
        data.append(record)
        f.seek(0)
        json.dump(data, f, indent=4)

    return accuracy, record


def run(category, args, num_shot=1):
    interval = 0
    num_shot = str(num_shot)
    
    
    visual_lr_list = [1e-6, 1e-5, 1e-4, 1e-3]
    adapter_lr_list = [1e-6, 1e-5, 1e-4, 1e-3]
    
    
    lora_r_list = [64, 32, 16]
    lora_scale_list = [64, 32]
    
    text_ratio = 1.0
    vis_ratio = 10.0
    rlw_list = [0.8]
    epochs = [10]
    drp_list = [0.0]
    batch_size = 64
    wd = [1e-4]
    
    code_file_name = "ct_lora_tuning_mixing.py"
    interval += run_phase_tuning(category,
                                visual_lr_list, adapter_lr_list, vis_ratio, text_ratio,
                                lora_r_list, lora_scale_list, rlw_list, drp_list,epochs, wd,
                                code_file_name, args.pretrain_weights,
                                 num_shots=num_shot, batch_size=batch_size)

    return interval


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    
    description = "ImagineFSL LoRA Fine-tuning"
    args_parser = get_args_parser(description=description)

    args = args_parser.parse_args()

    total_time = 0

    num_shots = [1,2,4,8,16]
    dataset = ['imagenet', 'caltech101','aircraft','cars',
               'food101','pets','flowers','dtd','eurosat', 
               'sun397', 'ucf101_frames']

    print(f"train {dataset}")
    for category in dataset:
        for num_shot in num_shots:
            print(f"train {dataset}, {num_shot} shots")
            interval = run(category, args, num_shot)

            total_time += interval

    print(f"train {dataset},\ntotal time: {total_time} hours")
