import os
import subprocess
import numpy as np
import json
import time
import argparse

parser = argparse.ArgumentParser(description='Fine-tuning')
parser.add_argument('--pretrain_weights', type=str, default='', help='pretrain_weights')
parser.add_argument('--exp_name', type=str, default='exp', help='exp_name')
args = parser.parse_args()

def run_phase_tuning(category, epoch_range, adapter_lr_list, text_ratio,
                     code_file_path, pretrain_weights, batch_size=64, num_shots=None):
    best_acc = 0
    start_time = time.time()
    data = []

    ckpt_path = "ckpts"
    max_patient_counter = 2


    json_file_path = f"./dinov2/eval/mix_train/{args.exp_name}/{category}/{category}_{num_shots}_shot.json"
    os.makedirs(f"./dinov2/eval/mix_train/{args.exp_name}/{category}", exist_ok=True)
    output_dir = f"./dinov2/eval/mix_train/{ckpt_path}/{args.exp_name}"

    weights = np.arange(1, 3, 2)
    if not os.path.exists(json_file_path):
        with open(json_file_path, 'w') as f:
            json.dump(data, f)

    print(f"Training with {num_shots} shots")
    for lr1 in adapter_lr_list:
        epochs = epoch_range

        lr2 = lr1 * 10
        lr4 = lr1 * text_ratio
        best_acc_lr = 0
        best_weight = 0.0
        for weight_v in weights:
            best_acc_epoch = 0.0
            patient_counter = 0
            for epoch in epochs:
                start_time_ = time.time()
                result = subprocess.run([
                    "python",
                    f'dinov2/eval/{code_file_path}',
                    "--learning-rates",
                    str(lr1), str(lr2), str(lr1), str(lr4),
                    "--epochs",
                    str(epoch),
                    "--num_shots",
                    str(num_shots),
                    "--num_workers", str(16),
                    "--batch-size", str(batch_size),
                    "--config-file",
                    "dinov2/configs/eval/clip_b16.yaml",
                    "--pretrained-weights", pretrain_weights,
                    "--clip-path",
                    "clip/ViT-B-16.pt",
                    "--output-dir", output_dir,                   
                    "--category", str(category),
                    "--phase", "syn",
                    "--range-params", "0.0", "1.0", "0.02",
                    "--template-type", "simple+cafo",
                    '--weight_v', str(weight_v)
                ],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(category)
                    print(result.stderr)
                    accuracy = 'error'

                accuracy, record = analyze_and_record(result, json_file_path, num_shots, lr1, lr2, lr4, epoch, weight_v, best_param=None)

                if accuracy > best_acc_epoch:
                    best_acc_epoch = accuracy
                    best_param_epoch = record
                    patient_counter = 0
                else:
                    patient_counter += 1

                if patient_counter >= max_patient_counter:
                    break

                if best_acc_epoch >= best_acc_lr:
                    best_acc_lr = best_acc_epoch
                    best_param_lr = best_param_epoch
                    print(f"{category}  weight: {best_param_lr['weight']}; acc: {best_param_lr['accuracy']}")
                    print(f"total time: {(time.time() - start_time) / 3600:.2f} hours")
                else:
                    print(f"{category}  weight: {best_param_lr['weight']}; acc: {best_param_lr['accuracy']}")
                    print(f"total time: {(time.time() - start_time) / 3600:.2f} hours")

            if best_acc_lr >= best_acc:
                best_acc = best_acc_lr
                best_param = best_param_lr
                print(
                    f"{category}  adapter lr: {best_param['adapter_lr']}; weight:{best_param['weight']}; acc: {best_param['accuracy']}")
                print(f"total time: {(time.time() - start_time) / 3600:.2f} hours")

    test_result = subprocess.run([
        "python",
        f'dinov2/eval/{code_file_path}',
        "--learning-rates",
        best_param["adapter_lr"], best_param["linear_lr"], best_param["adapter_lr"], best_param["text_lr"],
        "--epochs",
        best_param["epoch"],
        "--clip-path",
        "clip/ViT-B-16.pt",
        "--num_shots",
        str(num_shots),
        "--num_workers", str(16),
        "--batch-size", str(batch_size),
        "--config-file",
        "dinov2/configs/eval/clip_b16.yaml",
        "--pretrained-weights", pretrain_weights,
        "--output-dir", output_dir,
        "--category", str(category),
        "--phase", "syn",
        "--range-params", "0.0", "1.0", "0.02",
        "--template-type", "simple+cafo",
        '--weight_v', best_param['weight'],
        '--is_test', "True"
    ],
        capture_output=True,
        text=True
    )

    if test_result.returncode != 0:
        print(category)
        print(test_result.stderr)
        test_accuracy = 'error'
        return

    test_file_path = f"./dinov2/eval/mix_train/{args.exp_name}/{category}/test_results.json"
    os.makedirs(f"./dinov2/eval/mix_train/{args.exp_name}/{category}", exist_ok=True)
    test_accuracy, record = analyze_and_record(test_result, test_file_path, num_shots, best_param=best_param)

    end_time = time.time()
    interval = (start_time - end_time) / 3600
    print(f"{category} Final Acc: {test_accuracy}")
    print(f"{category} training time: {interval:.2f} hours")
    return interval

def analyze_and_record(result, json_file_path, shot, lr1=None, lr2=None, lr4=None, epoch=None, weight_v=None,
                       best_param=None):
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
            temp = float(line.split("FINAL_ACCURACY:")[1].split(",")[3].strip())
            break
        if line.startswith("BEST FUSE WEIGHT:"):
            best_fuse_weight = float(line.split("BEST FUSE WEIGHT:")[1].strip())

    if best_param:
        lr1 = best_param["adapter_lr"]
        lr2 = best_param["linear_lr"]
        lr4 = best_param["text_lr"]
        epoch = best_param["epoch"]
        weight_v = best_param['weight']

    record = {
        "adapter_lr": str(lr1),
        "linear_lr": str(lr2),
        "text_lr": str(lr4),
        "epoch": str(epoch),
        "accuracy": str(accuracy),
        "clip acc": str(clip_acc),
        "visual acc": str(visual_acc),
        "temperature": str(temp),
        "weight": str(weight_v),
        "shot": str(shot),
        "fuse": str(best_fuse_weight)
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


def run(category, num_shot=1):

    code_file_name = "ct_tuning_mixing.py"

    interval = 0
    num_shot = str(num_shot)
        
    adapter_lr_list = [1e-6, 1e-5, 1e-4, 1e-3]
    epoch_ranges = np.arange(10, 90, 10)

    text_ratio = 0.2
    print(epoch_ranges)
    print(adapter_lr_list)
    
    interval += run_phase_tuning(category, epoch_ranges, adapter_lr_list, text_ratio,
                                    code_file_name, args.pretrain_weights, num_shots=num_shot)

    return interval


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    
    description = "ImagineFSL Fine-tuning"

    total_time = 0

    num_shots = [1,2,4,8,16]
    dataset = ['imagenet', 'caltech101','aircraft','cars',
               'food101','pets','flowers','dtd','eurosat', 
               'sun397', 'ucf101_frames']

    print(f"train {dataset}")
    for category in dataset:
        for num_shot in num_shots:
            print(f"train {dataset}, {num_shot} shots")
            interval = run(category, num_shot)

            total_time += interval

    print(f"train {dataset},\ntotal time: {total_time} hours")
