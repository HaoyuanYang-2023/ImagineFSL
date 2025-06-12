<h2 align="center">ImagineFSL: Self-Supervised Pretraining Matters on Imagined Base Set for VLM-based Few-shot Learning</h2>

<!-- <h3 align="center">Haoyuan Yang &nbsp;&nbsp; Xiaoou Li &nbsp;&nbsp; Jiaming Lv &nbsp;&nbsp; Xianjun Cheng &nbsp;&nbsp; Qilong Wang &nbsp;&nbsp; Peihua Li</h3> -->

<!-- <h5 align="center">Dalian University of Technology &nbsp; Beijing University of Posts and Telecommunications &nbsp; Tianjin University</h5> -->

<h3 align="center">
  <img src="imgs/subtitle.png" width="70%"> 
  <!-- &nbsp; CVPR 2025 (Highlight)</span> -->
</h3>

<h3 align="center">
    <a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_ImagineFSL_Self-Supervised_Pretraining_Matters_on_Imagined_Base_Set_for_VLM-based_CVPR_2025_paper.pdf">[Paper]</a> •
    <a href="http://peihuali.org/ImagineFSL">[Project]</a>
</h3>

<div align="center"><img src="imgs/overview.gif" width="70%"></div>

## Introduction


This repository contains the official code for **"ImagineFSL: Self-Supervised Pretraining Matters on Imagined Base Set for VLM-based Few-shot Learning"** &nbsp;&nbsp;
(<img src="imgs/award1.png" width="10" style="vertical-align: middle;"> **CVPR 2025 Highlight**)

In this paper:


- We frame synthetic images as **standalone knowledge
repositories** and present **a CLIP adaptation methodology** that pretrains on purely synthetic images before fine-tuning for few-shot tasks. This marks a clear departure
from existing one-stage fine-tuning methods that simply
treat synthetic images as complements to real images.

- We propose **an improved Self-SL method based on
DINO**, specifically tailored for FSL. It introduces higher-order moments for image representation and employs
synthetic augmentation for effective view construction.

- We develop **a systematic and scalable pipeline for synthesizing both captions and images**, enabling generation
of large-scale base sets for pretraining and task-specific
datasets. Distinct from existing arts, **we leverage chain-of-though and in-conetext learning techniques** for diverse, realistic image generation.


## Installation

### 1. Clone this repository:

```
git clone https://github.com/HaoyuanYang-2023/ImagineFSL.git
cd ImagineFSL
```

### 2. Install dependencies:

> ⚠️ To ensure stable and reproducible code execution, we strongly recommend setting up the following environment for experiments.

We conduct experiments using PyTorch 2.2.2 and Python 3.10. The CUDA version is 12.1. Install the corresponding PyTorch environment using:

```
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies using:

```
pip install -r requirements.txt
```

**Note**: We use Meta's xformers library to accelerate Attention computation. Different hardware environments may require different versions of xformers. The installation command is provided in `requirements.txt`, which is validated on RTX 4090 and 3090. If installation fails, try different versions. For more information, refer to the [`offical website of xformers`](https://github.com/facebookresearch/xformers).


## Dataset

- **iBase Dataset**:
  
  The iBase dataset used for pretraining can be downloaded from the following links:
  
  [`Baidu Yun`](https://pan.baidu.com/s/1-a4oFKiPFdD_QRGJqAN9jA?pwd=r9ur) | [`Microsoft OneDrive`](https://maildluteducn-my.sharepoint.com/:u:/g/personal/yanghaoyuan_mail_dlut_edu_cn/EW67Bo9jyf5LtfQRjpFCnucB8wnoL3kPfCno4nGNSB5YHA?e=DFflEj)

- **10 Downstream Datasets (Real Images)**:

  We provide the following download links for the 10 downstream datasets used in our experiments (except ImageNet).
  *These datasets are identical to those provided by [`CoOp`](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) but with standardized file organization for PyTorch compatibility*.
  
   [`Baidu Yun`](https://pan.baidu.com/s/17k-xFrEtBdwh8taFtPcpwg?pwd=7fm5) | [`Microsoft OneDrive`](https://maildluteducn-my.sharepoint.com/:f:/g/personal/yanghaoyuan_mail_dlut_edu_cn/Emd9FEmohW9Mkj392CH8zjcBVnz1BmYTGNZHEIX3gjhdpg?e=WnChnu)



## Getting started

### 1. Synthesizing Captions & Images

Run the following command to get into the directory of synthesizing captions and images:

```
cd synthesizing
```

<h3> Querying GPT-4 to Analyze Key Factors</h3>

We query GPT-4 to analyze key factors for different datasets. You need to register an account on [`OpenAI`](https://platform.openai.com/docs/overview) and obtain an api key for GPT-4. For more details, refer to the [`OpenAI API documentation`](https://platform.openai.com/docs/quickstart).


Run the following command to analyze `attribute`:

```
python syn_attribute.py \
--api_key YOUR API_KEY \
--model gpt-4 \
--dataset DATASET_NAME \ 
``` 

Run the following command to analyze `background(BG)`:

```
python syn_background.py \
--api_key YOUR API_KEY \
--model gpt-4 \
--dataset DATASET_NAME \ 
```

We also provide the factors of `viewpoint`, `lighting condition (LC)` and `cause of
degradation of photos (CD)` in the `synthesizing/utils` folder.



<h3> Synthesize Examples Captions by GPT-4</h3>

Run the following command to high-quality exemplary captions for different datasets:

```
python syn_examples.py \
--api_key YOUR_API_KEY \
--model gpt-4 \
--dataset DATASET_NAME \ 
``` 

<h3> Synthesize Extensive Captions by Llama 3</h3>

We use Llama 3 8B to synthesize extensive captions. The weight files of Llama 3 8B can be downloaded [`here`](https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main/original).

You need to install additional dependencies required for Llama 3:

```
fire==0.3.0
fairscale==0.4.13
tiktoken==0.7.0
blobfile==0.3.0
```

Run the following command to synthesize extensive captions for different datasets:

```
LLAMA_FOLDER=YOUR_LLAMA_WEIGHT_FILE_FOLDER

torchrun --nproc_per_node 1 --master_port 12388 \
    syn_captions.py --ckpt_dir ${LLAMA_FOLDER} --tokenizer_path ${LLAMA_FOLDER}/tokenizer.model \
    --max_batch_size 16 --max_seq_len 400 --max_gen_len 100 \
    --total_captions 300 --seed 0 --category DATASET_NAME --temperature 0.8
```

<h3> Synthesize Images </h3>

We use Stable Diffusion 3 Medium accelerated by TensorRT to synthesize images. Refer to the [`code provided by NVIDIA`](https://github.com/NVIDIA/TensorRT/tree/release/10.8/demo/Diffusion) for details.

----
### 2. Pretraining  

Run the following command for pretraining:

```
sh run_pretrain.sh
```

You need to specify the hyperparameters for pretraining in the config files in the `dinov2/config/train` folder.


We provide download links for the pretrained model weights of CLIP ViT-B/16 and CLIP ViT-L/14:

- CLIP ViT-B/16: [`Baidu Yun`](https://pan.baidu.com/s/1i3txwvzpx9yJeGj8AFTZrw?pwd=app9) | [`Google Drive`](https://drive.google.com/file/d/1-5s6-RpLLTzbrZXbHyfPWSuLqQ3Qq4WG/view?usp=drive_link)

- CLIP ViT-L/14: [`Baidu Yun`](https://pan.baidu.com/s/1Ac8UqmtKnR9wFVhXeTjrbg?pwd=pp64) | [`Google Drive`](https://drive.google.com/file/d/1-5Zi1BrIPQW_iCwVvJ7wXIS1OlfWjShT/view?usp=drive_link)
   
----

### 3. Few-shot Fine-tuning

**ImagineFSL**:

Run the following command for ImagenFSL fine-tuning:

```
sh run_imaginefsl.sh
```
You need to set the hyperparameters for fine-tuning in `dinov2/eval/imgainefsl_tuning_pipline.py` folder and the dataset path in the `dinov2/eval/ct_tuning_mixing.py` first.

For evaluation, run the following command:

```
sh run_imaginefsl_eval.sh
```

**ImagineFSL_LoRA**:

Run the following command for ImagenFSL_LoRA fine-tuning:
```
sh run_imaginefsl_lora.sh
```
You need to set the hyperparameters for fine-tuning in `dinov2/eval/imgainefsl_lora_tuning_pipline.py` folder and the dataset path in the `dinov2/eval/ct_tuning_mixing_lora.py` first.


For evaluation, run the following command:

```
sh run_imaginefsl_lora_eval.sh
```

> **Note:** Due to the impact of randomness during training, the results on individual datasets may slightly differ from those in the paper. We recommend evaluating all methods across all 11 datasets and observing the average performance.

**Models**:

We provide download links for fine-tuned models on 1-/16-shot settings for ViT-B/16 across 11 datasets:

|Method|1-shot|16-shot|
|:-|:-:|:-:|
|ImagineFSL| 76.1 \| [`Baidu Yun`](https://pan.baidu.com/s/1Jpu45g3S3VizXuoz9_NMzQ?pwd=r5eq) \| [`Google Drive`](https://drive.google.com/drive/folders/1-6-kHsgYmXJAwbBn0HV4A2nYkd2bBgva?usp=drive_link) | 86.4 \| [`Baidu Yun`](https://pan.baidu.com/s/1JMLmzoJ8AqKRyV_ONv9vVg?pwd=5i5f) \| [ `Google Drive`](https://drive.google.com/drive/folders/1-5MMNE69OQKCcAMqwCpVeX0hyfn7kt6W?usp=drive_link) |
|ImagineFSL_LoRA|77.6 \| [`Baidu Yun`](https://pan.baidu.com/s/11P61q63LVbxxiX3ZsORpVA?pwd=a9md) \| [`Google Drive`](https://drive.google.com/drive/folders/10I_kpcFId7JQgByAgKasMeWsm7GaYq95?usp=sharing) | 87.6 \| [`Baidu Yun`]( https://pan.baidu.com/s/15XSGTHI_vF1sjCMgfELJAg?pwd=13ev) \| [`Google Drive`](https://drive.google.com/drive/folders/11nlNzFf4anzZH8Bqhfia0uriOXRYuEtU?usp=sharing)|
|||

> ##### *See `readme.txt` in the above links for more details of the models and hyperparameters for inference.*

**Detailed results of All K-shot settings can be found in [`here`](https://github.com/HaoyuanYang-2023/ImagineFSL/tree/main/results).**


## Acknowledgement

- We thank the authors of CLIP and DINOv2. This repository is built upon the official implementations of [`CLIP`](https://github.com/openai/CLIP) and [`DINOv2`](https://github.com/facebookresearch/dinov2).

- We are also grateful to the authors of CoOp for providing [`dataset instructions`](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md), DISEF for [`their codebase`](https://github.com/vturrisi/disef/tree/main), and SynCLR for [`their codebase`](https://github.com/google-research/syn-rep-learn/tree/main/SynCLR).

- We further acknowledge the contributions of other researchers who have made their code publicly available.


## Citation

If this repository or the paper "ImagineFSL: Self-Supervised Pretraining Matters on Imagined Base Set for VLM-based Few-shot Learning" is helpful for your research, please consider citing the paper:

```BibTeX
@InProceedings{ImagineFSL_CVPR25,
    author    = {Yang, Haoyuan and Li, Xiaoou and Lv, Jiaming and Cheng, Xianjun and Wang, Qilong and Li, Peihua},
    title     = {{ImagineFSL}: Self-Supervised Pretraining Matters on Imagined Base Set for {VLM}-based Few-shot Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2025},
}
```

## Contact

If you have any questions or suggestions, please contact us:

- Haoyuan Yang (yanghaoyuan@mail.dlut.edu.cn)
<!-- - Xiaoou Li (xiaoouli@bupt.edu.cn)
- Jiaming Lv (ljm_vlg@mail.dlut.edu.cn) -->
