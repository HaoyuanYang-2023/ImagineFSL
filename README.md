<h1 align="center">ImagineFSL: Self-Supervised Pretraining Matters on Imagined Base Set for VLM-based Few-shot Learning</h1>

<h3 align="center">Haoyuan Yang &nbsp;&nbsp; Xiaoou Li &nbsp;&nbsp; Jiaming Lv &nbsp;&nbsp; Xianjun Cheng &nbsp;&nbsp; Qilong Wang &nbsp;&nbsp; Peihua Li</h3>


<h3 align="center">CVPR 2025</h3>

<h4 align="center">
    <a href="https://arxiv.org/abs/2412.08139">[Paper]</a> •
    <a href="http://peihuali.org/ImagineFSL">[Project]</a>
</h4>

<div align="center"><img src="figures/overview.png" width="90%"></div>

## Introduction

In this paper:

- We frame synthetic images as standalone knowledge repositories and present **a CLIP adaptation methodology** that pretrains on purely synthetic images before fine-tuning for few-shot tasks.
- We propose **an improved Self-SL method** based on DINO. It introduces higher-order moments for image representation and employs synthetic augmentation for effective view construction.
- We develop **a systematic and scalable pipeline** for synthesizing both captions and images, enabling generation of large-scale base sets for pretraining and task-specific datasets.

## Dataset

- iBase Dataset:
  
  The iBase dataset used for pretraining can be downloaded from [here](https://).

- 10 Datasets (Real Images):

  We provide [download links](https://) for the 10 datasets used in our experiments (except ImageNet). These datasets are identical to those provided by [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) but with standardized file organization for PyTorch compatibility.

## Installation

All our experiments are conducted on a PC with an Intel Core i9-13900K CPU and GeForce RTX 4090 GPUs.

### 1. Clone this repo:

```
git clone https://github.com/ImagineFSL/ImagineFSL.git
cd ImagineFSL
```

### 2. Environments:

We conduct experiments using PyTorch 2.2.2 and Python 3.10. The CUDA used is 12.1

Install the corresponding PyTorch version using:

```
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies using:

```
pip install -r requirements.txt
```

**Note**: We use Facebook's xformers to accelerate Attention computation. Different hardware environments may require different versions. We provide the xformers installation command in requirements.txt, but successful installation is not guaranteed (verified working on RTX 4090 and 3090). If installation fails, try different versions.

## Getting started

### 1. Synthesizing Captions & Images

#### Require Factors by GPT

Run the following command to synthesize Factors:

```
python synthesizing/synthesize_factors.py --model gpt-4o --api_key YOUR_API_KEY --category DATASET_NAME
```

You need to register an account on [OpenAI](https://platform.openai.com/docs/guides/authentication) and obtain an API_KEY.

#### Synthesize Examples

运行如下命令合成Examples:

```
python synthesizing/synthesize_examples.py --model gpt-4o --api_key YOUR_API_KEY --category DATASET_NAME
```

#### Synthesize Captions

我们使用Llama 3合成文本，Llama 3 的权重文件可以在[这里](http://)下载。

需要额外安装Llama3推理所需的依赖:

```
fire==0.3.0
fairscale==0.4.13
tiktoken==0.7.0
blobfile==0.3.0
tqdm==4.66.5
```

运行如下命令：

```
python synthesizing/synthesize_captions.py
```

#### Synthesize Images

我们使用TensorrtRT加速的SD3合成图像，具体代码参考[NVIDA 提供的示例](https://github.com/NVIDIA/TensorRT/tree/release/10.8/demo/Diffusion)

### 2. Pretraining  

使用以下命令进行预训练：

```
sh run.sh
```

配置文件：dinov2/config/train/clip_b16.yaml

注意：您需要在配置文件中指定预训练的数据集路径。

我们提供了预训练模型的下载链接：

- CLIP-ViT-B/16: [https://](https://)
- CLIP-ViT-L/14: [https://](https://)

### 3. Few-shot Fine-tuning

**ImagineFSL**:

```
sh run_ct.sh
```

**ImagineFSL$_\text{LoRA}$**:

```
sh run_ct.sh
```

我们提供了11个数据集的微调模型的下载链接：

- ImagineFSL: [https://](https://)
- ImagineFSL$_\text{LoRA}$: Coming soon.

**Note that the results is slightly different to the results in paper due to randomness in traning.**

**We recommend evaluated all methods and models across 11 datasets.**

### 4. Evaluation

Code for evalution only is coming soon...

## Acknowledgement

- Thanks for CoOp (Dataset), DINOv2 (Pretraining), DISEF (Fine-tuning), SynCLR (Synthesizing Text)
- Thanks also go to authors of other papers who make their code publicly available.

## Citation

If this repo is helpful for your research, please consider citing the paper:

```BibTeX
@InProceedings{ImagineFSL_CVPR25,
    author    = {Yang, Haoyuan and Li, Xiaoou and Lv, Jiaming and Cheng, Xianjun and Wang, Qilong and Li, Peihua},
    title     = {ImagineFSL: Self-Supervised Pretraining Matters on Imagined Base Set for VLM-based Few-shot Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2025},
}
```

## Contact

If you have any questions or suggestions, please contact us:

- Haoyuan Yang (yanghaoyuan@mail.dlut.edu.cn)
- Xiaoou Li (xiaoouli@bupt.edu.cn)
- Jiaming Lv (ljm_vlg@mail.dlut.edu.cn)
