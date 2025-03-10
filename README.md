# Advancing Medical Image Segmentation via Self-supervised Instance-adaptive Prototype Learning (IJCAI 2024)

The repo contains PyTorch implementation of paper **Advancing Medical Image Segmentation via Self-supervised Instance-adaptive Prototype Learning**.

## Architecture overview of SIPL
<img src="framework.png" width = "600" height = "345" alt="" align=center />


## Installation
### 1. Create and activate conda environment
```bash
conda create --name SIPL python=3.10
conda activate SIPL
```
### 2. Install PyTorch and torchvision
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install 'monai[all]'
```
### 3.Install other dependencies
```bash
pip install -r requirements.txt
```

## Dataset
### BTCV  
1. Download the [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) dataset according to the dataset link and
set it into ./data/. 
2. Arrange the dataset according to the `dataset/dataset_list/PAOT.txt`.  
3. The following script can be used for generating one-hot label.
```bash
python -W ignore label_transfer.py
``` 

## Training and Evaluation
### 1. Training 
```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py --data_root_path ./data/ --num_samples 4 --cache_dataset --cache_rate 0.005 --uniform_sample
```
### 2. Evaluation
For BTCV, Paste the pre-trained model (e.g. epoch_500.pth) into the following path:
```
./out/BTCV_model/epoch_**.pth
```
Then, run
```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --resume ./out/epoch_***.pth --data_root_path /data/ --store_result --cache_dataset --cache_rate 0.005
``` 

## Acknowledgment
* [CLIP-Driven](https://github.com/ljwztc/CLIP-Driven-Universal-Model)
* [Mask2Former](https://bowenc0221.github.io/mask2former)
* [kMaX-DeepLab](https://github.com/bytedance/kmax-deeplab)

## Citation
If you find this repository useful, please consider citing this paper:
```
@inproceedings{ijcai2024p117,
  title     = {Advancing Medical Image Segmentation via Self-supervised Instance-adaptive Prototype Learning},
  author    = {Liang, Guoyan and Zhou, Qin and Chen, Jingyuan and Wang, Zhe and Yao, Chang},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24},
  pages     = {1056--1064},
  year      = {2024},
}
```



