## Class-Level Confidence Based 3D Semi-Supervised Learning
This is a Pytorch implementation of Detectaion part of Class-Level Confidence Based 3D Semi-Supervised Learning.

Paper link: https://arxiv.org/abs/2210.10138



## Installation

Preparation: A Ubuntu system with GPU.

Install Nvidia driver and CUDA Toolkit.
```
$ nvidia-smi  # check driver
$ nvcc --version # check toolkit
```

Install `Python` -- This repo is tested with Python 3.7.6.

Install `NumPy` -- This repo is tested with NumPy 1.18.5. Please make sure your NumPy version is at least 1.18.

Install `PyTorch` with `CUDA` -- This repo is tested with 
PyTorch 1.5.1, CUDA 10.1. It may work with newer versions, 
but that is not guaranteed. A lower version may be problematic.
```
pip install torch==1.5.1 torchvision==0.6.1
```
or
```
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
```

Install `TensorFlow` (for `TensorBoard`) -- This repo is tested with TensorFlow 2.2.0.

Compile the CUDA code for [PointNet++](https://arxiv.org/abs/1706.02413), which is used in the backbone network:
```
cd pointnet2
python setup.py install
```

If there is a problem, please refer to [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch#building-only-the-cuda-kernels)

Compile the CUDA code for general 3D IoU calculation in [OpenPCDet](https://github.com/open-mmlab/OpenPCDet):
```
cd OpenPCDet
python setup.py develop
```

I deleted the CUDA kernels except 3D IoU calculation in OpenPCDet
for faster installation.

Install dependencies:
```
pip install -r requirements.txt
```

## Installation on Windows
Preparation: GPU support CUDA 10.2, not use GeForce 30 series with Ampere micro-architecture, anaconda is recommended.
```
> nvidia-smi # check nvidia driver
> nvcc --version # check CUDA Toolkit version
> cl # check MSVC compiler, if no, add compile to environment path
```

1. create a virtual environment: `conda create --name iou python=3.7`
2. install `numpy`: `conda install numpy -y`
3. install `pytorch` and `cudatoolkit`: `conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch`
4. degrade setuptools: `pip install setuptools=59.6`
5. install `pointnet2`: `cd pointnet2; python ./setup.py install`
6. install `OpenPCDet`: `cd OpenPCDet; python ./setup.py develop`
7. install `tensorflow`: `pip install tensorflow` 
8. install `requirements`: `pip install -r ./requirements.txt`

## Datasets

### ScanNet
Please follow the instructions in `scannet/README.md`. using the download script with 
`-o $(pwd) --types _vh_clean_2.ply .aggregation.json _vh_clean_2.0.010000.segs.json .txt` options to download data. 
### SUNRGB-D
Please follow the instructions in `sunrgbd/README.md`. 

## Pre-training without resample

Please run:
```shell script
sh run_pretrain_ori.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST>
```

For example:
```shell script
sh run_pretrain_ori.sh 0 pretrain_scannet scannet scannetv2_train_0.1.txt
``` 

```shell script
sh run_pretrain_ori.sh 0 pretrain_sunrgbd sunrgbd sunrgbd_v1_train_0.05.txt
``` 

## Pre-training with resample (please use the pretrained model obtained above for training)

Please run:
```shell script
sh run_pretrain.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST>
```

For example:
```shell script
sh run_pretrain.sh 0 pretrain_scannet scannet scannetv2_train_0.1.txt
``` 

```shell script
sh run_pretrain.sh 0 pretrain_sunrgbd sunrgbd sunrgbd_v1_train_0.05.txt


## Training

Please run:
```shell script
sh run_train.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST> <PRETRAIN_CKPT>
```

For example, use the downloaded models:
```shell script
sh run_train.sh 0 train_scannet scannet scannetv2_train_0.1.txt ckpts/scan_0.1_pretrain.tar
``` 

```shell script
sh run_train.sh 0 train_sunrgbd sunrgbd sunrgbd_v1_train_0.05.txt ckpts/sun_0.05_pretrain.tar
``` 
You may modify the script by adding `--view_stats`  to load labels on unlabeled data and view the statistics on the unlabeled data (e.g. average IoU, class prediction accuracy).


## Evaluation

Please run:
```shell script
sh run_eval.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST> <CKPT>
```

For example, use the downloaded models:
```shell script
sh run_eval.sh 0 eval_scannet scannet scannetv2_train_0.1.txt ckpts/scan_0.1.tar
``` 

```shell script
sh run_eval.sh 0 eval_sunrgbd sunrgbd sunrgbd_v1_train_0.05.txt ckpts/sun_0.05.tar
