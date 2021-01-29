# PSPNet

This repository is a simplified version of official pytorch implement [semseg](https://github.com/hszhao/semseg), where multi machines distributed training function is removed and single-machine-multi-gpu training is maintained. Training on ADE20K, PSAVAL VOC 2012 and Cityscapes is supported.

## Usage

### 1. Requriments

- PyTorch >= 1.1.0
- Python 3.6

### 2. Train

- Download needed datasets and symlink the paths to them as follows:
```bash
cd PSPNet.pytorch
mkdir data
ln -s /path/to/ADEChallengeData2016 data/ADEChallengeData2016
ln -s /path/to/Cityscapes data/Cityscapes
ln -s /path/to/VOC2012 data/VOC2012
```

- Run the corresponding scipts under the fold `scripts` to generate required intermediate data.

- Download [pretrained models](https://drive.google.com/drive/folders/15wx9vOM0euyizq-M1uINgN0_wjVRf9J3) on ImageNet and put them under the fold `initmodel` for weights initialization.

- Train model in single-machine-single-gpu mode:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/cityscapes_pspnet101.yaml
```

- Train model with single-machine-multi-gpu mode:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_dist.py --config configs/cityscapes_pspnet101.yaml
```
Note that use flag `-r` for resuming training. Pretrained models are provided on [google drive](https://drive.google.com/drive/folders/1EnMH50ZkGHbw9acXdwNp5-IhQ6OKOhaA?usp=sharing). Besides, one can use official pretrained models to initialize the network and reproduce official provided results. Official provided pretrained models can be visited [here](https://drive.google.com/drive/folders/15wx9vOM0euyizq-M1uINgN0_wjVRf9J3).

## Performance

- **PASCAL VOC 2012**: train on **train_aug** (10582 images) set and test on **val** (1449 images) set.

| metrics  |  mIOU  |  mAcc  |  aAcc  |
| -------- | ------ | ------ | ------ |
| personal | 0.7801 | 0.8561 | 0.9498 |
| official | 0.7907 | 0.8636 | 0.9534 |

- **ADE20K**: train on ade challenge 2016 **train** (20210 images) set and test on **val** (2000 images) set.

| metrics  |  mIOU  |  mAcc  |  aAcc  |
| -------- | ------ | ------ | ------ |
| personal | 0.4304 | 0.5378 | 0.7989 |
| official | 0.4310 | 0.5375 | 0.8107 |

- **Cityscapes**: train on **fine_train** (2975 images) set and test on **fine_val** (500 images) set.

| metrics  |  mIOU  |  mAcc  |  aAcc  |
| -------- | ------ | ------ | ------ |
| personal | 0.7188 | 0.8035 | 0.9597 |
| official | 0.7863 | 0.8577 | 0.9614 |

## Acknowledgement

Thanks the official awesome implement [semseg](https://github.com/hszhao/semseg).