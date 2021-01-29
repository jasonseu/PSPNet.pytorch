# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-24
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import yaml
import numpy as np
import argparse
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from models.pspnet import PSPNet
from lib import transform
from lib.util import *
from lib.dataset import SegmentationDataset


DIST_BACKEND = 'nccl'
DIST_ADDR = '127.0.0.1'
DIST_PORT = '12345'
DIST_INIT_METHOD = 'tcp://{}:{}'.format(DIST_ADDR, DIST_PORT)

class Evaluator(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.model = PSPNet(layers=args.layers, classes=args.num_classes, zoom_factor=args.zoom_factor)
        self.model.cuda()
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[0])
        
        mean = [item * 255 for item in args.mean]
        std = [item * 255 for item in args.std]
        val_transform = transform.Compose([
            transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
        ])
        val_dataset = SegmentationDataset(args, split='val', data_path=args.val_path, transform=val_transform)
        val_sampler = DistributedSampler(val_dataset)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)

        self.loss_meter = AverageMeter()
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.target_meter = AverageMeter()

    def run(self):
        model_path = self.args.ckpt_best_path if self.args.ckpt_path == '' else self.args.ckpt_path
        print("=> loading checkpoint '{}'".format(model_path))
        model_dict = torch.load(model_path)
        self.model.load_state_dict(model_dict, strict=True)
        print("=> loaded checkpoint '{}'".format(model_path))
        self.validate()

    def validate(self):
        self.model.eval()
        self.loss_meter.reset()
        self.intersection_meter.reset()
        self.union_meter.reset()
        self.target_meter.reset()
        with torch.no_grad():
            for _, batch in enumerate(self.val_loader):
                input = batch[0].cuda(non_blocking=True)
                target = batch[1].cuda(non_blocking=True)
                output = self.model(input)
                if self.args.zoom_factor != 8:
                    output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
                loss = self.criterion(output, target)

                batch_size = input.size(0)
                loss *= batch_size
                count = target.new_tensor([batch_size], dtype=torch.long)
                dist.all_reduce(loss), dist.all_reduce(count)
                loss = loss / count.item()
                self.loss_meter.update(loss.cpu().detach().item())

                output = torch.argmax(output, dim=1)
                intersection, union, target = intersectionAndUnionGPU(output, target, self.args.num_classes, self.args.ignore_label)
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
                self.intersection_meter.update(intersection.cpu().numpy())
                self.union_meter.update(union.cpu().numpy())
                self.target_meter.update(target.cpu().numpy())

        loss = self.loss_meter.compute()
        iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
        accuracy_class = self.intersection_meter.sum / (self.target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        aAcc = sum(self.intersection_meter.sum) / (sum(self.target_meter.sum) + 1e-10)
        if dist.get_rank() == 0:
            print(f'Validation result: mIoU {mIoU:.4f} mAcc {mAcc:.4f} aAcc {aAcc:.4f} Loss {loss:.4f}')

def main_worker(local_rank, gpu_list, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list[local_rank]
    dist.init_process_group(backend=DIST_BACKEND, init_method=DIST_INIT_METHOD, world_size=len(gpu_list), rank=local_rank)
    evaluator = Evaluator(args)
    evaluator.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='configs/cityscapes_pspnet101.yaml')
    parser.add_argument('--ckpt_path', type=str, default='')
    args = parser.parse_args()
    yaml.add_constructor('!join', lambda loader, node: os.path.join(*loader.construct_sequence(node)))
    with open(args.config, 'r') as fr:
        cfg = yaml.load(fr)
    cfg['ckpt_path'] = args.ckpt_path
    args = Namespace(**cfg)
    print(args)
    
    gpu_list = list(map(str, range(torch.cuda.device_count())))
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'].split(',')

    mp.spawn(main_worker, args=(gpu_list, args,), nprocs=len(gpu_list))