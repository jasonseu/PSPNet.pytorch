# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-24
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import yaml
import random
import numpy as np
import argparse
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.pspnet import PSPNet
from lib import transform
from lib.util import *
from lib.dataset import SegmentationDataset


class Evaluator(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label, reduction='none')
        self.model = PSPNet(layers=args.layers, classes=args.num_classes, zoom_factor=args.zoom_factor)
        self.model.cuda()
        
        mean = [item * 255 for item in args.mean]
        std = [item * 255 for item in args.std]
        val_transform = transform.Compose([
            transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
        ])
        val_dataset = SegmentationDataset(args, split='val', data_path=args.val_path, transform=val_transform)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        self.loss_meter = AverageMeter()
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.target_meter = AverageMeter()

    def run(self):
        print("=> loading checkpoint '{}'".format(self.args.ckpt_best_path))
        model_dict = torch.load(self.args.ckpt_best_path)
        if next(iter(model_dict.keys())).startswith('module'):
            model_dict = {k[7:]:v for k, v in model_dict.items()}
        self.model.load_state_dict(model_dict, strict=True)
        print("=> loaded checkpoint '{}'".format(self.args.ckpt_best_path))
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

                self.loss_meter.update(loss.cpu().detach().item())

                output = torch.argmax(output, dim=1)
                intersection, union, target = intersectionAndUnionGPU(output, target, self.args.num_classes, self.args.ignore_label)
                self.intersection_meter.update(intersection.cpu().numpy())
                self.union_meter.update(union.cpu().numpy())
                self.target_meter.update(target.cpu().numpy())

        loss = self.loss_meter.compute()
        iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
        accuracy_class = self.intersection_meter.sum / (self.target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        aAcc = sum(self.intersection_meter.sum) / (sum(self.target_meter.sum) + 1e-10)
        print(f'Validation result: mIoU {mIoU:.4f} mAcc {mAcc:.4f} aAcc {aAcc:.4f} Loss {loss:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='configs/cityscapes_pspnet101.yaml')
    parser.add_argument('--ckpt_path', type=str, default='')
    args = parser.parse_args()
    yaml.add_constructor('!join', lambda loader, node: os.path.join(*loader.construct_sequence(node)))
    with open(args.config, 'r') as fr:
        cfg = yaml.load(fr)
    if args.ckpt_path is not None:
        cfg['ckpt_best_path'] = args.ckpt_path
    cfg['has_prediction'] = args.has_prediction
    args = Namespace(**cfg)
    print(args)

    evaluator = Evaluator(args)
    evaluator.run()