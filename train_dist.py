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
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from models.pspnet import PSPNet
from lib import transform
from lib.util import *
from lib.dataset import SegmentationDataset


# manual_seed = 2020
# random.seed(manual_seed)
# np.random.seed(manual_seed)
# torch.manual_seed(manual_seed)
# torch.cuda.manual_seed(manual_seed)
# torch.cuda.manual_seed_all(manual_seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

DIST_BACKEND = 'nccl'
DIST_ADDR = '127.0.0.1'
DIST_PORT = '12345'
DIST_INIT_METHOD = 'tcp://{}:{}'.format(DIST_ADDR, DIST_PORT)

class Trainer(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.writer = SummaryWriter(args.log_dir)
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.model = PSPNet(layers=args.layers, classes=args.num_classes, zoom_factor=args.zoom_factor)
        modules_ori = [self.model.layer0, self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        modules_new = [self.model.ppm, self.model.cls, self.model.aux]
        params_list = []
        for module in modules_ori:
            params_list.append(dict(params=module.parameters(), lr=args.base_lr))
        for module in modules_new:
            params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
        self.optimizer = SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.index_split = 5
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, verbose=True)
        self.model.cuda()
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[0])
        
        mean = [item * 255 for item in args.mean]
        std = [item * 255 for item in args.std]
        train_transform = transform.Compose([
            transform.RandScale([args.scale_min, args.scale_max]),
            transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
        ])
        train_dataset = SegmentationDataset(args, split='train', data_path=args.train_path, transform=train_transform)
        self.train_sampler = DistributedSampler(train_dataset)
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=self.train_sampler, drop_last=True)

        val_transform = transform.Compose([
            transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
        ])
        val_dataset = SegmentationDataset(args, split='val', data_path=args.val_path, transform=val_transform)
        val_sampler = DistributedSampler(val_dataset)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)

        self.global_step = 0
        self.loss_meter = AverageMeter()
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.target_meter = AverageMeter()

    def run(self):
        s_epoch, best_score = 0, 0.0
        if self.args.resume:
            print("=> loading checkpoint '{}'".format(self.args.ckpt_latest_path))
            s_epoch, global_step, best_score, optim_dict, model_dict = load_checkpoint(self.args, is_distributed=True)
            self.global_step = global_step
            self.optimizer.load_state_dict(optim_dict)
            self.model.load_state_dict(model_dict, strict=True)
            del optim_dict
            del model_dict
            print("=> loaded checkpoint '{}'".format(self.args.ckpt_latest_path))
        for epoch in range(s_epoch, self.args.max_epochs):
            self.train(epoch)
            mIoU = self.validate(epoch)
            self.lr_scheduler.step(mIoU)
            if dist.get_rank() == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'global_step': self.global_step + 1,
                    'best_score': best_score,
                    'model_dict': self.model.state_dict(),
                    'optim_dict': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, self.args.ckpt_latest_path)
                if mIoU > best_score:
                    best_score = mIoU
                    torch.save(self.model.state_dict(), self.args.ckpt_best_path)

    def train(self, epoch):
        self.intersection_meter.reset()
        self.union_meter.reset()
        self.target_meter.reset()
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        max_iter = self.args.max_epochs * len(self.train_loader)
        for step, batch in enumerate(self.train_loader):
            input, target = batch[0].cuda(), batch[1].cuda()
            if self.args.zoom_factor != 8:
                h = int((target.size()[1] - 1) / 8 * self.args.zoom_factor + 1)
                w = int((target.size()[2] - 1) / 8 * self.args.zoom_factor + 1)
                # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
                target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()
            output, aux = self.model(input)
            main_loss = self.criterion(output, target)
            aux_loss = self.criterion(aux, target)
            loss = main_loss + self.args.aux_weight * aux_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            current_iter = epoch * len(self.train_loader) + step + 1
            current_lr = poly_learning_rate(self.args.base_lr, current_iter, max_iter, power=self.args.power)
            for idx in range(self.index_split):
                self.optimizer.param_groups[idx]['lr'] = current_lr
            for idx in range(self.index_split, len(self.optimizer.param_groups)):
                self.optimizer.param_groups[idx]['lr'] = current_lr * 10

            batch_size = input.size(0)
            main_loss, aux_loss, loss = [l * batch_size for l in [main_loss, aux_loss, loss]]
            count = target.new_tensor([batch_size], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss), dist.all_reduce(count)
            main_loss, aux_loss, loss = main_loss / count.item(), aux_loss / count.item(), loss / count.item()

            output = torch.argmax(output, dim=1)
            intersection, union, target = intersectionAndUnionGPU(output, target, self.args.num_classes, self.args.ignore_label)
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = [t.cpu().numpy() for t in [intersection, union, target]]
            self.intersection_meter.update(intersection)
            self.union_meter.update(union)
            self.target_meter.update(target)

            accuracy = sum(intersection) / (sum(target) + 1e-10)
            main_loss, aux_loss, loss = [l.cpu().detach().item() for l in [main_loss, aux_loss, loss]]
            if dist.get_rank() == 0 and self.global_step % self.args.print_freq == 0:
                print(f'Train Epoch: [{epoch}][{step}/{len(self.train_loader)}] MainLoss {main_loss:.4f} AuxLoss {aux_loss:4f} Loss {loss:.4f} Accuracy {accuracy:.4f}')
                self.writer.add_scalar('train/loss', loss, self.global_step)
                self.writer.add_scalar('train/mIoU', np.mean(intersection / (union + 1e-10)), self.global_step)
                self.writer.add_scalar('train/mAcc', np.mean(intersection / (target + 1e-10)), self.global_step)
                self.writer.add_scalar('train/aAcc', accuracy, self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
            
            self.global_step += 1

        iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
        accuracy_class = self.intersection_meter.sum / (self.target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        aAcc = sum(self.intersection_meter.sum) / (sum(self.target_meter.sum) + 1e-10)
        if dist.get_rank() == 0:
            print(f'Train result at Epoch [{epoch}]: mIoU {mIoU:.4f} mAcc {mAcc:.4f} aAcc {aAcc:.4f}')

    def validate(self, epoch):
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
            print(f'Validation result at Epoch [{epoch}]: mIoU {mIoU:.4f} mAcc {mAcc:.4f} aAcc {aAcc:.4f} Loss {loss:.4f}')
            self.writer.add_scalar('val/loss', loss, epoch)
            self.writer.add_scalar('val/mIoU', mIoU, epoch)
            self.writer.add_scalar('val/mAcc', mAcc, epoch)
            self.writer.add_scalar('val/aAcc', aAcc, epoch)

        return mIoU

def main_worker(local_rank, gpu_list, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list[local_rank]
    dist.init_process_group(backend=DIST_BACKEND, init_method=DIST_INIT_METHOD, world_size=len(gpu_list), rank=local_rank)
    trainer = Trainer(args)
    trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='configs/cityscapes_pspnet101.yaml')
    parser.add_argument('-r', '--resume', action='store_true', help='resume training')
    args = parser.parse_args()
    yaml.add_constructor('!join', lambda loader, node: os.path.join(*loader.construct_sequence(node)))
    with open(args.config, 'r') as fr:
        cfg = yaml.load(fr)
    cfg['resume'] = args.resume
    args = Namespace(**cfg)
    print(args)
    check_makedirs(args.ckpt_dir)
    check_makedirs(args.log_dir)
    
    gpu_list = list(map(str, range(torch.cuda.device_count())))
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'].split(',')

    mp.spawn(main_worker, args=(gpu_list, args,), nprocs=len(gpu_list))