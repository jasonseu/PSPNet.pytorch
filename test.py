# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-24
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import yaml
import argparse
from argparse import Namespace

import numpy as np
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.pspnet import PSPNet
from lib import transform
from lib.util import *
from lib.dataset import SegmentationDataset


class Tester(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mean = [item * 255 for item in args.mean]
        self.std = [item * 255 for item in args.std]
        test_transform = transform.Compose([transform.ToTensor()])
        self.test_dataset = SegmentationDataset(args, split='val', data_path=args.test_path, transform=test_transform)

        self.colors = np.loadtxt(args.colors_path).astype('uint8')
        self.data_list = self.test_dataset.data_list
        self.label_map = self.test_dataset.label_map

    def run(self):
        if not self.args.has_prediction:
            self.infer()
        self.cal_metric()

    def infer(self):
        self.model = PSPNet(layers=args.layers, classes=args.num_classes, zoom_factor=args.zoom_factor)
        self.model.cuda()
        self.model.eval()

        print("=> loading checkpoint '{}'".format(self.args.ckpt_best_path))
        model_dict = torch.load(self.args.ckpt_best_path)
        if next(iter(model_dict.keys())).startswith('module'):
            model_dict = {k[7:]:v for k, v in model_dict.items()}
        self.model.load_state_dict(model_dict, strict=True)
        print("=> loaded checkpoint '{}'".format(self.args.ckpt_best_path))

        check_makedirs(self.args.gray_dir)
        check_makedirs(self.args.color_dir)
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        for i, (input, _) in enumerate(tqdm(test_loader)):
            input = np.squeeze(input.numpy(), axis=0)
            image = np.transpose(input, (1, 2, 0))
            h, w, _ = image.shape
            prediction = np.zeros((h, w, self.args.num_classes), dtype=float)
            for scale in self.args.scales:
                long_size = round(scale * self.args.base_size)
                new_h, new_w = long_size, long_size
                if h > w:
                    new_w = round(long_size / float(h) * w)
                else:
                    new_h = round(long_size / float(w) * h)
                image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                prediction += self.scale_process(image_scale, h, w)

            prediction /= len(self.args.scales)
            prediction = np.argmax(prediction, axis=2)

            gray = np.uint8(prediction)
            color = colorize(gray, self.colors)
            image_path, _ = self.data_list[i]
            image_name = image_path.split('/')[-1].split('.')[0]
            gray_path = os.path.join(self.args.gray_dir, image_name + '.png')
            color_path = os.path.join(self.args.color_dir, image_name + '.png')
            cv2.imwrite(gray_path, gray)
            color.save(color_path)

    def scale_process(self, image, h, w, stride_rate=2/3):
        ori_h, ori_w, _ = image.shape
        pad_h = max(self.args.test_h - ori_h, 0)
        pad_w = max(self.args.test_w - ori_w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.mean)
        new_h, new_w, _ = image.shape
        stride_h = int(np.ceil(self.args.test_h * stride_rate))
        stride_w = int(np.ceil(self.args.test_w * stride_rate))
        grid_h = int(np.ceil(float(new_h - self.args.test_h) / stride_h) + 1)
        grid_w = int(np.ceil(float(new_w - self.args.test_w) / stride_w) + 1)
        prediction_crop = np.zeros((new_h, new_w, self.args.num_classes), dtype=float)
        count_crop = np.zeros((new_h, new_w), dtype=float)
        for index_h in range(0, grid_h):
            for index_w in range(0, grid_w):
                s_h = index_h * stride_h
                e_h = min(s_h + self.args.test_h, new_h)
                s_h = e_h - self.args.test_h
                s_w = index_w * stride_w
                e_w = min(s_w + self.args.test_w, new_w)
                s_w = e_w - self.args.test_w
                image_crop = image[s_h:e_h, s_w:e_w].copy()
                count_crop[s_h:e_h, s_w:e_w] += 1
                prediction_crop[s_h:e_h, s_w:e_w, :] += self.net_process(image_crop)
        prediction_crop /= np.expand_dims(count_crop, 2)
        prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
        prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        return prediction

    def net_process(self, image, flip=True):
        input = torch.from_numpy(image.transpose((2, 0, 1))).float()
        for t, m, s in zip(input, self.mean, self.std):
            t.sub_(m).div_(s)
        input = input.unsqueeze(0).cuda()
        if flip:
            input = torch.cat([input, input.flip(3)], 0)
        with torch.no_grad():
            output = self.model(input)
        _, _, h_i, w_i = input.shape
        _, _, h_o, w_o = output.shape
        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
        output = F.softmax(output, dim=1)
        if flip:
            output = (output[0] + output[1].flip(2)) / 2
        else:
            output = output[0]
        output = output.data.cpu().numpy()
        output = output.transpose(1, 2, 0)
        return output

    def cal_metric(self):
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        for i, (image_path, target_path) in enumerate(self.data_list):
            image_name = image_path.split('/')[-1].split('.')[0]
            pred = cv2.imread(os.path.join(self.args.gray_dir, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
            target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
            if self.label_map is not None:
                temp = np.zeros(target.shape)
                for k, v in self.label_map.items():
                    t = target == k
                    target[t] = v
                    temp = np.logical_or(temp, t)
                target[~temp] = 255
            intersection, union, target = intersectionAndUnion(pred, target, self.args.num_classes)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            accuracy = sum(intersection) / (sum(target) + 1e-10)
            print(f'Evaluating {i+1}/{len(self.data_list)} on image {image_name}.png, accuracy {accuracy:.4f}.')
        
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        names = [line.strip() for line in open(self.args.names_path)]
        print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i, name in enumerate(names):
            print('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='configs/cityscapes_pspnet101.yaml')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('-p', '--has_prediction', action='store_true')
    args = parser.parse_args()
    yaml.add_constructor('!join', lambda loader, node: os.path.join(*loader.construct_sequence(node)))
    with open(args.config, 'r') as fr:
        cfg = yaml.load(fr)
    if args.ckpt_path is not None:
        cfg['ckpt_best_path'] = args.ckpt_path
    cfg['has_prediction'] = args.has_prediction
    args = Namespace(**cfg)
    print(args)

    tester = Tester(args)
    tester.run()