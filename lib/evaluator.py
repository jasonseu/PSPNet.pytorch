# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2020-12-22
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2020 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch


class Evaluator():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes).cuda()

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes).cuda()
    
    def pixel_accuracy(self):
        if self.confusion_matrix.sum() == 0:
            print("Attention: pixel_total is zero!!!")
            PA = 0
        else:
            PA = self.confusion_matrix.sum().diag() / self.confusion_matrix.sum().float()
        return PA

    def mean_pixel_accuracy(self):
        mPA = self.confusion_matrix.diag() / self.confusion_matrix.sum(dim=1).float()
        not_nan = ~torch.isnan(mPA)
        mPA = mPA[not_nan].sum() / not_nan.float().sum()
        return mPA

    def mean_intersection_over_union(self):
        temp = self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) - self.confusion_matrix.diag()
        IoUs = self.confusion_matrix.diag() / temp.float()
        not_nan = ~torch.isnan(IoUs)
        mIoU = IoUs[not_nan].sum() / not_nan.float().sum()
        return mIoU, IoUs

    def frequency_weighted_intersection_over_union(self):
        temp = self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) - self.confusion_matrix.diag()
        FWIoU = self.confusion_matrix.sum(dim=1) * self.confusion_matrix.diag() / temp.float()
        FWIoU = FWIoU[~FWIoU.isnan()].sum() / self.confusion_matrix.sum().float()
        return FWIoU

    def __generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].int() + pre_image[mask]
        count = torch.bincount(label.view(-1), minlength=self.num_classes**2)
        confusion_matrix = count.view(self.num_classes, self.num_classes)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert the size of two images are same
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self.__generate_matrix(gt_image, pre_image)

    def update(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix = self.__generate_matrix(gt_image, pre_image)