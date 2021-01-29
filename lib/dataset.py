# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-27
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import cv2
from PIL import Image
import numpy as np

from torch.utils.data import Dataset


def make_dataset(split='train', data_path=None):
    assert split in ['train', 'val', 'test']
    image_label_list = []
    list_read = open(data_path).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            image_path = line_split[0]
            label_path = image_path  # just set place holder for label_path, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_path = line_split[0]
            label_path = line_split[1]
        item = (image_path, label_path)
        image_label_list.append(item)
    return image_label_list


class SegmentationDataset(Dataset):
    def __init__(self, args, split='train', data_path=None, transform=None, ignore_label=255):
        self.args = args
        self.split = split
        self.transform = transform
        self.data_list = make_dataset(split, data_path)
        self.ignore_index = args.ignore_label
        if args.label_op == 'mapping':
            self.label_map = {c:i for i, c in enumerate(args.label_list)}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        label = np.array(Image.open(label_path).convert('L'))
        if self.args.label_op == 'mapping':
            temp = np.zeros(label.shape)
            for k, v in self.label_map.items():
                t = label == k
                label[t] = v
                temp = np.logical_or(temp, t)
            label[~temp] = 255
        elif self.args.label_op == 'shift':
            label = label - 1
        
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        
        return image, label
