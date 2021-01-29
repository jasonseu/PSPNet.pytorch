# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-27
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os

target_dir = 'temp/voc2012'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# data_dir = 'data/pascal_voc/VOC2012'
# seg_dir = os.path.join(data_dir, 'SegmentationClass')
# img_dir = os.path.join(data_dir, 'JPEGImages')

# train_set = [t.strip() for t in open(os.path.join(data_dir, 'ImageSets/Segmentation/train.txt'))]
# val_set = [t.strip() for t in open(os.path.join(data_dir, 'ImageSets/Segmentation/val.txt'))]

# train_data_list = []
# val_data_list = []
# for seg_name in os.listdir(seg_dir):
#     name = seg_name[:-4]
#     seg_path = os.path.join(seg_dir, seg_name)
#     img_path = os.path.join(img_dir, name+'.jpg')
#     if name in train_set:
#         train_data_list.append('{} {}\n'.format(img_path, seg_path))
#     elif name in val_set:
#         val_data_list.append('{} {}\n'.format(img_path, seg_path))


data_dir = 'data/VOC2012'

train_data_list = []
with open(os.path.join(data_dir, 'ImageSets/SegmentationAug/train_aug.txt'), 'r') as fr:
    for line in fr:
        temp = line.strip().split()
        img_path = os.path.join(data_dir, temp[0][1:])
        seg_path = os.path.join(data_dir, temp[1][1:])
        train_data_list.append('{} {}\n'.format(img_path, seg_path))

val_data_list = []
with open(os.path.join(data_dir, 'ImageSets/SegmentationAug/val.txt'), 'r') as fr:
    for line in fr:
        temp = line.strip().split()
        img_path = os.path.join(data_dir, temp[0][1:])
        seg_path = os.path.join(data_dir, temp[1][1:])
        val_data_list.append('{} {}\n'.format(img_path, seg_path))

# test_data_list = []
# with open(os.path.join(data_dir, 'ImageSets/SegmentationAug/test.txt'), 'r') as fr:
#     for line in fr:
#         temp = line.strip().split()
#         img_path = os.path.join(data_dir, temp[0][1:])
#         seg_path = os.path.join(data_dir, temp[1][1:])
#         test_data_list.append('{} {}\n'.format(img_path, seg_path))

with open(os.path.join(target_dir, 'train.txt'), 'w') as fw:
    fw.writelines(train_data_list)
with open(os.path.join(target_dir, 'val.txt'), 'w') as fw:
    fw.writelines(val_data_list)
# with open(os.path.join(target_dir, 'test.txt'), 'w') as fw:
#     fw.writelines(test_data_list)

