# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2020-12-23
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2020 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os

target_dir = 'temp/cityscapes'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

train_img_dir = 'data/Cityscapes/leftImg8bit/train'
train_seg_dir = 'data/Cityscapes/gtFine/train'
train_data = []
for city in os.listdir(train_img_dir):
    city_img_dir = os.path.join(train_img_dir, city)
    city_seg_dir = os.path.join(train_seg_dir, city)
    for name in os.listdir(city_img_dir):
        img_path = os.path.join(city_img_dir, name)
        seg_name = name.replace('leftImg8bit', 'gtFine_labelIds')
        seg_path = os.path.join(city_seg_dir, seg_name)
        if not os.path.exists(seg_path):
            print('{} not exist!'.format(seg_path))
        train_data.append('{} {}\n'.format(img_path, seg_path))
with open(os.path.join(target_dir, 'train.txt'), 'w') as fw:
    fw.writelines(train_data)

val_img_dir = 'data/Cityscapes/leftImg8bit/val'
val_seg_dir = 'data/Cityscapes/gtFine/val'
val_data = []
for city in os.listdir(val_img_dir):
    city_img_dir = os.path.join(val_img_dir, city)
    city_seg_dir = os.path.join(val_seg_dir, city)
    for name in os.listdir(city_img_dir):
        img_path = os.path.join(city_img_dir, name)
        seg_name = name.replace('leftImg8bit', 'gtFine_labelIds')
        seg_path = os.path.join(city_seg_dir, seg_name)
        if not os.path.exists(seg_path):
            print('{} not exist!'.format(seg_path))
        val_data.append('{} {}\n'.format(img_path, seg_path))
with open(os.path.join(target_dir, 'val.txt'), 'w') as fw:
    fw.writelines(val_data)