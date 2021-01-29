# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-27
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os

target_dir = 'temp/ade2016'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
 
train_img_dir = 'data/ADEChallengeData2016/images/training'
train_seg_dir = 'data/ADEChallengeData2016/annotations/training'
train_data = []
for name in os.listdir(train_img_dir):
    img_path = os.path.join(train_img_dir, name)
    seg_name = name[:-4] + '.png'
    seg_path = os.path.join(train_seg_dir, seg_name)
    if not os.path.exists(seg_path):
        print('{} not exist!'.format(seg_path))
    train_data.append('{} {}\n'.format(img_path, seg_path))
with open(os.path.join(target_dir, 'train.txt'), 'w') as fw:
    fw.writelines(train_data)


val_img_dir = 'data/ADEChallengeData2016/images/validation'
val_seg_dir = 'data/ADEChallengeData2016/annotations/validation'
val_data = []
for name in os.listdir(val_img_dir):
    img_path = os.path.join(val_img_dir, name)
    seg_name = name[:-4] + '.png'
    seg_path = os.path.join(val_seg_dir, seg_name)
    if not os.path.exists(seg_path):
        print('{} not exist!'.format(seg_path))
    val_data.append('{} {}\n'.format(img_path, seg_path))
with open(os.path.join(target_dir, 'val.txt'), 'w') as fw:
    fw.writelines(val_data)
