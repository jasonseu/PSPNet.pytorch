# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-27
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from collections import OrderedDict

model_path = 'officialmodel/train_epoch_200.pth'
# model_path = 'checkpoints/cityscapes/cityscapes_pspnet101_best_model.pth'
save_path = 'officialmodel/cityscapes.pth'
checkpoint = torch.load(model_path)
model_dict = checkpoint['state_dict']
torch.save(model_dict, save_path)