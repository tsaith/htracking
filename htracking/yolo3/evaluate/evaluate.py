# coding='utf-8'

import numpy as np

import argparse
import os
import sys
import time
import datetime
import json
import importlib
import logging
import shutil

import torch
import torch.nn as nn

import torchvision

import htracking
from htracking.datasets import VOCDetection
from htracking.transforms import ListToNumpy, NumpyToTensor
from htracking.utils import read_config
from htracking.yolo3 import ModelMain, YOLOLoss
from htracking.yolo3.common.utils import non_max_suppression, bbox_iou

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=False, default='config.yaml', help="Configuaration file")
args = vars(ap.parse_args())

config_name = args['config']


logging.basicConfig(level=logging.DEBUG,
                    format="[%(asctime)s %(filename)s] %(message)s")

cwd = os.getcwd()
config_path = os.path.join(cwd, config_name)
config = read_config(config_path)

gpu_devices = config["gpu_devices"]
num_gpus = len(gpu_devices)

batch_size = config["batch_size"] * num_gpus

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_devices))



is_training = False
classes = config["classes"]
num_classes = len(classes)

# Load and initialize network
net = ModelMain(config, is_training=is_training)
net.train(is_training)

# Set data parallel
net = nn.DataParallel(net)
net = net.cuda()

# Restore the pretrained model
model_pretrained = config["model_pretrained"]
if model_pretrained:
    logging.info("load checkpoint from {}".format(model_pretrained))
    state_dict = torch.load(model_pretrained)
    net.load_state_dict(state_dict)
else:
    raise Exception("missing the model pretrained!!!")


# YOLO loss with 3 scales
yolo_losses = []
for i in range(3):
    yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                num_classes, (config["img_w"], config["img_h"])))

    
# Dataset     
eval_images_path = config['eval_images_path']
eval_ann_path = config['eval_ann_path']  

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
target_transform = torchvision.transforms.Compose([ListToNumpy(), NumpyToTensor()])
dataset = VOCDetection(eval_images_path, eval_ann_path, transform=transform, target_transform=target_transform)

# Data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=32, pin_memory=True)      
    

# Start the eval loop
logging.info("Start eval.")
n_gt = 0
correct = 0
for step, samples in enumerate(dataloader):
    images, labels = samples["image"], samples["label"]
    labels = labels.cuda()
    with torch.no_grad():
        outputs = net(images)
        output_list = []
        for i in range(3):
            output_list.append(yolo_losses[i](outputs[i]))
        output = torch.cat(output_list, 1)
        output = non_max_suppression(output, num_classes, conf_thres=config["confidence_thresh"])
        #  calculate
        for sample_i in range(labels.size(0)):
            # Get labels for sample where width is not zero (dummies)
            target_sample = labels[sample_i, labels[sample_i, :, 3] != 0]
            for obj_cls, tx, ty, tw, th in target_sample:
                # Get rescaled gt coordinates
                tx1, tx2 = config["img_w"] * (tx - tw / 2), config["img_w"] * (tx + tw / 2)
                ty1, ty2 = config["img_h"] * (ty - th / 2), config["img_h"] * (ty + th / 2)
                n_gt += 1
                box_gt = torch.cat([coord.unsqueeze(0) for coord in [tx1, ty1, tx2, ty2]]).view(1, -1)
                sample_pred = output[sample_i]
                if sample_pred is not None:
                    # Iterate through predictions where the class predicted is same as gt
                    for x1, y1, x2, y2, conf, obj_conf, obj_pred in sample_pred[sample_pred[:, 6] == obj_cls]:
                        box_pred = torch.cat([coord.unsqueeze(0) for coord in [x1, y1, x2, y2]]).view(1, -1)
                        iou = bbox_iou(box_pred, box_gt)
                        #print("sample_i, obj_pred, obj_conf = {}, {}, {:4.2f}".format(sample_i, obj_pred, obj_conf))
                        if iou >= config["iou_thresh"]:
                            correct += 1
                            break
    if n_gt:
        logging.info('Batch [%d/%d] mAP: %.5f' % (step, len(dataloader), float(correct / n_gt)))

logging.info('Mean Average Precision: %.5f' % float(correct / n_gt))

