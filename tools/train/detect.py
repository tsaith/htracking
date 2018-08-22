import numpy as np

import os
import sys
import argparse
import yaml
import time
import datetime
import json
import importlib
import logging
import shutil
import cv2
import random

import torch
import torch.nn as nn

import torchvision

import htracking
from htracking.yolo3 import ModelMain, YOLOLoss
from htracking.yolo3.common.utils import non_max_suppression, bbox_iou
from htracking.utils import read_config, draw_bbox, get_rgb_colors

from PIL import Image


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
image_w = config["image_w"]
image_h = config["image_h"]
confidence_thresh = config['confidence_thresh']


print("Detecting images:")
print("gpu_devices = {}".format(gpu_devices))
print("confidence_thresh = {}".format(confidence_thresh))

# Start training
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_devices))

is_training = False
classes = config["classes"]
num_classes = len(classes)
detect_images_path = config["detect_images_path"]
detect_output_path = config["detect_output_path"]


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Load and initialize network
net = ModelMain(config, is_training=is_training)
net.train(is_training)

# Set data parallel
net = nn.DataParallel(net)
net = net.cuda()

# Restore pretrain model
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
                                num_classes, (image_w, image_h)))

# prepare images path
images_name = os.listdir(detect_images_path)
images_path = [os.path.join(detect_images_path, name) for name in images_name]
#images_path = images_path[:3]

if len(images_path) == 0:
    raise Exception("no image found in {}".format(detect_images_path))

# Start inference


if not os.path.isdir(detect_output_path):
    os.makedirs(detect_output_path)

colors = get_rgb_colors()

print("Detecting images under {}.".format(detect_images_path))
for path in images_path:

    image = Image.open(path).convert('RGB')
    #image = cv2.imread(path, cv2.IMREAD_COLOR)

    open_cv_image = np.array(image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    if image is None:
        logging.error("read path error: {}. skip it.".format(path))
        continue
    image_ori = open_cv_image  # save original one
    image = transform(image)
    image = image.unsqueeze(0) # Add dimention

    # inference
    with torch.no_grad():
        output = net(image)
        output_list = []
        for i in range(3):
            output_list.append(yolo_losses[i](output[i]))
        output = torch.cat(output_list, 1)
        detections = non_max_suppression(output, num_classes, conf_thres=confidence_thresh)
        detections = detections[0]

    # write result images. Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        #bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            # Rescale coordinates to original dimensions
            ori_h, ori_w = image_ori.shape[:2]
            pre_h, pre_w = image_h, image_w
            bbox_h = ((y2 - y1) / pre_h) * ori_h
            bbox_w = ((x2 - x1) / pre_w) * ori_w
            y1 = (y1 / pre_h) * ori_h
            x1 = (x1 / pre_w) * ori_w

            # Draw the bbox
            bbox = (x1, y1, x1+bbox_w, y1+bbox_h)
            cls_index = int(cls_pred)
            lb = "{}({:4.2f})".format(classes[cls_index], cls_conf)
            draw_bbox(image_ori, bbox, label=lb, color=colors[cls_index])


    output_path = os.path.join(detect_output_path, os.path.basename(path))
    cv2.imwrite(output_path, np.uint8(image_ori))

print("Output the detected images to {}.".format(detect_output_path))
print("done")
