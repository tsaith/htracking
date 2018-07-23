
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import yaml

import os
import sys


import htracking
from htracking.utils import read_config


import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil
import cv2
import random

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import torch
import torch.nn as nn

#yolo3_path = "/home/andrew/projects/htracking/htracking/yolo3"
#sys.path.insert(0, yolo3_path)

from htracking.yolo3 import ModelMain, YOLOLoss
from htracking.yolo3.common.utils import non_max_suppression, bbox_iou


cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]


def predict(config):
    is_training = False
    classes = config["classes"]
    num_classes = len(config["classes"])
    predict_images_path = config["predict_images_path"]
    predict_output_path = config["predict_output_path"]
    
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
                                    num_classes, (config["img_w"], config["img_h"])))

    # prepare images path
    images_name = os.listdir(predict_images_path)
    images_path = [os.path.join(predict_images_path, name) for name in images_name]
    if len(images_path) == 0:
        raise Exception("no image found in {}".format(predict_images_path))

    # Start inference
    batch_size = config["batch_size"]
    for step in range(0, len(images_path), batch_size):
        # preprocess
        images = []
        images_origin = []
        for path in images_path[step*batch_size: (step+1)*batch_size]:
            logging.info("processing: {}".format(path))
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                logging.error("read path error: {}. skip it.".format(path))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_origin.append(image)  # keep for save result
            image = cv2.resize(image, (config["img_w"], config["img_h"]),
                               interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            images.append(image)
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        # inference
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, num_classes,
                                                   conf_thres=config["confidence_thresh"])

        # write result images. Draw bounding boxes and labels of detections
        #classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]
        if not os.path.isdir(predict_output_path):
            os.makedirs(predict_output_path)
        for idx, detections in enumerate(batch_detections):
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(images_origin[idx])
            if detections is not None:
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Rescale coordinates to original dimensions
                    ori_h, ori_w = images_origin[idx].shape[:2]
                    pre_h, pre_w = config["img_h"], config["img_w"]
                    box_h = ((y2 - y1) / pre_h) * ori_h
                    box_w = ((x2 - x1) / pre_w) * ori_w
                    y1 = (y1 / pre_h) * ori_h
                    x1 = (x1 / pre_w) * ori_w
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                             edgecolor=color,
                                             facecolor='none')
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(x1, y1, s=classes[int(cls_pred)], color='white',
                             verticalalignment='top',
                             bbox={'color': color, 'pad': 0})
            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            saved_path = "{}/{}_{}.jpg".format(predict_output_path, step, idx)
            plt.savefig(saved_path, bbox_inches='tight', pad_inches=0.0)
            plt.close()

    logging.info("Save all results to {}".format(predict_output_path))


logging.basicConfig(level=logging.DEBUG,
                    format="[%(asctime)s %(filename)s] %(message)s")

root_path = "/home/andrew/projects/htracking/htracking/yolo3/predict"
config_name = "config.yaml"
config_path = os.path.join(root_path, config_name)
config = read_config(config_path)

config["batch_size"] *= len(config["parallels"])

# Start training
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
predict(config)
