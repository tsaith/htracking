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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from tensorboardX import SummaryWriter

import htracking
from htracking.datasets import VOCDetection
from htracking.transforms import ListToNumpy, NumpyToTensor
from htracking.utils import read_config
from htracking.yolo3 import ModelMain, YOLOLoss


def _save_checkpoint(state_dict, config, evaluate_func=None):
    # global best_eval_result
    checkpoint_path = os.path.join(config["sub_working_dir"], "model.pth")
    torch.save(state_dict, checkpoint_path)
    logging.info("Model checkpoint saved to %s" % checkpoint_path)


def _get_optimizer(config, net):
    optimizer = None

    # Assign different lr for each layer
    params = None
    base_params = list(
        map(id, net.backbone.parameters())
    )
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

    if not config["lr"]["freeze_backbone"]:
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
            {"params": net.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
        ]
    else:
        logging.info("freeze backbone's parameters.")
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
        ]

    # Initialize optimizer class
    if config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
    elif config["optimizer"]["type"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"],
                               amsgrad=True)
    elif config["optimizer"]["type"] == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])
    else:
        # Default to sgd
        logging.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))

    return optimizer


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=False, default='config.yaml', help="Configuaration file")
args = vars(ap.parse_args())

config_name = args['config']


logging.basicConfig(level=logging.DEBUG,
                    format="[%(asctime)s %(filename)s] %(message)s")

# Read the configuration file
cwd = os.getcwd()
config_path = os.path.join(cwd, config_name)
config = read_config(config_path)

gpu_devices = config["gpu_devices"]
num_gpus = len(gpu_devices)
batch_size = config["batch_size"] * num_gpus
image_w = config["image_w"]
image_h = config["image_h"]



# Show parameters
print("Start training:")
print("gpu_devices = {}".format(gpu_devices))
print("batch_size = {}".format(batch_size))

# Create sub_working_dir
sub_working_dir = '{}/{}/size{}x{}_try{}/{}'.format(
    config['working_dir'], config['model_params']['backbone_name'],
    image_w, image_h, config['try'],
    time.strftime("%Y%m%d%H%M%S", time.localtime()))
if not os.path.exists(sub_working_dir):
    os.makedirs(sub_working_dir)
config["sub_working_dir"] = sub_working_dir
logging.info("sub working dir: %s" % sub_working_dir)

# Creat tf_summary writer
config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
logging.info("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir))

# Start training
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_devices))


config["global_step"] = config.get("start_step", 0)
is_training = False if config.get("export_onnx") else True
classes = config['classes']
num_classes = len(classes)

# Load and initialize network
net = ModelMain(config, is_training=is_training)
net.train(is_training)

# Optimizer and learning rate
optimizer = _get_optimizer(config, net)
lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config["lr"]["decay_step"],
    gamma=config["lr"]["decay_gamma"])

# Set data parallel
net = nn.DataParallel(net)
net = net.cuda()

# Restore pretrained model
model_pretrained = config["model_pretrained"]
if model_pretrained:
    logging.info("Load pretrained weights from {}".format(model_pretrained))
    state_dict = torch.load(model_pretrained)
    net.load_state_dict(state_dict)


# YOLO loss with 3 scales
yolo_losses = []
for i in range(3):
    yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i], num_classes, (image_w, image_h)))

# Dataset
train_images_path = config['train_images_path']
train_ann_path = config['train_ann_path']

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
target_transform = torchvision.transforms.Compose([ListToNumpy(), NumpyToTensor()])
dataset = VOCDetection(train_images_path, train_ann_path, transform=transform, target_transform=target_transform)

# Data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=32, pin_memory=True)


# Start the training loop
logging.info("Start training.")
for epoch in range(config["epochs"]):
    for step, samples in enumerate(dataloader):
        images, labels = samples["image"], samples["label"]
        start_time = time.time()
        config["global_step"] += 1

        # Forward and backward
        optimizer.zero_grad()
        outputs = net(images)
        losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
        losses = []
        for _ in range(len(losses_name)):
            losses.append([])
        for i in range(3):
            _loss_item = yolo_losses[i](outputs[i], labels)
            for j, l in enumerate(_loss_item):
                losses[j].append(l)
        losses = [sum(l) for l in losses]
        loss = losses[0]

        loss.backward()
        optimizer.step()

        if step > 0 and step % 10 == 0:
            _loss = loss.item()
            duration = float(time.time() - start_time)
            example_per_second = batch_size / duration
            lr = optimizer.param_groups[0]['lr']
            logging.info(
                "epoch [%.3d] iter = %d loss = %.2f example/sec = %.3f lr = %.5f "%
                (epoch, step, _loss, example_per_second, lr)
            )
            config["tensorboard_writer"].add_scalar("lr",
                                                    lr,
                                                    config["global_step"])
            config["tensorboard_writer"].add_scalar("example/sec",
                                                    example_per_second,
                                                    config["global_step"])
            for i, name in enumerate(losses_name):
                value = _loss if i == 0 else losses[i]
                config["tensorboard_writer"].add_scalar(name,
                                                        value,
                                                        config["global_step"])

        if step > 0 and step % 1000 == 0:
            # net.train(False)
            _save_checkpoint(net.state_dict(), config)
            # net.train(True)

    lr_scheduler.step()

# net.train(False)
_save_checkpoint(net.state_dict(), config)
# net.train(True)
logging.info("Bye~")



