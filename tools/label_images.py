import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import cv2 # OpenCV
import time
import datetime

import torch

import htracking
from htracking.utils import read_config
from htracking.yolo3 import YOLO3_MODEL
from htracking.yolo3 import ModelMain, YOLOLoss
from htracking.datasets import ImageLabeler

from PIL import Image

import fire

def run():
    """
    Run the code.
    """

    # Main parameters
    gpu_devices = [3]

    # Model path
    model_path = "/home/andrew/projects/htracking/notebooks/posture_estimation/face_model.pth"


    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_devices))

    num_gpus = len(gpu_devices)
    if num_gpus > 0 and torch.cuda.is_available():
        use_cuda = True
        device = torch.device("cuda:0" )
    else:
        use_cuda = False
        device = torch.device("cpu")

    print("gpu_devices: {}, num_gpus: {}".format(gpu_devices, num_gpus))
    print("Running on the device: {}".format(device))

    cwd = "/home/andrew/projects/htracking/tools/detect_faces"
    config_name = "config.yaml"
    config_path = os.path.join(cwd, config_name)
    config = read_config(config_path)

    training = False

    # Model with data parallel
    model = YOLO3_MODEL(config, device, training=training)

    # Load the trained models
    print("Loading the trained model from {}".format(model_path))
    model.load_state(model_path)

    # Generate the annotation files

    ann_dir = "/home/andrew/projects/datasets/face_datasets/MYFACES/buffer/xml"
    image_dir = "/home/andrew/projects/datasets/face_datasets/MYFACES/buffer/images"

    labeler = ImageLabeler(ann_dir, image_dir, model)
    labeler.run()

if __name__ == '__main__':

    fire.Fire()

