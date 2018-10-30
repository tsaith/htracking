import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2 # OpenCV
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision 
from torchvision import transforms

import htracking
from htracking.yolo3 import YOLO3_MODEL
from htracking.yolo3 import ModelMain, YOLOLoss
from htracking.yolo3.common.utils import non_max_suppression, bbox_iou
from htracking.utils import wait_key
from htracking.utils import read_config, draw_bbox, get_rgb_colors
from htracking.utils import patch_extractor
from handpose.utils.pytorch import predict
from handpose.utils.pytorch import Conv2dSame, num_flat_features

from PIL import Image

# Main parameters

gpu_devices = [0]

# True: webcam, False: video file
use_webcam = False


image_width = 416
image_height = 416

patch_width = 128
patch_height = 128

# Video path
video_file = 'test_faces.mp4'
video_path = "/home/andrew/projects/datasets/faces/test/video/{}".format(video_file)
output_path = "output.mp4"

# Model path
face_model_path = "/home/andrew/projects/htracking/notebooks/posture_estimation/face_model_cpu.pth"
#face_model_path = "/home/andrew/projects/htracking/notebooks/posture_estimation/face_model.pth"


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
face_model = YOLO3_MODEL(config, device, training=training)

# Load the trained models
print("Loading the trained model from {}".format(face_model_path))
face_model.load_state(face_model_path)

transform = transforms.Compose([transforms.ToTensor()])


def pil_to_tensor(image):
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    tensor = transform(image)
    tensor = tensor.unsqueeze(0) # Add dimention
    
    return tensor
 
def draw_detection(image_cv, image_h, image_w, detections, classes, colors):    
    # write result images. Draw bounding boxes and labels of detections
    
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    #bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

        # Rescale coordinates to original dimensions
        ori_h, ori_w = image_cv.shape[:2]
        pre_h, pre_w = image_h, image_w
        bbox_h = ((y2 - y1) / pre_h) * ori_h
        bbox_w = ((x2 - x1) / pre_w) * ori_w
        y1 = (y1 / pre_h) * ori_h
        x1 = (x1 / pre_w) * ori_w

        # Draw the bbox
        bbox = (x1, y1, x1+bbox_w, y1+bbox_h)
        cls_index = int(cls_pred)
        lb = "{}({:4.2f})".format(classes[cls_index], cls_conf)
        draw_bbox(image_cv, bbox, label=lb, color=colors[cls_index])

    return bbox        

def draw_info(image, text="info"):

    #cv2.rectangle(image, (20, 60), (120, 160), (0, 255, 0), 2)
    cv2.putText(image, text, (0, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 1, cv2.LINE_AA)

face_classes = ["face"]
num_classes = len(face_classes)

colors = get_rgb_colors()


# Read frames form webcam or video file
video = video_path
if video is None: # Use the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height);

else:
    cap = cv2.VideoCapture(video)

# In case the video can't be open
if not cap.isOpened(): 
    print("The video cann't be open!")
    exit(0)

# Creat a window
win_name = 'test'
cv2.namedWindow(win_name)


_, first_frame = cap.read()

# Determine the width and height from the first image
height, width, channels = first_frame.shape

# Patch extractor
#extractor = patch_extractor(width, height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
fps_out = 3
out = cv2.VideoWriter(output_path, fourcc, fps_out, (width, height))

# Initialize the compresive tracker
rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

# loop over the frames of the video
print("Start to run.")
iframe = 0
while True:
    iframe += 1
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = cap.read()

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed: break

    # Process the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_pil = Image.fromarray(frame_rgb)
    image = transform(frame_pil)
    image = image.unsqueeze(0) # Add dimention

    # Detect the image
    detections = face_model.detect(image)

    if detections is not None:
        bbox = draw_detection(frame, width, height, detections, face_classes, colors)   

    print("iframe = ", iframe)

    # show the frame and record if the user presses a key
    cv2.imshow(win_name, frame)
    out.write(frame) # Write out frame to video

    # Wait for a while
    #time.sleep(0.1)

    # if the `q` key is pressed, breaqk from the lop
    #key = wait_key(1)
    #if key == ord("q"): break



# Release the resources and close any open windows
print("Finalize the application.")
cap.release()
out.release()
cv2.destroyAllWindows()
