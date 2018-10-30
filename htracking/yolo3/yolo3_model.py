import numpy as np

import torch
import torch.nn as nn

from .nets import ModelMain, YOLOLoss
from .common.utils import non_max_suppression, bbox_iou


class YOLO3_MODEL:
    # YOLO3 model.

    def __init__(self, config, device, training=True):

        self.config = config
        self.device = device
        self.training = training

        self.classes = self.config["classes"]
        self.num_classes = len(self.classes)

        self.image_c = 3
        self.image_h = self.config["image_h"]
        self.image_w = self.config["image_w"]

        self.yolo_losses = []
        for i in range(3):
            self.yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                               self.num_classes, (self.image_w, self.image_h)))

        # Construct the network
        self.net = ModelMain(config, is_training=self.training)
        self.net = nn.DataParallel(self.net)
        self.net = self.net.to(device)

        self.confidence = config["confidence"]
        self.confidence_thresh = config["confidence_thresh"]
        self.nms_thresh = config["nms_thresh"]
        self.iou_thresh = config["iou_thresh"]

    def train(self, training=True):
        self.training = training
        self.net.train(training)

    def load_state(self, f):
        state_dict = torch.load(f)
        self.net.load_state_dict(state_dict)

    def detect(self, image):
        """
        Detect the image.

        image: Tensor
            Input image.
        """

        with torch.no_grad():
            output = self.net(image)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_losses[i](output[i]))
            output = torch.cat(output_list, 1)
            detections = non_max_suppression(output, self.num_classes,
                                             conf_thres=self.confidence_thresh)
            detections = detections[0]

        return detections

    def objects_from(self, detections):
        # Convert to an annotation.

        objects = []

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            cls_index = int(cls_pred)

            obj = {}
            obj['name'] = self.classes[cls_index]
            obj['xmin'] = int(x1)
            obj['xmax'] = int(x2)
            obj['ymin'] = int(y1)
            obj['ymax'] = int(y2)

            objects.append(obj)

        return objects


