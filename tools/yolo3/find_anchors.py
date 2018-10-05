import numpy as np

import os
import random
import argparse

import torchvision
from torchvision import transforms

import htracking
from htracking.datasets import VOCDetection
from htracking.transforms import ListToNumpy, NumpyToTensor
from htracking.utils import read_config

def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)

def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n

def print_anchors(centroids, image_width=416, image_height=416):
    out_string = ''

    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    anchors_sorted = [[] for i in range(3)]
    for j, i in enumerate(sorted_indices):
        anchors_sorted[j%3].append([int(anchors[i,0]*image_width), int(anchors[i,1]*image_height)])

    print("anchors:")
    print(anchors_sorted)

def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        #assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=False, default='config.yaml', help="Configuaration file")
ap.add_argument("-a", "--anchors", required=False, default=9, help="Number of anchors")
args = vars(ap.parse_args())

config_name = args['config']
num_anchors = args['anchors']


# Read the configuration file
cwd = os.getcwd()
config_path = os.path.join(cwd, config_name)
config = read_config(config_path)

print("Finding the anchors:")

# Dataset
train_images_path = config['train_images_path']
train_ann_path = config['train_ann_path']

transform = transforms.Compose([transforms.Resize(416), transforms.ToTensor(), ])
target_transform = torchvision.transforms.Compose([ListToNumpy(), NumpyToTensor()])
dataset = VOCDetection(train_images_path, train_ann_path, transform=transform, target_transform=target_transform)

num_samples = len(dataset)

# run k_mean to find the anchors
annotation_dims = []
for i in range(num_samples):
#or i in range(36):
    #print("Sample({})".format(i))
    sample = dataset[i]
    anns = sample['target']
    for ann in anns:
        # Normalized width and height
        width = ann[3]
        height = ann[4]
        # Width and height must be greater than zero
        if width > 0 and height > 0:
            #print("ann: {}".format(ann))
            annotation_dims.append((width, height))


annotation_dims = np.array(annotation_dims)
centroids = run_kmeans(annotation_dims, num_anchors)

# write anchors to file
print('\naverage IOU for', num_anchors, 'anchors:', '%0.2f' % avg_IOU(annotation_dims, centroids))
print_anchors(centroids)

