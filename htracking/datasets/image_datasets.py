#import torch.utils.data as data

import numpy as np
from torch.utils.data import Dataset
import os
import sys
import glob
import copy
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

def extract_class(fpath):
    # Extract the class from the file path.
    # Assume the file name has the format like "fist_memo.jpg"
    fname = os.path.basename(fpath)
    class_name = fname.split("_")[0]
    return class_name

def class_map(anns_in, mapping):
    """
    Annotation class mapping.
    """

    anns = copy.deepcopy(anns_in)
    for ann in anns:
        name = ann['class']
        if name in mapping.keys():
           ann['class'] = mapping[name]

    return anns

def class_filter(anns_in, classes):
    """
    Annotation class filter.
    """
    anns = []
    for ann_in in anns_in:
        ann = copy.deepcopy(ann_in)
        name = ann['class']
        if name in classes:
            anns.append(ann)

    return anns

class ImageClassification(Dataset):
    '''
    Dataset of image classification.
    '''

    def __init__(self, image_dir, classes=None, class_mapping=None, transform=None, target_transform=None, normalize_coordinates=True):
        """
        image_dir: str
            Image directory path.
        classes: list
            List of expected classes.
        class_mapping: dict
            Mapping dict of classes.
        """

        self.image_dir = image_dir
        self.classes = classes
        self.class_mapping = class_mapping
        self.transform = transform
        self.target_transform = target_transform
        self.normalize_coordinates = normalize_coordinates

        if os.path.exists(image_dir):
            image_files = glob.glob(os.path.join(self.image_dir, '*.jpg'))
        else:
            print("Error: Image directory doesn't exist!")
            print("Error: Expected path is {}".format(image_dir))

        self.anns = []
        for f in image_files:
            ann = {} # Blank annotation
            # File name
            ann['filename'] = os.path.basename(f)
            # Class name
            ann['class'] = extract_class(f)
            # Store annotations
            self.anns.append(ann)

        # Class mapping
        if class_mapping:
            self.anns = class_map(self.anns, class_mapping)

        # Only keep the expected classes
        if classes:
            self.anns = class_filter(self.anns, classes)

        # Object names in the dataset
        self.names = self.prepare_object_names()

        # Name dictionary
        self.namedict = self.prepare_namedict()


    def __getitem__(self, index):

        image, target = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)

    def __len__(self):
        return len(self.anns)

    def get_raw_item(self, index):

        ann = self.anns[index]

        filename = ann['filename']
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        # Target format: [class]
        target = np.asarray(self.namedict[ann['class']])

        return image, target


    def prepare_object_names(self):
        # Prepare the object names
        if self.classes:
            names = self.classes
        else:
            names = []
            for ann in self.anns:
                names.append(ann['class'])
            names = sorted(list(set(names)))

        return names

    def prepare_namedict(self):
        # Prepare the name dictionary

        namedict = {}
        for i, name in enumerate(self.names):
            namedict[name] = i

        return namedict

    def info(self):

        num_names = len(self.names)
        num_instances = [0 for i in range(num_names)]

        for ann in self.anns:
            name = ann['class']
            index = self.namedict[name]
            num_instances[index] += 1

        print("There are {} instances in total.".format(len(self.anns)))

        for i in range(num_names):
            print("{}: {}".format(self.names[i], num_instances[i]))

    def imshow(self, index):
        # Show the image.
        # You might need to install ImageMagic for Image.show() to work properly.
        # e.g. sudo apt-get install imagemagick

        image, target = self.get_raw_item(index)
        image.show()


