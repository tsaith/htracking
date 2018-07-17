#import torch.utils.data as data

from torch.utils.data import Dataset
import os
#import os.path
import sys
import glob
import copy
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

def xml_annotations(filepath):
    '''
    Return the annotations of xml file.

    filepath: str
        xml file path.

    returns
    -------
    anns: dict
        anns['filename'] saves the filename of image.
        anns['objects'] saves information of objects. e.g. {[class_name, xmin, ymin, xmax, ymax]}
    '''

    root = ET.parse(filepath).getroot()

    anns = {}
    # Filename
    anns['filename'] = root.find('filename').text.strip()

    # Object information
    objects = []
    for obj in root.iter('object'):
        name = obj.find('name').text
        #name = obj[0].text.lower().strip()
        bndbox = obj.find('bndbox')
        #bbox = obj[4]
        bb = [bndbox.find('xmin').text, bndbox.find('ymin').text,
              bndbox.find('xmax').text, bndbox.find('ymax').text]
        # supposes the order is xmin, ymin, xmax, ymax
        # attention with indices
        bb = [int(v)-1 for v in bb]

        objects += [[name] + bb]

    anns['objects'] = objects

    return anns


class VOCDetection(Dataset):
    '''
    Dataset of Pascal VOC Detection.
    '''

    def __init__(self, image_dir, ann_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.target_transform = target_transform

        if os.path.exists(ann_dir):
            ann_files = glob.glob(os.path.join(self.ann_dir, '*.xml'))
        else:
            print("Annotation diretory doesn't exist!")

        self.anns = [xml_annotations(f) for f in ann_files]

        # Object names
        self.names = self.prepare_object_names()

        # Name dictionary
        self.namedict = self.prepare_namedict()


    def __getitem__(self, index):

        image, target = self.get_raw_item(index)

        for obj in target:
            obj[0] = self.namedict[obj[0]]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        sample = sample = {'image': image, 'label': target}

        return sample

    def __len__(self):
        return len(self.anns)

    def get_raw_item(self, index):

        ann = self.anns[index]

        filename = ann['filename']
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')

        target = copy.deepcopy(ann['objects'])

        return image, target


    def prepare_object_names(self):
        # Prepare the object names
        names = []
        for ann in self.anns:
            objects = ann['objects']
            for obj in objects:
                names.append(obj[0])
        return sorted(list(set(names)))

    def prepare_namedict(self):
        # Prepare the name dictionary

        namedict = {}
        for i, name in enumerate(self.names):
            namedict[name] = i

        return namedict

    def imshow(self, index):
        # Show the image with bbox
        # You might need to install ImageMagic for Image.show() to work properly.
        # e.g. sudo apt-get install imagemagick

        image, target = self.get_raw_item(index)

        for obj in target:
            obj[0] = self.namedict[obj[0]]

        #img, target = self.__getitem__(index)
        draw = ImageDraw.Draw(image)
        for obj in target:
            name = self.names[obj[0]]
            draw.rectangle(obj[1:5], outline=(255,0,0))
            draw.text(obj[1:3], name, fill=(0,255,0))

        image.show()



class TransformVOCDetectionAnnotation(object):
    def __init__(self, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            #name = obj.find('name').text
            name = obj[0].text.lower().strip()
            #bb = obj.find('bndbox')
            bbox = obj[4]
            #bndbox = [bb.find('xmin').text, bb.find('ymin').text,
            #    bb.find('xmax').text, bb.find('ymax').text]
            # supposes the order is xmin, ymin, xmax, ymax
            # attention with indices
            bndbox = [int(bb.text)-1 for bb in bbox]

            res += [bndbox + [name]]

        return res

class VOC2007Detection(Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        dataset_name = 'VOC2007'
        self._annopath = os.path.join(self.root, dataset_name, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self.root, dataset_name, 'ImageSets', 'Main', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()

        img = Image.open(self._imgpath % img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def show(self, index):
        img, target = self.__getitem__(index)
        draw = ImageDraw.Draw(img)
        for obj in target:
            draw.rectangle(obj[0:4], outline=(255,0,0))
            draw.text(obj[0:2], obj[4], fill=(0,255,0))
        img.show()


class VOC2007Segmentation(Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        dataset_name = 'VOC2007'
        self._annopath = os.path.join(self.root, dataset_name, 'SegmentationClass', '%s.png')
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self.root, dataset_name, 'ImageSets', 'Segmentation', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = Image.open(self._annopath % img_id)#.convert('RGB')

        img = Image.open(self._imgpath % img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)
