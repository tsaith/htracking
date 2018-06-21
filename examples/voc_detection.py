import htracking
from htracking.datasets import VOCDetection, xml_annotations
from htracking.datasets.voc_utils import voc_imshow

image_dir = '/home/andrew/projects/datasets/voc_2007/VOCdevkit/VOC2007/JPEGImages'
ann_dir = '/home/andrew/projects/datasets/voc_2007/VOCdevkit/VOC2007/Annotations'

# Define dataset
ds = VOCDetection(image_dir, ann_dir)
img, target = ds[1]

# Show image
ds.show(1)
