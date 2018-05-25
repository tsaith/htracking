import xml.etree.ElementTree as ET
import argparse
import os
import re

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--old_dataset", required=False, default='~/old_datasets', help="Old dataset path.")
ap.add_argument("-d", "--dataset", required=False, default='.', help="Dataset path.")
ap.add_argument("-a", "--annotation", required=False, default='train_annotations', help="Training annotation folder")
args = vars(ap.parse_args())

# Arguments
OLD_DATASET_PATH = args['old_dataset']
DATASET_PATH = args['dataset']
ANNOTATIONS_PATH= os.path.join(DATASET_PATH, args['annotation'])

print("Modifying the path within the annotation files ...")

if not os.path.isdir(ANNOTATIONS_PATH):
    print("Annotation folder: {} doesn't exist!".format(ANNOTATIONS_PATH))

for filename in os.listdir(ANNOTATIONS_PATH):

    filepath = os.path.join(ANNOTATIONS_PATH, filename)
    tree = ET.parse(filepath)
    root = tree.getroot()
    old_text = root.find('path').text
    root.find('path').text= old_text.replace(OLD_DATASET_PATH, DATASET_PATH)
    tree.write(filepath)

print("End of run.")
