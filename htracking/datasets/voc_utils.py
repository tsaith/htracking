from .voc_annotation import VOCAnnotation
import os
import csv
import copy
from PIL import Image, ImageDraw


def class_filter(anns_in, classes):
    """
    Annotation class filter.
    """
    anns = []
    for ann_in in anns_in:
        ann = copy.deepcopy(ann_in)
        #ann = ann_in.copy()
        objects = [] # New object list
        for obj in ann['objects']:
            name = obj[0]
            if name in classes:
                objects.append(obj)
        ann['objects'] = objects

        # If object exists
        if len(objects) > 0:
            anns.append(ann)

    return anns

def class_map(anns_in, mapping):
    """
    Annotation class mapping.
    """

    anns = copy.deepcopy(anns_in)
    for ann in anns:
        for obj in ann['objects']:
            name = obj[0]
            if name in mapping.keys():
                obj[0] = mapping[name]

    return anns

def csv_to_voc(csv_path, xml_dir_path):
    # Convert one csv file into xml files with Pascal VOC format.

    if not os.path.exists(xml_dir_path):
        os.makedirs(xml_dir_path)

    csv_file = open(csv_path, 'r')
    reader = csv.DictReader(csv_file)

    ann_saved = None
    for row in reader:

        filename = row['filename']
        width = row['width']
        height = row['height']
        class_name = row['class']
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']

        ann = VOCAnnotation(filename, width, height, depth=3)
        new_object = [class_name, 0, xmin, ymin, xmax, ymax]
        ann.add_object(new_object)

        # Merge the objects
        if not ann_saved is None:
            if ann.filename == ann_saved.filename:
                for obj in ann_saved.objects:
                    ann.add_object(obj)

        # Output annotation file
        xml_name = os.path.splitext(filename)[0] + '.xml'
        xml_path = os.path.join(xml_dir_path, xml_name)
        ann.output_xml(xml_path)

        # Save the previous annotation
        ann_saved = ann


    csv_file.close()


