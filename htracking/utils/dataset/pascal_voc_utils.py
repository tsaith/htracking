from htracking.utils.dataset import PascalVocAnnotation
import os
import csv


def csv_to_pascal_voc(csv_path, xml_dir_path):
    # Convert one csv file into xml files with Pascal Voc format.

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

        ann = PascalVocAnnotation(filename, width, height, depth=3)
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


