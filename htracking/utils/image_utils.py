import numpy as np

def grab_patch_bbox(bbox_in, image_width, image_height):
    # Return the bbox of image patch.

    bbox_x, bbox_y, bbox_width, bbox_height = bbox_in
    bbox_width = max(bbox_width, bbox_height)
    bbox_height = bbox_width

    if bbox_x+bbox_width > image_width:
        bbox_x = image_width - bbox_width

    if bbox_y+bbox_height > image_height:
        bbox_y = image_height - bbox_height

    return (bbox_x, bbox_y, bbox_width, bbox_height)

def grab_image_patch(bbox, image):
    # Grab image patch.

    x, y, width, height = bbox
    image_out = image[y:y+height+1, x:x+width+1]

    return image_out
