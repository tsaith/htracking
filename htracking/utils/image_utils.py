import numpy as np

class patch_extractor:
    """
    Image patch extractor.
    Assume that image has the format of (height, width, channels).
    """

    def __init__(self, width, height):

        self.image_w = width
        self.image_h = height
        self.bbox = None

    def locate(self, bbox):
        """
        Locate the patch.

        bbox: tuple
            Bounding box (xa, ya, width, height).
        """

        xa, ya, w, h = bbox

        assert w <= self.image_w
        assert h <= self.image_h

        xa_max = self.image_w - w
        ya_max = self.image_h - h

        if xa < 0:
            xa = 0
        if xa > xa_max:
            xa = xa_max
        if ya < 0:
            ya = 0
        if ya > ya_max:
            ya = ya_max

        xz = xa + w
        yz = ya + h

        self.bbox = (xa, ya, w, h)

    def extract(self, image):
        """
        Extract the patch from an image.
        """
        xa, ya, w, h = self.bbox
        patch = image[ya:ya+h, xa:xa+w]

        return patch

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
