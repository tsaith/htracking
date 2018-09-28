import numpy as np
import matplotlib.pyplot as plt
import cv2

def imshow_bgr(image):
    """
    Show the image with the channel format as BGR.
    """
    # Roll BGR to RGB
    img = np.roll(image, 1, axis=2)
    plt.imshow(img)


def imshow_tensor(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def draw_bbox(image, bbox, label=None, color=None, thickness=5):
    """
    Draw the bounding box with label.
        image: ndarray
            Image to be draw.
        bbox: tuple or list
            Bounding box with format (xmin, ymin, xmax, ymax).
        label: str
            label of bbox.
        color: tuple
            color of bbox with RGB format, e.g. (r, g, b).
    """

    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]

    assert xmin < xmax and ymin < ymax

    # Width and height of bbox
    #width = xmax - xmin
    #height = ymax - ymin

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
    text_width, text_height = text_size[0][0], text_size[0][1]

    # Draw the rectangle
    region = np.array([[xmin-3,             ymin],
                       [xmin-3,             ymin-text_height-26],
                       [xmin+text_width+13, ymin-text_height-26],
                       [xmin+text_width+13, ymin]], dtype='int32')

    cv2.fillPoly(img=image, pts=[region], color=color)
    cv2.rectangle(img=image, pt1=(xmin, ymin), pt2=(xmax, ymax),
                  color=color, thickness=thickness)

    # Put the text
    cv2.putText(img=image, text=label, org=(xmin+13, ymin-13),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1e-3 * image.shape[0],
                color=(0,0,0), thickness=2)

    return image
