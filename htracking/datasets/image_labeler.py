from .voc_utils import write_voc_file
from torchvision import transforms
import os
import glob
from PIL import Image

class ImageLabeler(object):

    def __init__(self, ann_dir, image_dir, model,
                 ann_ext='.xml', image_ext='.jpg', image_channels=3):

        self.ann_dir = ann_dir
        self.image_dir = image_dir
        self.model = model
        self.ann_ext = ann_ext
        self.image_ext = image_ext
        self.image_channels = image_channels
        self.image_transform = transforms.Compose([transforms.ToTensor()])

        # Image file paths
        if os.path.exists(image_dir):
            rexpress = "*"+self.image_ext
            self.image_paths = glob.glob(os.path.join(self.image_dir, rexpress))
        else:
            print("Error: Image directory doesn't exist!")
            print("Error: Expected path is {}".format(image_dir))


    def run(self):

        anns = []
        for image_path in self.image_paths:

            basename = os.path.basename(image_path)
            filename, ext = os.path.splitext(basename)

            ann_basename = filename + self.ann_ext
            ann_path =  os.path.join(self.ann_dir,ann_basename)

            # Detect the image to objects
            image = Image.open(image_path)
            width, height = image.size
            image = self.image_transform(image)
            image = image.unsqueeze(0) # Add dimention

            detections = self.model.detect(image)

            # Output the annotation file
            if detections is not None:
                objects = self.model.objects_from(detections)

                image_size = (height, width, self.image_channels)
                print("Output {}".format(ann_path))
                write_voc_file(objects, ann_path, filename, image_path, image_size)

        print("Bye~")
