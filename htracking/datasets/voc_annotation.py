from .voc_io import VOCWriter

class VOCAnnotation:
    '''
    Pascal VOC Annotation.
    '''

    def __init__(self, filename, width, height, depth=3, filepath=""):
        self.filename = filename
        self.filepath = filepath
        self.width = width
        self.height = height
        self.depth = depth

        # Objects to be detected
        self.objects = []

    def add_object(self, obj):
        # Add object with format as [class_name, difficulty, xmin, ymin, xmax, ymax]
        self.objects.append(obj)

    def output_xml(self, path):
        # Output xml file.

        dirname = 'images'
        image_size = [self.height, self.width, self.depth]
        image_path = self.filepath

        writer = VOCWriter(dirname, self.filename, imgSize=image_size, databaseSrc='Unknown', 
                                 localImgPath=image_path)
        root = writer.genXML()
        writer.appendObjects(root)

        # Bounding boxes
        for obj in self.objects:
            name, difficulty, xmin, ymin, xmax, ymax = obj
            writer.addBndBox(xmin, ymin, xmax, ymax, name, difficulty)

        writer.save(path)
