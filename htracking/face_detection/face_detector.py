import numpy as np
import cv2

class FaceDetector(object):
    """
    Face Detector based on OpenCV.
    """

    def __init__(self, prototxt_path, model_path, confidence_thresh=0.5):
        """
        prototxt_path: str
            Path to the prototxt which contains the architecture of network.
        model_path: str
            Path to the model file.
        confidence_thresh: float
            Confidence threshold.
        """

        self.prototxt_path = prototxt_path
        self.model_path = model_path
        self.confidence_thresh = confidence_thresh

        # Initialize the network
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def detect(self, image):
        """
        Detect the image.

        Parameters
        ----------
        image: array
            Image array of opencv format.
        """

        # load the input image and construct an input blob for the image and resize image to
        # fixed 300x300 pixels and then normalize it
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, 
                                     (300, 300), (103.93, 116.77, 123.68))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        outputs = self.net.forward()

        detections = []
        # loop over the detections
        for i in range(0, outputs.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = outputs[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.confidence_thresh:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = outputs[0, 0, i, 3:7] * np.array([w, h, w, h])
                bbox = box.astype("int")

                detection = [bbox, confidence]
                detections.append(detection)

        return detections

    def draw_info(self, image, detections):
        """
        Detect the image.

        Parameters
        ----------
        image: array
            Image array of opencv format.
        """

        for detection in detections:

            bbox, confidence = detection
            # draw the bounding box of the face along with the associated
            # probability
            (startX, startY, endX, endY) = bbox
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)



