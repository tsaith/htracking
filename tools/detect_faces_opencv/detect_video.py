import numpy as np
import cv2
import time

from htracking.face_detection import FaceDetector
from htracking.utils import wait_key


# Video path
video_file = 'test_faces.mp4'
video_path = "/home/andrew/projects/datasets/faces/test/video/{}".format(video_file)
output_path = "output.mp4"

# Net prototxt and model path
prototxt_path = "/home/andrew/projects/datasets/model_zoo/face_models/caffe/deploy.prototxt.txt"
model_path = "/home/andrew/projects/datasets/model_zoo/face_models/caffe/res10_300x300_ssd_iter_140000.caffemodel"

# Initialize the face detecotr
detector = FaceDetector(prototxt_path, model_path)

# Read frames form webcam or video file
video = video_path
if video is None: # Use the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height);

else:
    cap = cv2.VideoCapture(video)

# In case the video can't be open
if not cap.isOpened():
    print("The video cann't be open!")
    exit(0)

# Creat a window
win_name = 'test'
cv2.namedWindow(win_name)

_, first_frame = cap.read()

# Determine the width and height from the first image
height, width, channels = first_frame.shape

# Patch extractor
#extractor = patch_extractor(width, height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
fps_out = 3
out = cv2.VideoWriter(output_path, fourcc, fps_out, (width, height))

# Initialize the compresive tracker
rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

# loop over the frames of the video
print("Start to run.")
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = cap.read()

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed: break

    # Detect the image
    ta = time.time()
    detections = detector.detect(frame)
    dt = time.time() - ta
    #print("Sampleing rate is {}(Hz)".format(1.0/dt))

    # Draw info
    detector.draw_info(frame, detections)

    # show the frame and record if the user presses a key
    cv2.imshow(win_name, frame)
    out.write(frame) # Write out frame to video

    # Wait for a while
    #time.sleep(0.1)

    # if the `q` key is pressed, breaqk from the lop
    key = wait_key(1)
    if key == ord("q"): break

# Release the resources and close any open windows
print("Finalize the application.")
cap.release()
out.release()
cv2.destroyAllWindows()
