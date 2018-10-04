import cv2

def wait_key(delay):
    """
    Wait for a pressed key.

    Parameters
    ---------
    delay: Delay in milliseconds. 0 is the special value that means "forever"".
    """
    return cv2.waitKey(delay) & 0xFF

def define_rect(image):
    """
    Define a rectangular window by click and drag the mouse.

    Parameters
    ----------
    image: Input image.
    """

    clone = image.copy()
    rect_pts = [] # Starting and ending points
    win_name = "image" # Window name

    def select_points(event, x, y, flags, param):

        nonlocal rect_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]

        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))

            # draw a rectangle around the region of interest
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow(win_name, clone)
        key = wait_key(0)
        #key = cv2.waitKey(0) & 0xFF

        if key == ord("r"): # Hit 'r' to replot the image
            clone = image.copy()

        elif key == ord("c"): # Hit 'c' to confirm the selection
            break

    if len(rect_pts) != 2:
        print("Error: please define a bounding box by your mouse.")

    # close the open windows
    cv2.destroyWindow(win_name)

    return rect_pts
