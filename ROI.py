import cv2
import time
from copy import deepcopy

THICKNESS = 3
COLOR_MAP = {
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0)
}


def on_mouse_select_rectangle(event, x, y, flags, param):
    """
    This function aims for set the mouse call back to select a rectangle
    """
    global img, point1, point2, left_button_up
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        # when left mouse button is pressed, get a point
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        # when hold left mouse button, draw a rectangle
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), thickness=THICKNESS)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_LBUTTONUP:
        # when release the left mouse button, get a rectangle
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), thickness=THICKNESS)
        left_button_up = False # notice that it has been released the left mouse button


def on_mouse_select_line(event, x, y, flags, param):
    """
    This function aims for set the mouse call back to select a line
    """
    global img, point, min_y, left_button_up, pt1, pt2
    img2 = img.copy()
    pt1 = (x,min_y)
    pt2 = (x,min_y+height)
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        pt1 = (x, min_y)
        pt2 = (x, min_y + height)
        left_button_up = 0
    elif event == cv2.EVENT_MOUSEMOVE:
        cv2.line(img2,pt1=pt1, pt2=pt2,color=(0, 0, 255), thickness=THICKNESS)
        cv2.imshow('image', img2)


def select_ROI(capture):
    """
    This function aims for drawing a region of interest in the canvas for counting vehicles
    :param capture:
    :return: roi_start_point, roi_end_point are the top left and bottom right points of target rectangle
             left_line_start_point, left_line_end_point are the ends of the left line for counting in vehicles
             right_line_start_point, right_line_end_point are the ends of the right line for counting out vehicles
    """
    _, frame = capture.read()
    time.sleep(0.5)  # waiting for opening the camera
    _, frame = capture.read()
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    global img, left_button_up, min_x, min_y, width, height, pt1, pt2
    img = frame

    # get the target rectangle
    left_button_up = True
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse_select_rectangle)
    cv2.imshow('image', img)
    cv2.moveWindow("image", 0, 0)
    while left_button_up:
        key = cv2.waitKey(1)

    if point1 != point2:
        min_x = min(point1[0], point2[0])
        if min_x > frame_width:
            min_x = frame_width
        min_y = min(point1[1], point2[1])
        if min_y > frame_height:
            min_y = frame_height
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])


    roi_start_point = (min_x, min_y)
    roi_end_point = (min_x + width, min_y + height)

    if roi_end_point[0] > frame_width:
        roi_end_point = (frame_width, roi_end_point[1])
    if roi_end_point[1] > frame_height:
        roi_end_point = (roi_end_point[0], frame_height)

    # get the target lines for counting in or out vehicle
    left_button_up = True
    cv2.rectangle(img=frame, pt1=roi_start_point, pt2=roi_end_point, color=COLOR_MAP["green"], thickness=THICKNESS)
    cv2.setMouseCallback('image', on_mouse_select_line)
    cv2.imshow('image', img)
    while left_button_up:
        cv2.waitKey(1)

    left_line_start_point = pt1
    left_line_end_point = pt2

    left_button_up = True

    cv2.rectangle(img=frame, pt1=roi_start_point, pt2=roi_end_point, color=COLOR_MAP["green"], thickness=THICKNESS)
    cv2.line(img=frame, pt1=left_line_start_point, pt2=left_line_end_point, color=COLOR_MAP["red"], thickness=THICKNESS)
    cv2.imshow('image', img)

    while left_button_up:
        cv2.waitKey(1)

    right_line_start_point = pt1
    right_line_end_point = pt2
    cv2.destroyAllWindows()

    if left_line_start_point[0] > right_line_start_point[0]:
        # if the left line is on the right of right line
        t = deepcopy(left_line_start_point)
        left_line_start_point = deepcopy(right_line_start_point)
        right_line_start_point = deepcopy(t)
        t = deepcopy(left_line_end_point)
        left_line_end_point = deepcopy(right_line_end_point)
        right_line_end_point = deepcopy(t)

    if left_line_start_point[0] < roi_start_point[0]:
        # if the left line is on the left of roi
        left_line_start_point = (roi_start_point[0], left_line_start_point[1])
        left_line_end_point = (roi_start_point[0], left_line_end_point[1])
    elif left_line_start_point[0] > roi_end_point[0]:
        # if the left line is on the right of roi
        left_line_start_point = (roi_end_point[0], right_line_start_point[1])
        left_line_end_point = (roi_end_point[0], right_line_end_point[1])

    if right_line_start_point[0] > roi_end_point[0]:
        # if the right line is on the right of roi
        right_line_start_point = (roi_end_point[0], right_line_start_point[1])
        right_line_end_point = (roi_end_point[0], right_line_end_point[1])
    elif right_line_start_point[0] < roi_start_point[0]:
        # if the right line is on the left of roi
        right_line_start_point = (roi_start_point[0], left_line_start_point[1])
        right_line_end_point = (roi_start_point[0], left_line_end_point[1])

    return roi_start_point, roi_end_point, left_line_start_point, left_line_end_point, \
           right_line_start_point, right_line_end_point
