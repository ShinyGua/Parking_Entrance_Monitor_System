import numpy as np
import cv2

COLOR_MAP = {
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (255, 255, 0),
    "black": (0, 0, 0)
}

LINE_THICKNESS = 3
TEXT_THICKNESS = 2
BOXES_THICKNESS = 1


def draw_frame_ui(img, roi, line1, line2, total_org, in_org, out_org, vehicle_count, in_vehicle_count,
                  out_vehicle_count,fps):
    """
    This function aims to draw UI on frame
    :param img: frame
    :param roi: the start point and end point of ROI, e.g.: (start_point, end_point)
    :param line1: the start point and end point of left line, e.g.: (start_point, end_point)
    :param line2: the start point and end point of right line, e.g.: (start_point, end_point)
    :param total_org: the position of total number of cars text
    :param in_org: the position of total number of in cars text
    :param out_org: the position of total number of out cars text
    :param vehicle_count: total number of cars
    :param in_vehicle_count: total number of in cars
    :param out_vehicle_count: total number of out cars
    :return: None
    """

    cv2.rectangle(img=img, pt1=roi[0], pt2=roi[1], color=COLOR_MAP["red"], thickness=LINE_THICKNESS)
    cv2.line(img=img, pt1=line1[0], pt2=line1[1], color=COLOR_MAP["green"], thickness=LINE_THICKNESS)
    cv2.line(img=img, pt1=line2[0], pt2=line2[1], color=COLOR_MAP["green"], thickness=LINE_THICKNESS)

    cv2.putText(img=img, text="Total: " + str(vehicle_count), org=total_org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=COLOR_MAP["red"], thickness=TEXT_THICKNESS)
    cv2.putText(img=img, text="FPS: " + str(fps), org=(total_org[0],total_org[1]+30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=COLOR_MAP["red"], thickness=TEXT_THICKNESS)
    cv2.putText(img=img, text="In: " + str(in_vehicle_count), org=in_org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=COLOR_MAP["red"], thickness=TEXT_THICKNESS)
    cv2.putText(img=img, text="Out: " + str(out_vehicle_count), org=out_org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=COLOR_MAP["red"], thickness=TEXT_THICKNESS)



def draw_fg_mask_ui(img, line1, line2):
    """
    This function aims to draw UI on fg_mask
    :param img: fg_mask
    :param line1: the start point and end point of left line on the fg_mask, e.g.: (start_point, end_point)
    :param line2: the start point and end point of right line on the fg_mask, e.g.: (start_point, end_point)
    :return: None
    """
    cv2.line(img=img, pt1=line1[0], pt2=line1[1], color=COLOR_MAP["white"], thickness=LINE_THICKNESS)
    cv2.line(img=img, pt1=line2[0], pt2=line2[1], color=COLOR_MAP["white"], thickness=LINE_THICKNESS)


def draw_boxes(frame, fg_mask, matches, offset):
    """
    This function aims to draw a bounding box around detected objects
    :param frame:
    :param fg_mask:
    :param matches: the detected objects
    :param offset: the offset between the frame and the region of interest
    :return: None
    """
    if not matches:
        return
    for match in matches:
        fg_mask_start_point = (match[0][0], match[0][1])
        fg_mask_end_point = (match[0][0] + match[0][2], match[0][1] + match[0][3])

        frame_start_point = (match[0][0] + offset[0], match[0][1] + offset[1])
        frame_end_point = (match[0][0] + offset[0] + match[0][2],
                           match[0][1] + offset[1] + match[0][3])
        if match[2]:
            cv2.rectangle(frame, frame_start_point, frame_end_point, COLOR_MAP["red"], BOXES_THICKNESS)
        else:
            cv2.rectangle(frame, frame_start_point, frame_end_point, COLOR_MAP["green"], BOXES_THICKNESS)
        cv2.rectangle(fg_mask, fg_mask_start_point, fg_mask_end_point, COLOR_MAP["white"], BOXES_THICKNESS)


def draw_paths(img, paths, offset):
    """
    This function aims to draw paths of vehicles on the frame
    :param img: the target image
    :param paths: the vehicles' paths
    :param offset: the offset between the frame and the region of interest
    :return: None
    """
    for i, path in enumerate(paths):
        path = (np.array(path)[:, 1]).tolist()
        path = (np.array(path) + np.array(offset)).tolist()

        for point in path:
            cv2.circle(img, (point[0], point[1]), 2, COLOR_MAP["yellow"], -1)
            cv2.polylines(img, [np.int32(path)], False, COLOR_MAP["yellow"], 1)



