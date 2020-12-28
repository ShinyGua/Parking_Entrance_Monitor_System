import cv2
import math
from copy import deepcopy
from model import *
import numpy as np

from tensorflow import lite

EXTRA_SECTION = 30


def distance(x, y):
    return math.sqrt(float((x[0] - y[0]) ** 2) + float((x[1] - y[1]) ** 2))


def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)


def filter_mask(img):
    """
        This filters are hand-picked just based on visual tests
    """


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    #img = cv2.medianBlur(img, 5)

    # Fill any small holes
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=2)

    return dilation


def detect_vehicles(roi_frame, fg_mask, min_contour_width, min_contour_height, max_contour_width, max_contour_height):

    """
    This function aims to find external contours in the fg_mask
    :param fg_mask:
    :param min_contour_width: min bounding rectangle width
    :param min_contour_height: min bounding rectangle height
    :return: matches including the left bottom position, width, height and centroid position of detected objects
             e.g.: ((x, y, w, h), (centroid_x,centroid_y))
    """

    matches = []

    _, thresh = cv2.threshold(fg_mask, 127, 255, 0)

    # finding external contours
    #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    if len(contours) > 10:
        return

    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height) \
                        and (w <= max_contour_width) and (h <= max_contour_height)

        if not contour_valid:
            continue

        centroid = get_centroid(x, y, w, h)

        img = roi_frame[y:(y+h),x:(x+w)]
        matches.append(((x, y, w, h), centroid, predict(img)))


    return matches


def creat_paths(matches, paths, x_left_border, x_right_border, max_des):
    """
    This function aims to creat detected objects
    :param matches: detected objects
    :param paths: the old paths of detected objects
    :param x_left_border: the left border of checking cars in or out
    :param x_right_border: the left border of checking cars in or out
    :param max_des: the max number of path points in each path
    :return: the new paths based on new detected objects
    """
    points = np.array(matches)[:, 0:3]
    points = points[points[:,2] == 0]
    points = points[:,0:2]
    points = points.tolist()

    if not paths:
        # if paths is empty, create a new paths
        for p in points:
            if x_left_border - EXTRA_SECTION < p[1][0] < x_right_border + EXTRA_SECTION:
                # only recording point in and around the left border and right border
                paths.append([p])
    else:
        new_paths = []

        for path in paths:
            _min = float('inf')
            _match = None
            d = float('inf')
            for p in points:
                if not (x_left_border - EXTRA_SECTION < p[1][0] < x_right_border + EXTRA_SECTION):
                    # only recording point in and around the left border and right border
                    continue
                if len(path) != 1:
                    # based on 2 prev points predict next point and calculate
                    # distance from predicted next point to current
                    xn = 2 * path[-1][0][0] - path[-2][0][0]
                    yn = 2 * path[-1][0][1] - path[-2][0][1]
                    d = distance(p[0], (xn, yn))
                else:
                    # distance from last point to current
                    d = distance(p[0], path[-1][0])

                if d < _min:
                    _min = d
                    _match = p

            if _match and _min <= max_des:
                points.remove(_match)
                path.append(_match)
                new_paths.append(path)

            # do not drop path if current frame has no matches
            if _match is None:
                new_paths.append(path)

        paths = new_paths

        if len(points):
            for p in points:
                if not (x_left_border - EXTRA_SECTION < p[1][0] < x_right_border + EXTRA_SECTION):
                    # only recording point in and around the left border and right border
                    continue
                paths.append([p])

    return deepcopy(paths)
