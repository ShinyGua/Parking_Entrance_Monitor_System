import cv2
from ROI import select_ROI
from detect_vehicle import *
from drew import *
import time


VIDEO_SOURCE = "data/13550828_test.h264" # 0 for using camera
MANUAL_SELECT_ROI = False
# True for manually selecting the region of interest, False for hard-code setting the region

# hard-code parameter of region of interest
WIDTH_ROI_RATE = .75
HEIGHT_ROI_RATE = .6
LEFT_LINE_RATE =  .33
RIGHT_LINE_RATE = .43 

# set for skipping every JUMP_FRAME_RATE frame to speed up processing
SKIP_FRAME_RATE = 1
varThreshold = 50

# set the bounding rectangle width and height of a car
MIN_CONTOUR_WIDTH = 35/2
MIN_CONTOUR_HEIGHT = 25/2
MAX_CONTOUR_WIDTH_RATE = 0.7
MAX_CONTOUR_HEIGHT_RATE = 0.7

# set the max distance between two path points
MAX_DES = 25

# set the max number of path points in each path
PATH_SIZE = 150

APPROXIMATE_SECTION = 25

FPS = int(24)
FRAME_WIDTH = int(640/1)
FRAME_HEIGHT = int(360/1)

COLOR_MAP = {
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue" :(255, 0, 0)
}



def main():
    current_time = int(time.time())

    vehicle_count = 0
    in_vehicle_count = 0
    out_vehicle_count = 0

    # set the parameters for the video
    capture = cv2.VideoCapture(VIDEO_SOURCE)
    # capture.set(cv2.CAP_PROP_FPS, FPS)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # width = int(capture.get(3))
    # height = int(capture.get(4))
    width = FRAME_WIDTH
    height = FRAME_HEIGHT

    if MANUAL_SELECT_ROI:
        # if manually select the region of interest
        # the left and right borders used for checking cars in or out
        roi_start_point, roi_end_point, left_line_start_point, left_line_end_point, \
        right_line_start_point, right_line_end_point = select_ROI(capture)
    else:
        # if hard-code set the region of interest
        width_roi_rate = 1 - WIDTH_ROI_RATE
        height_roi_rate = 1 - HEIGHT_ROI_RATE

        # compute the position of ROI and left border and right border on the frame
        # the left and right borders used for checking cars in or out
        roi_start_point = (int(width * 0.5 * width_roi_rate),int(height * height_roi_rate))
        roi_end_point = (int(width * (1 - 0.5 * width_roi_rate)),height)
        left_line_start_point = (int(width * LEFT_LINE_RATE), int(height * height_roi_rate))
        left_line_end_point = (int(width * LEFT_LINE_RATE), height)
        right_line_start_point = (int(width * RIGHT_LINE_RATE), int(height * height_roi_rate))
        right_line_end_point = (int(width * RIGHT_LINE_RATE), height)

    # compute the position of ROI and left border and right border on the fg_mask.
    # There is a offset, roi_start_point, between ROI and frame
    left_sub_line_start_point = (left_line_start_point[0] - roi_start_point[0], 0)
    left_sub_line_end_point = (left_line_start_point[0] - roi_start_point[0], roi_end_point[1] - roi_start_point[1])
    right_sub_line_start_point = (right_line_start_point[0] - roi_start_point[0], 0)
    right_sub_line_end_point = (right_line_start_point[0] - roi_start_point[0], roi_end_point[1] - roi_start_point[1])

    backSub = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=varThreshold, detectShadows=False)
    cv2.namedWindow('capture')
    cv2.namedWindow('FG Mask')
    cv2.moveWindow("capture", 0, 0)
    cv2.moveWindow("FG Mask", 0, height+50)
    count = 1

    paths = []

    x_left_border = left_line_start_point[0] - roi_start_point[0]
    x_right_border = right_line_start_point[0] - roi_start_point[0]

    max_contour_width = MAX_CONTOUR_WIDTH_RATE * (roi_end_point[0] - roi_start_point[0])
    max_contour_height = MAX_CONTOUR_WIDTH_RATE * height * (roi_end_point[1] - roi_start_point[1])

    while True:
        _, frame = capture.read()
        frame = cv2.resize(frame,(width,height))
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        count = count + 1
        # skipping every JUMP_FRAME_RATE frame to speed up processing

        if count % SKIP_FRAME_RATE != 0:
            continue

        current_fps = round(1/(time.time() - current_time),2)
        current_time = time.time()


        if frame is None:
            break

        # trim the region of interest from frame
        roi_frame = frame[roi_start_point[1]:roi_end_point[1], roi_start_point[0]:roi_end_point[0]]

        fg_mask = backSub.apply(roi_frame, None, 0.0005)
        fg_mask[fg_mask < 200] = 0
        fg_mask = filter_mask(fg_mask)

        matches = detect_vehicles(roi_frame, fg_mask, MIN_CONTOUR_WIDTH, MIN_CONTOUR_HEIGHT, max_contour_width, max_contour_height)

        draw_boxes(frame, fg_mask, matches, roi_start_point)

        if matches:

            paths = creat_paths(matches, deepcopy(paths), x_left_border, x_right_border, MAX_DES)

            new_paths = []

            for i, path in enumerate(paths):
                paths[i] = paths[i][PATH_SIZE * -1:]  # save only last N points in path

                if x_left_border < path[len(path)-1][1][0] < x_right_border:
                    # if the last point is the between the left and right boundary, add to the paths
                    # otherwise drop these point is in the out of the left and right boundary
                    new_paths.append(path)

                if path[len(path)-1][1][0] <= x_left_border and path[0][1][0] > x_right_border - APPROXIMATE_SECTION:
                    # if the last point is on the left of the left border and the first point is around right border,
                    # the detected object is coming into the parking space
                    in_vehicle_count = in_vehicle_count + 1
                    vehicle_count = in_vehicle_count - out_vehicle_count
                    print("A vehicle drives into parking space. There is ", vehicle_count)

                if path[len(path)-1][1][0] >= x_right_border and path[0][1][0] < x_left_border + APPROXIMATE_SECTION:
                    # if the last point is on the right of the right border and the first point is around left border,
                    # the detected object is leaving the parking space
                    out_vehicle_count = out_vehicle_count + 1
                    vehicle_count = in_vehicle_count - out_vehicle_count
                    print("A vehicle drives out of parking space. There is ", vehicle_count)

            paths = new_paths

        draw_paths(img=frame, paths=paths, offset=roi_start_point)

        draw_frame_ui(img=frame, roi=(roi_start_point, roi_end_point), line1=(left_line_start_point, left_line_end_point),
                      line2=(right_line_start_point, right_line_end_point), total_org=(int(width / 2 - 70), 30), in_org=(30, 30),
                      out_org=(int(width-130),30), vehicle_count=vehicle_count, in_vehicle_count=in_vehicle_count,
                      out_vehicle_count=out_vehicle_count,fps=current_fps)

        draw_fg_mask_ui(img=fg_mask, line1=(left_sub_line_start_point, left_sub_line_end_point), line2=(right_sub_line_start_point, right_sub_line_end_point))

        cv2.imshow("capture", frame)
        cv2.imshow("FG Mask", fg_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
