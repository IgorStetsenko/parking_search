import cv2
import argparse
import time
from yolov5 import detect
from run_camera import Setting_camera, Camera_work
from data_calculate import Data_calculate
from sms import Sms_delivery

if __name__ == "__main__":  # main program algorithm
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--source', type=str, default="yolov5/test2.mp4",
                            help='source')  # file/folder, 0,1,2 for webcam
        opt = parser.parse_args()  # parse terminal
        source_file = opt.source
        cap = cv2.VideoCapture(source_file)
        Setting_camera().set_cap(cap)  # set camera
        status_frame_init, frame_init = cap.read()
        frame_init = Data_calculate().crop_image(frame_init, 200, 960, 0, 720)
        parking_space_box = detect.detection_function(frame_init)  # init #the park space box

        while True:
            status_frame_1, frame1 = cap.read()
            status_frame_2, frame2 = cap.read()
            frame1 = Data_calculate().crop_image(frame1, 200, 960, 0, 720)
            frame2 = Data_calculate().crop_image(frame2, 200, 960, 0, 720)
            move_detector, contour_area, frame_orig = Data_calculate().contours_search_and_filter(frame1, frame2)
            if move_detector:  # if move_detector == True (Box>5000 px)
                box_auto = detect.detection_function(frame_orig)
                parking_space_box = Data_calculate().update_parking_space_box(parking_space_box, box_auto)
                print(len(parking_space_box))
                time.sleep(10)

                if Data_calculate().get_status_space(parking_space_box, box_auto):
                    Sms_delivery().pull_sms()

    except NameError:
        print("Give source image, video or stream")
