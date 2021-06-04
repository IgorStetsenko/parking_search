import cv2
import argparse

import time
import run_recognition

from run_camera import Camera_work
from time import sleep
from yolov5 import detect

from run_camera import Setting_camera
from data_calculate import Data_calculate

if __name__ == "__main__":  # main program algorithm
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--source', type=str, default="yolov5/test2.mp4",
                            help='source')  # file/folder, 0,1,2 for webcam
        opt = parser.parse_args()  # parse terminal
        source_file = opt.source
        cap = cv2.VideoCapture(source_file)
        Setting_camera().set_cap(cap)  # set camera

        while True:
            ret, frame1 = cap.read()
            ret2, frame2 = cap.read()
            move_detector, contour_area, frame_orig = Data_calculate().contours_search_and_filter(frame1, frame2)
            if move_detector:
                box = detect.detection_function(frame_orig)
                for i in box:
                    cv2.rectangle(frame_orig, i[0], i[1], (255, 0, 0), 2)
                cv2.imshow(str(contour_area), frame_orig)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                #print(box)
            #time.sleep(10)


        cap.release()
        cv2.destroyAllWindows()





    except NameError:
        print("Give source image, video or stream")

        #
        # source_image = opt.source
        # cap = cv2.VideoCapture(source_image)
        # # cap = cv2.VideoCapture(0)  # видео поток с веб камеры
        # cap.set(3, 240)  # установка размера окна
        # cap.set(4, 480)
        # cap.set(cv2.CAP_PROP_FPS, 25)
        # ret, frame1 = cap.read()
        # ret, frame2 = cap.read()
        # stop_detection = True
        # move_detector = Frame_utils()
        # cam_work = Camera_work()
        #
        # while True:  # метод isOpened() выводит статус видеопотока
        #     frame1 = cv2.rectangle(frame1, (1, 1), (960, 200), (0, 0, 0), -1) #Временное решение
        #     frame2 = cv2.rectangle(frame2, (1, 1), (960, 200), (0, 0, 0), -1)
        #     contours_filter, contour_area = move_detector.contours_search_and_filter(frame1,frame2)
        #     print(contours_filter, contour_area)
        #     stop_detection = [True if contours_filter else False]
        #     #cv2.imshow("frame1", frame1)
        #     sleep(0.01)
        #     frame1 = frame2
        #     ret, frame2 = cap.read()
        #     if cv2.waitKey(40) == 27:
        #         cap.release()
        #         cv2.destroyAllWindows()
        #         break
        #     if contours_filter and stop_detection:
        #         print(contours_filter,contour_area)
        #         frame2 = cv2.rectangle(frame2, (1, 1), (960, 200), (0, 0, 0), -1)
        #         prediction_boxes = detection_function(frame2)
        #         for i in prediction_boxes:
        #             cv2.rectangle(frame2, i[0], i[1], (255, 0, 0), 2)
        #         cam_work.image_show(frame2)
        #         print(prediction_boxes, "prediction_boxes")
        #     else:
        #         sleep(0.001)
        #         print("пауза")
        # cap.release()
        # cv2.destroyAllWindows()

    # boxes = Data_calculate()
    # boxes_intersection_array = boxes.boxes_intersection_search(prediction_boxes)

    # print(boxes_intersection_array)

    # sms = Sms_delivery()
    # sms.pull_sms()

    # prediction_boxes = detection_function(source="yolov5/videoplayback.avi")
    # prediction_boxes = detection_function(source_image)

    # cam = Camera_work()
    # cam.run_camera("yolov5/videoplayback.mp4")

    # main algoritm
