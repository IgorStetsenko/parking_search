import cv2
import argparse
from yolov5.utils.datasets import LoadImages
from yolov5.detect import detection_function

test_image_path = 'screen.jpg'

class Camera_work():

    def run_camera(self):
        '''Doc'''
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            cv2.imshow("camera", img)
            if cv2.waitKey(10) == 27: # Клавиша Esc
                break
        cap.release()
        cv2.destroyAllWindows()

    def screen_photo(self):
        '''Doc'''
        pass

    def write_photo(self):
        '''Doc'''
        pass

    def write_video(self):
        '''Doc'''
        pass

class Data_calculate():
    def calculate_iou(self, boxes):
        """Calculate iou-metrics function.
            imput: boxes predicted
            output: iouarray"""
        pass

    def boxes_intersection_search(self):
        '''This function do boxes intersection search'''
        pass



class Sms_delivery():

    def pull_sms(self):
        '''Doc'''
        pass

class Detect():
    def run_detect(self):
        '''Doc'''
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0,1,2 for webcam
    opt = parser.parse_args()

    prediction_boxes = detection_function()
    print(prediction_boxes)