import cv2
import argparse
from yolov5.utils.datasets import LoadImages
from yolov5.detect import detection_function
from twilio.rest import Client




class Camera_work():

    def run_camera(self, source):
        """Doc"""
        cap = cv2.VideoCapture(source)
        while True:
            ret, img = cap.read()
            cv2.imshow("camera", img)
            if cv2.waitKey(10) == 27:  # Клавиша Esc
                break
        cap.release()
        cv2.destroyAllWindows()

    def read_video(self):
        """Doc"""

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

    def iou_calculate(self, box1, box2):
        """Функция рассчитывает метрику IoU и проверяет пересечение прямоугольников"""
        flag = False
        ax1, ay1, ax2, ay2 = box1[0][0], box1[0][1], box1[1][0], box1[1][1]
        bx1, by1, bx2, by2 = box2[0][0], box2[0][1], box2[1][0], box1[1][1]
        if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
            flag = True
            sq_a = abs(ax1 - ax2) * abs(ay1 - ay2)
            sq_b = abs(bx1 - bx2) * abs(by1 - by2)
            interArea = abs(max([ax1, ax2]) - min([bx1, bx2])) * abs(max([ay1, ay2]) - min([by1, by2]))
            iou = interArea / float(abs(abs(sq_a + sq_b) - interArea))
            return iou, flag
        else:
            flag = False
            return flag

    def boxes_intersection_search(self, prediction_boxes):
        """This function do boxes intersection search"""
        j = 0
        for n in prediction_boxes:
            i = prediction_boxes.index(n)
            if j == len(prediction_boxes) - 1:
                break
            elif i != len(prediction_boxes) - 1:
                while i != len(prediction_boxes) - 1:
                    a = prediction_boxes[i + 1]
                    self.iou_calculate(n, a)
                    #                 print(n == a, n, a)
                    i += 1
                j += 1
            else:
                print('Error')

class Sms_delivery():

    def pull_sms(self):
        '''The function sms delivery'''
        # Twilio account details
        twilio_account_sid = ''
        twilio_auth_token = ''
        twilio_source_phone_number = '+'

        # Create a Twilio client object instance
        client = Client(twilio_account_sid, twilio_auth_token)

        # Send an SMS
        message = client.messages.create(
            body="Seat is free!!!",
            from_=twilio_source_phone_number,
            to=""
        )

def main_algorithm(prediction_boxes):
    """

    :param prediction_boxes:
    :return:
    """
    temp_parking_seat = []
    iou=Data_calculate()
    temp_parking_seat = prediction_boxes.copy()

    for box_original in prediction_boxes:
        i=0
        temp_parking_seat=temp_parking_seat[i]
        print(temp_parking_seat,i, '----')
        print(box_original,i, '---+')
        iou.iou_calculate(box_original, temp_parking_seat)
        i+=1

    pass


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--source', type=str, default='yolov5/default_img.jpg', help='source')  # file/folder, 0,1,2 for webcam
        opt = parser.parse_args()

        source_image = opt.source

        if source_image == '0' or '1' or '2':
            prediction_boxes = detection_function(source_image)   #run video
            # print(prediction_boxes)
            main_algorithm(prediction_boxes)
        else:
            prediction_boxes = detection_function('rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa')

    except NameError:
        print("Give source image, video or stream")





    # boxes = Data_calculate()
    # boxes_intersection_array = boxes.boxes_intersection_search(prediction_boxes)


    # print(boxes_intersection_array)








    # sms = Sms_delivery()
    # sms.pull_sms()

    # prediction_boxes = detection_function(source="yolov5/videoplayback.avi")
    # prediction_boxes = detection_function(source_image)

    # cam = Camera_work()
    # cam.run_camera("yolov5/videoplayback.mp4")

    #main algoritm