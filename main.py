import cv2
import argparse
from yolov5.utils.datasets import LoadImages
from yolov5.detect import detection_function
from twilio.rest import Client


test_image_path = 'screen.jpg'


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
        flag = True
        ax1, ay1, ax2, ay2 = box1[0][0], box1[0][1], box1[1][0], box1[1][1]
        bx1, by1, bx2, by2 = box2[0][0], box2[0][1], box2[1][0], box1[1][1]
        if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
            print("пересекаются")
            sq_a = abs(ax1 - ax2) * abs(ay1 - ay2)
            sq_b = abs(bx1 - bx2) * abs(by1 - by2)
            interArea = abs(max([ax1, ax2]) - min([bx1, bx2])) * abs(max([ay1, ay2]) - min([by1, by2]))
            iou = interArea / float(abs(abs(sq_a + sq_b) - interArea))
            return iou
        else:
            print("не пересекаются")
            return 0

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
        '''Doc'''
        # Twilio account details
        twilio_account_sid = 'AC76115b7fa91b315f7d21c9640caa54f6'
        twilio_auth_token = 'a2ce2eb68dcf6137930a5db1d2cba18b'
        twilio_source_phone_number = '+19288131669'

        # Create a Twilio client object instance
        client = Client(twilio_account_sid, twilio_auth_token)

        # Send an SMS
        message = client.messages.create(
            body="This is my SMS message!",
            from_=twilio_source_phone_number,
            to="+79381424592"
        )


class Detect():
    def run_detect(self):
        '''Doc'''
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0,1,2 for webcam
    opt = parser.parse_args()

    prediction_boxes = detection_function(source="yolov5/videoplayback.avi")
    # print(prediction_boxes)
    # cam = Camera_work()
    # cam.run_camera("yolov5/videoplayback.mp4")
