import cv2
import argparse
from yolov5.utils.datasets import LoadImages
from yolov5.detect import detection_function
from twilio.rest import Client
from time import sleep


class Camera_work():

    def run_camera(self, source):
        """Doc"""
        cap = cv2.VideoCapture(source)
        while True:
            ret, img = cap.read()
            if ret:
                cv2.imshow("camera", img)
            if cv2.waitKey(10) == 27:
                cap.release()
                cv2.destroyAllWindows()
                break

    def detection_move(self, source):
        """
        :param source:
        :return:
        """
        cap = cv2.VideoCapture(source)
        cap.set(3, 240)  # установка размера окна
        cap.set(4, 480)
        cap.set(cv2.CAP_PROP_FPS, 25)
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        flag = False

        while cap.isOpened():  # метод isOpened() выводит статус видеопотока
            frame1 = cv2.rectangle(frame1, (1, 1), (960, 200), (0, 0, 0), -1)
            frame2 = cv2.rectangle(frame2, (1, 1), (960, 200), (0, 0, 0), -1)
            diff = cv2.absdiff(frame1,
                               frame2)  # нахождение разницы двух кадров, которая проявляется лишь при изменении одного из них, т.е. с этого момента наша программа реагирует на любое движение.
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # перевод кадров в черно-белую градацию
            blur = cv2.GaussianBlur(gray, (5, 5), 0)  # фильтрация лишних контуров
            _, thresh = cv2.threshold(blur, 20, 255,
                                      cv2.THRESH_BINARY)  # метод для выделения кромки объекта белым цветом
            dilated = cv2.dilate(thresh, None,
                                 iterations=3)  # данный метод противоположен методу erosion(), т.е. эрозии объекта, и расширяет выделенную на предыдущем этапе область
            сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # нахождение массива контурных точек
            cv2.imshow("frame1", frame1)
            sleep(0.1)
            frame1 = frame2  #
            ret, frame2 = cap.read()  #
            if сontours == []:
                flag = False
            else:
                flag = True
            print(flag)
            return flag
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

    def diff_frame(self, frame1, frame2):
        '''

        :param frame1:
        :param frame2:
        :return:
        '''
        diff = cv2.absdiff(frame1, frame2)
        return diff








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


# def main_algorithm(prediction_boxes):
#     """
#
#     :param prediction_boxes:
#     :return:
#     """
#     temp_parking_seat = []
#     iou = Data_calculate()
#     temp_parking_seat = prediction_boxes.copy()
#
#     number_box = 0
#     for box_original in prediction_boxes:
#         #     print(temp_parking_seat[number_box],number_box, '----')
#         #     print(box_original,number_box, '---+')
#         num, flag = iou.iou_calculate(box_original, temp_parking_seat[number_box])
#         if
#         number_box += 1
#         print(flag, num)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--source', type=str, default="yolov5/test2.mp4",
                            help='source')  # file/folder, 0,1,2 for webcam
        opt = parser.parse_args()

        source_image = opt.source

        move = Camera_work()
        dviz = move.detection_move(source_image)
        print(dviz, "+++")

        # if dviz:
        #     print("dviz")
        # if source_image == ('0' or '1' or '2'):
        #     prediction_boxes = detection_function(source_image)  # run video
        #     # print(prediction_boxes)
        #     main_algorithm(prediction_boxes)
        # else:
        #     prediction_boxes = detection_function(source_image)
        #     main_algorithm(prediction_boxes)




        # cam = Camera_work()
        # ret, img = cam.run_camera("yolov5/test.mp4")




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

    # main algoritm
