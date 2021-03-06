import cv2
import numpy as np


class Data_calculate():

    def parking_space_monitor(self):
        """

        :return:
        """
        pass

    def euclidean_distance(self, x1, x2):
        ans = np.linalg.norm(x1 + x2)
        return ans

    def iou_calculate(self, box1, box2):
        """Функция рассчитывает метрику IoU и проверяет пересечение прямоугольников
            Ордината смотрит вниз!"""
        flag = False
        iou = 0
        ax1, ay1, ax2, ay2 = box1[0][0], box1[0][1], box1[1][0], box1[1][1]
        bx1, by1, bx2, by2 = box2[0][0], box2[0][1], box2[1][0], box1[1][1]
        if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
            flag = True
            sq_a = abs(ax1 - ax2) * abs(ay1 - ay2)
            sq_b = abs(bx1 - bx2) * abs(by1 - by2)
            interArea = abs(max([ax1, ax2]) - min([bx1, bx2])) * abs(max([ay1, ay2]) - min([by1, by2]))
            iou = interArea / float(abs(abs(sq_a + sq_b) - interArea))

        return iou, flag

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

    def contours_search_and_filter(self, frame1, frame2, draw=False):
        """
        Frame contours search and filter function.
        return: flag (True or False)"""
        flag = False  # Motion is none

        contour_area = 0  #
        diff = cv2.absdiff(frame1,
                           frame2)  # Subtraction function the frame1 and frame2
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # Frame to grayscale
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Contours filtration
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)  # Draw white contours
        dilated = cv2.dilate(thresh, None, iterations=3)  # Contours delate
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Contours  search
        if contours:
            for contour in contours:
                contour_area = int(cv2.contourArea(contour))
                if contour_area >= 1000:  # If contour_area >= 5000 flag == True
                    flag = True
                if draw:
                    cv2.drawContours(frame2, contours, -1, (0, 255, 0), 2)  # Draw contours

        return flag, contour_area, frame2

    def update_parking_space_box(self, parking_space_box, box_auto):
        """Update parking space box function"""
        box_flag = []
        j = 0
        for car in box_auto:
            i = 0
            while i <= len(parking_space_box) - 1:
                iou, flag = self.iou_calculate(car, parking_space_box[i])
                box_flag.append(flag)
                i += 1
            box_average = ((sum(box_flag)) / (len(box_flag)))
            box_flag = []
            if box_average == 0:
                parking_space_box.append(car)
        return parking_space_box

    def get_status_space(self, parking_space_box, box_auto):
        """Update parking space box function"""
        box_flag = []
        status_space = True
        for space in parking_space_box:
            i = 0
            while i <= len(box_auto) - 1:
                iou, flag = self.iou_calculate(space, box_auto[i])
                box_flag.append(flag)
                i += 1
            box_average = ((sum(box_flag)) / (len(box_flag)))
            box_flag = []
            if box_average == 0:
                status_space = False
        print(status_space)
        return status_space

    def crop_image(self, frame, y1, y2, x1, x2):
        """
        The function crop input frame
        :input frame:
        :return: crop frame
        """
        frame = frame[y1:y2, x1:x2]
        return frame
