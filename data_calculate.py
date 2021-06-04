import cv2


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

    def contours_search_and_filter(self,frame1,frame2):
        """
        :param frame2:
        :param frame1:
        :return:
        """

        contour_area = 0
        flag = False
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
        #print(сontours, "contour")
        cv2.drawContours(frame1, сontours, -1, (0, 255, 0), 2) #также можно было просто нарисовать контур объекта
        cv2.imshow("image, bitch", frame2)
        cv2.waitKey(1)
        #cv2.destroyAllWindows()
        for contour in сontours:
            # (x, y, w, h) = cv2.boundingRect(contour)  # преобразование массива из предыдущего этапа в кортеж из четырех координат
            # метод contourArea() по заданным contour точкам, здесь кортежу,
            # вычисляет площадь зафиксированного объекта в каждый момент времени, это можно проверить
            #
            contour_area = cv2.contourArea(contour)
            if contour_area>500:  # условие при котором площадь выделенного объекта меньше 700 px
                flag = True

        return flag, contour_area