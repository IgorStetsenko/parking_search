a = [((251, 134), (298, 180)),
     ((95, 192), (201, 293)),
     ((13, 196), (53, 272))]

b = [((251, 134), (298, 180)),
     ((95, 192), (201, 293)),
     ((13, 196), (53, 272)), ((649, 241), (713, 298)), ((100, 100), (150, 150))]


def iou_calculate(box1, box2):
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


def update_parking_space_box(parking_space_box, box_auto):
    """Update parking space box function"""
    box_flag = []
    j = 0
    for car in box_auto:
        i = 0
        while i <= len(parking_space_box) - 1:
            iou, flag = iou_calculate(car, parking_space_box[i])
            box_flag.append(flag)
            i += 1
        box_average = ((sum(box_flag)) / (len(box_flag)))
        box_flag = []
        if box_average == 0:
            parking_space_box.append(car)
    print(parking_space_box)
    return parking_space_box


update_parking_space_box(a, b)
