import cv2

class Video():

    def write_video(self):
        '''Doc'''
        cap = cv2.VideoCapture(0)
            while True:
                ret, img = cap.read()
                cv2.imshow("camera", img)
                cv2.imwrite('screen.jpg', img)
                if cv2.waitKey(10) == 27: # Клавиша Esc
                    break
            cap.release()
            cv2.destroyAllWindows()


def main_algoritm():
    pass

if __name__ == "__main__":

    main_algoritm()