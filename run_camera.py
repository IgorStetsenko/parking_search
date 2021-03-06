import cv2

class Setting_camera():
    """
    Set camera class
    """
    def set_cap(self, cap):
        """
        imput: cap
        :return: set cap
        """
        cap.set(3, 240)  # size window set
        cap.set(4, 480)
        cap.set(cv2.CAP_PROP_FPS, 25)
        return cap

class Camera_work():

    def run_video(self, source):
        """Doc"""
        cap = cv2.VideoCapture(source)
        while True:
            ret, img = cap.read()
            print("++++++++++++")
            if ret:
                #cv2.imshow("camera", img)
                return img
            if cv2.waitKey(10) == 27:
                cap.release()
                cv2.destroyAllWindows()
                break

    def image_show(self, image):
        cv2.imshow("image, bitch", image)
        cv2.waitKey(0)
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