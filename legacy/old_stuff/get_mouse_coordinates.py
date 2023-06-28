import cv2
import matplotlib.pyplot as plt


class TestClass():
    def __init__(self):
        self.fname = r'C:\ran_data\RAMBAM\SlideID_images9\Box_1_1.png'
        self.img = cv2.imread(self.fname)
        self.point = ()

    def getCoord(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(self.img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return self.point

    def __onclick__(self,click):
        self.point = (click.xdata,click.ydata)
        return self.point

q = TestClass()
q.getCoord()