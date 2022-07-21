import cv2
import matplotlib.pyplot as plt

class TestClass():
    def __init__(self):
        self.fname = r'C:\ran_data\RAMBAM\SlideID_images9\Box_1_1.jpg'
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

'''
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10)
y = x ** 2

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(x, y)
plt.imshow(r'C:\ran_data\RAMBAM\SlideID_images9\Box_1_1.jpg')

coords = []


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print('x = ', str(ix), ', y = ', str(iy))

    global coords
    coords.append((ix, iy))

    if len(coords) == 2:
        fig.canvas.mpl_disconnect(cid)

    return coords


cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
'''