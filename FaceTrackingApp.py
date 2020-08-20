import sys
import cv2

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5 import uic

# load "ui" file with the graphic template
ui, wnd = uic.loadUiType('FaceTrackingUi.ui')

class Window(wnd,ui):
    # open main window
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # emit constant signals for run_app function, decide if it run or not
        self.timer.timeout.connect(self.run_app)
        # connect clicking RunButton with the controlTimer
        self.RunButton.clicked.connect(self.controlTimer)

    # show a view from the camera and search for faces
    def run_app(self):
        # pre-trained face classifier
        classifier_file_face = 'face_detector.xml'
        # create classifier
        fac_tracker = cv2.CascadeClassifier(classifier_file_face)

        # capture frame-by-frame
        ret, frame = self.cap.read()

        # change image from RGB to gray
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect face
        fac = fac_tracker.detectMultiScale(grayscaled_frame)

        # draw rectangles with detected faces
        for (x, y, w, h) in fac:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # get image information
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from frame
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # reverse RGB format
        qImg = qImg.rgbSwapped()
        # display image
        self.ImgLabel.setPixmap(QPixmap.fromImage(qImg))

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped -> run video
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(50)
            # change RunButton text
            self.RunButton.setText("Stop")
        # if timer is started -> stop video
        else:
            print("stop")
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # change RunButton text
            self.RunButton.setText("Start")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mwin = Window()
    mwin.show()
    app.exec_()