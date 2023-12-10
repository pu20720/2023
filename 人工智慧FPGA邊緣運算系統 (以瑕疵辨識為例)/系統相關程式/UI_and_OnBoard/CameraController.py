from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from CameraUI import Ui_MainWindow
import cv2
import Normal_utils as N
import sys
import xir
import time
import numpy as np

class MainWindow_controller(QtWidgets.QMainWindow):
    returnSignal = pyqtSignal()
    def __init__(self, model, parent = None):
        super(MainWindow_controller, self).__init__(parent) # in python3, super(Class, self).xxx = super().xxx
        self.model = model
        self.returnSignal.connect(self.show_camera)
        self.graph = xir.Graph.deserialize(model)      # Deserialize Model Graph
        self.subgraphs = N.get_child_subgraph_dpu(self.graph)
        self.dpu = N.dpu_runner(2, self.subgraphs)     # Construct DPU based on model subgraph
        self.input_scale = N.input_scale(self.dpu)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.timer_camera = QTimer() # Initial Timer
        self.timer_camera.setTimerType(Qt.PreciseTimer)
        self.cap = cv2.VideoCapture() # Initial Camera
        self.CAM_NUM = 0
        self.image_list = []
        self.predict = []
        self.button_control()
    # Button Control
    def button_control(self):
        self.timer_camera.timeout.connect(self.show_camera)
        self.ui.OP_Button.clicked.connect(self.OpenCamera)
        self.ui.SR_Button.clicked.connect(self.getframe)
        self.ui.exit_Button.clicked.connect(self.ExitCamera)
    # Show Current Camera Screen
    def show_camera(self): 
        ret, frame = self.cap.read()
        show = frame.copy()
        x = 320
        y = 240
        w = h = 80
        #Image Crop & Resize & RGB Convert
        img1 = cv2.resize(show[y:y+2*h,x-h:x+h],(360, 330))
        img2 = cv2.resize(show[y-2*h:y,x-h:x+h],(360, 330))
        img3 = cv2.resize(show[y-h:y+h,x:x+2*h],(360, 330))
        img4 = cv2.resize(show[y-h:y+h,x-2*h:x],(360, 330))

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

        showImage1 = QImage(img1, img1.shape[1], img1.shape[0], QImage.Format_RGB888)
        showImage2 = QImage(img2, img2.shape[1], img2.shape[0], QImage.Format_RGB888)
        showImage3 = QImage(img3, img3.shape[1], img3.shape[0], QImage.Format_RGB888)
        showImage4 = QImage(img4, img4.shape[1], img4.shape[0], QImage.Format_RGB888)    
        self.ui.camera_label_1.setPixmap(QPixmap.fromImage(showImage1))
        self.ui.camera_label_2.setPixmap(QPixmap.fromImage(showImage2))
        self.ui.camera_label_3.setPixmap(QPixmap.fromImage(showImage3))
        self.ui.camera_label_4.setPixmap(QPixmap.fromImage(showImage4))
    #Get Screen Shot
    def getframe(self):
        self.predict = []
        self.image_list = []
        ret, frame = self.cap.read()
        show = frame.copy()
        x = 320
        y = 240
        w = h = 80
        #Image Crop
        img1 = show[y:y+2*h,x-h:x+h]
        img2 = show[y-2*h:y,x-h:x+h]
        img3 = show[y-h:y+h,x:x+2*h]
        img4 = show[y-h:y+h,x-2*h:x]

        cv2.imwrite("000.png",img[0])
        cv2.imwrite("001.png",img[1])
        cv2.imwrite("002.png",img[2])
        cv2.imwrite("003.png",img[3])

        self.image_list.append(img1)
        self.image_list.append(img2)
        self.image_list.append(img3)
        self.image_list.append(img4)
        # Total 4 image per OP
        runTotal = len(self.image_list)
        img = []
        for i in range(runTotal):
            img.append(N.preprocess_fn(self.image_list[i], self.input_scale))
        for i in range(len(img)):
            result = N.runDPU(self.dpu[1], img[i])
            self.predict.append(result)

        self.ui.result_label_1.setText("Result is : " + self.predict[0])
        self.ui.result_label_2.setText("Result is : " + self.predict[1])
        self.ui.result_label_3.setText("Result is : " + self.predict[2])
        self.ui.result_label_4.setText("Result is : " + self.predict[3])
    # Camera Setting
    def OpenCamera(self):
        flag = self.cap.open(self.CAM_NUM, cv2.CAP_V4L2)

        if flag == False:
             msg = QMessageBox.Warning(self, u'Warning', u'Please check the connection of the camera', buttons = QMessageBox.Ok, defaultButton = QMessageBox.Ok)
        else:
             self.timer_camera.start(30)
    # Exit Camera
    def ExitCamera(self):
        self.timer_camera.isActive()
        self.timer_camera.stop()
        self.cap.release()
        self.ui.exit_Button.clicked.connect(self.close)
        sys.exit()
        

