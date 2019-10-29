from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt,QThread
from PyQt5 import QtCore,QtGui,QtWidgets,uic
import cv2
import os
import sys
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skimage
import threading
import time
import queue
from mrcnn import visualize
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from pose import OpenPoseImage
from pose.OpenPoseImage import GetPersonPoint

matplotlib.use('Agg')

# epoch number
EPOCH_NUM = 10
SP_EPOCH = 100

ROOT_DIR = os.path.abspath("../../")
DATASET_DIR = ""
WEIGHTS_PATH = ""
FILE_PATH = ""

active =False
running = False
capture_thread = None
form_class = uic.loadUiType("simple.ui")[0]
q = queue.Queue()
q2 = queue.Queue()
t_model = None
num=0
num2=0
table_size = 10
sys.path.append(ROOT_DIR)

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR,"logs")

lheight=[]
lweight=[]
lshoulder=[]
lwaist=[]

class VThread(QThread):
    def __init__(self):
        QThread.__init__(self)

    def run(self):
        global q
        global q2
        global active
        global FILE_PATH
        global running
        global t_model
        global num
        num = 0

        capture = cv2.VideoCapture(0)

        class_names = ['BG','person']
        colors = visualize.random_colors(len(class_names))
        while(running):
            frame = {}

            plt.clf()
            
            retval, img = capture.read()

            if not retval:
                break

            r = t_model.detect([img],verbose=0)[0]
            splash,height,weight = visualize.display_instances(img,r['rois'], r['masks'], r['class_ids'],
                     class_names, r['scores'], colors=colors,making_video=True)

            frame["img"] = splash
            frame["height"] = height
            frame["weight"] = weight

            #When we clicked measure button
            #input data in new queue
            if active:
                q2.put(frame)
                num+=1

                if num==table_size-1:
                    num = 0
                    active = False

            q.put(frame)
            print("q1size: ",q.qsize())
        capture.release()

class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None
        self.setMinimumSize(800,600)

    def setImage(self, image):
        self.image = image

        self.update()
              
    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
            print("Paint Event")
        qp.end()

class TestWindow(QDialog,form_class):
    def measureClicked(self):
        global active
        global running
        self.table.clear()
        self.table2.clear()
        if running:
            active=True
        
    def runClicked(self):
        global running
        running = True
        global WEIGHTS_PATH
        global FILE_PATH
        global t_model
        t_model = self.model

        FILE_PATH = "/home/default/Desktop/Test/pigvod.mp4"
        WEIGHTS_PATH = "/home/default/logs/person20191011T1414/mask_rcnn_person_0532.h5"

        self.model.load_weights(WEIGHTS_PATH,by_name=True)
        self.model.keras_model._make_predict_function()

        if self.capture_thread.isRunning():
            self.capture_thread.terminate()
            self.capture_thread.wait()
            self.capture_thread.start()
        else:
            self.capture_thread.start()


    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        global q
        global q2
        q.queue.clear()
        q2.queue.clear()

        class InferenceConfig(PersonConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        #Make inference model
        self.capture_thread = VThread()
        self.config = InferenceConfig()
        self.config.display()
        self.model = modellib.MaskRCNN(mode="inference",config = self.config,model_dir = DEFAULT_LOGS_DIR)
        
        #Widget
        self.label1 = QLabel("CAM")
        self.label2 = QLabel("TABLE")
        self.label3 = QLabel("RESULT")

        self.label1.setAlignment(Qt.AlignCenter)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label3.setAlignment(Qt.AlignCenter)

        self.table = QTableWidget(table_size,4,self)
        self.table.setHorizontalHeaderLabels(["Height","Weight(px)","Shoulder(cm)","Waist(cm)"])

        self.table2 = QTableWidget(1,4,self)
        self.table2.setHorizontalHeaderLabels(["Height","Weight(kg)","Size(Top)","Size(Pants)"])

        self.play_button = QPushButton("Play Video")
        self.play_button.clicked.connect(self.runClicked)
        self.play_button.setToolTip('Get the frame by webcam') 

        self.measure_button = QPushButton("Measure")
        self.measure_button.clicked.connect(self.measureClicked)
        self.measure_button.setToolTip('Measuring instance profile')
        
        self.ImgWidget = OwnImageWidget()
        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()

        self.retranslateUi(self) 
        QtCore.QMetaObject.connectSlotsByName(self) 

        #Layout
        layout = QGridLayout()
        layout.addWidget(self.label1,0,0)
        layout.addWidget(self.label2,0,1)
        layout.addWidget(self.ImgWidget,1,0)

        verLayout = QVBoxLayout()
        verLayout.addWidget(self.table)
        verLayout.addWidget(self.label3)
        verLayout.addWidget(self.table2)
        layout.addLayout(verLayout,1,1)

        horLayout = QHBoxLayout() 
        horLayout.addStretch(1) 
        horLayout.addWidget(self.play_button) 
        horLayout.addStretch(1)
        layout.addLayout(horLayout,2,0)

        horLayout2 = QHBoxLayout() 
        horLayout2.addStretch(1) 
        horLayout2.addWidget(self.measure_button) 
        horLayout2.addStretch(1)
        layout.addLayout(horLayout2,2,1)

        self.setLayout(layout)
        self.setGeometry(200,200,1350,650)

	    #Timer
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    def retranslateUi(self,TestQFileDialog):
        _translate = QtCore.QCoreApplication.translate
        TestQFileDialog.setWindowTitle(_translate("TestQFileDialog","Dialog"))

    def update_frame(self):
        if not q.empty():
            frame = q.get()
            img = frame["img"]
            
            img = cv2.resize(img, None, fx=1.0, fy=1.0, interpolation = cv2.INTER_NEAREST)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.setImage(image)

        if q2.qsize() > 0:
            global num2
            
            frame = q2.get()
            img = frame["img"]
            img = cv2.resize(img, None, fx=1.0, fy=1.0, interpolation = cv2.INTER_NEAREST)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #get value by pixel
            height = frame["height"]
            weight = frame["weight"]
            shoulder,waist = GetPersonPoint(img)
 
            #reflect on table
            cv2.imwrite('testimage%d.jpg' % num2, img)
            self.table.setItem(num2, 0, QTableWidgetItem(str(height)))
            self.table.setItem(num2, 1, QTableWidgetItem(str(weight)))
            self.table.setItem(num2, 2, QTableWidgetItem(str(shoulder)))
            self.table.setItem(num2, 3, QTableWidgetItem(str(waist)))

            num2+=1

            if height>0 and weight>0 and shoulder>0 and waist>0:
                lheight.append(height)
                lweight.append(weight)
                lshoulder.append(shoulder)
                lwaist.append(waist)
            
            if num2==table_size-1:
                num2=0

                if len(lheight) > 0:
                    avg_height = str(sum(lheight)/len(lheight))
                    avg_weight = str(sum(lweight)/len(lweight))
                    avg_shoulder = str(max(lshoulder))
                    avg_waist = str(max(lwaist))
                else:
                    avg_height = "No instance detecting!"
                    avg_weight = "No instance detecting!"
                    avg_shoulder = "No instance detecting!"
                    avg_waist = "No instance detecting!"


                self.table.setItem(table_size-1, 0, QTableWidgetItem(avg_height))
                self.table.setItem(table_size-1, 1, QTableWidgetItem(avg_weight))
                self.table.setItem(table_size-1, 2, QTableWidgetItem(avg_shoulder))
                self.table.setItem(table_size-1, 3, QTableWidgetItem(avg_waist))

                lheight.clear()
                lweight.clear()
                lshoulder.clear()
                lwaist.clear()
    
    def closeEvent(self,event): 
        global running
        running = False
        print("end testing")
    
#train         
class TrainWindow(QDialog):
    def _open_file_dialog1(self):
        directory = str(QFileDialog.getExistingDirectory())
        self.lineEdit1.setText('{}'.format(directory))
        
    def _open_file_dialog2(self):
        directory = QFileDialog.getOpenFileName(self,'Open file','/home/default')
        self.lineEdit2.setText(directory[0])

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        #train model
        self.config = PersonConfig()
        self.config.display()
        self.model = modellib.MaskRCNN(mode = "training", config = self.config, model_dir = DEFAULT_LOGS_DIR)

        #pyqt
        layout = QGridLayout()
        boxlayout = QBoxLayout(QBoxLayout.TopToBottom,self)

        self.toolButtonOpenDialog1 = QToolButton(self)
        self.toolButtonOpenDialog1.setGeometry(QtCore.QRect(210,10,25,19))
        self.toolButtonOpenDialog1.setObjectName("toolButtonOpenDialog")
        self.toolButtonOpenDialog1.clicked.connect(self._open_file_dialog1)

        self.toolButtonOpenDialog2 = QToolButton(self)
        self.toolButtonOpenDialog2.setGeometry(QtCore.QRect(210,10,25,19))
        self.toolButtonOpenDialog2.setObjectName("toolButtonOpenDialog2")
        self.toolButtonOpenDialog2.clicked.connect(self._open_file_dialog2)
            
        self.lineEdit1 = QLineEdit(self) 
        self.lineEdit1.setEnabled(False) 
        self.lineEdit1.setGeometry(QtCore.QRect(10, 10, 191, 20)) 
        
        self.lineEdit2 = QLineEdit(self) 
        self.lineEdit2.setEnabled(False) 
        self.lineEdit2.setGeometry(QtCore.QRect(10, 10, 191, 20)) 
         
        self.retranslateUi(self) 
        QtCore.QMetaObject.connectSlotsByName(self) 

        self.setGeometry(1100,200,300,100)
        self.setWindowTitle("train")
        
        btn1 = QPushButton("Start Training")
        btn2 = QPushButton("Close")

        btn1.clicked.connect(self.trainClicked)
        btn2.clicked.connect(self.closeClicked)

        self.groupbox = QGroupBox("",self)
        self.groupbox.setLayout(boxlayout)

        self.chk1 = QRadioButton("COCO",self)
        self.chk2 = QRadioButton("LAST",self)
        self.chk3 = QRadioButton("Other",self)

        boxlayout.addWidget(self.chk1)
        boxlayout.addWidget(self.chk2)
        boxlayout.addWidget(self.chk3)

        label1 = QLabel("Epoch : ")
        label2 = QLabel("Numbers per epoch : ")
        label3 = QLabel("Location of dataset : ")
        label4 = QLabel("Weight : ")

        self.spinBox1 = QSpinBox(self)
        self.spinBox2 = QSpinBox(self)

        self.spinBox1.setMaximum(1000)
        self.spinBox2.setMaximum(1000)
        
        self.spinBox1.setValue(10)
        self.spinBox2.setValue(100)

        layout.addWidget(label1,0,0)
        layout.addWidget(label2,1,0)
        layout.addWidget(label3,2,0)
        layout.addWidget(label4,4,0)

        layout.addWidget(self.spinBox1,0,1)
        layout.addWidget(self.spinBox2,1,1)
        layout.addWidget(self.toolButtonOpenDialog1,2,1)
        layout.addWidget(self.lineEdit1,3,0,1,2)
        layout.addWidget(self.groupbox,4,1)
        layout.addWidget(self.toolButtonOpenDialog2,5,1)
        layout.addWidget(self.lineEdit2,6,0,1,2)

        layout.addWidget(btn1,7,0)
        layout.addWidget(btn2,7,1)
        self.setLayout(layout)
        self.setGeometry(300,300,300,200)

    def trainClicked(self):
        global EPOCH_NUM
        global SP_EPOCH
        
        EPOCH_NUM = self.spinBox1.value()
        SP_EPOCH = self.spinBox2.value()
        self.config.STEPS_PER_EPOCH = SP_EPOCH

        #dataset path
        global DATASET_DIR 
        DATASET_DIR = self.lineEdit1.text()

        if DATASET_DIR == "":
            QMessageBox.about(self,"error","Set dataset directory!")
     
        #weights path
        global WEIGHTS_PATH
        WEIGHTS_PATH = self.lineEdit2.text()

        if self.chk1.isChecked():
                WEIGHTS_PATH = COCO_WEIGHTS_PATH
        elif self.chk2.isChecked():
                WEIGHTS_PATH = self.model.find_last()
        elif self.chk3.isChecked():
            if WEIGHTS_PATH == "":
                QMessageBox.about(self,"error","Set weight directory!")
        else:
            QMessageBox.about(self,"error","Check weight!")

        #load weights
        if DATASET_DIR == "" or WEIGHTS_PATH == "":
            print("Check before run!")
        else:
            print("epoch_num is : ",EPOCH_NUM)
            print("steps per epoch is : ",SP_EPOCH)
            print("dataset directory is : ",DATASET_DIR)
            print("weights directory is : ",WEIGHTS_PATH)

            if WEIGHTS_PATH == COCO_WEIGHTS_PATH:
                self.model.load_weights(WEIGHTS_PATH,by_name = True, exclude=["mrcnn_class_logits","mrcnn_bbox_fc","mrcnn_bbox","mrcnn_mask"])
            else:
                self.model.load_weights(WEIGHTS_PATH,by_name=True)
            train(self.model)

    def closeClicked(self):
        self.close()
              
    def retranslateUi(self,TestQFileDialog):
        _translate = QtCore.QCoreApplication.translate
        TestQFileDialog.setWindowTitle(_translate("TestQFileDialog","Dialog"))
        self.toolButtonOpenDialog1.setIcon(QtGui.QIcon('./icon.png'))
        self.toolButtonOpenDialog2.setIcon(QtGui.QIcon('./icon.png'))
    
class ExWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):

        label = QLabel('Mask R-CNN Person',self)
        label.setAlignment(Qt.AlignCenter)

        font = label.font()
        font.setBold(True)
        
        btn1 = QPushButton('Train',self)
        btn2 = QPushButton('Test',self)
        btn3 = QPushButton('Close',self)
        
        btn1.clicked.connect(self.trainEvent)
        btn2.clicked.connect(self.testEvent)
        btn3.clicked.connect(self.close)

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addWidget(btn1)
        vbox.addWidget(btn2)
        vbox.addWidget(btn3)

        self.setLayout(vbox)

        self.setGeometry(800,200,300,300)
        self.show()

    def testEvent(self):
        dia = TestWindow()
        dia.exec_()

    def trainEvent(self):
        dia = TrainWindow()
        dia.exec_()

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?',QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

class PersonConfig(Config):

    NAME = "person"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1  # Background + objects

    STEPS_PER_EPOCH = SP_EPOCH

    DETECTION_MIN_CONFIDENCE = 0.9

class PersonDataset(utils.Dataset):
    def load_VIA(self, dataset_dir, subset, hc=False):

        self.add_class("person", 1, "person")
        #self.add_class("pig", 2, "lying_pig")

        assert subset in ["train","val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations1.values())  
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
# Get the x, y coordinaets of points of the polygons that make up
# the outline of each object instance. There are stores in the
# shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            #names = [r['region_attributes'] for r in a['regions'].values()]
# load_mask() needs the image size to convert polygons to masks.
# Unfortunately, VIA doesn't include it in JSON, so we must read
# the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "person",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)
                #names=names)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "person":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        #class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        #class_ids = np.zeros([len(info["polygons"])])

        #for i, p in enumerate(class_names):
            #if p['name'] == 'standing_pig':
                #class_ids[i] = 1
            #elif p['name'] == 'lying_pig':
                #class_ids[i] = 2
#assert code here to extend to other labels
        #class_ids = class_ids.astype(int)
# Return mask, and array of class IDs of each instance. Since we have
# one class ID only, we return an array of 1s
        
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        #return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "person":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def detect_and_color_splash(model, image_path=None, video_path=None, out_dir=''):
    assert image_path or video_path

    class_names = ['BG', 'person']

# Image or video?
    if image_path:
# Read image
        image = skimage.io.imread(FILE_PATH)
# Detect objects
        r = model.detect([image], verbose=1)[0]
# Color splash and save
        masked_image = visualize.display_instances2(image, r['rois'], r['masks'], r['class_ids'],
            class_names, r['scores'],"image")
        return masked_image

    elif video_path:
# Video capture
        vcapture = cv2.VideoCapture(video_path)
# width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = 0
        success = True
#For video, we wish classes keep the same mask in frames, generate colors for masks
        colors = visualize.random_colors(len(class_names))
        while success:
            print("frame: ", count)
            plt.clf()
            plt.close()
            success, image = vcapture.read()
            if success:
        
# OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
# Detect objects
                r = model.detect([image], verbose=0)[0]
# Color splash
# splash = color_splash(image, r['masks'])
                
                splash = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
            class_names, r['scores'], colors=colors,making_video=True)
# Add image to video writer
                cv2.imshow('img',splash)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                count += 1
        vcapture.release()

def train(model):
    dataset_train = PersonDataset()
    dataset_train.load_VIA(DATASET_DIR, "train")
    dataset_train.prepare()

    dataset_val = PersonDataset()
    dataset_val.load_VIA(DATASET_DIR, "val")
    dataset_val.prepare()
    print("Training network heads")
    model.train(dataset_train, dataset_val,
            learning_rate=PersonConfig().LEARNING_RATE,
            epochs=model.epoch+EPOCH_NUM,
            layers='heads')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ExWindow()
    sys.exit(app.exec_())

