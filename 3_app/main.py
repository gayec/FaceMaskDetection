from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QMessageBox
from maskdetection import face_mask_prediction

class VideoCapture(qtc.QThread):
    change_signal = qtc.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)

        while self.run_flag:
            ret , frame = cap.read() 
            predicted_image = face_mask_prediction(frame)

            if ret == True:
                self.change_signal.emit(predicted_image)
                
        predicted_image = 127+np.zeros((450,600,3), dtype = np.uint8)
        self.change_signal.emit(predicted_image)
        cap.release()
        
    def stop(self):
        self.run_flag = False
        self.wait()    
    
class mainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(qtg.QIcon('D:\\deneme\\Face Mask Detection\\3_app\\images\\face-mask.png')) # window icon
        self.setWindowTitle('  ')
        self.setFixedSize(900,900)

        # Adding Widgets
        label = qtw.QLabel('<h2>Real Time Face Mask Detection </h2>')
        label.setAlignment(qtc.Qt.AlignCenter)

        self.camButton = qtw.QPushButton('Turn On Camera',clicked=self.camButtonClick, checkable=True) 

        #button
        self.info = QPushButton(self, flat = True)
        self.info.clicked.connect(self.show_popup)
        self.info.setFixedSize(50,30)
        self.info.setIcon(qtg.QIcon('D:\\deneme\\Face Mask Detection\\3_app\\images\\deneme.png'))
        self.info.setIconSize(qtc.QSize(30,30))
        self.info.setLayoutDirection(qtc.Qt.RightToLeft)

        # screen
        self.screen = qtw.QLabel()
        self.screen.setScaledContents(True)
        self.image = qtg.QPixmap(900,780)
        self.image.fill(qtg.QColor('darkGrey'))
        self.screen.setPixmap(self.image)

        # Layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.info)
        layout.addWidget(self.camButton)
        layout.addWidget(self.screen)
        self.setLayout(layout)
        self.show()
    
    def show_popup(self):
        msg = QMessageBox()
        msg.setWindowTitle("Information")
        msg.setIcon(QMessageBox.Information)
        msg.setText("Pressing the turn on camera button will start face detection system, if you want to turn off camera, just press the turn off button. If you want to close the application, press the close button. \nThis system was developed by Gaye Celik in April 2022.\ngithub:https://github.com/gayec\nlinkedin:https://www.linkedin.com/in/gayeeminecelik/")

        x = msg.exec_()

    def camButtonClick(self):
        print('clicked')
        status = self.camButton.isChecked()
        print(status)
        
        if status == True:
            self.camButton.setText('Turn Off Camera')
            self.capt = VideoCapture()
            self.capt.change_signal.connect(self.updateImage)
            self.capt.start()         

        elif status == False:
            self.camButton.setText('Turn On Camera')
            self.capt.stop()
            
    @qtc.pyqtSlot(np.ndarray)
    def updateImage(self,arr):
        image = cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)
        h,w, ch = image.shape
        bytes = ch*w
        converted = qtg.QImage(image.data,w,h,bytes,qtg.QImage.Format_RGB888)
        scaled=converted.scaled(900,780)
        qt = qtg.QPixmap.fromImage(scaled)
        self.screen.setPixmap(qt)


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    mainwindow = mainWindow()
    sys.exit(app.exec())