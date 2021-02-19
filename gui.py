# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets

import warnings
warnings.filterwarnings(action='ignore')

import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Input, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from skimage.transform import pyramid_expand
from Subpixel import Subpixel


base_path = r'C:\Super_Resolution\LP\processed'
# base_path = r'C:\Super_Resolution\LP\processed_gray'


upscale_factor = 4

model = Sequential()


#1st convolution layer
model.add(Conv2D(64,(5,5),padding='same',strides=(1,1), input_shape = (15,50,3), activation='relu'))

#2nd convolution layer
model.add(Conv2D(64,(3,3), padding='same', strides=(1,1), activation='relu'))

#3rd convolution layer
model.add(Conv2D(32,(3,3), padding='same', strides=(1,1), activation='relu'))

#4th convolution layer
model.add(Conv2D(16, (3,3), padding='same', strides=(1,1), activation='relu'))

model.add(Subpixel(3, (3,3), r=upscale_factor, padding='same', activation='relu'))

#add line
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)


model.summary()
model.load_weights(r"C:\Super_Resolution\models\model_LPR_color_gray.h5")

x_test_list = sorted(glob.glob(os.path.join(base_path, 'x_test', '*.npy')))
y_test_list = sorted(glob.glob(os.path.join(base_path, 'y_test', '*.npy')))

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(160, 200, 80, 30))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.printTextEdit)

        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(150, 110, 100, 30))
        self.textEdit.setObjectName("textEdit")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)


    def printTextEdit(self):
        test_idx = int(self.textEdit.toPlainText())

        # 저해상도 이미지(input)
        x1_test = np.load(x_test_list[test_idx])

        # 저해상도 이미지 확대시킨 이미지
        x1_test_resized = pyramid_expand(x1_test, 4, multichannel=True)  # 색깔 채널 조건 추가.

        # 정답 이미지
        y1_test = np.load(y_test_list[test_idx])

        y_pred = model.predict(x1_test.reshape((1, 15, 50, 3)))

        # 모델이 예측한 이미지(output)
        y_pred_process = activation_model.predict(x1_test.reshape((1, 15, 50, 3)))

        y_pred_1 = y_pred_process[0]
        y_pred_2 = y_pred_process[1]
        y_pred_3 = y_pred_process[2]
        y_pred_4 = y_pred_process[3]


        x1_test = (x1_test * 255).astype(np.uint8)
        x1_test_resized = (x1_test_resized * 255).astype(np.uint8)
        y1_test = (y1_test * 255).astype(np.uint8)
        y_pred = np.clip(y_pred.reshape((60, 200, 3)), 0, 1)
        y_pred = (y_pred * 255).astype(np.uint8)

        x1_test = cv2.cvtColor(x1_test,
                               cv2.COLOR_BGR2RGB)

        x1_test_resized = cv2.cvtColor(x1_test_resized,
                                       cv2.COLOR_BGR2RGB)

        y1_test = cv2.cvtColor(y1_test,
                               cv2.COLOR_BGR2RGB)

        y_pred = cv2.cvtColor(y_pred,
                              cv2.COLOR_BGR2RGB)

        figs, ax = plt.subplots(2, 4, figsize=(15, 10))

        ax[0, 0].set_title('input')
        ax[0, 0].imshow(x1_test)

        ax[0, 1].set_title('resized')
        ax[0, 1].imshow(x1_test_resized)
        psnr1 = cv2.PSNR(y1_test, x1_test_resized)
        ax[0, 1].set_xlabel("{:.2f} db".format(psnr1))

        ax[0, 2].set_title('output')
        ax[0, 2].imshow(y_pred)
        psnr2 = cv2.PSNR(y_pred, y1_test)
        ax[0, 2].set_xlabel("{:.2f} db".format(psnr2))

        ax[0, 3].set_title('groundtruth')
        ax[0, 3].imshow(y1_test)

        ax[1, 0].set_title('1st convolution')
        ax[1, 0].matshow(y_pred_1[0, :, :, 2], cmap="viridis")

        ax[1, 1].set_title('2nd convolution')
        ax[1, 1].matshow(y_pred_2[0, :, :, 2], cmap="viridis")

        ax[1, 2].set_title('3rd convolution')
        ax[1, 2].matshow(y_pred_3[0, :, :, 2], cmap="viridis")

        ax[1, 3].set_title('4th convolution')
        ax[1, 3].matshow(y_pred_4[0, :, :, 2], cmap="viridis")

        plt.show()


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "CHECK"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
