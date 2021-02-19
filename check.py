
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


base_path = r'C:\Super_Resolution\LP\processed_gray'


upscale_factor = 4
model = Sequential()

# input layer
# inputs = Input(shape=(15, 50, 3))

#1st convolution layer
model.add(Conv2D(64,(5,5),padding='same',strides=(1,1), input_shape = (15,50,3), activation='relu'))

# net = Conv2D(filters=64,
#              kernel_size=5,
#              strides=1,
#              padding='same',
#              activation='relu')(inputs)

#2nd convolution layer
model.add(Conv2D(64,(3,3), padding='same', strides=(1,1), activation='relu'))
# net = Conv2D(filters=64,
#              kernel_size=3,
#              strides=1,
#              padding='same',
#              activation='relu')(net)

#3rd convolution layer
model.add(Conv2D(32,(3,3), padding='same', strides=(1,1), activation='relu'))

# net = Conv2D(filters=32,
#              kernel_size=3,
#              strides=1,
#              padding='same',
#              activation='relu')(net)

#4th convolution layer
model.add(Conv2D(16, (3,3), padding='same', strides=(1,1), activation='relu'))

# net = Conv2D(filters=upscale_factor**2,
#              kernel_size=3,
#              strides=1,
#              padding='same',
#              activation='relu')(net)
model.add(Subpixel(3, (3,3), r=upscale_factor, padding='same', activation='relu'))
# net = Subpixel(filters=3,
#                kernel_size=3,
#                r=upscale_factor,
#                padding='same')(net)

# outputs = Activation('relu')(net)


#add line
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# model = Model(inputs=inputs, outputs=outputs)


model.summary()
model.load_weights(r"C:\Super_Resolution\models\model_LPR_color_gray.h5")

x_test_list = sorted(glob.glob(os.path.join(base_path, 'x_test', '*.npy')))
y_test_list = sorted(glob.glob(os.path.join(base_path, 'y_test', '*.npy')))

# print(len(x_test_list), len(y_test_list))
# print(y_test_list[0])


PSNR1 = 0
PSNR2 = 0



for i in range(1):
    test_idx = i+1

    # test_idx = 852

    # 저해상도 이미지(input)
    x1_test = np.load(x_test_list[test_idx])

    # 저해상도 이미지 확대시킨 이미지
    x1_test_resized = pyramid_expand(x1_test, 4, multichannel=True)  # 색깔 채널 조건 추가.

    # 정답 이미지
    y1_test = np.load(y_test_list[test_idx])

    y_pred = model.predict(x1_test.reshape((1, 15, 50, 3)))

    # 모델이 예측한 이미지(output)
    y_pred_process = activation_model.predict(x1_test.reshape((1, 15, 50, 3)))
    # print(y_pred)
    # print(y_pred.shape)
    y_pred_1 = y_pred_process[0]
    y_pred_2 = y_pred_process[1]
    y_pred_3 = y_pred_process[2]
    y_pred_4 = y_pred_process[3]

    # print(y_pred_1.shape)
    # plt.matshow(y_pred_1[0, :, :, 2], cmap='viridis')
    # plt.show()
    # plt.matshow(y_pred_2[0, :, :, 2], cmap='viridis')
    # plt.show()
    # plt.matshow(y_pred_3[0, :, :, 2], cmap='viridis')
    # plt.show()
    # plt.matshow(y_pred_4[0, :, :, 2], cmap='viridis')
    # plt.show()


    # y_pred_1 = y_pred[0]
    # y_pred_3 = y_pred[2]
    # y_pred_4 = y_pred[3]
    # print(x1_test.shape, y1_test.shape)

    x1_test = (x1_test * 255).astype(np.uint8)
    x1_test_resized = (x1_test_resized * 255).astype(np.uint8)
    y1_test = (y1_test * 255).astype(np.uint8)
    y_pred = np.clip(y_pred.reshape((60, 200, 3)), 0, 1)
    y_pred = (y_pred * 255).astype(np.uint8)

    # y_pred_2 = cv2.cvtColor(y_pred[1], cv2.COLOR_BGR2RGB)




    x1_test = cv2.cvtColor(x1_test,
                           cv2.COLOR_BGR2RGB)

    x1_test_resized = cv2.cvtColor(x1_test_resized,
                                   cv2.COLOR_BGR2RGB)

    y1_test = cv2.cvtColor(y1_test,
                           cv2.COLOR_BGR2RGB)

    y_pred = cv2.cvtColor(y_pred,
                          cv2.COLOR_BGR2RGB)





    figs, ax = plt.subplots(2, 4, figsize=(15, 10))
    # ax = ax.ravel()

    ax[0, 0].set_title('input')
    ax[0, 0].imshow(x1_test)

    ax[0, 1].set_title('resized')
    ax[0, 1].imshow(x1_test_resized)
    psnr1 = cv2.PSNR(y1_test, x1_test_resized)
    ax[0, 1].set_xlabel("{:.2f} db".format(psnr1))
    PSNR1 = psnr1+PSNR1


    ax[0, 2].set_title('output')
    ax[0, 2].imshow(y_pred)
    psnr2 = cv2.PSNR(y_pred, y1_test)
    ax[0, 2].set_xlabel("{:.2f} db".format(psnr2))
    PSNR2 = psnr2+PSNR2
    # print(psnr2)

    ax[0, 3].set_title('groundtruth')
    ax[0, 3].imshow(y1_test)



    ax[1, 0].matshow(y_pred_1[0, :, :, 2], cmap="viridis")

    ax[1, 1].matshow(y_pred_2[0, :, :, 2], cmap="viridis")

    ax[1, 2].matshow(y_pred_3[0, :, :, 2], cmap="viridis")

    ax[1, 3].matshow(y_pred_4[0, :, :, 2], cmap="viridis")



    plt.show()

# print("PSNR1 avg:",PSNR1/2000)
# print("PSNR2 avg:",PSNR2/2000)
