
import warnings
warnings.filterwarnings(action='ignore')

import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Input, Activation
from tensorflow.keras.models import Model
from skimage.transform import pyramid_expand
from Subpixel import Subpixel


base_path = r'C:\Super_Resolution\celeba_dataset\processed'


upscale_factor = 4

inputs = Input(shape=(44, 44, 3))

net = Conv2D(filters=64,
             kernel_size=5,
             strides=1,
             padding='same',
             activation='relu')(inputs)

net = Conv2D(filters=64,
             kernel_size=3,
             strides=1,
             padding='same',
             activation='relu')(net)

net = Conv2D(filters=32,
             kernel_size=3,
             strides=1,
             padding='same',
             activation='relu')(net)

net = Conv2D(filters=upscale_factor**2,
             kernel_size=3,
             strides=1,
             padding='same',
             activation='relu')(net)

net = Subpixel(filters=3,
               kernel_size=3,
               r=upscale_factor,
               padding='same')(net)

outputs = Activation('relu')(net)

model = Model(inputs=inputs, outputs=outputs)

model.load_weights(r"C:\Super_Resolution\models\model.h5")

x_test_list = sorted(glob.glob(os.path.join(base_path, 'x_test', '*.npy')))
y_test_list = sorted(glob.glob(os.path.join(base_path, 'y_test', '*.npy')))

print(len(x_test_list), len(y_test_list))
print(y_test_list[0])


test_idx = 80

# 저해상도 이미지(input)
x1_test = np.load(x_test_list[test_idx])

# 저해상도 이미지 확대시킨 이미지
x1_test_resized = pyramid_expand(x1_test, 
                                 4,
                                 multichannel=True) # 색깔 채널 조건 추가.

# 정답 이미지
y1_test = np.load(y_test_list[test_idx])

# 모델이 예측한 이미지(output)
y_pred = model.predict(x1_test.reshape((1, 44, 44, 3)))

print(x1_test.shape, y1_test.shape)

x1_test = (x1_test * 255).astype(np.uint8)
x1_test_resized = (x1_test_resized * 255).astype(np.uint8)
y1_test = (y1_test * 255).astype(np.uint8)
y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1)

x1_test = cv2.cvtColor(x1_test, 
                       cv2.COLOR_BGR2RGB)

x1_test_resized = cv2.cvtColor(x1_test_resized, 
                               cv2.COLOR_BGR2RGB)

y1_test = cv2.cvtColor(y1_test, 
                       cv2.COLOR_BGR2RGB)

y_pred = cv2.cvtColor(y_pred, 
                      cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1,4,figsize=(15, 10))
ax = ax.ravel()

ax[0].set_title('input')
ax[0].imshow(x1_test)


ax[1].set_title('resized')
ax[1].imshow(x1_test_resized)
psnr1 = cv2.PSNR(y1_test,x1_test_resized)
ax[1].set_xlabel(psnr1+"db")

ax[2].set_title('output')
ax[2].imshow(y_pred)

# print(type(y_pred))
# print(type(y1_test))
# psnr2 = cv2.PSNR(y_pred, y1_test)
# ax[2].set_xlabel(psnr2)

ax[3].set_title('groundtruth')
ax[3].imshow(y1_test)

plt.show()