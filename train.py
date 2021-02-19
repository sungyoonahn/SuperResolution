import warnings
warnings.filterwarnings(action='ignore')

import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.transform import pyramid_expand
from Subpixel import Subpixel
from DataGenerator import DataGenerator


base_path = r'C:\Super_Resolution\LP\processed_color_gray'

x_train_list = sorted(glob.glob(os.path.join(base_path, 'x_train', '*.npy')))
x_val_list = sorted(glob.glob(os.path.join(base_path, 'x_val', '*.npy')))

print(len(x_train_list), len(x_val_list))
print(x_train_list[0])

x_train_list


x1 = np.load(x_train_list[0])
x2 = np.load(x_val_list[0])

print(x1.shape, x2.shape)

plt.subplot(1, 2, 1)
plt.imshow(x1)
plt.subplot(1, 2, 2)
plt.imshow(x2)
plt.show()

train_gen = DataGenerator(list_IDs=x_train_list,
                          labels=None,
                          batch_size=4,
                          dim=(15,50),
                          n_channels=3,
                          n_classes=None,
                          shuffle=True)

val_gen = DataGenerator(list_IDs=x_val_list,
                        labels=None,
                        batch_size=4,
                        dim=(15,50),
                        n_channels=3,
                        n_classes=None,
                        shuffle=False)

upscale_factor = 4

inputs = Input(shape=(15, 50, 3))

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

model.compile(optimizer='adam', loss='mse')

model.summary()


history = model.fit(train_gen, validation_data=val_gen, epochs=100, verbose=1,
                              callbacks=[ModelCheckpoint(r'C:\Super_Resolution\models\model_LPR_color_gray.h5',
                                                         monitor='val_loss',
                                                         verbose=1,
                                                         save_best_only=True)])



# history = model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)
# model.save(r'C:\Super_Resolution\models\model_LPR_gray.h5')