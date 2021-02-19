# Load Library
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import pyramid_reduce

os.chdir('C:\Super_Resolution')
base_path = 'C:\Super_Resolution\LP'
img_base_path = os.path.join(base_path, 'images')
img_save_pth = os.path.join(base_path, 'images_cropped')
target_img_path = os.path.join(base_path, 'processed')

for name in os.listdir("C:\Super_Resolution\LP\images"):
    path = os.path.join(img_base_path, name)
    img_sample = cv2.imread(path)
    h, w, _ = img_sample.shape
    print(img_sample.shape)
    crop_sample = cv2.resize(img_sample, (1600, 1000))
    crop_sample_1 = crop_sample[:, 150:650]
    crop_sample_2 = crop_sample[:, 950:1450]
    new = np.concatenate((crop_sample_1, crop_sample_2), axis=1)
    print(new.shape)

    new_1 = new[0:200, 0:200]
    new_2 = new[200:400, 0:200]
    new_3 = new[400:600, 0:200]
    new_4 = new[600:800, 0:200]
    new_5 = new[800:1000, 0:200]
    new_6 = new[0:200, 200:400]
    new_7 = new[200:400, 200:400]
    new_8 = new[400:600, 200:400]
    new_9 = new[600:800, 200:400]
    new_10 = new[800:1000, 200:400]
    new_11 = new[0:200, 400:600]
    new_12 = new[200:400, 400:600]
    new_13 = new[400:600, 400:600]
    new_14 = new[600:800, 400:600]
    new_15 = new[800:1000, 400:600]
    new_16 = new[0:200, 600:800]
    new_17 = new[200:400, 600:800]
    new_18 = new[400:600, 600:800]
    new_19 = new[600:800, 600:800]
    new_20 = new[800:1000, 600:800]
    new_21 = new[0:200, 800:1000]
    new_22 = new[200:400, 800:1000]
    new_23 = new[400:600, 800:1000]
    new_24 = new[600:800, 800:1000]
    new_25 = new[800:1000, 800:1000]


    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_01.jpg"), new_1)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_02.jpg"), new_2)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_03.jpg"), new_3)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_04.jpg"), new_4)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_05.jpg"), new_5)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_06.jpg"), new_6)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_07.jpg"), new_7)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_08.jpg"), new_8)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_09.jpg"), new_9)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_10.jpg"), new_10)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_11.jpg"), new_11)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_12.jpg"), new_12)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_13.jpg"), new_13)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_14.jpg"), new_14)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_15.jpg"), new_15)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_16.jpg"), new_16)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_17.jpg"), new_17)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_18.jpg"), new_18)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_19.jpg"), new_19)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_20.jpg"), new_20)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_21.jpg"), new_21)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_22.jpg"), new_22)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_23.jpg"), new_23)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_24.jpg"), new_24)
    cv2.imwrite(os.path.join(img_save_pth,name.split(".")[0]+"_25.jpg"), new_25)



# img_sample = cv2.imread(os.path.join(img_base_path,img_path))
#
# h, w, _ = img_sample.shape
# print(img_sample.shape)
# crop_sample = cv2.resize(img_sample,(1600,1000))
# crop_sample_1 = crop_sample[ : , 150:650]
# crop_sample_2 = crop_sample[ : , 950:1450]
#
# print(crop_sample_2.shape)
#
#
# new = np.concatenate((crop_sample_1, crop_sample_2), axis=1)
# print(new.shape)
#
# new_01 = new[0:200, 0:200]
# new_02 = new[200:400, 0:200]
# new_03 = new[400:600, 0:200]
# new_04 = new[600:800, 0:200]
# new_05 = new[800:1000, 0:200]
# new_06 = new[0:200, 200:400]
# new_07 = new[200:400, 200:400]
# new_08 = new[400:600, 200:400]
# new_09 = new[600:800, 200:400]
# new_10 = new[800:1000, 200:400]
# new_11 = new[0:200, 400:600]
# new_12 = new[200:400, 400:600]
# new_13 = new[400:600, 400:600]
# new_14 = new[600:800, 400:600]
# new_15 = new[800:1000, 400:600]
# new_16 = new[0:200, 600:800]
# new_17 = new[200:400, 600:800]
# new_18 = new[400:600, 600:800]
# new_19 = new[600:800, 600:800]
# new_20 = new[800:1000, 600:800]
# new_21 = new[0:200, 800:1000]
# new_22 = new[200:400, 800:1000]
# new_23 = new[400:600, 800:1000]
# new_24 = new[600:800, 800:1000]
# new_25 = new[800:1000, 800:1000]
#
# # test = np.concatenate((new_01,new_02,new_03,new_04,new_05,
# #                        new_06, new_07, new_08, new_09, new_10,
# #                        new_11, new_12, new_13, new_14, new_15,
# #                        new_16, new_17, new_18, new_19, new_20,
# #                        new_21, new_22, new_23, new_24, new_25), axis=1)
# #
# # cv2.imshow("new_image", test)
# #
# #
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# cv2.imwrite(os.path.join(img_save_pth,"new_01.jpg"), new_01)
