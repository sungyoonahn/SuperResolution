import os
import cv2
import random
import numpy as np


path = "C:\\Super_Resolution\\LP\\images_color"


def brightness(img_b, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255] = 255
    hsv[:,:,2] = hsv[:,:,2]*value
    hsv[:,:,2][hsv[:,:,2]>255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img_b = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img_b



def rotate(img_r):
    (h, w) = img_r.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    D = random.randrange(40)
    D = int(D)+340
    # rotate our image by 45 degrees
    M = cv2.getRotationMatrix2D((cX, cY), D, 1.0)
    img_r = cv2.warpAffine(img_r, M, (w, h))
    return img_r

path_list=[]
for file in os.listdir(path):
    file = path+"\\"+file
    path_list.append(file)


x = random.sample(range(0,19766),10000)
for i in range(10000):
    img = cv2.imread(path_list[x[i]])
    img = brightness(img, 2, 3)
    file_name = path_list[x[i]].split("\\")
    file_name[3] = "images_aug"
    img_name = file_name[4].split(".")
    new_name = img_name[0] + "_B.jpg"
    file_name[4] = new_name
    save_path = ("\\".join(file_name))

    cv2.imwrite(save_path, img)
    print(i)


x = random.sample(range(0,19766),10000)
for i in range(10000):
    img = cv2.imread(path_list[x[i]])
    img = rotate(img)
    file_name = path_list[x[i]].split("\\")
    file_name[3] = "images_aug"
    img_name = file_name[4].split(".")
    new_name = img_name[0] + "_R.jpg"
    file_name[4] = new_name
    save_path = ("\\".join(file_name))

    cv2.imwrite(save_path, img)
    print(i)


x = random.sample(range(0,19766),10000)
for i in range(10000):
    img = cv2.imread(path_list[x[i]])
    img = rotate(img)
    img = brightness(img,2,3)
    file_name = path_list[x[i]].split("\\")
    file_name[3] ="images_aug"
    img_name = file_name[4].split(".")
    new_name = img_name[0]+"_B_R.jpg"
    file_name[4] = new_name
    save_path = ("\\".join(file_name))

    cv2.imwrite(save_path,img)
    print(i)


