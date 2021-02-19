import os
import cv2

path = "C:\\Super_Resolution\\LP\\images"


for name in os.listdir(path):
    name = path+"\\"+name
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    file_name = name.split("\\")
    file_name[3] = "images_gray"
    img_name = file_name[4].split(".")
    new_name = img_name[0] + "_G.jpg"
    file_name[4] = new_name
    save_path = ("\\".join(file_name))
    cv2.imwrite(save_path,gray)
