import os
import csv


path="C:\Super_Resolution\LP\images_color_gray"


with open('C:\Super_Resolution\LP\list_eval_partition_LP_color_gray.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_id", "partition"])

    file = os.listdir(path)
    print(len(file))
    for i in range(len(file)):
        if i < len(file) * 0.75:
            writer.writerow([file[i], "0"])

        elif len(file) * 0.7 < i < len(file) * 0.95:
            writer.writerow([file[i], "1"])

        else:
            writer.writerow([file[i], "2"])