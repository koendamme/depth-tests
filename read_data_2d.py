import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
import numpy as np
import cv2
import open3d

def path_to_date(path):
    date_str = path.split("_")[-1].replace(".png", "")
    date_float = float(date_str)
    return datetime.fromtimestamp(date_float)


# paths = glob.glob(os.path.join("techmed_test", "test1", "depth", "*.png"))
#
# start = paths[0]
# finish = paths[-1]
#
# x = path_to_date(start)
# y = path_to_date(finish)
#
# print(y - x)

fps = 30

test_name = "test4"

for path in glob.glob(os.path.join("techmed_test", test_name, "depth", "*.png")):
    filename = path.split("\\")[-1]

    depth = cv2.imread(os.path.join("techmed_test", test_name, "depth", filename))
    rgb = cv2.imread(os.path.join("techmed_test", test_name, "rgb", filename))
    concatenated = np.concatenate((depth, rgb), axis=1)

    cv2.imshow('current_img', concatenated)
    key = cv2.waitKey(int(round(1000 / fps)))
    if key == 27:
        break










