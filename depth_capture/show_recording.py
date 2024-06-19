import glob
import os
import cv2
import numpy as np


def main():
    fps = 30
    paths = glob.glob(os.path.join("C:", os.sep, "data", "test", "4", "rgbd", "rgb", "*.png"))
    paths.sort(key=lambda p: int(p.split("\\")[-1].split("_")[1]))

    for path in paths:
        img = cv2.imread(path)
        min, max = np.min(img), np.max(img)
        img = (img - min)/(max-min)

        cv2.imshow('current_img', img)
        key = cv2.waitKey(int(round(1000 / fps)))
        if key == 27:
            break


if __name__ == '__main__':
    main()










