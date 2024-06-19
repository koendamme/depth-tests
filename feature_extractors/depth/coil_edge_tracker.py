import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
mpl.use("QtAgg")
import matplotlib.pyplot as plt

def nothing(x):
    pass


def main():
    cv2.namedWindow('Frame')
    cv2.createTrackbar('Threshold1','Frame',0,100, nothing)
    cv2.createTrackbar('Threshold2','Frame',0,200, nothing)

    paths = glob.glob(os.path.join("C:", os.sep, "data", "B_raw", "session1", "rgbd", "rgb", "*.png"))
    paths.sort(key=lambda p: int(p.split("\\")[-1].replace(".png", "").split("_")[1]))

    waveform = []
    for p in tqdm(paths):
        img = cv2.imread(p)
        img = img[230:280, 280:400]
        img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        # edges = cv2.Canny(blurred, cv2.getTrackbarPos('Threshold1','Frame'), cv2.getTrackbarPos('Threshold2', 'Frame'))
        edges = cv2.Canny(blurred, 0, 16)
        img[edges != 0] = [0, 0, 255]
        cv2.line(img, [img.shape[1]//2, 0], [img.shape[1]//2, img.shape[0]], [255, 0, 0], 1)

        v = edges[:, img.shape[1]//2]
        edges_on_line = np.where(v == 255)[0]
        waveform.append(edges_on_line[0])

        cv2.circle(img, [img.shape[1]//2, edges_on_line[0]], 3, [0, 255, 0], 3)
        cv2.imshow("Frame", img)
        cv2.waitKey(1)

    plt.plot(waveform)
    plt.show()


if __name__ == "__main__":
    main()