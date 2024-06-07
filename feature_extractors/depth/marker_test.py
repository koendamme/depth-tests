import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt



def main():
    paths = glob.glob(os.path.join("D:", os.sep, "MRI-28-5", "depth", "session1_rgb", "*.png"))
    paths.sort(key=lambda p: int(p.split("\\")[-1].replace(".png", "").split("_")[1]))

    old_points = np.array([[(102, 56)]], dtype=np.float32)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    old = cv2.imread(paths[0])[350:450, 550:750]
    old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
    y_values = []
    for p in paths[1:]:
        img = cv2.imread(p)
        img = img[350:450, 550:750]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, img_gray, old_points, None, **lk_params)
        y_values.append(new_points.squeeze()[1])
        old_gray = img_gray.copy()
        old_points = new_points

        x, y = new_points.ravel()
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

        cv2.imshow("Frame", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    plt.plot(y_values)
    plt.show()


if __name__ == "__main__":
    main()