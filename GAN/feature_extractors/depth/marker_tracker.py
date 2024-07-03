import glob
import os
import cv2
import numpy as np
import matplotlib as mpl
mpl.use("QtAgg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


def nothing(x):
    pass

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


def track_marker(root, grayscale_threshold, x_line, roi, show=False):
    paths = glob.glob(os.path.join(root, "*.png"))
    paths.sort(key=lambda p: float(p.split("_")[-1].replace(".png", "")))

    waveform = []
    for p in tqdm(paths):
        img = cv2.imread(p)
        img = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

        img = cv2.resize(img, None, fx=6, fy=6, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, grayscale_threshold[0], grayscale_threshold[1], cv2.THRESH_BINARY)
        red_mask = np.zeros_like(img)
        red_mask[:, :, 2] = mask

        overlay = cv2.addWeighted(img, .7, red_mask, .3, 0)

        intersection = np.where(mask[:, x_line] != 0)[0][0]
        waveform.append(intersection)

        if show:
            cv2.line(overlay, [x_line, 0], [x_line, overlay.shape[0]], [255, 0, 0], 2)
            cv2.circle(overlay, [x_line, intersection], 2, [0, 255, 0], 2)
            cv2.imshow("image", overlay)
            cv2.imwrite("coil.png", overlay)
            cv2.waitKey(0)

    with open("tracked_coil.pickle", "wb") as f:
        pickle.dump(waveform, f)

    cv2.destroyAllWindows()

    return waveform


roi_coords = []
line_x_pos = 0

def select_roi(event, x, y , flags, param):
    global roi_coords

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_coords = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        roi_coords.append((x, y))

        cv2.rectangle(param, roi_coords[0], roi_coords[1], (0, 255, 0), 2)
        cv2.imshow("Select ROI", param)


def get_roi(img_path):
    global roi_coords

    img = cv2.imread(img_path)

    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", select_roi, img)

    while True:
        cv2.imshow("Select ROI", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return roi_coords


def select_line_position(event, x, y, flags, param):
    global line_x_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        line_x_pos = x

        cv2.line(param, [line_x_pos, 0], [line_x_pos, param.shape[0]], [255, 0, 0], 2)
        cv2.imshow("Select X line", param)


def get_line_position(img_path, roi):
    global line_x_pos

    img = cv2.imread(img_path)
    img = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
    img = cv2.resize(img, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)

    cv2.namedWindow("Select X line")
    cv2.setMouseCallback("Select X line", select_line_position, img)

    while True:
        cv2.imshow("Select X line", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return line_x_pos


def get_grayscale_thresholds(img_path, roi):
    img = cv2.imread(img_path)
    img = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
    img = cv2.resize(img, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)

    cv2.namedWindow("Find thresholds")
    cv2.createTrackbar("Lower", "Find thresholds", 0, 255, nothing)
    cv2.createTrackbar("Upper", "Find thresholds", 0, 255, nothing)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    while True:
        _, mask = cv2.threshold(gray,
                                cv2.getTrackbarPos("Lower", "Find thresholds"),
                                cv2.getTrackbarPos("Upper", "Find thresholds"),
                                cv2.THRESH_BINARY)
        red_mask = np.zeros_like(img)
        red_mask[:, :, 2] = mask
        overlay = cv2.addWeighted(img, .7, red_mask, .3, 0)
        cv2.imshow("Find thresholds", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    lower = cv2.getTrackbarPos("Lower", "Find thresholds")
    upper = cv2.getTrackbarPos("Upper", "Find thresholds")

    cv2.destroyAllWindows()
    return lower, upper


def tracking_pipline(root):
    paths = glob.glob(os.path.join(root, "*.png"))
    roi = get_roi(paths[0])
    lower, upper = get_grayscale_thresholds(paths[0], roi)
    x = get_line_position(paths[0], roi)

    wave = track_marker(root=root,
                 grayscale_threshold=[lower, upper],
                 roi = roi,
                 x_line=x,
                 show=True)

    return wave


def main():
    wave = tracking_pipline(os.path.join("C:", os.sep, "data", "A_raw", "session2 rerun", "rgbd", "rgb"))
    plt.plot(wave)
    plt.show()


if __name__ == '__main__':
    main()