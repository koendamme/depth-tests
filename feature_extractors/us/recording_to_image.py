import pandas as pd
import numpy as np
import matplotlib as mpl
import cv2
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

def show_window(full_image, min_depth, max_depth, min_time, max_time):
    # full_image = full_image[500:1000, 3800:3900]
    full_image = full_image[min_depth:max_depth, min_time:max_time]
    # full_image = np.clip(full_image, a_min=0, a_max=None)
    max, min = np.max(full_image), np.min(full_image)
    full_image = (full_image - min) / (max - min)

    plt.imshow(full_image, cmap="gray")
    plt.xlabel("Time")
    plt.ylabel("Depth")
    plt.show()


def slide_through_image(full_image, min_depth, max_depth):
    window_width = 1000

    for i in range(0, full_image.shape[1] - window_width, 7):

        curr_window = full_image[min_depth:max_depth, i:i+window_width]
        # curr_window = np.abs(curr_window)
        # curr_window = np.clip(curr_window, a_min=0, a_max=None)

        max, min = np.max(curr_window), np.min(curr_window)

        curr_window = (curr_window - min) / (max - min)

        cv2.imshow("window", curr_window)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


def get_full_image(path):
    data = pd.read_pickle(path)
    times = pd.read_pickle(path.replace(".pickle", "_times.pickle"))

    full_image = np.zeros((len(data), len(data[0][0])))
    for i in range(len(data)):
        full_image[i, :] = data[i][0]

    full_image = full_image.T
    return full_image, times


def main():

    # data = pd.read_pickle(r"C:\dev\ultrasound\mri_experiment\test1\2024-05-14 10,56,09.pickle")
    data = pd.read_pickle(r"C:\dev\ultrasound\mri_experiment\test1\2024-05-14 11,06,37.pickle")

    # data = pd.read_pickle(r"C:\dev\ultrasound\data\2024-05-13 16,46,13.pickle")

    full_image = np.zeros((len(data), len(data[0][0])))
    print(full_image[0, :].shape)
    print(data[0][0].shape)
    for i in range(len(data)):
        full_image[i, :] = data[i][0]

    full_image = full_image.T

    # slide_through_image(full_image, 0, 1000)
    show_window(full_image, 0, 1000, 4400, 5400)


if __name__ == '__main__':
    main()