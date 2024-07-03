import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from feature_extractors.us.load_us_data import load_to_memory

# data = pd.read_pickle(r"C:\dev\ultrasound\mri_experiment\test1\2024-05-14 10,56,09.pickle")
# data = pd.read_pickle(r"C:\dev\ultrasound\data\2024-05-13 16,46,13.pickle")
# data = pd.read_pickle(r"D:\mri_us_experiments_14-5\us\2024-05-14 11,06,37.pickle")
# data = pd.read_pickle(r"C:\dev\ultrasound\data\test1.pickle")


data, _ = load_to_memory(r"C:\data\A_raw\session2 rerun\us\session.pickle")

for i in range(len(data)):
    row = data[i]
    fig = plt.figure()
    plt.plot(row)
    plt.ylim(-.1, .1)

    # fig = plt.gcf()
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    # Convert RGBA to BGR
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)

    # Display the image in a window using OpenCV
    cv2.imshow('Plot', image_array)
    cv2.waitKey(1)

cv2.destroyAllWindows()
