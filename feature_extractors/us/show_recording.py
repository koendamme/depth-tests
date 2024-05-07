import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

data = pd.read_pickle(r"C:\dev\ultrasound\data\pretty_long_cable.pickle")


for i in range(len(data)):
    row = data[i][0]
    fig = plt.figure()
    plt.plot(row)
    plt.ylim(-.05, .05)

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

