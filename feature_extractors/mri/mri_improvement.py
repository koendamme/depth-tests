import pickle
import cv2
import numpy as np
import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import os


def nothing(x):
    pass


def main():
    with open("../mri/mr.pickle", "rb") as file:
        mr = np.array(pickle.load(file)["images"])
        mr = np.clip(mr, a_min=0, a_max=255)

    waveform = []
    for img in mr:
        img = np.uint8(img)
        _, binary_mask = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

        kernel_size = 5  # Adjust this value based on your needs
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        line = cleaned_mask[:, 80]

        border = np.where(line != 0)[0][0]
        waveform.append(border * 1.9)
        color_image = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        cv2.line(color_image, [80, 0], [80, color_image.shape[0]], [0, 0, 255], 2)
        cv2.circle(color_image, (80, border), 3, [255, 0, 0], 2)

        cv2.imshow("Frame", color_image)
        key = cv2.waitKey(20)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    plt.plot(waveform)
    plt.title("Liver Deformation")
    plt.ylabel("Deformation (mm)")
    plt.xlabel("Frame")
    plt.show()




if __name__ == "__main__":
    main()