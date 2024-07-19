import pickle
import cv2
import numpy as np
import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import os
import json


def nothing(x):
    pass


def select_line_position(event, x, y, flags, param):
    global line_x_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        line_x_pos = x

        cv2.line(param, [line_x_pos, 0], [line_x_pos, param.shape[0]], [255, 0, 0], 2)
        cv2.imshow("Select X line", param)


def get_line_position(img):
    global line_x_pos

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.namedWindow("Select X line")
    cv2.setMouseCallback("Select X line", select_line_position, color_img)

    while True:
        cv2.imshow("Select X line", color_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return line_x_pos

def get_grayscale_thresholds(img):
    cv2.namedWindow("Find thresholds")
    cv2.createTrackbar("Thresh", "Find thresholds", 0, 255, nothing)

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    while True:
        _, mask = cv2.threshold(img,
                                cv2.getTrackbarPos("Thresh", "Find thresholds"),
                                255,
                                cv2.THRESH_BINARY)

        red_mask = np.zeros_like(color_img)
        red_mask[:, :, 2] = mask
        overlay = cv2.addWeighted(color_img, .7, red_mask, .3, 0)
        cv2.imshow("Find thresholds", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    thresh = cv2.getTrackbarPos("Thresh", "Find thresholds")

    cv2.destroyAllWindows()
    return thresh


def get_current_border_position(img, thresh, x):
    _, binary_mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    line = cleaned_mask[30:, x]

    border = np.where(line != 0)[0][0] + 30

    return border, cleaned_mask


def get_waveform(path):
    with open(os.path.join(path, "mr.pickle"), "rb") as file:
        mr = np.array(pickle.load(file)["images"])
        # mr = np.clip(mr, a_min=0, a_max=255)
        # mr = cv2.addWeighted(mr, 1.7, np.zeros(mr.shape, mr.dtype), 0, 0)

        mr = np.clip(mr, a_min=0, a_max=255).astype(np.uint8)
        mr = cv2.addWeighted(mr, 1.7, np.zeros(mr.shape, mr.dtype), 0, 0)
        # mr = np.uint8(mr)

    thresh = get_grayscale_thresholds(mr[100])
    # thresh = 35
    x = get_line_position(mr[100])

    waveform = []
    for img in mr:
        border, cleaned_mask = get_current_border_position(img, thresh, x)
        
        waveform.append(border*1.9)
        color_image = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        cv2.line(color_image, [x, 0], [x, color_image.shape[0]], [0, 0, 255], 2)
        cv2.circle(color_image, (x, border), 3, [255, 0, 0], 2)
        cv2.imshow("Frame", color_image)
        key = cv2.waitKey(20)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    return waveform, thresh, x


def main():
    s = "B1"
    path = os.path.join("C:", os.sep, "data", "Formatted_datasets", s)
    waveform, thresh, x = get_waveform(path)

    print(thresh, x)
    plt.plot(waveform)
    plt.show()

    # with open(os.path.join(path, "mr_wave_updated.pickle"), "wb") as file:
    #     pickle.dump({"mri_waveform": waveform}, file)

    # # Step 1: Load the JSON file
    # with open(os.path.join(path, "settings.json"), 'r') as file:
    #     data = json.load(file)

    # # Step 2: Modify the data
    # data["MRI"]['Updated_Waveform_parameters'] = {"Threshold": thresh, "x": x}

    # # Step 3: Save the data back to the JSON file, overwriting the existing file
    # with open(os.path.join(path, "settings.json"), 'w') as file:
    #     json.dump(data, file, indent=4)



if __name__ == "__main__":
    main()