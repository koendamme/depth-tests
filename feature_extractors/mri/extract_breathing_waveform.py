import json
import os
import cv2
import numpy as np
from feature_extractors.mri.utils import preprocess_image
import matplotlib.pyplot as plt


line_x, line_middlepoint_y = None, None


def mouse_callback(event, x, y, flags, param):
    global line_x, line_middlepoint_y
    if event == cv2.EVENT_LBUTTONDOWN:
        line_x = x
        line_middlepoint_y = y


def get_current_liver_border_position(edge, line_x, line_top_point_y):
    v = edge[line_top_point_y:, line_x]

    edges_on_line = np.where(v == 255)[0]

    if len(edges_on_line) != 0:
        return line_x, line_top_point_y + edges_on_line[0]
    else:
        return line_x, -1


def main():
    # Initialize global variables
    global line_x, line_middlepoint_y

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_callback)

    json_dir = os.path.join("D:", os.sep, "mri_us_experiments_14-5", "mri", "session3", "images.json")

    with open(json_dir, 'r') as file:
        data = json.load(file)
    imgs = data['images']

    line_length = 50

    waveform = []

    for i, img in enumerate(imgs):
        img = preprocess_image(img)

        blurred = cv2.GaussianBlur(img, (15, 15), 2)
        edges = cv2.Canny(blurred, 31, 120)

        color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color_image[edges != 0] = [0, 0, 255]

        if line_x is not None:
            color_image = cv2.line(color_image, (line_x, line_middlepoint_y - line_length//2), (line_x, line_middlepoint_y + line_length//2), (0, 255, 0), 1)
            x, y = get_current_liver_border_position(edges, line_x, line_middlepoint_y - line_length//2)
            color_image = cv2.circle(color_image, (x, y), 3, (255, 0, 0), 3)

            waveform.append(y - line_middlepoint_y - line_length//2)

        cv2.imshow("Frame", color_image)
        cv2.waitKey(0 if i == 0 else 1)

    cv2.destroyAllWindows()

    plt.plot(waveform)
    plt.show()


if __name__ == '__main__':
    main()