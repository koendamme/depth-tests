import matplotlib as mpl
mpl.use('Qt5Agg')
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


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


def set_vertical_line_position(img):
    global line_x, line_middlepoint_y
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_callback)

    while line_x is None:
        cv2.imshow('Frame', img)
        cv2.waitKey(10)


def extract_waveform(mr_images, save=False, show_result=False):
    line_length = 50
    waveform = []

    set_vertical_line_position(np.uint8(mr_images[0]))
    line_x, line_middlepoint_y = 83, 76
    print(line_x, line_middlepoint_y)
    for i, img in enumerate(mr_images):
        img = np.array(img)
        img = np.uint8(img)

        blurred = cv2.GaussianBlur(img, (15, 15), 2)
        edges = cv2.Canny(blurred, 51, 83)

        color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color_image[edges != 0] = [0, 0, 255]

        color_image = cv2.line(color_image, (line_x, line_middlepoint_y - line_length//2), (line_x, line_middlepoint_y + line_length//2), (0, 255, 0), 1)
        x, y = get_current_liver_border_position(edges, line_x, line_middlepoint_y - line_length//2)
        color_image = cv2.circle(color_image, (x, y), 3, (255, 0, 0), 3)

        waveform.append(y - (line_middlepoint_y - line_length//2))

        cv2.imshow("Frame", color_image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    if save:
        with open("waveform_data.pickle", 'wb') as file:
            pickle.dump(waveform, file)

    if show_result:
        plt.plot(waveform)
        plt.xlabel("Frame")
        plt.ylabel("Distance (pixels)")
        plt.show()

    return waveform


def main():
    json_dir = os.path.join("C:", os.sep, "data", "MRI-28-5", "MRI", "images.json")

    with open(json_dir, 'r') as file:
        data = json.load(file)
    imgs = data['images']
    extract_waveform(imgs, show_result=True, save=True)



if __name__ == '__main__':
    main()