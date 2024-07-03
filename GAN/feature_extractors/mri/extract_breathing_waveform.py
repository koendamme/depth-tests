import matplotlib as mpl
mpl.use('Qt5Agg')
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from feature_extractors.mri.extract_images_from_dicom import extract_images


line_x, line_middlepoint_y = None, None

def select_line_position(event, x, y, flags, param):
    global line_x, line_middlepoint_y

    if event == cv2.EVENT_LBUTTONDOWN:
        line_x = x
        line_middlepoint_y = y

        cv2.line(param[0],
                 [line_x, line_middlepoint_y - param[1]//2],
                 [line_x, line_middlepoint_y + param[1]//2],
                 [255, 0, 0],
                 2)
        cv2.imshow("Select X line", param[0])


def get_current_liver_border_position(edge, line_x, line_top_point_y, line_length):
    v = edge[line_top_point_y:line_top_point_y+line_length, line_x]

    edges_on_line = np.where(v == 255)[0]

    if len(edges_on_line) != 0:
        return line_x, line_top_point_y + edges_on_line[0]
    else:
        print("No edges on line")
        return line_x, -1


def get_vertical_line_position(img, line_length):
    global line_x, line_middlepoint_y

    cv2.namedWindow("Select X line")
    cv2.setMouseCallback("Select X line", select_line_position, (img, line_length))

    while True:
        cv2.imshow("Select X line", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return line_x, line_middlepoint_y



def extract_waveform(mr_images, line_length, min_edge_length, canny_thresh1=51, canny_thresh2=83, line_middle_position=(83, 76), save=False, show_result=False):
    waveform = []
    line_x, line_middlepoint_y = line_middle_position
    for i, img in enumerate(mr_images):
        img = np.array(img)
        img = np.uint8(img)

        blurred = cv2.GaussianBlur(img, (15, 15), 2)
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_edges = np.zeros_like(edges)
        for contour in contours:
            if cv2.arcLength(contour, False) >= min_edge_length:
                cv2.drawContours(filtered_edges, [contour], -1, 255, 1)

        color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color_image[filtered_edges != 0] = [0, 0, 255]

        color_image = cv2.line(color_image, (line_x, line_middlepoint_y - line_length//2), (line_x, line_middlepoint_y + line_length//2), (0, 255, 0), 1)
        x, y = get_current_liver_border_position(filtered_edges, line_x, line_middlepoint_y - line_length//2, line_length)
        color_image = cv2.circle(color_image, (x, y), 3, (255, 0, 0), 3)

        waveform.append(y - (line_middlepoint_y - line_length//2))

        cv2.imshow("Frame", color_image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    if save:
        with open("mri_waveform.pickle", 'wb') as file:
            pickle.dump(waveform, file)

    if show_result:
        plt.plot(waveform)
        plt.xlabel("Frame")
        plt.ylabel("Distance (pixels)")
        plt.show()

    return waveform

def nothing(x):
    pass

def find_edge_thresholds(images):
    cv2.namedWindow('image')
    cv2.createTrackbar('Threshold1', 'image', 0, 100, nothing)
    cv2.createTrackbar('Threshold2', 'image', 0, 200, nothing)
    cv2.createTrackbar('Min Length', 'image', 0, 200, nothing)

    index = 0
    while True:
        img = images[index]
        img = np.clip(img, 0, 255)

        blurred_image = cv2.GaussianBlur(img, (15, 15), 2)
        edges = cv2.Canny(blurred_image,
                          cv2.getTrackbarPos('Threshold1', 'image'),
                          cv2.getTrackbarPos('Threshold2', 'image'))

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_edges = np.zeros_like(edges)
        for contour in contours:
            if cv2.arcLength(contour, False) >= cv2.getTrackbarPos('Min Length', 'image'):
                cv2.drawContours(filtered_edges, [contour], -1, 255, 1)

        color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color_image[filtered_edges != 0] = [0, 0, 255]
        cv2.imshow('image', color_image)

        key = cv2.waitKey(20)
        if key == ord('w'):
            index += 1
        elif key == ord('q'):
            break

    thresh1 = cv2.getTrackbarPos('Threshold1', 'image')
    thresh2 = cv2.getTrackbarPos('Threshold2', 'image')
    min_length = cv2.getTrackbarPos('Min Length', 'image')
    cv2.destroyAllWindows()

    return thresh1, thresh2, min_length


def extract_waveform_pipeline(images, thresh1=None, thresh2=None, line_length=50, line_middle_position=None, min_edge_length=0):
    thresh1, thresh2, min_edge_length = find_edge_thresholds(images) if thresh1 is None or thresh2 is None else (thresh1, thresh2, min_edge_length)
    line_middle_position = get_vertical_line_position(images[0], line_length) if line_middle_position is None else line_middle_position

    w = extract_waveform(mr_images=images,
                         line_length=line_length,
                         min_edge_length=min_edge_length,
                         line_middle_position=line_middle_position,
                         canny_thresh1=thresh1,
                         canny_thresh2=thresh2,
                         show_result=True)

    return w, (thresh1, thresh2, line_middle_position, min_edge_length)


def main():
    mr_path = os.path.join("C:", os.sep, "data", "A_raw", "session3 (2 rerun)", "mr")
    mri_images = extract_images(root_dir=mr_path)
    mri_images = np.array(mri_images).astype(np.uint8)

    extract_waveform_pipeline(mri_images)
    # extract_waveform(imgs, show_result=True, save=True)



if __name__ == '__main__':
    main()