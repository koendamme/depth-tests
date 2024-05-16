import cv2
import os
import json
from feature_extractors.mri.utils import preprocess_image


def nothing(x):
    pass

def main():
    cv2.namedWindow('image')
    cv2.createTrackbar('Threshold1','image',0,100, nothing)
    cv2.createTrackbar('Threshold2','image',0,200, nothing)

    json_dir = os.path.join("D:", os.sep, "mri_us_experiments_14-5", "mri", "session2", "images.json")

    with open(json_dir, 'r') as file:
        data = json.load(file)

    index = 0
    while True:
        img = data['images'][index]
        img = preprocess_image(img)

        blurred_image = cv2.GaussianBlur(img, (15, 15), 2)
        edges = cv2.Canny(blurred_image, cv2.getTrackbarPos('Threshold1','image'), cv2.getTrackbarPos('Threshold2', 'image'))

        color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        color_image[edges != 0] = [0, 0, 255]

        cv2.imshow('image', color_image)

        index = index + 1 if cv2.waitKey(20) == ord('q') else index


if __name__ == '__main__':
    main()