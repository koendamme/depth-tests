import pydicom
import glob
import os
import numpy as np
import json
import cv2


def min_max_scale(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def extract_images(root_dir):
    paths = os.listdir(root_dir)
    folders = [folder for folder in paths if os.path.isdir(os.path.join(root_dir, folder))]

    n_frames = len(folders)
    images = []
    img_shape = (192, 192)

    for current_frame in range(3, n_frames + 1):
        path = glob.glob(os.path.join(root_dir, str(current_frame), "DICOM", "*.dcm"))[0]
        try:
            dcm = pydicom.dcmread(path)
            # img = np.array(dcm.pixel_array)
            images.append(dcm.pixel_array.tolist())

        except AttributeError:
            images.append(np.zeros(img_shape).tolist())
            print("No pixel data")

    return images


def main():
    root_dir = os.path.join("D:", os.sep, "mri_us_experiments_14-5", "mri", "session3")
    images = extract_images(root_dir)
    with open(os.path.join(root_dir, "images.json"), 'w') as json_file:
        json.dump({'images': images}, json_file, indent=4)
    print(f"JSON file created successfully.")


if __name__ == "__main__":
    main()





