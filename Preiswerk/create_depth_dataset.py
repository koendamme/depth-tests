import h5py
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import flood_fill


def extract_edge(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0.15, 1, cv2.THRESH_BINARY)
    thresh = thresh[50:180, :]
    cv2.imshow("w3", thresh)
    filled = flood_fill(thresh, (thresh.shape[0] // 2, thresh.shape[1] - 1), 1, footprint=None, connectivity=None,
                        tolerance=None, in_place=False)
    fill_area = filled - thresh

    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    edge = cv2.filter2D(fill_area, -1, filter)

    new_edge = np.zeros(edge.shape)
    for j in range(edge.shape[0]):
        row = edge[j, :]
        idxs = np.where(row != 0)[0]
        new_pos = np.sum(idxs) // len(idxs)
        new_edge[j, new_pos] = 1

    return new_edge


def main():
    patient_paths = [
        # os.path.join("datasets", "A", "1"),
        os.path.join("datasets", "B", "2"),
        # os.path.join("datasets", "H", "2")
    ]



    for p in patient_paths:
        print(p)
        with h5py.File(os.path.join(p, "mr_data.h5"), "r") as f:
            dataset = f['mr_data']['I']
            depth_dataset = np.zeros((dataset.shape[0], 180-50))
            for i in range(dataset.shape[0]):
                d = dataset[i, 0]
                x_min = np.min(d)
                x_max = np.max(d)

                scaled = (d - x_min) / (x_max - x_min)
                rotated = cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE)
                edge = np.zeros(rotated.shape)
                edge[50:180, :] = extract_edge(rotated)

                rgb = np.repeat(rotated[:, :, None], 3, axis=2)
                color_edge = np.stack([np.zeros(edge.shape), np.zeros(edge.shape), edge], axis=-1)

                cv2.imshow("window with edge", rgb + color_edge)

                cv2.waitKey(200)
                for j in range(50, 180):
                    curr_pos = np.where(edge[j] != 0)[0]
                    depth_dataset[i, j - 50] = curr_pos

            with open(os.path.join(p, "depth_data.npy"), "wb") as file:
                np.save(file, depth_dataset)


if __name__ == '__main__':
    main()