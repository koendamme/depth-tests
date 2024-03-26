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
    filled = flood_fill(thresh, (thresh.shape[0] // 2, thresh.shape[1] - 1), 1, footprint=None, connectivity=None,
                        tolerance=None, in_place=False)
    fill_area = filled - thresh

    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    edge = cv2.filter2D(fill_area, -1, filter)

    new_edge = np.zeros(edge.shape)
    for j in range(edge.shape[0]):
        row = edge[j, :]
        idxs = np.where(row != 0)[0]
        # new_pos = np.sum(idxs) // len(idxs)
        new_pos = np.max(idxs)
        new_edge[j, new_pos] = 1

    return new_edge


filename = os.path.join("datasets", "B", "2", "mr_data.h5")

with h5py.File(filename, "r") as f:
    dataset = f['mr_data']['I']
    ref_edge = None
    final_dists = []
    for i in range(dataset.shape[0]):
        d = dataset[i, 0]
        x_min = np.min(d)
        x_max = np.max(d)

        scaled = (d - x_min) / (x_max - x_min)
        rotated = cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE)

        edge = extract_edge(rotated)

        if i == 0:
            ref_edge = edge.copy()

        dists = []
        for j in range(edge.shape[0]):
            curr_pos = np.where(edge[j] != 0)[0]
            ref_pos = np.where(ref_edge[j] != 0)[0]

            dist = abs(curr_pos - ref_pos)
            dists.append(dist)

        final_dists.append(sum(dists)/len(dists))

        print(i)
        cv2.imshow("image1", rotated)
        cv2.imshow("image2", edge)
        cv2.waitKey(200)

plt.plot(final_dists)
plt.show()
cv2.destroyAllWindows()