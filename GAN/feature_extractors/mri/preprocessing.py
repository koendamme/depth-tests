import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch


def main():
    # with open("C:\data\A\mr.pickle", "rb") as f:
    #     mr = np.array(pickle.load(f)["images"])
    #
    # mr = np.clip(mr, a_min=0, a_max=255)
    # enhanced = cv2.addWeighted(mr.copy(), 2, np.zeros(mr.shape, mr.dtype), 0, 0)
    #
    # cv2.imshow("Original", mr[100]/255)
    # cv2.imshow("Enhanced", enhanced[100]/255)
    # cv2.waitKey(0)

    with open("C:\data\A\mr.pickle", 'rb') as file:
        mr = pickle.load(file)["images"]
        mr = np.clip(mr, a_min=0, a_max=255).astype(np.uint8)
        mr = cv2.addWeighted(mr, 2, np.zeros(mr.shape, mr.dtype), 0, 0)
        # mr = torch.tensor(mr, dtype=torch.uint8)
        mr = torch.from_numpy(mr).float()
        # mr = torch.clip(mr, min=0, max=255) * 2 / 255 - 1
        mr = mr * 2 / 255 - 1

        cv2.imshow("Enhanced", (mr[100].numpy() + 1)/2)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()