import pydicom
import glob
import os
import numpy as np
import cv2
import json


def min_max_scale(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def main():
    json_dir = os.path.join("C:", os.sep, "data", "MRI-28-5", "MRI", "images.json")

    with open(json_dir, 'r') as file:
        data = json.load(file)
        imgs = data['images']

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 15.0
        out = cv2.VideoWriter("video session2.mp4", fourcc, fps, (192, 192), 0)

        current_frame = 0
        paused = False
        while current_frame < len(imgs):
            img = np.array(imgs[current_frame])
            # img = min_max_scale(img) if np.sum(img) != 0 else img
            img = np.uint8(img)
            out.write(img)

            cv2.imshow("Frame", img)
            current_frame = current_frame + 1 if not paused else current_frame
            key = cv2.waitKey(1 if not paused else 0) & 0xFF

            if key == ord('q'):
                break
            elif key == 32:
                paused = not paused
            elif key == ord('d'):
                current_frame = min(current_frame + 1, len(imgs))
            elif key == ord('a'):
                current_frame = max(current_frame - 1, 0)

        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()