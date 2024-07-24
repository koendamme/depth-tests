import os
import numpy as np
import cv2
import pickle


def main():
    s = "E3"
    file_dir = os.path.join("C:", os.sep, "data", "Formatted_datasets", s, "mr.pickle")

    with open(file_dir, 'rb') as file:
        imgs = pickle.load(file)["images"]

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 15.0
        out = cv2.VideoWriter(f"{s}.mp4", fourcc, fps, (192, 192), 0)

        current_frame = 0
        paused = False
        while current_frame < len(imgs):
            img = imgs[current_frame]
            img = np.clip(img, a_min=0, a_max=255).astype(np.uint8)
            img = cv2.addWeighted(img, 2, np.zeros(img.shape, img.dtype), 0, 0)
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