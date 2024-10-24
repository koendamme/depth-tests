import os
import numpy as np
import cv2
import pickle


def main():
    s = "A2"
    file_dir = f"F:\\Formatted_datasets\\{s}\\mr.pickle"

    with open(file_dir, 'rb') as file:
        imgs = pickle.load(file)["images"]

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # fps = 1/.35560
        fps = 10
        out = cv2.VideoWriter(f"{s}_preproc.mp4", fourcc, fps, (128, 128), 0)

        max_value = np.max(imgs)
        current_frame = 0
        paused = False
        while current_frame < len(imgs):
            img = imgs[current_frame]
            # img = np.uint8(img/max_value*255)
            img = np.clip(img, a_min=0, a_max=255).astype(np.uint8)
            img = cv2.addWeighted(img, 2, np.zeros(img.shape, img.dtype), 0, 0)
            img = img[:128, 32:160]
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