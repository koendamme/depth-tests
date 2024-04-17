import glob
import os
import cv2
import numpy as np

dir = "17-04-1124"

writer = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter("patient A.mp4", writer, 3, (512, 512))

paths = glob.glob(os.path.join(dir, "*.png"))
# print(int(paths[0].split("_")[1].split(".")[0]))

paths.sort(key=lambda x: int(x.split("Epoch")[1].split("_")[0]))
# print(paths)

for i in range(0, 4000, 4):
    fake_0 = cv2.imread(paths[i])
    real_0 = cv2.imread(paths[i+1])

    fake_1 = cv2.imread(paths[i+2])
    real_1 = cv2.imread(paths[i+3])

    row0 = np.concatenate([fake_0, real_0], axis=1)
    row1 = np.concatenate([fake_1, real_1], axis=1)

    final = np.concatenate([row0, row1], axis=0)

    video.write(final)

cv2.destroyAllWindows()
video.release()

