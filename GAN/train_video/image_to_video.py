import glob
import os
import cv2

dir = "3-4-1057"

writer = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter("video.avi", writer, 3, (512, 256))

paths = glob.glob(os.path.join(dir, "*.png"))
# print(int(paths[0].split("_")[1].split(".")[0]))

paths.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
print(paths)

for path in paths:
    image = cv2.imread(path)
    video.write(image)

cv2.destroyAllWindows()
video.release()

