import glob
import os
from datetime import datetime


# dir_path = os.path.join("..", "random_tests", "rgbd_test", "640", "depth")
dir_path = os.path.join("data", "depth")


def main():
    paths = glob.glob(os.path.join(dir_path, "*.png"))

    df = [""] * len(paths)
    for path in paths:
        n = int(path.split("\\")[-1].split("_")[1])
        t = path.split("\\")[-1].split("_")[2].replace(".png", "")
        date_float = float(t)
        df[n] = datetime.fromtimestamp(date_float)

    fps = len(df) / (df[-1] - df[0]).total_seconds()
    print(f"FPS: {round(fps, 2)}")


if __name__ == '__main__':
    main()