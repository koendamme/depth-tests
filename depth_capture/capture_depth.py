from depth_camera import DepthCamera
import os
import cv2
import time


def main():
    # d_width, d_height = 1024, 768
    d_width, d_height = 640, 480
    depth_camera = DepthCamera(do_align=False, rgbd=False, d_width=d_width, d_height=d_height)

    i = 0
    while True:
        depth = depth_camera.get_depth_frame()
        cv2.imwrite(os.path.join("C:", os.sep, "dev", "depth-testing", "experiment-13-5", "Test2", "depth", f"frame_{i}_{time.time()}.png"), depth)
        i += 1


if __name__ == '__main__':
    main()
