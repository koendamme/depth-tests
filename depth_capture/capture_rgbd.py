from depth_camera import DepthCamera
import os
import cv2
import time


def main():
    # rgb_width, rgb_height = 1280, 720
    rgb_width, rgb_height = 640, 480
    # d_width, d_height = 1024, 768
    d_width, d_height = 640, 480
    depth_camera = DepthCamera(do_align=False,
                               rgbd=True,
                               d_width=d_width,
                               d_height=d_height,
                               rgb_width=rgb_width,
                               rgb_height=rgb_height)

    i = 0
    while True:
        depth, rgb = depth_camera.get_rgbd_frame()
        t = time.time()
        # cv2.imwrite(os.path.join("..", "random_tests", "rgbd_test", "1024+640", "depth", f"frame_{i}_{t}.png"), depth)
        # cv2.imwrite(os.path.join("..", "random_tests", "rgbd_test", "1024+640", "rgb", f"frame_{i}_{t}.png"), rgb)

        root_dir = os.path.join("C:", os.sep, "data", "E_raw", "session3", "rgbd")
        # root_dir = os.path.join("C:", os.sep, "data", "test3", "rgbd")
        cv2.imwrite(os.path.join(root_dir, "depth", f"frame_{i}_{t}.png"), depth)
        cv2.imwrite(os.path.join(root_dir, "rgb", f"frame_{i}_{t}.png"), rgb)
        i += 1


if __name__ == '__main__':
    main()