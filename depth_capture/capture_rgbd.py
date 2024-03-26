from depth_camera import DepthCamera
import os
import cv2
import time


def main():
    rgb_width, rgb_height = 640, 480
    # rgb_width, rgb_height = 640, 480
    # d_width, d_height = 1024, 768
    d_width, d_height = 640, 480
    depth_camera = DepthCamera(do_align=True, rgbd=True, d_width=d_width, d_height=d_height, rgb_width=rgb_width, rgb_height=rgb_height)

    i = 0
    while True:
        depth, rgb = depth_camera.get_rgbd_frame()
        t = time.time()
        # cv2.imwrite(os.path.join("..", "random_tests", "rgbd_test", "1024+640", "depth", f"frame_{i}_{t}.png"), depth)
        # cv2.imwrite(os.path.join("..", "random_tests", "rgbd_test", "1024+640", "rgb", f"frame_{i}_{t}.png"), rgb)

        cv2.imwrite(os.path.join("D:", os.sep, "koen_mri_room", "test6_rgbd", "depth", f"frame_{i}_{t}.png"), depth)
        cv2.imwrite(os.path.join("D:", os.sep, "koen_mri_room", "test6_rgbd", "rgb", f"frame_{i}_{t}.png"), rgb)
        i += 1


if __name__ == '__main__':
    main()