from depth_camera import DepthCamera
import numpy as np
import cv2


def main():
    fps = 30
    rgb_width, rgb_height = 640, 480
    d_width, d_height = 640, 480
    depth_camera = DepthCamera(do_align=True, rgbd=True, d_width=d_width, d_height=d_height, rgb_width=rgb_width, rgb_height=rgb_height)

    while True:
        depth, rgb = depth_camera.get_rgbd_frame()

        depth_3d = np.stack((depth, depth, depth), axis=-1)
        concatenated = np.concatenate((depth_3d, rgb/255), axis=1)

        cv2.imshow('current_img', concatenated)
        key = cv2.waitKey(int(round(1000 / fps)))
        if key == 27:
            depth_camera.pipeline.stop()
            break


if __name__ == '__main__':
    main()