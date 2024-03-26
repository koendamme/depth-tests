import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time


class DepthCamera(object):
    def __init__(self, do_align=True, rgbd=False, d_width=640, d_height=480, rgb_width=1280, rgb_height=720):
        self.do_align = do_align
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, d_width, d_height, rs.format.z16, 30)
        if rgbd:
            config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, 30)

        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)

    def get_rgbd_frame(self):
        frames = self.pipeline.wait_for_frames()

        if self.do_align:
            frames = self.align.process(frames)

        depth = frames.get_depth_frame()
        depth = np.asanyarray(depth.get_data())

        rgb = frames.get_color_frame()
        rgb = np.asanyarray(rgb.get_data())

        return depth, rgb

    def get_depth_frame(self):
        frames = self.pipeline.wait_for_frames()

        depth = frames.get_depth_frame()
        depth_image = np.asanyarray(depth.get_data())

        return depth_image


def main():
    fps = 60
    depth_range = (.25, 3)
    depth_camera = DepthCamera(depth_range, scale=False, do_align=False, width=1024, height = 768)
    # save_dir = os.path.join("cable_test", "test1")

    record = False
    i = 0
    print(f"Start recording...")
    while True:
        depth = depth_camera.get_frame()

        cv2.imwrite(os.path.join("../cable_test", "test2_1024", f"frame_{i}_{time.time()}.png"), depth)
        i+=1

        # if record:
        #     t = time.time()
        #     cv2.imwrite(os.path.join(save_dir, "depth", f"frame_{t}.png"), depth)
        #     cv2.imwrite(os.path.join(save_dir, "rgb", f"frame_{t}.png"), color)
        #
        # depth_3d = np.stack((depth, depth, depth), axis=-1)
        # concatenated = np.concatenate((depth_3d, color/255), axis=1)
        #
        # cv2.imshow('current_img', concatenated)
        # key = cv2.waitKey(int(round(1000 / fps)))
        # if key == 27:
        #     depth_camera.pipeline.stop()
        #     break
        #
        # if key == ord("a"):
        #     print("Started recording")
        #     record = True
            # cv2.imwrite(os.path.join(f"volume_tests/aruco_volume_test_depth{i}.png"), depth)
            # cv2.imwrite(os.path.join(f"volume_tests/aruco_volume_test_color{i}.png"), color * 255)
            # i+=1


if __name__ == '__main__':
    main()