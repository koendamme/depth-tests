import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time


class DepthCamera(object):
    def __init__(self, depth_range=None, scale=False, do_align=True):
        self.depth_range = depth_range
        self.do_align = do_align
        self.scale = scale
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()

        if self.do_align:
            frames = self.align.process(frames)

        depth = frames.get_depth_frame()
        color = frames.get_color_frame()

        depth_image = np.asanyarray(depth.get_data())

        if self.scale:
            depth_image = depth_image * self.depth_scale
            depth_image = self._scale_depth_image(depth_image, self.depth_range)

        color_image = np.asanyarray(color.get_data())

        return depth_image, color_image

    def _scale_depth_image(self, depth_image, depth_range):
        mask = (depth_image > depth_range[0]) & (depth_image < depth_range[1])
        depth_in_range = np.zeros(depth_image.shape)
        depth_in_range[mask] = depth_image[mask]

        return (depth_in_range - depth_range[0]) / (depth_range[1] - depth_range[0])



def main():
    fps = 60
    depth_range = (.25, 3)
    depth_camera = DepthCamera(depth_range, scale=False, do_align=True)
    save_dir = os.path.join("techmed_test", "test4")

    i = 0
    start = False
    while True:
        depth, color = depth_camera.get_frame()

        if start:
            t = time.time()
            cv2.imwrite(os.path.join(save_dir, "depth", f"frame_{t}.png"), depth)
            cv2.imwrite(os.path.join(save_dir, "rgb", f"frame_{t}.png"), color)

        depth_3d = np.stack((depth, depth, depth), axis=-1)
        concatenated = np.concatenate((depth_3d, color/255), axis=1)

        cv2.imshow('current_img', concatenated)
        key = cv2.waitKey(int(round(1000 / fps)))
        if key == 27:
            depth_camera.pipeline.stop()
            break

        if key == ord("a"):
            print("Started recording")
            start = True
            # cv2.imwrite(os.path.join(f"volume_tests/aruco_volume_test_depth{i}.png"), depth)
            # cv2.imwrite(os.path.join(f"volume_tests/aruco_volume_test_color{i}.png"), color * 255)
            # i+=1


if __name__ == '__main__':
    main()