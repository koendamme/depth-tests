import glob
import os
import cv2
import numpy as np
import open3d
from depth_capture.intrinsics import Intrinsics
import time


def main():
    # fps = 30
    scale = 0.00025
    paths = glob.glob(os.path.join("D:", os.sep, "techmed_test_2703", "Test4", "depth", "*.png"))
    paths.sort(key=lambda p: int(p.split("\\")[-1].split("_")[1]))
    intr = Intrinsics(width=640, height=480)

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    geometry = open3d.geometry.PointCloud()
    vis.add_geometry(geometry)

    for i, path in enumerate(paths):
        depth = np.asanyarray(open3d.io.read_image(path)) * scale
        depth = open3d.geometry.Image(depth.astype(np.float32))
        pcd = open3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic=intr.get_intrinsics())
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        geometry.points = pcd.points

        if i == 0:
            vis.add_geometry(geometry)

        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()


if __name__ == '__main__':
    main()