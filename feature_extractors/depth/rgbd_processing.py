import glob
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from depth_capture.intrinsics import Intrinsics


def main():
    depth_paths = glob.glob(r"D:\depth-parameter-experiment\Test11rgbd2\depth\*.png")
    depth_paths.sort(key=lambda p: int(p.split("\\")[-1].replace(".png", "").split("_")[1]))
    rgb_paths = glob.glob(r"D:\depth-parameter-experiment\Test11rgbd2\rgb\*.png")
    rgb_paths.sort(key=lambda p: int(p.split("\\")[-1].replace(".png", "").split("_")[1]))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    vis.add_geometry(geometry)
    vis.add_geometry(axes)

    first_frame = True
    for depth_path, rgb_path in zip(depth_paths, rgb_paths):
        # depth_raw = np.asanyarray(o3d.io.read_image(depth_path)) * 0.00025
        # depth_raw = o3d.geometry.Image(depth_raw.astype(np.float32))
        depth_raw = o3d.io.read_image(depth_path)
        color_raw = o3d.io.read_image(rgb_path)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=color_raw,
                                                                        depth=depth_raw,
                                                                        depth_scale=2500, depth_trunc=100)

        intr = Intrinsics(width=640, height=480)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic=intr.get_intrinsics())
        # pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, intrinsic=intr.get_intrinsics())
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        geometry.points = pcd.points
        geometry.colors = pcd.colors

        if first_frame:
            vis.add_geometry(geometry)
            first_frame = False

        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()



if __name__ == "__main__":
    main()