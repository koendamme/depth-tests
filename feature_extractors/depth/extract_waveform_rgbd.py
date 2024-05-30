import glob
import os
import open3d as o3d
import cv2
import numpy as np
import matplotlib as mpl
from depth_capture.intrinsics import Intrinsics
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import pickle
import pandas as pd
import copy


def select_plane_points(path, scale):
    # Read the image
    depth = np.asanyarray(o3d.io.read_image(path)) * scale
    depth = depth.astype(np.float32)
    # depth = open3d.geometry.Image(depth.astype(np.float32))
    scaled = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    cv2.imshow("Image", scaled)

    positions = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            positions.append((x, y))
            cv2.circle(scaled, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", scaled)

    # Set mouse callback
    cv2.setMouseCallback("Image", mouse_callback)

    # Wait for three clicks
    while len(positions) < 3:
        cv2.waitKey(1)

    # Close the window
    cv2.destroyAllWindows()
    return positions


def apply_bounding_box(pcd):
    # Test 1
    # p1 = p1 + np.array([1, 1, 10])
    # p2 = p2 + np.array([-1, -1, -10])

    # 320x240
    p1 = np.array([0, 0, 2])
    p2 = np.array([1, .5, 5])

    points_to_filter = np.asarray(pcd.points)

    x_bound = (points_to_filter[:, 0] > p1[0]) & (points_to_filter[:, 0] < p2[0])
    z_bound = (points_to_filter[:, 2] > p1[2]) & (points_to_filter[:, 2] < p2[2])
    y_bound = (points_to_filter[:, 1] > p1[1]) & (points_to_filter[:, 1] < p2[1])

    return pcd.select_by_index(np.where(x_bound & z_bound & y_bound)[0])


def get_rotation_and_translation(depth_image, points, intrinsics):
    points3d = []
    for p in points:
        d = depth_image[p[1], p[0]]
        u = p[0]
        v = p[1]

        z = d
        x = (u - intrinsics.cx) * z / intrinsics.fx
        y = (v - intrinsics.cy) * z / intrinsics.fy

        points3d.append([x, y, z])

    points3d = np.array(points3d)

    u_1 = points3d[1] - points3d[2]
    u_2 = points3d[0] - points3d[2]

    n = np.cross(u_1, u_2)
    R = Rotation.align_vectors([0, 1, 0], n)[0].as_matrix()
    return R, points3d[2]


def get_reference(path, rotation, translation, scale, intrinsics):
    depth_reference = np.asanyarray(o3d.io.read_image(path)) * scale
    depth_reference = o3d.geometry.Image(depth_reference.astype(np.float32))
    pcd_reference = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_reference,
                                                                           intrinsic=intrinsics)

    pcd_reference.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_reference.translate(translation)
    pcd_reference.rotate(rotation, center=(0, 0, 0))

    return pcd_reference


def create_pcd(rgb_path, depth_path, scale, intrinsics, translation, rotation):
    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=color_raw,
                                                                    depth=depth_raw,
                                                                    depth_scale=2500,
                                                                    depth_trunc=100)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic=intrinsics)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd.translate(translation)
    pcd.rotate(rotation, center=(0, 0, 0))
    return pcd


def main():
    scale = 0.00025
    depth_paths = glob.glob(r"D:\depth-parameter-experiment\Test11rgbd3\depth\*.png")
    depth_paths.sort(key=lambda p: int(p.split("\\")[-1].replace(".png", "").split("_")[1]))
    rgb_paths = glob.glob(r"D:\depth-parameter-experiment\Test11rgbd3\rgb\*.png")
    rgb_paths.sort(key=lambda p: int(p.split("\\")[-1].replace(".png", "").split("_")[1]))

    # points = select_plane_points(depth_paths[0], scale)
    points = [(178, 16), (204, 124), (393, 17)]
    print(points)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.io.read_image(rgb_paths[0]),
                                                                    depth=o3d.io.read_image(depth_paths[0]),
                                                                    depth_scale=2500,
                                                                    depth_trunc=100)

    npdepth = np.asanyarray(rgbd_image.depth)

    intr = Intrinsics(width=640, height=480)
    R, translation = get_rotation_and_translation(npdepth, points, intr)

    pcd_reference = create_pcd(rgb_paths[0], depth_paths[0], scale, intr.get_intrinsics(), translation, R)
    pcd_reference = apply_bounding_box(pcd_reference)
    # pcd_reference.voxel_down_sample(voxel_size=1)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([pcd_reference, axes])

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    geometry = o3d.geometry.PointCloud()
    vis.add_geometry(geometry)
    vis.add_geometry(axes)

    diff = []
    first_frame = True
    for depth_path, rgb_path in tqdm(zip(depth_paths, rgb_paths), total=len(depth_paths)):
        pcd = create_pcd(rgb_path, depth_path, scale, intr.get_intrinsics(), translation, R)
        pcd = apply_bounding_box(pcd)
        # pcd.voxel_down_sample(voxel_size=0.1)

        dists = pcd.compute_point_cloud_distance(pcd_reference)
        m_dists = np.mean(np.asarray(dists))
        diff.append(m_dists)

        geometry.points = pcd.points
        geometry.colors = pcd.colors

        if first_frame:
            vis.add_geometry(geometry)
            first_frame = False

        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

    plt.plot(diff)
    plt.show()


if __name__ == '__main__':
    main()