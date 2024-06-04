import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os.path
import glob
from tqdm import tqdm
from datetime import datetime

# paths = glob.glob(r"D:\depth-parameter-experiment\*.bag")
# fname = int(paths[0].split("\\")[-1].replace(".bag", "").split("_")[-1])
# paths.sort(key=lambda p: int(p.split("\\")[-1].replace(".bag", "").split("_")[-1]))

# test_number = 11
# os.mkdir(os.path.join("D:", os.sep, "depth-parameter-experiment", f"Test{test_number}"))

try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    align = rs.align(rs.stream.depth)

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    # rs.config.enable_device_from_file(config, paths[test_number])
    rs.config.enable_device_from_file(config, r"D:\MRI-28-5\depth\20240528_141637.bag", repeat_playback=False)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

    # Start streaming from file
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    i = 0
    playback.resume()
    print(playback.current_status())
    while playback.current_status() == rs.playback_status.playing:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth = frames.get_depth_frame()
        # depth = rs.temporal_filter().process(depth)
        depth = np.asanyarray(depth.get_data())

        rgb = frames.get_color_frame()
        rgb = np.asanyarray(rgb.get_data())

        root = os.path.join("D:", os.sep, "MRI-28-5", "depth", "session1_no_temperal")
        cv2.imwrite(os.path.join(root, "depth", f"frame_{i}.png"), depth)
        cv2.imwrite(os.path.join(root, "rgb", f"frame_{i}.png"), rgb)
        i += 1

finally:
    pass