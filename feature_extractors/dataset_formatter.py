from mri.extract_images_from_dicom import extract_images
from breath_hold_synchronizer import synchronize
import os
import pickle
from feature_extractors.us.extract_us_wave import get_wave_updated
from feature_extractors.mri.extract_breathing_waveform import extract_waveform_pipeline
from feature_extractors.surrogate_synchronizer import synchronize_signals
import numpy as np
import json


def main():
    TR = 0.35560
    us_freq, mri_freq = 50, 1 / TR
    depth_roi = (200, 1000)

    root_raw = os.path.join("C:", os.sep, "data", "C_raw", "session3")
    save_root = os.path.join("C:", os.sep, "data", "Formatted", "C3")

    heat_path = os.path.join(root_raw, "heat", "raw_waveform.csv")
    us_path = os.path.join(root_raw, "us", "session.pickle")
    mr_path = os.path.join(root_raw, "mr")
    rgb_path = os.path.join(root_raw, "rgbd", "rgb")

    mri_images = extract_images(root_dir=mr_path)
    line_length = 70
    mri_waveform, params = extract_waveform_pipeline(np.array(mri_images).astype(np.uint8),
                                                     line_length=line_length)
                                                     # thresh1=37,
                                                     # thresh2=112,
                                                     # line_middle_position=(136, 64),
                                                     # min_edge_length=107)

    with open(os.path.join(save_root, "settings.json"), "w") as json_file:
        json.dump({
            "MRI": {
                "TR": TR,
                "Waveform parameters": {
                    "Threshold1": params[0],
                    "Threshold2": params[1],
                    "Tracking line position": params[2],
                    "Minimum edge length": params[3],
                    "Tracking line length": line_length
                }
            },
            "US": {
                "ROI": depth_roi
            }
        }, json_file, indent=4)

    if os.path.exists(os.path.join(save_root, "surrogates.pickle")):
        with open(os.path.join(save_root, "surrogates.pickle"), "rb") as file:
            surrogates = pickle.load(file)
    else:
        surrogates = synchronize_signals(heat_path, us_path, rgb_path)
        with open(os.path.join(save_root, "surrogates.pickle"), 'wb') as file:
            pickle.dump(surrogates, file)

    us_waveform = get_wave_updated(surrogates["us"], depth_roi[0], depth_roi[1], smooth=True)
    mr2us = synchronize(mri_waveform, us_waveform, us_freq, mri_freq)

    with open(os.path.join(save_root, "mr.pickle"), 'wb') as file:
        pickle.dump({"images": mri_images}, file)

    with open(os.path.join(save_root, "mr_wave.pickle"), 'wb') as file:
        pickle.dump({"mri_waveform": mri_waveform}, file)
    with open(os.path.join(save_root, "mr2us.pickle"), 'wb') as file:
        pickle.dump({"mr2us": mr2us}, file)





if __name__ == '__main__':
    main()