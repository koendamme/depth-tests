import numpy as np
from feature_extractors.us.load_us_data import load_to_memory
from feature_extractors.us.extract_us_wave import get_wave_updated
from feature_extractors.depth.marker_tracker import tracking_pipline
import pickle
import os
from scipy.interpolate import interp1d
from datetime import datetime
import glob
from feature_extractors.heat.waveform_from_csv import read_heat_waveform
import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt


def get_us(us_file_path):
    us_data, us_timestamps = load_to_memory(us_file_path)
    us_data = np.array(us_data)
    us_timestamps = np.array(us_timestamps)
    return us_timestamps, us_data


def get_coil_waveform(rgb_path):
    waveform = tracking_pipline(rgb_path)
    paths = glob.glob(os.path.join(rgb_path, "*.png"))
    timestamps = [float(p.split("_")[-1].replace(".png", "")) for p in paths]
    timestamps.sort()

    return np.array(timestamps), waveform

def synchronize_signals(heat_file_path, us_file_path, rgb_path):
    heat_timestamps, heat_data = read_heat_waveform(heat_file_path)
    us_timestamps, us_data = get_us(us_file_path)
    coil_timestamps, coil_data = get_coil_waveform(rgb_path)
    # with open("tracked_coil_temp.pickle", "wb") as f:
    #     pickle.dump({"ts": coil_timestamps, "coil": coil_data}, f)

    # with open("tracked_coil_temp.pickle", "rb") as f:
    #     temp = pickle.load(f)
    #     coil_timestamps, coil_data = temp["ts"], temp["coil"]

    heat_interpolator = interp1d(heat_timestamps, heat_data, kind='linear', bounds_error=False)
    coil_interpolator = interp1d(coil_timestamps, coil_data, kind='linear', bounds_error=False)

    heat_data_interpolated = heat_interpolator(us_timestamps)
    coil_data_interpolated = coil_interpolator(us_timestamps)

    valid_heat_idxs = np.where(~np.isnan(heat_data_interpolated))[0]
    valid_coil_idxs = np.where(~np.isnan(coil_data_interpolated))[0]
    start = np.max([valid_heat_idxs[0], valid_coil_idxs[0]])
    end = np.min([valid_heat_idxs[-1], valid_coil_idxs[-1]])

    valid_us_data = us_data[start:end, :]
    valid_heat_data = heat_data_interpolated[start:end]
    valid_coil_data = coil_data_interpolated[start:end]
    valid_timestamps = us_timestamps[start:end]

    return {"us": valid_us_data, "heat": valid_heat_data, "coil": valid_coil_data, "ts": valid_timestamps}


def main():
    root_raw = os.path.join("C:", os.sep, "data", "A_raw", "session3 (2 rerun)")

    heat_path = os.path.join(root_raw, "heat", "raw_waveform.csv")
    us_path = os.path.join(root_raw, "us", "session.pickle")
    rgb_path = os.path.join(root_raw, "rgbd", "rgb")
    save_path = "synchronized_surrogates.pickle"

    d = synchronize_signals(heat_path, us_path, rgb_path)
    us_wave = get_wave_updated(d["us"], 300, 1000, smooth=True)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    ax1.set_title("US")
    ax1.plot(d["ts"], us_wave)
    ax2.set_title("Heat")
    ax2.plot(d["ts"], d["heat"])
    ax3.set_title("Coil")
    ax3.plot(d["ts"], d["coil"])
    plt.show()

    plt.show()


if __name__ == '__main__':
    main()