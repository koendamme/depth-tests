import numpy as np
from feature_extractors.heat.extract_heat_wave import extract_heat
from feature_extractors.us.load_us_data import load_to_memory
import pickle
import os
from scipy.interpolate import interp1d


def synchronize_signals(heat_file_path, us_file_path, save_path=None):
    if os.path.exists("heat\heat_wave.pickle"):
        # Load the dictionary from the file
        with open("heat\heat_wave.pickle", 'rb') as file:
            dict = pickle.load(file)
            heat_timestamps, heat_data = dict['ts'], dict['data']
    else:
        heat_timestamps, heat_data = extract_heat(heat_file_path)

        with open("heat\heat_wave.pickle", 'wb') as file:
            pickle.dump({"ts": heat_timestamps, "data": heat_data}, file)

    us_data, us_timestamps = load_to_memory(us_file_path)
    us_data = np.array(us_data)
    us_timestamps = np.array(us_timestamps)
    heat_timestamps = np.array(heat_timestamps)
    heat_data = np.array(heat_data)

    interpolator = interp1d(heat_timestamps, heat_data, kind='linear', bounds_error=False)
    heat_data_interpolated = interpolator(us_timestamps)
    valid_idxs = np.where(~np.isnan(heat_data_interpolated))[0]

    valid_us_data = us_data[valid_idxs, :]
    valid_heat_data = heat_data_interpolated[valid_idxs]
    valid_timestamps = us_timestamps[valid_idxs]

    d = {"us": valid_us_data, "heat": valid_heat_data, "ts": valid_timestamps}

    if save_path is not None:
        with open(save_path, 'wb') as file:
            pickle.dump(d, file)

    return d


def main():
    heat_path = r'C:\Users\kjwdamme\Desktop\Rec-000017.seq'
    us_path = r"C:\data\MRI-28-5\session1.pickle"
    save_path = "synchronized_surrogates.pickle"

    synchronize_signals(heat_path, us_path, save_path)


if __name__ == '__main__':
    main()