import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from feature_extractors.us.load_us_data import load_to_memory
from scipy.ndimage import gaussian_filter1d


def get_wave_updated(data, depth_min, depth_max, smooth=False):
    data = np.array(data).T
    data = np.abs(data)
    data = data[depth_min:depth_max]
    index = np.arange(data.shape[0])[None, :]
    a = index @ data
    b = np.sum(data, axis=0)[None, :]
    c = a / b

    c = gaussian_filter1d(c[0], sigma=11) if smooth else c[0]

    return c


def get_wave(path, depth_min, depth_max):
    data = pd.read_pickle(path)

    full_image = np.zeros((len(data), len(data[0][0])))

    for i in range(len(data)):
        full_image[i, :] = data[i][0]

    full_image = full_image.T
    full_image = full_image[depth_min:depth_max]
    # full_image = np.absolute(full_image)
    full_image = np.clip(full_image, 0, None)
    index = np.arange(full_image.shape[0])[None, :]
    a = index @ full_image
    b = np.sum(full_image, axis=0)[None, :]
    c = a/b
    return c[0]


def main():
    # data = pd.read_pickle(r"C:\dev\ultrasound\mri_experiment\test1\2024-05-14 11,06,37.pickle")
    # path = r"C:\data\mri_us_experiments_14-5\us\2024-05-14 11,06,37.pickle"
    # path = r"C:\data\MRI-28-5\session1.pickle"
    # c = get_wave(path, 200, 800)
    # data, _ = load_to_memory(path)
    data, _ = load_to_memory(r"C:\data\A_raw\session3 (2 rerun)\us\session.pickle")
    c = get_wave_updated(data, 300, 1000, smooth=True)

    plt.plot(c)
    plt.show()


if __name__ == '__main__':
    main()

