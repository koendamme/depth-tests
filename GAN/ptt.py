from dataset import CustomDataset
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def main():
    root = os.path.join("C:", os.sep, "data", "Formatted_datasets")
    
    dataset = CustomDataset(root, "A2")

    pattern = "Deep Breathing"
    start = dataset.splits[pattern]["start"]
    end = dataset.splits[pattern]["end"]

    x = dataset.mr_wave[start:end]
    # x_smoothed = gaussian_filter1d(x, 1.2)
    peaks, _ = find_peaks(x, distance=20, prominence=.5)
    troughs, _ = find_peaks(-x, distance=20, prominence=.5)

    shortest_length = min(peaks.shape[0], troughs.shape[0])

    peaks = peaks[:shortest_length]
    troughs = troughs[:shortest_length]

    ptt = (x[peaks]/x[troughs]).mean()
    print(ptt)

    plt.plot(x)
    # plt.plot(x_smoothed)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(troughs, x[troughs], "x")
    plt.show()
        


if __name__ == '__main__':
    main()