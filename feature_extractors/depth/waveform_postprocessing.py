import pickle
import matplotlib as mpl
mpl.use("QtAgg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def main():
    with open("tracked_coil.pickle", "rb") as f:
        tracked_coil = pickle.load(f)
        # smoothed = gaussian_filter1d(tracked_coil, sigma=1)

        plt.plot(tracked_coil)
        # plt.plot(smoothed)
        plt.show()


if __name__ == "__main__":
    main()