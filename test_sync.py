import matplotlib.pyplot as plt
import pickle

def main():
    with open("F:\\Formatted_datasets\\D1\\surrogates.pickle", "rb") as file:
        surrogates = pickle.load(file)
        heat = surrogates["heat"]
        coil = surrogates["coil"]

    with open("F:\\Formatted_datasets\\D1\\us_wave_detrended.pickle", "rb") as file:
        us = pickle.load(file)


    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)

    ax1.plot(heat)
    ax2.plot(coil)
    ax3.plot(us)
    plt.show()


if __name__ == "__main__":
    main()