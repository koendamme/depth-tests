import pickle
import matplotlib.pyplot as plt


def main():
    path = "/Volumes/T9/Formatted_datasets/A3/mr_wave.pickle"

    with open(path, "rb") as file:
        mr_wave = pickle.load(file)["mri_waveform"]

        plt.plot(mr_wave)
        plt.show()

if __name__ == "__main__":
    main()