import matplotlib.pyplot as plt
import pickle
import os
from matplotlib.backend_bases import MouseButton


def split_dataset(mri_waveform):
    patterns = ["Deep BH", "Shallow Breathing", "Half Exhale BH", "Regular Breathing", "Full Exhale BH", "Deep Breathing"]

    data = {}
    for i, pattern in enumerate(patterns):
        plt.plot(mri_waveform)
        plt.title(pattern)
        if i == 0:
            points = plt.ginput(2, mouse_add=MouseButton.RIGHT, mouse_pop=None, timeout=100)
            data[pattern] = {"start": int(points[0][0]), "end": int(points[1][0])}
        else:
            points = plt.ginput(1, mouse_add=MouseButton.RIGHT, mouse_pop=None, timeout=100)
            data[pattern] = {"start": data[patterns[i-1]]["end"] + 1, "end": int(points[0][0])}

        plt.close()

    return data


def main():
    root = os.path.join("F:", os.sep, "Formatted_datasets")

    for s in ["F4", "G2", "G3", "G4"]:
        with open(os.path.join(root, s, "mr_wave.pickle"), 'rb') as f:
            mri_waveform = pickle.load(f)["mri_waveform"]

        d = split_dataset(mri_waveform)
        with open(os.path.join(root, s, "splits.pickle"), 'wb') as file:
            print(f"Successfully saved splits for subject {s}")
            pickle.dump(d, file)


if __name__ == '__main__':
    main()
