from feature_extractors.mri.extract_images_from_dicom import extract_images
from feature_extractors.mri.extract_breathing_waveform import extract_waveform
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib.backend_bases import MouseButton


def main():
    if not os.path.exists("temp_waveform.pickle"):
        mri_images = extract_images(root_dir=r"C:\data\MRI-28-5\MRI")
        mri_waveform = extract_waveform(mri_images)
        with open('temp_waveform.pickle', 'wb') as f:
            pickle.dump(mri_waveform, f)
    else:
        with open('temp_waveform.pickle', 'rb') as f:
            mri_waveform = pickle.load(f)

    d = split_dataset(mri_waveform)
    print(d)


def split_dataset(mri_waveform):
    patterns = ["Deep BH", "Shallow Breathing", "Half Exhale BH", "Regular Breathing", "Full Exhale BH", "Deep Breathing"]

    data = {}
    for i, pattern in enumerate(patterns):
        plt.plot(mri_waveform)
        plt.title(pattern)
        if i == 0:
            points = plt.ginput(2, mouse_add=MouseButton.RIGHT, mouse_pop=None)
            data[pattern] = {"start": int(points[0][0]), "end": int(points[1][0])}
        else:
            points = plt.ginput(1, mouse_add=MouseButton.RIGHT, mouse_pop=None)
            data[pattern] = {"start": data[patterns[i-1]]["end"] + 1, "end": int(points[0][0])}

        plt.close()

    return data


if __name__ == '__main__':
    main()
