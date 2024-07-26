from mri.extract_images_from_dicom import extract_images
import os
import pickle
from mri.mri_improvement import get_waveform_from_raw_data
from surrogate_synchronizer import synchronize_signals
import json
import matplotlib.pyplot as plt
from us.madore_wave_extraction import get_wave_from_us


def main():
    TR = 0.35560
    subject = "E"
    session = 3

    root_raw = os.path.join("C:", os.sep, "data", f"{subject}_raw", f"session{str(session)}")
    save_root = os.path.join("C:", os.sep, "data", "Formatted", f"{subject+str(session)}")

    print(f"Loading data from:    {root_raw}")
    print(f"Saving data to:       {save_root}")

    heat_path = os.path.join(root_raw, "heat", "waveform.csv")
    us_path = os.path.join(root_raw, "us", "session.pickle")
    mr_path = os.path.join(root_raw, "mr")
    rgb_path = os.path.join(root_raw, "rgbd", "rgb")

    mri_images = extract_images(root_dir=mr_path)
    mri_waveform, thresh, x = get_waveform_from_raw_data(mri_images)

    plt.plot(mri_waveform)
    plt.show()

    with open(os.path.join(save_root, "settings.json"), "w") as json_file:
        json.dump({
            "MRI": {
                "TR": TR,
                "Updated_waveform_parameters": {
                    "Threshold": thresh,
                    "x": x
                }
            },
        }, json_file, indent=4)

    surrogates = synchronize_signals(heat_path, us_path, rgb_path)
    us_wave = get_wave_from_us(surrogates["us"].T, (0, 1000))

    fig, axs = plt.subplots(3, sharex=True)
    axs[0].plot(surrogates["heat"])
    axs[1].plot(surrogates["coil"])
    axs[2].plot(us_wave)
    plt.show()

    with open(os.path.join(save_root, "surrogates.pickle"), 'wb') as file:
        pickle.dump(surrogates, file)

    with open(os.path.join(save_root, "mr.pickle"), 'wb') as file:
        pickle.dump({"images": mri_images}, file)

    with open(os.path.join(save_root, "mr_wave.pickle"), 'wb') as file:
        pickle.dump({"mri_waveform": mri_waveform}, file)


if __name__ == '__main__':
    with open("C:\data\Formatted\E3\surrogates.pickle", "rb") as file:
        surrogates = pickle.load(file)
        print()
    # main()