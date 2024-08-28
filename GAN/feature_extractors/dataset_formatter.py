from mri.extract_images_from_dicom import extract_images
import os
import pickle
from mri.mri_improvement import get_waveform_from_raw_data
from surrogate_synchronizer import synchronize_signals
import json
import matplotlib.pyplot as plt
from us.madore_wave_extraction import get_wave_from_us
import numpy as np


def main():
    TR = 0.35560
    subject = "G"
    session = 4

    # root_raw = os.path.join("C:", os.sep, "data", f"{subject}_raw", f"session{str(session)}")
    # save_root = os.path.join("C:", os.sep, "data", "Formatted", f"{subject+str(session)}")
    root_raw = f"/Volumes/T9/{subject}_raw/session{str(session)}"
    save_root = f"/Volumes/T9/Formatted_datasets/{subject+str(session)}"

    print(f"Loading data from:    {root_raw}")
    print(f"Saving data to:       {save_root}")

    # heat_path = os.path.join(root_raw, "heat", "raw_waveform.csv")
    # us_path = os.path.join(root_raw, "us", "session2.pickle")
    mr_path = os.path.join(root_raw, "mr")
    # rgb_path = os.path.join(root_raw, "rgbd", "rgb")

    mri_images = extract_images(root_dir=mr_path)
    mri_waveform, thresh, x = get_waveform_from_raw_data(mri_images)

    print(f"Threshold: {thresh}")
    print(f"x-position: {x}")

    plt.plot(mri_waveform)
    plt.show()

    with open(os.path.join(save_root, "mr.pickle"), 'wb') as file:
        pickle.dump({"images": mri_images}, file)

    with open(os.path.join(save_root, "mr_wave.pickle"), 'wb') as file:
        pickle.dump({"mri_waveform": mri_waveform}, file)


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

    # surrogates = synchronize_signals(heat_path, us_path, rgb_path)
    # us_wave = get_wave_from_us(surrogates["us"].T, (0, 1000))

    with open(os.path.join(save_root, "surrogates.pickle"), "rb") as file:
        surrogates = pickle.load(file)

    _, axs = plt.subplots(2, sharex=True)
    axs[0].plot(np.linspace(start=0, stop=len(mri_waveform), num=len(surrogates["coil"])), surrogates["coil"])
    axs[1].plot(np.linspace(start=0, stop=len(mri_waveform), num=len(mri_waveform)), mri_waveform)
    plt.show()

    with open(os.path.join(save_root, "surrogates.pickle"), 'wb') as file:
        pickle.dump(surrogates, file)

if __name__ == '__main__':
    main()