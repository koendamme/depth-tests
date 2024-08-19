import pandas as pd
import os
import json
import glob
import cv2
from GAN.feature_extractors.mri.mri_improvement import get_current_border_position
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle


def get_tracking_settings(path):
    with open(path) as file:
        settings = json.load(file)
        threshold = settings["MRI"]["Updated_Waveform_parameters"]["Threshold"]
        x = settings["MRI"]["Updated_Waveform_parameters"]["x"]

    return threshold, x


def main():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 13})
    model = "combined_model"
    # subject = "C3"
    # fig, axs = plt.subplots(nrows=3, ncols=6, sharey=True, figsize=(15, 5))
    for i_subject, subject in enumerate(["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]):
        print(subject)
        breathing_patterns = ["Deep Breathing", "Shallow Breathing", "Regular Breathing", "Half Exhale BH",
                              "Full Exhale BH", "Deep BH"]

        settings_path = f"/Volumes/T9/Formatted_datasets/{subject}/settings.json"
        threshold, x = get_tracking_settings(settings_path)
        ssim_breathing_data = None

        for i, pattern in enumerate(breathing_patterns):
            # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
            image_paths = glob.glob(os.path.join("/Volumes/T9/results", model, subject, pattern, "*.png"))
            real_waveform, fake_waveform, ssims = [], [], []
            image_paths.sort(key=lambda p: int(p.split("/")[-1].split(".")[0]))
            for path in image_paths:
                img = cv2.imread(path)
                assert np.any(img[:, :, 0] == img[:, :, 1]) and np.any(img[:, :, 1] == img[:, :, 2])
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                fake = img_gray[:, img_gray.shape[1] // 2:]
                real = img_gray[:, :img_gray.shape[1] // 2]
                y_real, _ = get_current_border_position(real, threshold, x)
                y_fake, _ = get_current_border_position(fake, threshold, x)
                real_waveform.append(y_real * 1.9)
                fake_waveform.append(y_fake * 1.9)
                ssims.append(ssim(real, fake, data_range=255, win_size=11))

            real_waveform, fake_waveform, ssims = np.array(real_waveform), np.array(fake_waveform), np.array(ssims)

            concatenated = np.concatenate([real_waveform[:, None], ssims[:, None]], axis=1)
            ssim_breathing_data = concatenated if ssim_breathing_data is None else np.concatenate([ssim_breathing_data, concatenated], axis=0)
        print(np.max(ssim_breathing_data[:, 1]))

            # m = np.min(np.concatenate([real_waveform, fake_waveform]))

            # fig.suptitle(f"{pattern}_{subject}")
            # ax1.plot(real_waveform - m, label="Real")
            # ax1.plot(fake_waveform - m, label="Synthetic")
            # ax1.set_ylim([-20, 110])
            # ax2.plot(ssims)
            # plt.show()

            # if i_subject == 0:
            #     axs[i_subject, i].set_title(f"{pattern}")
            # axs[i_subject, i].plot(real_waveform - m, label="Real")
            # axs[i_subject, i].plot(fake_waveform - m, label="Synthetic")
            # if i == 0:
            #     axs[i_subject, i].set_ylabel(f"Subject {subject[0]}")
            # axs[i_subject, i].set_ylim([-20, 110])


    # fig.supxlabel('Frame')
    # fig.supylabel('Liver Displacement (mm)')
    # fig.tight_layout()
    # fig.subplots_adjust(left=0.09,
    #                     bottom=0.1,
    #                     right=0.9,
    #                     top=.93,
    #                     wspace=0.1,
    #                     hspace=0.3)
    # plt.savefig("mae_combined.png")
    # plt.show()

    plt.scatter(ssim_breathing_data[:, 0], ssim_breathing_data[:, 1])
    plt.ylabel("SSIM")
    plt.xlabel("Liver position")
    plt.show()


if __name__ == "__main__":
    main()